"""extxyz dataset loading and batching for the Plan C student."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

from .descriptors import DescriptorConfig, species_to_index
from .neighbors import NeighborData, build_neighbor_list

try:
    from ase.io import read as ase_read
except Exception:  # pragma: no cover
    ase_read = None


@dataclass
class Frame:
    pos: torch.Tensor          # [N,3]
    cell: torch.Tensor         # [3,3]
    z: torch.Tensor            # [N] atomic numbers
    zi: torch.Tensor           # [N] model element indices
    energy: float
    forces: torch.Tensor       # [N,3]
    nl_rad: NeighborData
    nl_ang: NeighborData
    charges: Optional[torch.Tensor] = None   # [N], teacher reference charges
    name: str = ""
    info: dict = field(default_factory=dict)
    n_atoms: int = 0

    def __post_init__(self):
        self.n_atoms = int(self.pos.shape[0])


def load_frames(path: str, cfg: DescriptorConfig, max_frames: Optional[int] = None,
                dtype=torch.float64, verbose: bool = False) -> List[Frame]:
    if ase_read is None:
        raise RuntimeError("ASE is required to read extxyz")
    atoms_list = ase_read(path, index=":")
    if max_frames is not None:
        atoms_list = atoms_list[:max_frames]
    frames: List[Frame] = []
    for i, atoms in enumerate(atoms_list):
        pos = torch.tensor(atoms.get_positions(), dtype=dtype)
        cell = torch.tensor(atoms.get_cell().array, dtype=dtype)
        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        energy = atoms.info.get("REF_energy", atoms.info.get("energy", None))
        if energy is None:
            energy = atoms.get_potential_energy()
        if "REF_forces" in atoms.arrays:
            forces = torch.tensor(atoms.arrays["REF_forces"], dtype=dtype)
        elif "forces" in atoms.arrays:
            forces = torch.tensor(atoms.arrays["forces"], dtype=dtype)
        else:
            forces = torch.tensor(atoms.get_forces(), dtype=dtype)
        charges = None
        if "REF_charges" in atoms.arrays:
            charges = torch.tensor(atoms.arrays["REF_charges"], dtype=dtype)
        zi = species_to_index(z)
        nl_rad = build_neighbor_list(pos, cell, cfg.r_cut_rad)
        nl_ang = build_neighbor_list(pos, cell, cfg.r_cut_ang)
        frames.append(Frame(pos=pos, cell=cell, z=z, zi=zi,
                            energy=float(energy), forces=forces,
                            nl_rad=nl_rad, nl_ang=nl_ang, charges=charges,
                            name=str(atoms.info.get("config_type", f"frame{i}")),
                            info=dict(atoms.info)))
        if verbose and (i + 1) % 200 == 0:
            print(f"  loaded {i+1}/{len(atoms_list)}", flush=True)
    return frames


def _merge_nl(nls: Sequence[NeighborData], offsets: Sequence[int],
              n_total: int, batch: torch.Tensor,
              cell: torch.Tensor, device) -> NeighborData:
    kmax = max(nl.nbr.shape[1] for nl in nls)
    centers, nbrs, masks, shifts = [], [], [], []
    for nl, off in zip(nls, offsets):
        k = nl.nbr.shape[1]
        pad = kmax - k
        centers.append(nl.center + off)
        nbrs.append(torch.nn.functional.pad(nl.nbr + off, (0, pad)))
        masks.append(torch.nn.functional.pad(nl.mask, (0, pad), value=False))
        shifts.append(torch.nn.functional.pad(nl.shift, (0, 0, 0, pad)))
    return NeighborData(center=torch.cat(centers).to(device),
                        nbr=torch.cat(nbrs).to(device),
                        mask=torch.cat(masks).to(device),
                        shift=torch.cat(shifts).to(device),
                        batch=batch, n_atoms=n_total, cell=cell)


def collate(frames: Sequence[Frame], indices: Sequence[int], device="cpu"):
    """Merge frames into one disjoint graph.  All frames must share a cell."""
    sel = [frames[i] for i in indices]
    cell0 = sel[0].cell
    for f in sel:
        if not torch.allclose(f.cell, cell0):
            raise ValueError("collate requires identical cells; group bulk frames by cell")
    pos = torch.cat([f.pos for f in sel], dim=0).to(device)
    z = torch.cat([f.z for f in sel], dim=0).to(device)
    zi = torch.cat([f.zi for f in sel], dim=0).to(device)
    batch = torch.cat([torch.full((f.n_atoms,), i, dtype=torch.long)
                       for i, f in enumerate(sel)]).to(device)
    cell = cell0.unsqueeze(0).expand(len(sel), -1, -1).contiguous().to(device)
    offsets = []
    off = 0
    for f in sel:
        offsets.append(off)
        off += f.n_atoms
    nl_rad = _merge_nl([f.nl_rad for f in sel], offsets, off, batch, cell, device)
    nl_ang = _merge_nl([f.nl_ang for f in sel], offsets, off, batch, cell, device)
    energy = torch.tensor([f.energy for f in sel], dtype=pos.dtype, device=device)
    forces = torch.cat([f.forces for f in sel], dim=0).to(device)
    sizes = torch.tensor([f.n_atoms for f in sel], dtype=torch.long, device=device)
    charges = None
    if all(f.charges is not None for f in sel):
        charges = torch.cat([f.charges for f in sel], dim=0).to(device)
    return {"pos": pos, "cell": cell, "z": z, "zi": zi, "batch": batch,
            "nl_rad": nl_rad, "nl_ang": nl_ang, "energy": energy,
            "forces": forces, "charges": charges, "sizes": sizes, "G": len(sel)}


def per_element_reference(frames: Sequence[Frame], indices: Sequence[int]):
    """Least-squares per-element reference energies (eV/atom)."""
    import numpy as np
    A = np.zeros((len(indices), 4))
    b = np.zeros(len(indices))
    for row, i in enumerate(indices):
        f = frames[i]
        for z in f.z.tolist():
            from .descriptors import ELEMENT_INDEX
            A[row, ELEMENT_INDEX[z]] += 1.0
        b[row] = f.energy
    # ridge for elements absent from the training split
    reg = 1e-8 * np.eye(4)
    counts = A.sum(axis=0)
    w, *_ = np.linalg.lstsq(A.T @ A + reg, A.T @ b, rcond=None)
    # for absent elements fall back to the mean per atom
    w = np.where(counts > 0, w, b.mean() / max(A.sum(axis=1).mean(), 1.0))
    return torch.tensor(w, dtype=torch.float64)
