#!/usr/bin/env python3
"""Collect the missing density-coverage tranches.

1. Dense homogeneous: rho=0.035 molecules/A^3, 64-molecule boxes (L=12.23 A,
   above 2*r_cut=12 A), T=300/400/500 K, all six species.
2. Inhomogeneous: liquid-vapour water slabs at T=300/400/500 K (128 molecules,
   cell elongated 3x in z), plus CO2/NH3 at 300 K.

MD proposals use MACE-OFF23 medium (fast, stable); teacher q/mu/E/F labels are
added by a separate MACE-POLAR pass.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import ase.io
import numpy as np
from ase import Atoms, units
from ase.build import molecule as ase_molecule
from ase.md.langevin import Langevin

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
OUT = REPO / "data_20261025_new"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = ("/workspace/group/shared/bza/task3_sampling_database/mlip/models/"
         "MACE-OFF23_medium.model")
SPECIES = ["H2O", "NH3", "CH4", "C2H4", "C2H6", "CO2"]


def build_box(species: str, n_mol: int, rho: float, seed: int) -> Atoms:
    rng = np.random.default_rng(seed)
    mol = ase_molecule(species)
    base = mol.get_positions() - mol.get_positions().mean(axis=0)
    numbers = mol.numbers
    L = (n_mol / rho) ** (1.0 / 3.0)
    # grid placement with jitter: reliable even near the jamming limit
    n_side = int(math.ceil(n_mol ** (1.0 / 3.0)))
    spacing = L / n_side
    sites = [((i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing)
             for i in range(n_side) for j in range(n_side) for k in range(n_side)]
    rng.shuffle(sites)
    centers = [np.asarray(s) + rng.uniform(-0.25 * spacing, 0.25 * spacing, 3)
               for s in sites[:n_mol]]
    pos = []
    for c in centers:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])
        pos.append(base @ R.T + c + rng.normal(scale=0.05, size=base.shape))
    return Atoms(numbers=np.tile(numbers, n_mol),
                 positions=np.vstack(pos), cell=[L, L, L], pbc=True)


def make_slab(species: str, n_mol: int, rho: float, seed: int) -> Atoms:
    a = build_box(species, n_mol, rho, seed)
    L = float(a.cell[0, 0])
    pos = a.get_positions()
    # compress the block into half the box height and centre it
    z = pos[:, 2]
    z = (z - z.min()) / max(z.max() - z.min(), 1e-9) * (L * 0.5)
    pos[:, 2] = z + L * 0.25
    a.set_positions(pos)
    a.set_cell([L, L, 3.0 * L])
    a.set_pbc(True)
    return a


def run_state(atoms, tag, T, steps, dump_every, out_path, calc):
    atoms.calc = calc
    e0 = float(atoms.get_potential_energy())
    dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=T,
                   friction=0.01, fixcm=False)
    t0 = time.time()
    written = 0
    for i in range(steps):
        dyn.run(1)
        if (i + 1) % 100 == 0:
            e = float(atoms.get_potential_energy())
            temp = float(atoms.get_temperature())
            if not np.isfinite(e) or temp > 3000:
                raise SystemExit(f"[collect] unstable {tag} step {i+1}: T={temp} E={e}")
        if (i + 1) >= steps // 2 and (i + 1) % dump_every == 0:
            a = atoms.copy()
            a.info.update({"state_id": tag, "temperature_K": T,
                           "density_mol_A3": atoms.info.get("rho", 0.0),
                           "step": i + 1, "time_ps": (i + 1) * 1e-3})
            # Append across states: never truncate at the first frame of a
            # new state (that silently dropped earlier states pre-2026-10-27).
            ase.io.write(str(out_path), a, format="extxyz",
                         append=out_path.exists())
            written += 1
    print(f"[collect] {tag} T={T} E0={e0:.3f} frames={written} "
          f"wall={time.time()-t0:.1f}s", flush=True)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mode", choices=["all", "dense", "slabs"], default="all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=MODEL, device=args.device,
                          default_dtype="float32")

    dense_path = OUT / "dense_0035_64mol.xyz"
    slab_path = OUT / "slabs_validation.xyz"
    if args.mode in ("all", "dense") and dense_path.exists():
        dense_path.unlink()
    if args.mode in ("all", "slabs") and slab_path.exists():
        slab_path.unlink()

    # dense homogeneous
    for si, sp in enumerate(SPECIES if args.mode in ("all", "dense") else []):
        for T in (300, 400, 500):
            if args.smoke and not (sp == "H2O" and T == 300):
                continue
            atoms = build_box(sp, 64, 0.035, seed=20261027 + si * 10 + T)
            atoms.info["rho"] = 0.035
            run_state(atoms, f"dense-{sp}-0.035-{T}K", T,
                      steps=400 if args.smoke else 2000,
                      dump_every=200, out_path=dense_path, calc=calc)
            if args.smoke:
                return

    # inhomogeneous slabs
    slab_species = [("H2O", 125), ("H2O", 125), ("H2O", 125),
                    ("CO2", 125), ("NH3", 125)]
    slab_T = [300, 400, 500, 300, 300]
    if args.mode not in ("all", "slabs"):
        slab_species, slab_T = [], []
    for (sp, n), T in zip(slab_species, slab_T):
        atoms = make_slab(sp, n, 0.035, seed=20261028 + n + T)
        atoms.info["rho"] = 0.035
        run_state(atoms, f"slab-{sp}-0.035-{T}K", T,
                  steps=3000, dump_every=150, out_path=slab_path, calc=calc)

    print("done", dense_path, slab_path, flush=True)


if __name__ == "__main__":
    main()
