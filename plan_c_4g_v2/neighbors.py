"""Periodic neighbor lists for the Plan C 4G-HDNNP prototype.

Everything is plain PyTorch (FP64-capable, autograd-friendly).  A cell list is
built once per geometry; the *geometry* (distances / displacement vectors) is
recomputed from positions so that forces can be obtained by autograd while the
topology (index arrays, integer lattice shifts) stays fixed.

Conventions
-----------
cell: [3, 3] with rows = lattice vectors a1, a2, a3.
cart = frac @ cell ; frac = cart @ inv(cell).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


class _GeometryFn(torch.autograd.Function):
    """dist/vec with an index_add-based backward.

    PyTorch's generic advanced-indexing backward uses FP64 atomics into the
    position tensor (one entry per edge, many duplicates).  At 5,184 atoms and
    50 neighbours this measured tens of ms; index_add is ~100x faster.
    """

    @staticmethod
    def forward(ctx, pos, nbr, center, shift_cart, mask, n_atoms):
        vec = pos[nbr] - pos[center].unsqueeze(1) + shift_cart
        dist = torch.linalg.norm(vec, dim=-1)
        ctx.save_for_backward(vec, dist, nbr, center, mask)
        ctx.n_atoms = n_atoms
        return dist, vec

    @staticmethod
    def backward(ctx, grad_dist, grad_vec):
        vec, dist, nbr, center, mask = ctx.saved_tensors
        g = grad_vec if grad_vec is not None else torch.zeros_like(vec)
        if grad_dist is not None:
            safe = torch.where(dist > 1e-8, dist, torch.ones_like(dist))
            g = g + grad_dist.unsqueeze(-1) * vec / safe.unsqueeze(-1)
        g = g * mask.unsqueeze(-1)
        gp = torch.zeros(ctx.n_atoms, 3, device=vec.device, dtype=vec.dtype)
        gp.index_add_(0, center, -g.sum(dim=1))
        gp.index_add_(0, nbr.reshape(-1), g.reshape(-1, 3))
        return gp, None, None, None, None, None


@dataclass
class NeighborData:
    center: torch.Tensor      # [N] long, global atom index of the centre
    nbr: torch.Tensor         # [N, K] long, global neighbour indices (padded)
    mask: torch.Tensor        # [N, K] bool
    shift: torch.Tensor       # [N, K, 3] long, lattice shift applied to nbr
    batch: torch.Tensor       # [N] long, graph id of each atom
    n_atoms: int
    cell: torch.Tensor        # [G, 3, 3]

    def geometry(self, pos: torch.Tensor, cell: Optional[torch.Tensor] = None):
        """Return (dist [N,K], vec [N,K,3]) recomputed from positions.

        vec = pos[nbr] - pos[center] + shift @ cell[batch].
        Autograd flows through pos and (if provided) cell.
        """
        if cell is None:
            cell = self.cell
        if cell.dim() == 2:
            cell = cell.unsqueeze(0)
        cell_c = cell[self.batch]                       # [N,3,3] (row vectors)
        shift_cart = torch.einsum("nkj,nji->nki", self.shift.to(pos.dtype), cell_c)
        if pos.requires_grad:
            return _GeometryFn.apply(pos, self.nbr, self.center, shift_cart,
                                     self.mask, self.n_atoms)
        vec = pos[self.nbr] - pos[self.center].unsqueeze(1) + shift_cart
        dist = torch.linalg.norm(vec, dim=-1)
        return dist, vec

    def subset(self, dist: torch.Tensor, rc: float) -> "NeighborData":
        """Return a compacted neighbour list containing only dist < rc.

        ``dist`` must be the geometry of ``self``; no positions are needed.
        """
        keep = self.mask & (dist < rc)
        counts = keep.sum(dim=1)
        kmax = int(counts.max().item())
        n = self.n_atoms
        k = self.nbr.shape[1]
        # rank of each kept entry inside its row
        rank = torch.cumsum(keep.to(torch.int64), dim=1) - 1
        rows = torch.arange(n, device=self.center.device).unsqueeze(1).expand(n, k)
        flat_row = rows[keep]
        flat_rank = rank[keep]
        new_nbr = torch.zeros((n, max(kmax, 1)), dtype=torch.long,
                              device=self.center.device)
        new_shift = torch.zeros((n, max(kmax, 1), 3), dtype=torch.long,
                                device=self.center.device)
        new_mask = torch.zeros((n, max(kmax, 1)), dtype=torch.bool,
                               device=self.center.device)
        new_nbr[flat_row, flat_rank] = self.nbr[keep]
        new_shift[flat_row, flat_rank] = self.shift[keep]
        new_mask[flat_row, flat_rank] = True
        return NeighborData(center=self.center, nbr=new_nbr, mask=new_mask,
                            shift=new_shift, batch=self.batch,
                            n_atoms=self.n_atoms, cell=self.cell)


def _as_cell(cell: torch.Tensor) -> torch.Tensor:
    if cell.dim() == 2:
        return cell.unsqueeze(0)
    return cell


def _cell_grid(cell: torch.Tensor, rc: float):
    """Number of cell-list divisions along each lattice vector."""
    # perpendicular width is the safe choice for triclinic cells
    inv = torch.linalg.inv(cell)
    widths = 1.0 / torch.linalg.norm(inv, dim=0)   # distance between planes
    n = torch.floor(widths / rc).to(torch.long).clamp(min=1)
    return n


def build_neighbor_list(pos: torch.Tensor, cell: torch.Tensor, rc: float,
                        max_neighbors: Optional[int] = None,
                        batch: Optional[torch.Tensor] = None,
                        capacities: Optional[torch.Tensor] = None) -> NeighborData:
    """Build a padded periodic neighbour list for one graph.

    pos : [N,3]; cell : [3,3] or [1,3,3]; batch : optional [N] (all zeros).
    ``capacities`` allows a caller to force the padding width (needed when
    batching several graphs with different neighbour counts).
    """
    device, dtype = pos.device, pos.dtype
    cell = _as_cell(cell).to(device=device, dtype=dtype)
    if cell.shape[0] != 1:
        raise ValueError("build_neighbor_list handles one graph; loop in build_batch")
    cell = cell[0]
    n_atoms = pos.shape[0]
    if batch is None:
        batch = torch.zeros(n_atoms, dtype=torch.long, device=device)

    if n_atoms == 0:
        raise ValueError("empty graph")

    inv_cell = torch.linalg.inv(cell)
    frac = pos @ inv_cell
    n = _cell_grid(cell, rc)                          # [3]
    cid = torch.floor(frac * n).to(torch.long) % n
    ncell = int(n[0] * n[1] * n[2])
    flat_cid = (cid[:, 0] * n[1] + cid[:, 1]) * n[2] + cid[:, 2]

    order = torch.argsort(flat_cid)
    counts = torch.bincount(flat_cid, minlength=ncell)
    max_count = int(counts.max().item())
    starts = torch.cumsum(counts, dim=0) - counts

    # padded per-cell atom table
    cell_atoms = torch.zeros((ncell, max_count), dtype=torch.long, device=device)
    rank = torch.arange(n_atoms, device=device) - starts[flat_cid[order]]
    cell_atoms[flat_cid[order], rank] = order
    cell_mask = torch.arange(max_count, device=device).unsqueeze(0) < counts.unsqueeze(1)

    # 27 neighbour cells; with few divisions the offsets alias, so keep the
    # unique set (otherwise the same atom is visited several times).
    offs = torch.cartesian_prod(
        torch.arange(-1, 2, device=device),
        torch.arange(-1, 2, device=device),
        torch.arange(-1, 2, device=device),
    )
    offs = torch.unique(offs % n.unsqueeze(0), dim=0)
    nx, ny, nz = int(n[0]), int(n[1]), int(n[2])
    cand_idx = []
    cand_mask_list = []
    for off in offs.tolist():
        cn = (cid + torch.tensor(off, device=device)) % n
        fcn = (cn[:, 0] * ny + cn[:, 1]) * nz + cn[:, 2]
        cand_idx.append(cell_atoms[fcn])                       # [N, max_count]
        cand_mask_list.append(cell_mask[fcn])
    cand = torch.cat(cand_idx, dim=1)                          # [N, 27*M]
    cand_mask = torch.cat(cand_mask_list, dim=1)               # [N, 27*M]

    # exact minimum-image geometry for the candidates
    cand_vec0 = pos[cand] - pos.unsqueeze(1)
    frac_d = cand_vec0 @ inv_cell
    frac_round = torch.round(frac_d)
    cand_vec = (frac_d - frac_round) @ cell
    cand_shift = (-frac_round).to(torch.long)
    cand_dist = torch.linalg.norm(cand_vec, dim=-1)
    same = cand == torch.arange(n_atoms, device=device).unsqueeze(1)
    keep = cand_mask & (~same) & (cand_dist < rc) & (cand_dist > 1e-8)

    counts_n = keep.sum(dim=1)
    kmax = int(counts_n.max().item()) if n_atoms else 0
    if capacities is not None:
        kmax = int(capacities)
    if max_neighbors is not None:
        kmax = min(kmax, int(max_neighbors))
    kmax = max(kmax, 1)

    rank_n = torch.cumsum(keep.to(torch.int64), dim=1) - 1
    rows = torch.arange(n_atoms, device=device).unsqueeze(1).expand_as(keep)
    sel = keep & (rank_n < kmax)
    flat_row = rows[sel]
    flat_rank = rank_n[sel]

    nbr = torch.zeros((n_atoms, kmax), dtype=torch.long, device=device)
    shift = torch.zeros((n_atoms, kmax, 3), dtype=torch.long, device=device)
    mask = torch.zeros((n_atoms, kmax), dtype=torch.bool, device=device)
    nbr[flat_row, flat_rank] = cand[sel]
    shift[flat_row, flat_rank] = cand_shift[sel]
    mask[flat_row, flat_rank] = True

    return NeighborData(center=torch.arange(n_atoms, device=device), nbr=nbr,
                        mask=mask, shift=shift, batch=batch, n_atoms=n_atoms,
                        cell=_as_cell(cell))


def build_batch(pos_list, cell_list, rc: float, batch_list=None,
                max_neighbors: Optional[int] = None) -> NeighborData:
    """Build a disjoint padded neighbour list for a list of graphs."""
    if len(pos_list) == 0:
        raise ValueError("no graphs")
    device = pos_list[0].device
    dtype = pos_list[0].dtype
    # first pass to learn the padding width
    per = []
    kmax = 1
    for i, (p, c) in enumerate(zip(pos_list, cell_list)):
        n = build_neighbor_list(p, c, rc, max_neighbors=max_neighbors)
        per.append(n)
        kmax = max(kmax, n.nbr.shape[1])

    centers, nbrs, masks, shifts, batches = [], [], [], [], []
    cells = []
    offset = 0
    for i, (n, c) in enumerate(zip(per, cell_list)):
        k = n.nbr.shape[1]
        pad = kmax - k
        centers.append(n.center + offset)
        nbrs.append(torch.nn.functional.pad(n.nbr + offset, (0, pad)))
        masks.append(torch.nn.functional.pad(n.mask, (0, pad), value=False))
        shifts.append(torch.nn.functional.pad(n.shift, (0, pad, 0, 0)))
        batches.append(torch.full((n.n_atoms,), i, dtype=torch.long, device=device))
        cells.append(_as_cell(c).reshape(1, 3, 3))
        offset += n.n_atoms

    return NeighborData(
        center=torch.cat(centers),
        nbr=torch.cat(nbrs),
        mask=torch.cat(masks),
        shift=torch.cat(shifts),
        batch=torch.cat(batches),
        n_atoms=offset,
        cell=torch.cat(cells, dim=0).to(device=device, dtype=dtype),
    )
