"""Tkatchenko-Scheffler van der Waals dispersion for the 4G student.

Why this module exists
----------------------
The RuNNer 2.0 4G-HDNNP total energy is

    E_4G = sum_i E_i(G_i, q_i) + E_elec(q) + E_vdW(V)         (paper Eq. 5)

with a Hirshfeld-volume-scaled Tkatchenko-Scheffler dispersion term
(paper Sec. II F 1, Eqs. 39-43).  The first implementation in this repository
had the chi/J networks, the QEq solve and E_elec, but **no E_vdW**.  The
density-resolved error table (43 meV/atom / 87 meV/A overall, but 339 meV/A at
rho = 2e-2) is exactly the signature of missing long-range attraction: it grows
with density and it is largest in the force.

This module restores the missing term.  It is an independent PyTorch
implementation, not the RuNNer Fortran code.

Formulation
-----------
    R_a^eff = V_a^(1/3) R_a^free
    C6_aa   = V_a^2 C6_aa,free
    C6_ab   = 2 (C6_aa,free C6_bb,free) / ( (alpha_b/alpha_a) C6_aa,free
                                            + (alpha_a/alpha_b) C6_bb,free ) * V_a V_b
    E_vdW   = - sum_{a<b} C6_ab f_damp(r_ab) r_ab^-6 f_cut(r_ab)
    f_damp  = 1 / (1 + exp(-d (r_ab / (sR R_ab^eff) - 1)))

V_a is the Hirshfeld volume ratio (dimensionless, 1 for the free atom), emitted
by a small per-element network from the same local descriptors used by the rest
of the model.  Because V depends on the geometry, the force carries a
d(V)/dR contribution; the model obtains it by plain autograd, and the QEq force
trick is unaffected because E_vdW does not depend on the charges.
"""
from __future__ import annotations

import torch

# Bohr radius in Angstrom (CODATA 2018), same constant as the rest of the package.
BOHR_TO_ANG = 0.529177210903
HARTREE_TO_EV = 27.211386245988

# Free-atom Tkatchenko-Scheffler parameters, PRL 102, 073005 (2009), Table I.
# C6 in hartree * bohr^6, alpha in bohr^3, R_vdW in bohr.
# Keyed by atomic number; only the elements this student supports.
FREE_ATOM_TS = {
    1: {"C6": 6.50, "alpha": 4.50, "RvdW": 3.10},   # H
    6: {"C6": 46.6, "alpha": 12.0, "RvdW": 3.63},   # C
    7: {"C6": 24.2, "alpha": 7.40, "RvdW": 3.36},   # N
    8: {"C6": 15.6, "alpha": 5.40, "RvdW": 3.19},   # O
}

C6_BOHR6_TO_EV_ANG6 = HARTREE_TO_EV * BOHR_TO_ANG ** 6


def _ts_tables(z_supported, device, dtype):
    """Return (C6_free, alpha_free, RvdW_free) tensors indexed by model element."""
    c6 = torch.tensor([FREE_ATOM_TS[int(z)]["C6"] for z in z_supported],
                      device=device, dtype=dtype)
    alpha = torch.tensor([FREE_ATOM_TS[int(z)]["alpha"] for z in z_supported],
                         device=device, dtype=dtype)
    rvdw = torch.tensor([FREE_ATOM_TS[int(z)]["RvdW"] for z in z_supported],
                        device=device, dtype=dtype) * BOHR_TO_ANG
    return c6, alpha, rvdw


class TkatchenkoSchefflerDispersion(torch.nn.Module):
    """Pairwise C6/r^6 dispersion with Hirshfeld-volume scaling.

    cutoff: real-space cutoff for the dispersion sum (must be converged; the
            teacher's Ewald-based long-range physics extends further, so this
            is a real approximation and is recorded as such).
    sR:     global damping scale (0.94 is the TS default).
    d:      damping steepness (20 is the TS default).
    """

    def __init__(self, z_supported, cutoff: float = 12.0, sR: float = 0.94,
                 d: float = 20.0):
        super().__init__()
        self.z_supported = tuple(int(z) for z in z_supported)
        self.cutoff = float(cutoff)
        self.sR = float(sR)
        self.d = float(d)
        c6, alpha, rvdw = _ts_tables(self.z_supported, "cpu", torch.float64)
        self.register_buffer("c6_free", c6)
        self.register_buffer("alpha_free", alpha)
        self.register_buffer("rvdw_free", rvdw)

    def forward(self, pos, cell, batch, zi, n_atoms_per_graph: int):
        """Return (E_vdw_total, atom_vdw [N]) given per-atom volume ratios.

        pos:   [N,3]
        cell:  [G,3,3]
        batch: [N]
        zi:    [N] model element index
        v_ratio: passed separately by the caller as an argument to `energy`.
        """
        raise NotImplementedError("call energy()")

    def energy(self, pos, cell, batch, zi, v_ratio, neighbor_fn=None,
               max_neighbors: int = 0):
        """Total dispersion energy and its per-atom split.

        A dense all-pairs evaluation inside each graph is used.  For the
        training batches (<= 8 x 256 atoms) this is 262k pairs per graph and is
        cheap; for large single graphs the caller should pass a neighbour list
        via `neighbor_fn`.
        """
        dtype, device = pos.dtype, pos.device
        c6_free = self.c6_free.to(device=device, dtype=dtype)
        alpha_free = self.alpha_free.to(device=device, dtype=dtype)
        rvdw_free = self.rvdw_free.to(device=device, dtype=dtype)

        v = torch.clamp(v_ratio, 0.2, 5.0)
        c6_ii = (v ** 2) * c6_free[zi]                       # [N]
        r_eff = (v.clamp_min(1e-3) ** (1.0 / 3.0)) * rvdw_free[zi]   # [N]

        n = pos.shape[0]
        graph = torch.zeros(n, dtype=torch.long, device=device)
        if batch is not None:
            graph = batch

        e_total = pos.new_zeros(())
        atom_e = pos.new_zeros(n)

        # group atoms by graph id
        order = torch.argsort(graph)
        g_sorted = graph[order]
        bounds = torch.searchsorted(g_sorted, torch.arange(int(graph.max()) + 2,
                                                           device=device))
        for g in range(int(graph.max()) + 1):
            lo, hi = int(bounds[g]), int(bounds[g + 1])
            if hi - lo < 2:
                continue
            idx = order[lo:hi]
            p = pos[idx]
            d = p[:, None, :] - p[None, :, :]
            # minimum image for this graph's cell
            inv = torch.linalg.inv(cell[g])
            fr = d @ inv
            fr = fr - torch.round(fr)
            dv = fr @ cell[g]
            r = torch.linalg.norm(dv, dim=-1)
            iu = torch.triu_indices(len(idx), len(idx), offset=1, device=device)
            ra = r[iu[0], iu[1]]
            ia, ib = idx[iu[0]], idx[iu[1]]
            mask = (ra > 1e-3) & (ra < self.cutoff)
            if not bool(mask.any()):
                continue
            ra = ra[mask]
            ia, ib = ia[mask], ib[mask]

            c6a = c6_ii[ia]
            c6b = c6_ii[ib]
            c6a_free = c6_free[zi[ia]]
            c6b_free = c6_free[zi[ib]]
            aa = alpha_free[zi[ia]]
            ab = alpha_free[zi[ib]]
            # Eq. 41 of the paper (London combination rule with volume scaling)
            denom = (ab / aa) * c6a_free + (aa / ab) * c6b_free
            c6_ab = 2.0 * c6a_free * c6b_free / denom * v[ia] * v[ib]

            r_ab_eff = (r_eff[ia] + r_eff[ib])
            damp = 1.0 / (1.0 + torch.exp(-self.d * (ra / (self.sR * r_ab_eff) - 1.0)))
            # smooth cutoff so the sum is continuous at self.cutoff
            fc = 0.5 * (torch.cos(torch.pi * ra / self.cutoff) + 1.0)
            contrib = -c6_ab * damp * ra.pow(-6) * fc
            # convert hartree*bohr^6 -> eV*Angstrom^6
            contrib = contrib * C6_BOHR6_TO_EV_ANG6
            e_total = e_total + contrib.sum()
            atom_e.index_add_(0, ia, 0.5 * contrib)
            atom_e.index_add_(0, ib, 0.5 * contrib)

        return e_total, atom_e

    def extra_repr(self) -> str:
        return (f"cutoff={self.cutoff} A, sR={self.sR}, d={self.d}, "
                f"elements={self.z_supported}")
