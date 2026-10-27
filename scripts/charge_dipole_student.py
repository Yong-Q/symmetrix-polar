#!/usr/bin/env python3
"""Inference package: MACE student + q/mu heads + canonical frame + Ewald diagnostic.

Important: the MACE student was distilled on the teacher's TOTAL energy, which
already contains the long-range term.  The q/mu head and the Ewald diagnostic
are separate outputs for charge/polarization analysis; do NOT add E_elec to
the MACE total energy unless the backbone is retrained on the short-range
residual.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import ase

REPO = Path(__file__).resolve().parents[1]
GRAPH_LR = REPO.parent / "external-repos/graph_electrostatics-0.4.0"
QH = REPO / "bench/plan_d_bulk_2026-10-25/qmu_heads/qmu_heads.pt"
MR = REPO / "bench/plan_d_bulk_2026-10-25/qmu_heads/mu_ridge.npz"
MODEL = REPO / "bench/plan_d_bulk_2026-10-25/models/bulk/plan_d_bulk.model"


def canonical_rotation(cell):
    """Rotation Q with cell @ Q upper triangular (det +1)."""
    Q, R = np.linalg.qr(np.asarray(cell, dtype=float).T)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


class ChargeDipoleStudent:
    def __init__(self, device="cuda", with_ewald=True):
        from mace.calculators import MACECalculator
        self.device = device
        self.calc = MACECalculator(model_paths=str(MODEL), device=device,
                                   default_dtype="float32")
        self.model = self.calc.models[0]
        self._cap = {}
        self.model.readouts[0].register_forward_hook(
            lambda m, args, out: self._cap.__setitem__("x", args[0].detach()))
        ck = torch.load(QH, map_location="cpu")
        self.qhead = nn.Sequential(nn.Linear(128, 256), nn.SiLU(),
                                   nn.Linear(256, 128), nn.SiLU(),
                                   nn.Linear(128, 1))
        self.qhead.load_state_dict(ck["q_head"])
        self.qhead.to(device).eval()
        self.qm0, self.qs0 = ck["mu0"].to(device), ck["sd0"].to(device)
        rz = np.load(MR)
        self.W = torch.tensor(rz["W_full"], dtype=torch.float32, device=device)
        self.mm = torch.tensor(rz["mu"], dtype=torch.float32, device=device)
        self.ms = torch.tensor(rz["sd"], dtype=torch.float32, device=device)
        self.ewald = None
        if with_ewald:
            import sys
            sys.path.insert(0, str(GRAPH_LR))
            from graph_longrange.energy import GTOElectrostaticEnergy
            from graph_longrange.gto_utils import gto_basis_kspace_cutoff
            sigma = 1.5
            cutoff = 1.5 * gto_basis_kspace_cutoff([sigma], 1)
            self.ewald = GTOElectrostaticEnergy(
                density_max_l=1, density_smearing_width=sigma,
                kspace_cutoff=cutoff, include_self_interaction=True,
                include_pbc_corrections=True).to(device).double()
            self.ewald_cutoff = cutoff

    def canonicalize(self, atoms):
        Qc = canonical_rotation(atoms.cell)
        a = atoms.copy()
        a.set_positions(atoms.get_positions() @ Qc)
        a.set_cell(atoms.cell @ Qc, scale_atoms=False)
        return a, Qc

    def _features(self, atoms):
        atoms.calc = self.calc
        atoms.get_potential_energy()
        x = self._cap["x"]
        n = len(atoms)
        return x[:, :128], x[:, 128:512].reshape(n, 128, 3)

    def predict(self, atoms, canonical=True):
        a = atoms.copy()
        Qc = np.eye(3)
        if canonical:
            a, Qc = self.canonicalize(atoms)
        x0, x1 = self._features(a)
        with torch.no_grad():
            q = self.qhead((x0.float() - self.qm0) / self.qs0).squeeze(-1)
            q = (q.double() / (4 * math.pi)).cpu().numpy()
            F = ((x1.float() - self.mm) / self.ms).transpose(1, 2).reshape(len(x1), -1)
            mu = (F @ self.W).double().cpu().numpy() / (4 * math.pi)
        out = {"q": q, "mu_canonical": mu, "Qc": Qc, "atoms_canonical": a}
        # per-atom dipole in the original global frame (label basis permutation)
        out["mu_global"] = (mu @ np.asarray(Qc).T)
        if self.ewald is not None:
            out["E_elec"] = self.ewald_energy(a, np.concatenate([q[:, None], mu], 1))
        return out

    def ewald_energy(self, atoms, coeffs):
        import sys
        sys.path.insert(0, str(GRAPH_LR))
        from graph_longrange.kspace import compute_k_vectors_flat
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=self.device)
        cell = torch.tensor(np.asarray(atoms.cell), dtype=torch.float64,
                            device=self.device).unsqueeze(0)
        rc = 2 * np.pi * torch.linalg.inv(cell.mT)
        vol = torch.abs(torch.linalg.det(cell))
        kv, kn, kb, k0 = compute_k_vectors_flat(self.ewald_cutoff, cell, rc)
        batch = torch.zeros(len(atoms), dtype=torch.long, device=self.device)
        pbc = torch.ones(1, 3, dtype=torch.bool, device=self.device)
        src = torch.tensor(coeffs, dtype=torch.float64, device=self.device)
        return float(self.ewald(kv, kn, kb, k0, src, pos, batch, vol, pbc).sum())

    def energy_forces(self, atoms):
        a = atoms.copy()
        a.calc = self.calc
        e = float(a.get_potential_energy())
        f = a.get_forces()
        return e, f
