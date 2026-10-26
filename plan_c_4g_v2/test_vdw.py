#!/usr/bin/env python3
"""Numerical checks for the Tkatchenko-Scheffler dispersion term.

Four checks, all of which must pass before the term is trusted in training:

  1. finite-difference forces match autograd forces for E_vdw alone;
  2. V = 1 (free atom) reproduces the analytic dimer C6/r^6 behaviour at long
     range, including the sign;
  3. E_vdw is extensive: doubling a non-interacting (well-separated) box
     doubles the energy;
  4. E_vdw is continuous at the real-space cutoff (no jump when a pair crosses
     it).

Run:  python plan_c_4g_v2/test_vdw.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plan_c_4g_v2.vdw import C6_BOHR6_TO_EV_ANG6, TkatchenkoSchefflerDispersion
from plan_c_4g_v2.descriptors import Z_SUPPORTED

torch.set_default_dtype(torch.float64)


def make_pair(r, box=30.0):
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * box
    batch = torch.zeros(2, dtype=torch.long)
    zi = torch.tensor([0, 0], dtype=torch.long)  # element index 0 = hydrogen (Z=1)
    return pos, cell.unsqueeze(0), batch, zi


def test_forces():
    vdw = TkatchenkoSchefflerDispersion(Z_SUPPORTED, cutoff=12.0)
    pos, cell, batch, zi = make_pair(3.2)
    pos = pos.clone().requires_grad_(True)
    v = torch.tensor([1.0, 1.2], dtype=torch.float64)
    e, atom = vdw.energy(pos, cell, batch, zi, v)
    g = torch.autograd.grad(e, pos)[0]

    eps = 1e-5
    fd = np.zeros((2, 3))
    for i in range(2):
        for k in range(3):
            p = pos.detach().clone()
            p[i, k] += eps
            ep, _ = vdw.energy(p, cell, batch, zi, v)
            p[i, k] -= 2 * eps
            em, _ = vdw.energy(p, cell, batch, zi, v)
            fd[i, k] = float((ep - em) / (2 * eps))
    err = float(np.max(np.abs(fd - g.detach().numpy())))
    print(f"[1] max |FD - autograd| = {err:.3e} eV/A")
    assert err < 1e-6, err
    # sum of per-atom split must equal the total
    split_err = float(abs(atom.sum() - e))
    print(f"[1b] |sum(atom) - E| = {split_err:.3e} eV")
    assert split_err < 1e-10


def test_asymptotic():
    cutoff = 20.0
    vdw = TkatchenkoSchefflerDispersion(Z_SUPPORTED, cutoff=cutoff)
    r = 8.0
    pos, cell, batch, zi = make_pair(r, box=40.0)
    v = torch.ones(2, dtype=torch.float64)
    e, _ = vdw.energy(pos, cell, batch, zi, v)
    # analytic: -C6_HH * r^-6 in eV A^6, times the smooth cutoff factor.
    # The damping function is ~1 at this separation (verified below).
    from plan_c_4g_v2.vdw import FREE_ATOM_TS
    c6_hh_free = FREE_ATOM_TS[1]["C6"]
    fc = 0.5 * (np.cos(np.pi * r / cutoff) + 1.0)
    expected = -c6_hh_free * C6_BOHR6_TO_EV_ANG6 / r ** 6 * fc
    ratio = float(e) / expected
    print(f"[2] E(8 A) = {float(e):.6e} eV, analytic(-C6/r^6 * fcut) "
          f"{expected:.6e}, ratio {ratio:.4f} (fcut {fc:.4f})")
    assert float(e) < 0, "dispersion must be attractive"
    assert 0.999 < ratio < 1.001, ratio


def test_extensive():
    vdw = TkatchenkoSchefflerDispersion(Z_SUPPORTED, cutoff=12.0)
    pos1, cell, batch, zi = make_pair(3.0, box=40.0)
    e1, _ = vdw.energy(pos1, cell, batch, zi, torch.ones(2, dtype=torch.float64))
    # second copy 20 A away (beyond the cutoff): non-interacting pair
    pos2 = torch.cat([pos1, pos1 + torch.tensor([20.0, 0.0, 0.0],
                                                dtype=torch.float64)], dim=0)
    batch2 = torch.zeros(4, dtype=torch.long)
    zi2 = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    cell2 = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 80.0
    # note: a bigger box changes the pair distance via minimum image only if
    # the pair is nearer the boundary; 3 A and 20 A are both well inside 80 A.
    e2, _ = vdw.energy(pos2, cell2, batch2, zi2,
                       torch.ones(4, dtype=torch.float64))
    ratio = float(e2) / float(e1)
    print(f"[3] E(2 copies) / E(1 copy) = {ratio:.6f}")
    assert 1.95 < ratio < 2.05, ratio


def test_cutoff_continuity():
    vdw = TkatchenkoSchefflerDispersion(Z_SUPPORTED, cutoff=6.0)
    vals = []
    for r in (5.999, 6.0, 6.001):
        pos, cell, batch, zi = make_pair(r, box=30.0)
        e, _ = vdw.energy(pos, cell, batch, zi, torch.ones(2, dtype=torch.float64))
        vals.append(float(e))
    jump = abs(vals[2] - vals[0])
    print(f"[4] E(6.001) - E(5.999) = {jump:.3e} eV "
          f"(values {vals[0]:.3e}, {vals[1]:.3e}, {vals[2]:.3e})")
    assert jump < 1e-9, jump


if __name__ == "__main__":
    test_forces()
    test_asymptotic()
    test_extensive()
    test_cutoff_continuity()
    print("all vdW checks passed")
