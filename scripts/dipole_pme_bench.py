#!/usr/bin/env python3
"""Benchmark a direct 4-channel (charge + dipole) PME energy/force kernel.

This is the long-range part of a *linear MACE + l=1 head* hybrid:
the multipoles are predicted by a head (no self-consistent CG), so the
kernel is one 4-channel scatter/FFT/gather plus the position adjoint.

The MACE backbone is not included here; add its measured engine cost
separately (12.3 us/atom FP64 reference at 5,184 atoms).
"""
import json
import math
import sys
import time
from pathlib import Path

import ase.io
import numpy as np
import torch

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
sys.path.insert(0, str(REPO))
from plan_c_4g.qeq import PMECoulomb  # noqa: E402

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    REPO / "bench/plan_d_md_1152_ase_2026-10-26/traj_50ps.xyz")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else (
    REPO / "bench/plan_d_bulk_2026-10-25/results/dipole_pme_bench.json")

atoms = ase.io.read(str(DATA), index=0)
pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, device="cuda",
                   requires_grad=True)
cell = torch.tensor(np.asarray(atoms.cell)[None, :, :], dtype=torch.float64,
                    device="cuda")
batch = torch.zeros(len(atoms), dtype=torch.long, device="cuda")
N = len(atoms)
G = 1
torch.manual_seed(0)
q = (torch.randn(N, dtype=torch.float64, device="cuda") * 0.4).detach()
mu = (torch.randn(N, 3, dtype=torch.float64, device="cuda") * 0.1).detach()

coul = PMECoulomb(sigma=0.6, h_target=0.8)
st = coul.build_stencil(pos.detach(), cell.detach(), batch, G)
plan = coul._plans_by_n(st)
n = plan["n"]
influence = plan["influence"]  # [n1,n2,n3] includes 4pi/k^2 and B-spline factors
volume = float(torch.abs(torch.linalg.det(cell[0])))

# reciprocal vectors (same convention as PMECoulomb._plan)
inv = torch.linalg.inv(cell[0])
m1 = torch.fft.fftfreq(n[0], device="cuda", dtype=torch.float64) * n[0]
m2 = torch.fft.fftfreq(n[1], device="cuda", dtype=torch.float64) * n[1]
m3 = torch.fft.rfftfreq(n[2], device="cuda", dtype=torch.float64) * n[2]
b1, b2, b3 = inv[0], inv[1], inv[2]
k1, k2, k3 = 2*math.pi*m1, 2*math.pi*m2, 2*math.pi*m3
kx = k1[:, None, None]*b1[0] + k2[None, :, None]*b2[0] + k3[None, None, :]*b3[0]
ky = k1[:, None, None]*b1[1] + k2[None, :, None]*b2[1] + k3[None, None, :]*b3[1]
kz = k1[:, None, None]*b1[2] + k2[None, :, None]*b2[2] + k3[None, None, :]*b3[2]


def energy(pos_in):
    st_in = coul.build_stencil(pos_in, cell, batch, G)
    gq = st_in.scatter(q)
    gx = st_in.scatter(mu[:, 0])
    gy = st_in.scatter(mu[:, 1])
    gz = st_in.scatter(mu[:, 2])
    Fq = torch.fft.rfftn(gq, dim=(1, 2, 3))
    Fx = torch.fft.rfftn(gx, dim=(1, 2, 3))
    Fy = torch.fft.rfftn(gy, dim=(1, 2, 3))
    Fz = torch.fft.rfftn(gz, dim=(1, 2, 3))
    D = kx*Fx + ky*Fy + kz*Fz
    S2 = (Fq.real**2 + Fq.imag**2) + (D.real**2 + D.imag**2) \
        - 2.0*(Fq.conj()*D).imag
    e_k = (influence.unsqueeze(0) * S2).sum() / (2.0 * volume)
    # Gaussian self terms (timing-irrelevant but keep the energy physical)
    e_self = 0.5 * coul.self_energy_diag() * (q*q).sum() \
        + (1.0/(3.0*coul.sigma**3*math.sqrt(math.pi))) * (mu*mu).sum()
    return e_k + e_self


# warm-up
for _ in range(3):
    e = energy(pos)
    f = torch.autograd.grad(e, pos)[0]
torch.cuda.synchronize()

times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    e = energy(pos)
    f = torch.autograd.grad(e, pos)[0]
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1e3)
times = np.asarray(times)
res = {
    "data": str(DATA),
    "n_atoms": N,
    "grid": list(n),
    "median_ms": float(np.median(times)),
    "min_ms": float(times.min()),
    "all_ms": times.tolist(),
    "median_us_per_atom": float(np.median(times) / N * 1e3),
    "min_us_per_atom": float(times.min() / N * 1e3),
    "energy_eV": float(e),
    "force_rms": float(f.square().mean().sqrt()),
    "note": "direct 4-channel q+mu PME, no self-consistent CG; add MACE backbone separately",
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
