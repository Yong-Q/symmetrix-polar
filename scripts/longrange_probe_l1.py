#!/usr/bin/env python3
"""State-split accuracy comparison: MACE vs +monopole vs +l=1 multipoles.

Uses teacher charges and l=1 atomic dipoles (oracle) and the direct 4-channel
PME kernel.  The fitted intercept and slope absorb unit conventions and the
per-species constant offset; the held-out slope gain is the physical signal.
"""
import json
import math
import sys
from pathlib import Path

import ase.io
import numpy as np
import torch

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
sys.path.insert(0, str(REPO))
from plan_c_4g.qeq import PMECoulomb  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])

frames = ase.io.read(str(DATA), index=":")
print(f"[l1] {len(frames)} frames", flush=True)
calc = MACECalculator(
    model_paths=str(REPO / "bench/plan_d_bulk_2026-10-25/models/bulk/plan_d_bulk.model"),
    device="cpu", default_dtype="float32")
coul = PMECoulomb(sigma=0.6, h_target=0.8)


def reciprocal(cell):
    n = coul._plan(cell[0], cell.device, cell.dtype)["n"]
    inv = torch.linalg.inv(cell[0])
    m1 = torch.fft.fftfreq(n[0], device=cell.device, dtype=cell.dtype) * n[0]
    m2 = torch.fft.fftfreq(n[1], device=cell.device, dtype=cell.dtype) * n[1]
    m3 = torch.fft.rfftfreq(n[2], device=cell.device, dtype=cell.dtype) * n[2]
    b1, b2, b3 = inv[0], inv[1], inv[2]
    k1, k2, k3 = 2*math.pi*m1, 2*math.pi*m2, 2*math.pi*m3
    kx = k1[:, None, None]*b1[0] + k2[None, :, None]*b2[0] + k3[None, None, :]*b3[0]
    ky = k1[:, None, None]*b1[1] + k2[None, :, None]*b2[1] + k3[None, None, :]*b3[1]
    kz = k1[:, None, None]*b1[2] + k2[None, :, None]*b2[2] + k3[None, None, :]*b3[2]
    return kx, ky, kz


def elec_energies(pos, cell, batch, q, mu):
    st = coul.build_stencil(pos, cell, batch, 1)
    gq = st.scatter(q)
    Fq = torch.fft.rfftn(gq, dim=(1, 2, 3))
    plan = coul._plans_by_n(st)
    influence = plan["influence"]
    volume = float(torch.abs(torch.linalg.det(cell[0])))
    e_mono = 0.5 * (influence.unsqueeze(0) * (Fq.real**2 + Fq.imag**2)).sum() / volume
    kx, ky, kz = reciprocal(cell)
    gx, gy, gz = st.scatter(mu[:, 0]), st.scatter(mu[:, 1]), st.scatter(mu[:, 2])
    Fx = torch.fft.rfftn(gx, dim=(1, 2, 3))
    Fy = torch.fft.rfftn(gy, dim=(1, 2, 3))
    Fz = torch.fft.rfftn(gz, dim=(1, 2, 3))
    D = kx*Fx + ky*Fy + kz*Fz
    S2 = (Fq.real**2 + Fq.imag**2) + (D.real**2 + D.imag**2) \
        - 2.0*(Fq.conj()*D).imag
    e_l1 = 0.5 * (influence.unsqueeze(0) * S2).sum() / volume
    return float(e_mono), float(e_l1)


rows = []
for i, atoms in enumerate(frames):
    q = np.asarray(atoms.get_array("REF_charges"), dtype=np.float64)
    mu = np.asarray(atoms.get_array("REF_atomic_dipoles"), dtype=np.float64)
    atoms.calc = calc
    e_mace = float(atoms.get_potential_energy())
    e_ref = float(atoms.info["REF_energy"])
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)
    cell = torch.tensor(np.asarray(atoms.cell)[None, :, :], dtype=torch.float64)
    batch = torch.zeros(len(atoms), dtype=torch.long)
    qt = torch.tensor(q, dtype=torch.float64)
    mt = torch.tensor(mu, dtype=torch.float64)
    e_mono, e_l1 = elec_energies(pos, cell, batch, qt, mt)
    rows.append({
        "species": atoms.info.get("config_type", "").split("_")[0],
        "state_id": str(atoms.info.get("state_id", "unknown")),
        "n_atoms": len(atoms), "e_mace": e_mace, "e_ref": e_ref,
        "e_mono": e_mono, "e_l1": e_l1,
        "r": e_mace - e_ref,
    })
    if (i + 1) % 100 == 0:
        print(f"[l1] {i+1}/{len(frames)}", flush=True)

r = np.array([x["r"] for x in rows])
em = np.array([x["e_mono"] for x in rows])
el = np.array([x["e_l1"] for x in rows])
n = np.array([x["n_atoms"] for x in rows])
sp = np.array([x["species"] for x in rows])
state = np.array([x["state_id"] for x in rows])
states = sorted(set(state))
rng = np.random.default_rng(20261028)
rng.shuffle(states)
fit_states = set(states[: len(states) // 2])
fit = np.array([s in fit_states for s in state])
test = ~fit


def fit_line(x, mask):
    A = np.stack([np.ones(mask.sum()), x[mask]], axis=1)
    c, a = np.linalg.lstsq(A, r[mask], rcond=None)[0]
    return float(c), float(a)


def mae(v, mask):
    return float(np.mean(np.abs(v[mask]) / n[mask]) * 1e3)


per_sp = {}
for s in sorted(set(sp)):
    mf = (sp == s) & fit
    mt = (sp == s) & test
    if mf.sum() < 4 or mt.sum() < 2:
        continue
    c0 = float(np.mean(r[mf]))
    c_m, a_m = fit_line(em, mf)
    c_l, a_l = fit_line(el, mf)
    per_sp[s] = {
        "n": int((sp == s).sum()), "n_test": int(mt.sum()),
        "before": mae(r, mt),
        "offset": mae(r - c0, mt),
        "monopole": mae(r - (c_m + a_m * em), mt),
        "l1": mae(r - (c_l + a_l * el), mt),
        "alpha_mono": a_m, "alpha_l1": a_l,
    }

res = {"n_frames": len(rows), "n_states": len(states),
       "n_fit": int(fit.sum()), "n_test": int(test.sum()),
       "per_species": per_sp}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
