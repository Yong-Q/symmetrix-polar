#!/usr/bin/env python3
"""State-split long-range residual probe.

Improvements over v1:
  * split by state_id (whole trajectories), not even/odd frames, so smooth
    time-series correlation cannot leak across the split;
  * fit both an intercept and a slope: r = c + alpha * E_elec.  The intercept
    is a legitimate per-species offset correction; the slope measures the
    configuration-dependent long-range signal.
  * report offset-only, slope-only (centred), and combined held-out MAE.
"""
import json
import sys
from collections import defaultdict
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
print(f"[probe] {len(frames)} frames", flush=True)
calc = MACECalculator(
    model_paths=str(REPO / "bench/plan_d_bulk_2026-10-25/models/bulk/plan_d_bulk.model"),
    device="cpu", default_dtype="float32")
coul = PMECoulomb(sigma=0.6, h_target=0.8)

rows = []
for i, atoms in enumerate(frames):
    q = np.asarray(atoms.get_array("REF_charges"), dtype=np.float64)
    if q.size != len(atoms):
        continue
    atoms.calc = calc
    e_mace = float(atoms.get_potential_energy())
    e_ref = float(atoms.info["REF_energy"])
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)
    cell = torch.tensor(np.asarray(atoms.cell)[None, :, :], dtype=torch.float64)
    batch = torch.zeros(len(atoms), dtype=torch.long)
    qt = torch.tensor(q, dtype=torch.float64)
    st = coul.build_stencil(pos, cell, batch, 1)
    ecq = coul.apply_stencil(st, qt)
    e_elec = 0.5 * float((qt * ecq).sum()) + 0.5 * coul.self_energy_diag() * float((qt * qt).sum())
    rows.append({
        "index": i,
        "species": atoms.info.get("config_type", "").split("_")[0],
        "state_id": str(atoms.info.get("state_id", atoms.info.get("traj_id", "unknown"))),
        "n_atoms": len(atoms),
        "e_mace": e_mace,
        "e_ref": e_ref,
        "e_elec": e_elec,
        "r_eV": e_mace - e_ref,
    })
    if (i + 1) % 100 == 0:
        print(f"[probe] {i+1}/{len(frames)}", flush=True)

r = np.array([x["r_eV"] for x in rows])
e = np.array([x["e_elec"] for x in rows])
n = np.array([x["n_atoms"] for x in rows])
sp = np.array([x["species"] for x in rows])
state = np.array([x["state_id"] for x in rows])

states = sorted(set(state))
rng = np.random.default_rng(20261027)
rng.shuffle(states)
fit_states = set(states[: len(states) // 2])
fit = np.array([s in fit_states for s in state])
test = ~fit

def fit_line(mask):
    """r = c + alpha e, least squares."""
    if mask.sum() < 2:
        return 0.0, 0.0
    A = np.stack([np.ones(mask.sum()), e[mask]], axis=1)
    c, a = np.linalg.lstsq(A, r[mask], rcond=None)[0]
    return float(c), float(a)

def mae(v):
    return float(np.mean(np.abs(v)) * 1e3)

per_sp = {}
for s in sorted(set(sp)):
    m_fit = (sp == s) & fit
    m_test = (sp == s) & test
    if m_fit.sum() < 4 or m_test.sum() < 2:
        continue
    c, a = fit_line(m_fit)
    # offset-only: c fitted on train, no slope
    c0 = float(np.mean(r[m_fit] - 0.0 * e[m_fit]))
    # centred slope: remove train means, fit slope without intercept
    rc = r[m_fit] - r[m_fit].mean()
    ec = e[m_fit] - e[m_fit].mean()
    a_c = float(np.sum(rc * ec) / np.sum(ec * ec)) if np.sum(ec * ec) > 0 else 0.0
    r_test = r[m_test]
    e_test = e[m_test]
    n_test = n[m_test]
    per_sp[s] = {
        "n": int((sp == s).sum()),
        "n_test": int(m_test.sum()),
        "n_states": len(set(state[sp == s])),
        "corr_centred": float(np.corrcoef(r[m_test] - r[m_fit].mean(),
                                          e[m_test] - e[m_fit].mean())[0, 1]),
        "c_eV": c,
        "alpha": a,
        "offset_only_mae_meV_atom": float(np.mean(np.abs(r_test - c0) / n_test) * 1e3),
        "slope_only_mae_meV_atom": float(np.mean(np.abs(r_test - r[m_fit].mean()
                                                        - a_c * (e_test - e[m_fit].mean())) / n_test) * 1e3),
        "combined_mae_meV_atom": float(np.mean(np.abs(r_test - (c + a * e_test)) / n_test) * 1e3),
        "before_mae_meV_atom": float(np.mean(np.abs(r_test) / n_test) * 1e3),
    }

# global fit
c_g, a_g = fit_line(fit)
c0_g = float(np.mean(r[fit]))
global_res = {
    "n_frames": len(rows),
    "n_states": len(states),
    "n_fit_frames": int(fit.sum()),
    "n_test_frames": int(test.sum()),
    "alpha_global": a_g,
    "c_global_eV": c_g,
    "before_test_meV_atom": float(np.mean(np.abs(r[test]) / n[test]) * 1e3),
    "offset_only_test_meV_atom": float(np.mean(np.abs(r[test] - c0_g) / n[test]) * 1e3),
    "combined_test_meV_atom": float(np.mean(np.abs(r[test] - (c_g + a_g * e[test])) / n[test]) * 1e3),
    "per_species": per_sp,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(global_res, indent=1))
print(json.dumps(global_res, indent=1))
