#!/usr/bin/env python3
"""Estimate the student->POLAR reweighting ESS from per-frame energies."""
import json
from collections import defaultdict
from pathlib import Path

import ase.io
import numpy as np

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
HERE = Path(__file__).resolve().parent
VALID = REPO / "data_20261025/valid.xyz"
MODEL = HERE / "models/bulk/plan_d_bulk.model"
OUT = HERE / "results/ess_student_vs_polar.json"
KB = 8.617333262e-5

from mace.calculators import MACECalculator  # noqa: E402

calc = MACECalculator(model_paths=str(MODEL), device="cuda",
                      default_dtype="float64")
rows = []
for i, atoms in enumerate(ase.io.iread(str(VALID), index=":")):
    atoms.calc = calc
    e = float(atoms.get_potential_energy())
    rows.append({
        "e_err": e - float(atoms.info["REF_energy"]),
        "n_atoms": len(atoms),
        "species": str(atoms.info.get("config_type", "")).split("_")[0],
        "dataset": str(atoms.info.get("dataset", "")),
        "state": str(atoms.info.get("state_id", "")),
    })
print("frames", len(rows), flush=True)


def ess(dE, T=300.0):
    b = 1.0 / (KB * T)
    dE = np.asarray(dE, dtype=float)
    dE = dE - dE.mean()
    lw = -b * dE
    lw -= lw.max()
    w = np.exp(lw)
    w /= w.sum()
    return {"n": int(len(dE)), "sigma_beta": float(np.std(b * dE)),
            "neff": float(1.0 / np.sum(w * w)),
            "ess_fraction": float(1.0 / np.sum(w * w) / len(dE))}


res = {"raw": ess([r["e_err"] for r in rows]),
       "raw_gas": ess([r["e_err"] for r in rows if r["dataset"] == "gas"]),
       "raw_bulk": ess([r["e_err"] for r in rows if r["dataset"] == "bulk"])}
# remove per-state constant offsets (they cancel in normalized weights)
g = defaultdict(list)
for i, r in enumerate(rows):
    g[r["state"]].append(i)
e2 = np.array([r["e_err"] for r in rows])
for idx in g.values():
    idx = np.array(idx)
    e2[idx] -= e2[idx].mean()
res["state_centered"] = ess(e2)
# per-frame energy-error scale
ea = np.array([r["e_err"] / r["n_atoms"] for r in rows]) * 1e3
res["frame_stats"] = {
    "mean_abs_meV_per_atom": float(np.mean(np.abs(ea))),
    "std_meV_per_atom": float(np.std(ea)),
    "std_eV_per_96_atoms": float(np.std([r["e_err"] / r["n_atoms"] * 96
                                         for r in rows])),
}
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
