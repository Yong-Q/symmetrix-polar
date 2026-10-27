#!/usr/bin/env python3
"""Per-species E/F validation of the distilled Plan D bulk backbone.

Runs the exported student on the full 3,465-frame held-out set and reports
energy (meV/atom) and force (meV/Angstrom) MAE per molecule and per dataset.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import ase.io
import numpy as np

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
HERE = Path(__file__).resolve().parent
VALID = REPO / "data_20261025/valid.xyz"
MODEL = HERE / "models/bulk/plan_d_bulk.model"
OUT = HERE / "results/valid_by_species.json"

from mace.calculators import MACECalculator  # noqa: E402

calc = MACECalculator(model_paths=str(MODEL), device="cuda",
                      default_dtype="float64")
frames = ase.io.read(str(VALID), index=":")
print(f"[valid] {len(frames)} frames", flush=True)

acc = defaultdict(lambda: {"n_frames": 0, "n_atoms": 0,
                           "se_e": 0.0, "sf": 0.0, "nf": 0})
t0 = time.time()
for i, atoms in enumerate(frames):
    atoms.calc = calc
    e_pred = float(atoms.get_potential_energy())
    e_ref = float(atoms.info["REF_energy"])
    f_pred = atoms.get_forces()
    f_ref = np.asarray(atoms.arrays["REF_forces"])
    de = abs(e_pred - e_ref) / len(atoms) * 1e3
    # MACE convention: mean absolute Cartesian component error, not the
    # vector norm (the norm is sqrt(8/pi)/sqrt(2/pi) = 2x larger for
    # isotropic errors).
    df_mean = float(np.abs(f_pred - f_ref).mean()) * 1e3
    species = str(atoms.info.get("config_type", "")).split("_")[0]
    dataset = str(atoms.info.get("dataset", ""))
    for key in ("ALL", f"species:{species}", f"dataset:{dataset}",
                f"cross:{species}|{dataset}"):
        a = acc[key]
        a["n_frames"] += 1
        a["n_atoms"] += len(atoms)
        a["se_e"] += de * len(atoms)
        a["sf"] += df_mean * len(atoms)
        a["nf"] += len(atoms)
    if (i + 1) % 250 == 0:
        print(f"[valid] {i+1}/{len(frames)} {time.time()-t0:.0f}s", flush=True)

res = {}
for key, a in acc.items():
    res[key] = {
        "n_frames": a["n_frames"],
        "n_atoms": a["n_atoms"],
        "mae_e_meV_per_atom": a["se_e"] / a["n_atoms"],
        "mae_f_meV_per_A": a["sf"] / a["nf"],
    }
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                      for kk, vv in v.items()} for k, v in res.items()
                  if k.startswith(("ALL", "species:", "dataset:"))}, indent=1))
print("wrote", OUT, flush=True)
