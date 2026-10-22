#!/usr/bin/env python3
"""Measure numerical rank / low-rank compressibility of the six PolarMACE UVU tables.

Each table is stored flattened as 524288 = 2 paths x 512 x 512 (or 1024 x 512).
We report:
  - zero fraction
  - singular-value spectrum of the stacked 1024 x 512 matrix
  - numerical rank at relative tolerances
  - rank needed for 90/99/99.9/99.99% spectral energy
  - per-path 512 x 512 spectra
This is an analysis-only script; it does not modify the engine.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

MODEL = "/workspace/group/shared/bza/task3_sampling_database/mlip/models/mace-polar/MACE-POLAR-1-M.model"
OUT = Path("/workspace/group/shared/bza/symmetrix-polar/bench/theory_rank_2026-10-22")
sys.path.insert(0, "/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0")

KEYS = [
    "field_dependent_charges_maps.0.dot_products.weight",
    "field_dependent_charges_maps.0.tp_out.weight",
    "field_dependent_charges_maps.1.dot_products.weight",
    "field_dependent_charges_maps.1.tp_out.weight",
    "local_electron_energy.dot_products_q.weight",
    "local_electron_energy.dot_products_v.weight",
]

m = torch.load(MODEL, map_location="cpu", weights_only=False)
sd = m.state_dict()

def spectrum(v):
    v = v.to(torch.float64)
    s = torch.linalg.svdvals(v).cpu().numpy()
    return s

rows = []
per_path = []
for key in KEYS:
    w = sd[key].to(torch.float64)
    n = w.numel()
    rows.append({"key": key, "numel": n, "shape_stored": list(sd[key].shape)})
    assert n == 1024 * 512, (key, n)
    w2 = w.reshape(1024, 512)
    zero_frac = float((w2 == 0).double().mean().item())
    s = spectrum(w2)
    s2 = s ** 2
    energy = s2 / s2.sum()
    cum = np.cumsum(energy)
    def rank_at(rel):
        return int((s > rel * s[0]).sum())
    row = {
        "key": key,
        "zero_fraction": zero_frac,
        "s_max": float(s[0]),
        "s_min": float(s[-1]),
        "rank_1e-3": rank_at(1e-3),
        "rank_1e-4": rank_at(1e-4),
        "rank_1e-5": rank_at(1e-5),
        "rank_1e-6": rank_at(1e-6),
        "energy_rank_90": int(np.searchsorted(cum, 0.90) + 1),
        "energy_rank_99": int(np.searchsorted(cum, 0.99) + 1),
        "energy_rank_999": int(np.searchsorted(cum, 0.999) + 1),
        "energy_rank_9999": int(np.searchsorted(cum, 0.9999) + 1),
        "top10_singular": [float(x) for x in s[:10]],
    }
    rows[-1].update(row)
    print(json.dumps(row, indent=1), flush=True)

    for p in range(2):
        wp = w2[p * 512:(p + 1) * 512, :]
        sp = spectrum(wp)
        cum_p = np.cumsum(sp ** 2 / (sp ** 2).sum())
        per_path.append({
            "key": key, "path": p,
            "rank_1e-4": int((sp > 1e-4 * sp[0]).sum()),
            "rank_1e-6": int((sp > 1e-6 * sp[0]).sum()),
            "energy_rank_90": int(np.searchsorted(cum_p, 0.90) + 1),
            "energy_rank_99": int(np.searchsorted(cum_p, 0.99) + 1),
            "top5_singular": [float(x) for x in sp[:5]],
        })

(OUT / "uvu_rank.json").write_text(json.dumps(
    {"model": MODEL, "keys": KEYS, "stacked": rows, "per_path": per_path}, indent=1))
print("wrote", OUT / "uvu_rank.json")
