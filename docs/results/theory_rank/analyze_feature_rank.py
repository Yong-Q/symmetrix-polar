#!/usr/bin/env python3
"""Measure the effective rank of node features entering the PolarMACE polar UVU.

If X (N x C) has effective rank r << C, then W X can be evaluated as
(W U_r)(U_r^T X), reducing O(N C^2) to O(N C r). Analysis only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import ase.io

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, "/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0")
import mace_md as mm  # noqa: E402
from mace.calculators import mace_polar  # noqa: E402

POLAR = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"
OUT = Path("/workspace/group/shared/bza/symmetrix-polar/bench/theory_rank_2026-10-22")
OUT.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

calc = mace_polar(model=POLAR, device="cuda", default_dtype="float64")
model = calc.models[0]
model.eval()

targets = [
    ("fdcm0", model.field_dependent_charges_maps[0]),
    ("fdcm0_node_feats", model.field_dependent_charges_maps[0].potential_embedding.node_feats_linear),
    ("fdcm0_dot", model.field_dependent_charges_maps[0].dot_products),
    ("fdcm0_tp", model.field_dependent_charges_maps[0].tp_out),
    ("le_q", model.local_electron_energy.dot_products_q),
    ("le_v", model.local_electron_energy.dot_products_v),
]
print("TARGET TYPES")
for name, mod in targets:
    print(name, type(mod).__name__)

captures = {name: [] for name, _ in targets}

def make_hook(name):
    def hook(module, inputs, output):
        for x in inputs:
            if torch.is_tensor(x) and x.dim() >= 2:
                captures[name].append(x.detach())
                break
    return hook

for name, mod in targets:
    mod.register_forward_hook(make_hook(name))

def load_atoms():
    # 512 atoms (128 NH3) keeps the PyTorch graph small enough for a fast
    # feature-rank measurement while still allowing rank up to 512.
    base, _, _ = mm.build_gas("NH3", 128, 0.01, 770600, 2.0)
    print("LOADED_NH3", len(base))
    return base

atoms = load_atoms()
atoms.info["charge"] = 0
atoms.info["spin"] = 1
atoms.calc = calc
_ = atoms.get_potential_energy()
torch.cuda.synchronize()

def rank_stats(x):
    x = x.detach().to(torch.float64).cpu()
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    s = torch.linalg.svdvals(x)
    s2 = s ** 2
    cum = np.cumsum((s2 / s2.sum()).numpy())
    def er(frac):
        return int(np.searchsorted(cum, frac) + 1)
    return {
        "shape": list(x.shape),
        "rank_max": int(min(x.shape)),
        "rank_1e-3": int((s > 1e-3 * s[0]).sum()),
        "rank_1e-4": int((s > 1e-4 * s[0]).sum()),
        "rank_1e-5": int((s > 1e-5 * s[0]).sum()),
        "rank_1e-6": int((s > 1e-6 * s[0]).sum()),
        "energy_rank_90": er(0.90),
        "energy_rank_99": er(0.99),
        "energy_rank_999": er(0.999),
        "energy_rank_9999": er(0.9999),
        "s_top5": [float(v) for v in s[:5]],
        "s_min": float(s[-1]),
    }

rows = []
for name, arrs in captures.items():
    if not arrs:
        print("NO_CAPTURE", name)
        continue
    for i, x in enumerate(arrs):
        try:
            st = rank_stats(x)
        except Exception as exc:
            print("STATS_FAILED", name, i, repr(exc))
            continue
        st.update({"module": name, "call": i, "dtype": str(x.dtype)})
        rows.append(st)
        print(json.dumps(st, indent=1), flush=True)

(OUT / "feature_rank.json").write_text(json.dumps({"atoms": len(atoms), "rows": rows}, indent=1))
print("wrote", OUT / "feature_rank.json")
