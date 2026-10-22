#!/usr/bin/env python3
"""Rank-truncated inference gate for the low-rank polar activation hypothesis.

Capture the 2048-wide inputs entering the field-map and local-electron polar
operators, compute an oracle rank-r basis, replace each input by its best
rank-r reconstruction, and measure dE/dF versus the full model.

This is the go/no-go for the low-rank activation mechanism. The basis is an
oracle basis from the same snapshot; a production basis would need to be
calibrated/stable.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, "/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0")
import mace_md as mm  # noqa: E402
from mace.calculators import mace_polar  # noqa: E402

POLAR = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"
OUT = Path("/workspace/group/shared/bza/symmetrix-polar/docs/results/theory_rank")
OUT.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

RANKS = [32, 64, 128, 256]

calc = mace_polar(model=POLAR, device="cuda", default_dtype="float64")
model = calc.models[0]
model.eval()

targets = {
    "fdcm0": model.field_dependent_charges_maps[0].potential_embedding.node_feats_linear,
    "fdcm1": model.field_dependent_charges_maps[1].potential_embedding.node_feats_linear,
    "le_q": model.local_electron_energy.dot_products_q,
    "le_v": model.local_electron_energy.dot_products_v,
}

base, _, _ = mm.build_gas("NH3", 128, 0.01, 770600, 2.0)
atoms = base.copy()
atoms.info["charge"] = 0
atoms.info["spin"] = 1
atoms.calc = calc

captures = {}

def capture_hook(name):
    def hook(module, inputs, output):
        if name in captures:
            return
        for x in inputs:
            if torch.is_tensor(x) and x.dim() == 2 and x.shape[-1] == 2048:
                captures[name] = x.detach()
                return
    return hook

handles = []
for name, mod in targets.items():
    handles.append(mod.register_forward_hook(capture_hook(name)))

atoms.calc.results = {}
e0 = float(atoms.get_potential_energy())
f0 = np.asarray(atoms.get_forces(), dtype=np.float64)
for h in handles:
    h.remove()
print("FULL", e0, flush=True)
print("CAPTURED", {k: tuple(v.shape) for k, v in captures.items()}, flush=True)

bases = {}
for name, x in captures.items():
    x = x.to(torch.float64)
    # right singular vectors; x = U S Vh
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    bases[name] = (s.cpu(), vh.cpu())  # Vh: min(N,C) x C
    print("BASIS", name, tuple(x.shape), "s0", float(s[0]), flush=True)

def project_hook(Vr):
    def hook(module, args):
        out = []
        for a in args:
            if torch.is_tensor(a) and a.dim() == 2 and a.shape[-1] == Vr.shape[0]:
                a = (a @ Vr) @ Vr.T
            out.append(a)
        return tuple(out)
    return hook

rows = []
for r in RANKS:
    hs = []
    for name, mod in targets.items():
        if name not in bases:
            continue
        s, vh = bases[name]
        Vr = vh[:r].T.contiguous().to("cuda")
        hs.append(mod.register_forward_pre_hook(project_hook(Vr)))
    atoms.calc.results = {}
    e = float(atoms.get_potential_energy())
    f = np.asarray(atoms.get_forces(), dtype=np.float64)
    for h in hs:
        h.remove()
    d = np.abs(f - f0)
    row = {
        "rank": r,
        "e": e,
        "dE_eV": e - e0,
        "max_dF_eV_A": float(d.max()),
        "rms_dF_eV_A": float(np.sqrt((d ** 2).mean())),
        "max_abs_F": float(np.abs(f0).max()),
        "modules_projected": [k for k in targets if k in bases],
    }
    rows.append(row)
    print(json.dumps(row, indent=1), flush=True)

(OUT / "rank_truncated_error.json").write_text(
    json.dumps({"full_e": e0, "rows": rows}, indent=1))
print("wrote", OUT / "rank_truncated_error.json")
