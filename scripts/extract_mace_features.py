#!/usr/bin/env python3
"""Extract MACE 0e/1o node features and teacher q/mu labels for head training."""
import json
import sys
from pathlib import Path

import ase.io
import numpy as np
import torch
from e3nn import o3

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
MODEL = REPO / "bench/plan_d_bulk_2026-10-25/models/bulk/plan_d_bulk.model"
DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])

from mace.calculators import MACECalculator  # noqa: E402

frames = ase.io.read(str(DATA), index=":")
print(f"[feat] {len(frames)} frames", flush=True)
calc = MACECalculator(model_paths=str(MODEL), device="cuda",
                      default_dtype="float32")
model = calc.models[0]
readout = model.readouts[0]
irreps = o3.Irreps(readout.linear.irreps_in)
sl = irreps.slices()
s0 = [s for s, (mul, ir) in zip(sl, irreps) if ir == o3.Irrep(0, 1)][0]
s1 = [s for s, (mul, ir) in zip(sl, irreps) if ir == o3.Irrep(1, -1)][0]
print("[feat] 0e", s0, "1o", s1, flush=True)
n0 = (s0.stop - s0.start)
n1 = (s1.stop - s1.start) // 3
print("[feat] channels 0e", n0, "1o", n1, flush=True)

cap = {}

def hook(module, args, output):
    cap["x"] = args[0].detach()

handle = readout.register_forward_hook(hook)
X0, X1, YQ, YMU, FID, E, EREF = [], [], [], [], [], [], []
states = []
state_map = {}
for i, atoms in enumerate(frames):
    atoms.calc = calc
    e = float(atoms.get_potential_energy())
    x = cap["x"]
    if isinstance(x, tuple):
        x = x[0]
    n = len(atoms)
    X0.append(x[:, s0.start:s0.stop].half().cpu().numpy())
    X1.append(x[:, s1.start:s1.stop].reshape(n, n1, 3).half().cpu().numpy())
    YQ.append(np.asarray(atoms.get_array("REF_charges"), dtype=np.float32))
    YMU.append(np.asarray(atoms.get_array("REF_atomic_dipoles"), dtype=np.float32))
    FID.append(np.full(n, i, dtype=np.int32))
    E.append(e)
    EREF.append(float(atoms.info["REF_energy"]))
    st = str(atoms.info.get("state_id", "unknown"))
    states.append(st)
    state_map.setdefault(st, len(state_map))
    if (i + 1) % 500 == 0:
        print(f"[feat] {i+1}/{len(frames)}", flush=True)
handle.remove()

np.savez_compressed(
    OUT, X0=np.concatenate(X0), X1=np.concatenate(X1),
    yq=np.concatenate(YQ), ymu=np.concatenate(YMU),
    fid=np.concatenate(FID), e=np.asarray(E, dtype=np.float32),
    eref=np.asarray(EREF, dtype=np.float32),
    states=np.asarray(states),
    state_map=np.asarray(json.dumps(state_map)))
print(f"[feat] wrote {OUT} atoms={sum(len(x) for x in X0)}", flush=True)
