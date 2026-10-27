#!/usr/bin/env python3
"""Can the existing 128-channel MACE scalar features predict teacher charges?

This is the cheapest possible charge-head probe: freeze the bulk student,
capture the invariant (l=0) node features at the readout, and fit a ridge
readout to the teacher REF_charges.  Frame-level train/test split.

If the held-out q MAE is ~0.01 e, a charge head on this backbone is viable;
if it stays at the 4G level (~0.4 e), the charges need a different model.
"""
import json
import sys
from pathlib import Path

import ase.io
import numpy as np
import torch

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
MODEL = REPO / "bench/plan_d_bulk_2026-10-25/models/bulk/plan_d_bulk.model"
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    REPO / "bench/plan_d_bulk_2026-10-25/charges_balanced1200.xyz")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else (
    REPO / "bench/plan_d_bulk_2026-10-25/results/charge_readout_probe.json")

from mace.calculators import MACECalculator  # noqa: E402
from e3nn import o3  # noqa: E402

frames = ase.io.read(str(DATA), index=":")
print(f"[readout] {len(frames)} frames", flush=True)
calc = MACECalculator(model_paths=str(MODEL), device="cuda",
                      default_dtype="float32")
model = calc.models[0]
readout = model.readouts[0]
print("[readout] type:", type(readout).__name__, flush=True)
for attr in ("irreps_in",):
    if hasattr(readout, attr):
        irreps = o3.Irreps(getattr(readout, attr))
        break
else:
    if hasattr(readout, "linear"):
        irreps = o3.Irreps(readout.linear.irreps_in)
    elif hasattr(readout, "linear_1"):
        irreps = o3.Irreps(readout.linear_1.irreps_in)
    else:
        raise AttributeError(f"cannot find input irreps on {type(readout)}")
sl = irreps.slices()
scalar_slices = [s for s, (mul, ir) in zip(sl, irreps) if ir == o3.Irrep(0, 1)]
print("[readout] irreps:", irreps, "scalar slices:", scalar_slices, flush=True)
assert scalar_slices, "no scalar irreps at readout input"
sl0 = scalar_slices[0]

captured = {}

def hook(module, args, output):
    captured["x"] = args[0].detach()

handle = readout.register_forward_hook(hook)

Xs, ys, frame_ids = [], [], []
for i, atoms in enumerate(frames):
    q = np.asarray(atoms.get_array("REF_charges"), dtype=np.float64)
    if q.size != len(atoms):
        continue
    atoms.calc = calc
    atoms.get_potential_energy()
    x = captured["x"]
    if isinstance(x, tuple):
        x = x[0]
    X = x[:, sl0.start:sl0.stop].float().cpu().numpy()
    Xs.append(X)
    ys.append(q)
    frame_ids.append(np.full(len(atoms), i))
    if (i + 1) % 100 == 0:
        print(f"[readout] {i+1}/{len(frames)}", flush=True)

handle.remove()
X = np.concatenate(Xs, axis=0).astype(np.float64)
y = np.concatenate(ys, axis=0)
fid = np.concatenate(frame_ids)
print("[readout] X", X.shape, "y", y.shape, flush=True)

# frame-level split: even/odd frame ids
fit_mask = fid % 2 == 0
test_mask = ~fit_mask
mu = X[fit_mask].mean(axis=0)
sd = X[fit_mask].std(axis=0) + 1e-8
Xf = (X - mu) / sd
A = Xf[fit_mask].T @ Xf[fit_mask] + 1e-3 * np.eye(X.shape[1])
b = Xf[fit_mask].T @ y[fit_mask]
w = np.linalg.solve(A, b)
pred = Xf[test_mask] @ w
err = pred - y[test_mask]

# baseline: per-species mean charge
def mae(v):
    return float(np.mean(np.abs(v)))

res = {
    "data": str(DATA),
    "n_frames": len(frames),
    "n_atoms": int(len(y)),
    "n_features": int(X.shape[1]),
    "q_mae_test_e": mae(err),
    "q_rmse_test_e": float(np.sqrt(np.mean(err ** 2))),
    "q_mae_train_e": mae(Xf[fit_mask] @ w - y[fit_mask]),
    "q_std_test_e": float(y[test_mask].std()),
    "r2_test": float(1.0 - np.sum(err ** 2) / np.sum((y[test_mask] - y[test_mask].mean()) ** 2)),
    "mean_abs_q_test_e": float(np.mean(np.abs(y[test_mask]))),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
