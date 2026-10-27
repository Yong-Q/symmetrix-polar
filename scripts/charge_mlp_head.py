#!/usr/bin/env python3
"""Train a small MLP charge head on the frozen 128-channel MACE features.

Frame-level split.  Reports held-out q MAE/RMSE/R2 and saves the head.
"""
import json
import sys
from pathlib import Path

import ase.io
import numpy as np
import torch
import torch.nn as nn

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
MODEL = REPO / "bench/plan_d_bulk_2026-10-25/models/bulk/plan_d_bulk.model"
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    REPO / "bench/plan_d_bulk_2026-10-25/charges_balanced1200.xyz")
OUTDIR = Path(sys.argv[2]) if len(sys.argv) > 2 else (
    REPO / "bench/plan_d_bulk_2026-10-25/charge_head")
OUTDIR.mkdir(parents=True, exist_ok=True)

from mace.calculators import MACECalculator  # noqa: E402
from e3nn import o3  # noqa: E402

FEAT = OUTDIR / "features.npz"
if FEAT.exists():
    z = np.load(FEAT)
    X, y, fid = z["X"], z["y"], z["fid"]
    print("[mlp] loaded cached features", X.shape, flush=True)
else:
    frames = ase.io.read(str(DATA), index=":")
    print(f"[mlp] {len(frames)} frames", flush=True)
    calc = MACECalculator(model_paths=str(MODEL), device="cuda",
                          default_dtype="float32")
    model = calc.models[0]
    readout = model.readouts[0]
    irreps = o3.Irreps(readout.linear.irreps_in)
    sl = irreps.slices()
    sl0 = [s for s, (mul, ir) in zip(sl, irreps) if ir == o3.Irrep(0, 1)][0]
    print("[mlp] scalar slice", sl0, flush=True)
    captured = {}

    def hook(module, args, output):
        captured["x"] = args[0].detach()

    handle = readout.register_forward_hook(hook)
    Xs, ys, fids = [], [], []
    for i, atoms in enumerate(frames):
        q = np.asarray(atoms.get_array("REF_charges"), dtype=np.float64)
        atoms.calc = calc
        atoms.get_potential_energy()
        x = captured["x"]
        if isinstance(x, tuple):
            x = x[0]
        Xs.append(x[:, sl0.start:sl0.stop].float().cpu().numpy())
        ys.append(q)
        fids.append(np.full(len(atoms), i))
        if (i + 1) % 200 == 0:
            print(f"[mlp] {i+1}/{len(frames)}", flush=True)
    handle.remove()
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    fid = np.concatenate(fids)
    np.savez_compressed(FEAT, X=X, y=y, fid=fid)
    print("[mlp] saved", FEAT, X.shape, flush=True)

fit = fid % 2 == 0
test = ~fit
mu = X[fit].mean(0)
sd = X[fit].std(0) + 1e-8
Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device="cuda")
yt = torch.tensor(y, dtype=torch.float32, device="cuda")
fit_t = torch.tensor(fit, device="cuda")
test_t = torch.tensor(test, device="cuda")

head = nn.Sequential(
    nn.Linear(X.shape[1], 256), nn.SiLU(),
    nn.Linear(256, 128), nn.SiLU(),
    nn.Linear(128, 1),
).cuda()
opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
loss_fn = nn.MSELoss()
for ep in range(60):
    head.train()
    perm = torch.randperm(int(fit_t.sum()), device="cuda")
    idx = fit_t.nonzero(as_tuple=True)[0][perm]
    tot = 0.0
    for k in range(0, len(idx), 8192):
        b = idx[k:k + 8192]
        pred = head(Xt[b]).squeeze(-1)
        loss = loss_fn(pred, yt[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += float(loss) * len(b)
    sched.step()
    if (ep + 1) % 10 == 0:
        head.eval()
        with torch.no_grad():
            p = head(Xt[test_t]).squeeze(-1)
            e = p - yt[test_t]
            print(f"[mlp] ep {ep+1} train_mse {tot/len(idx):.3e} "
                  f"test_q_mae {e.abs().mean():.5f} rmse {torch.sqrt((e*e).mean()):.5f}",
                  flush=True)

head.eval()
with torch.no_grad():
    p = head(Xt[test_t]).squeeze(-1).cpu().numpy()
    t = yt[test_t].cpu().numpy()
e = p - t
res = {
    "data": str(DATA),
    "n_atoms": int(len(y)),
    "n_features": int(X.shape[1]),
    "q_mae_test_e": float(np.mean(np.abs(e))),
    "q_rmse_test_e": float(np.sqrt(np.mean(e ** 2))),
    "r2_test": float(1 - np.sum(e ** 2) / np.sum((t - t.mean()) ** 2)),
    "mean_abs_q_test_e": float(np.mean(np.abs(t))),
}
torch.save({"state_dict": head.state_dict(), "mu": mu, "sd": sd,
            "n_features": X.shape[1]}, OUTDIR / "charge_head.pt")
(OUTDIR / "charge_head.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
