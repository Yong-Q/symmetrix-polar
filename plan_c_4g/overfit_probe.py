#!/usr/bin/env python3
"""Decisive diagnostic: can the 4G student overfit a handful of dense frames?

If it cannot drive the training error to ~0 on 8 frames, the problem is the
model (descriptor or capacity) and no amount of data or active learning will
help.  If it can, the problem is generalization/coverage and active learning is
the right tool.

Reports the training loss and the per-epoch force MAE on the *same* frames it
is optimising, for both a dilute and a dense state.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plan_c_4g.data import collate, load_frames
from plan_c_4g.descriptors import DescriptorConfig
from plan_c_4g.model import FourGStudent, ModelConfig


def pick(frames, density, n):
    hits = [i for i, f in enumerate(frames)
            if abs(float(f.info.get("density_mol_A3", -1)) - density) < 1e-6]
    return hits[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data_20261025_subset/bulk_train.xyz")
    ap.add_argument("--density", type=float, default=0.02)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=160)
    ap.add_argument("--no-vdw", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    device = args.device if torch.cuda.is_available() else "cpu"
    cfg = DescriptorConfig()
    frames = load_frames(args.train, cfg, verbose=False)
    idx = pick(frames, args.density, args.n_frames)
    if len(idx) < 2:
        raise SystemExit(f"only {len(idx)} frames at density {args.density}")
    print(f"[overfit] {len(idx)} frames at rho={args.density}, "
          f"{frames[idx[0]].n_atoms} atoms each", flush=True)

    model = FourGStudent(cfg, ModelConfig(hidden=args.hidden,
                                          trunk_hidden=int(args.hidden * 0.75),
                                          use_vdw=not args.no_vdw)).to(device)
    # Per-element reference energies.  A single-density probe has a rank-1
    # composition matrix, so the least-squares fit is degenerate; fall back to
    # the dataset-wide reference values (printed by every training log) so the
    # network is not forced to absorb ~2 keV/atom in its local energy head.
    DATASET_E0 = {1: -15.99000358581543, 6: -1037.1678466796875,
                  7: -1490.378173828125, 8: -2047.592529296875}
    else_used = False
    A = np.array([[np.sum(frames[i].z.cpu().numpy() == z) for z in (1, 6, 7, 8)]
                  for i in idx], dtype=float)
    y = np.array([frames[i].energy for i in idx], dtype=float)
    if np.linalg.matrix_rank(A) >= 4:
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    else:
        sol = np.array([DATASET_E0[z] for z in (1, 6, 7, 8)])
        else_used = True
    eps.copy_(torch.tensor(sol, dtype=torch.float64, device=device))
    print(f"[overfit] E0 source: {'dataset-wide (rank-deficient probe)' if else_used else 'fitted on probe frames'}", flush=True)
    model.set_element_reference(eps)
    model.fit_scaler(frames, idx)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    model.train()
    f0 = None
    for step in range(args.steps + 1):
        b = collate(frames, idx, device)
        out = model.predict(b, compute_forces=True, create_graph=True)
        e_err = (out["E_graph_train"] - b["energy"]) / b["sizes"]
        f_err = out["F"] - b["forces"]
        loss = (e_err ** 2).mean() + 50.0 * (f_err ** 2).mean()
        if step == 0:
            f0 = float(f_err.abs().mean())
        if step % 100 == 0:
            print(f"  step {step:4d}  E_MAE {float(e_err.abs().mean())*1e3:9.2f} meV/atom"
                  f"  F_MAE {float(f_err.abs().mean())*1e3:9.2f} meV/A"
                  f"  (initial F_MAE {f0*1e3:.1f})", flush=True)
        if step == args.steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

    ef = float(e_err.abs().mean()) * 1e3
    ff = float(f_err.abs().mean()) * 1e3
    verdict = "CAN overfit" if (ef < 2.0 and ff < 20.0) else \
              "CANNOT overfit -> model/descriptor limitation"
    print(f"[overfit] final E {ef:.2f} meV/atom, F {ff:.2f} meV/A -> {verdict}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
