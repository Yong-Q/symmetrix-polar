"""Final held-out evaluation for the Plan C student.

- gas valid split (never trained)
- bulk pilot held-out states (same deterministic state split as training)
- bulk2 C2H4 stratified held-out states (never used for training)
- six-molecule smoke boxes (optional OOD)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plan_c_4g.data import load_frames  # noqa: E402
from plan_c_4g.evaluate import load_model  # noqa: E402
from plan_c_4g.train import evaluate  # noqa: E402
from plan_c_4g.train_bulk import split_states  # noqa: E402

PILOT = "/workspace/group/shared/bza/symmetrix-polar/bench/path_b_bulk_q4_2026-10-23/attempt-01/data"
GAS = "/workspace/group/shared/bza/symmetrix-polar/bench/path_b_data2_q4_2026-10-22/attempt-01/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda"
    model, ck = load_model(args.ckpt, device)
    cfg = model.dcfg
    dt = next(model.parameters()).dtype
    res = {"checkpoint": args.ckpt, "step": ck.get("step"),
           "params": int(sum(p.numel() for p in model.parameters()))}

    # gas valid
    gas = load_frames(os.path.join(GAS, "valid.xyz"), cfg, dtype=dt)
    res["gas_valid"] = evaluate(model, gas, list(range(len(gas))),
                                args.batch_size, device)
    print("gas_valid", json.dumps({k: v for k, v in res["gas_valid"].items()
                                   if k != "groups"}), flush=True)

    # bulk pilot: reproduce the deterministic state-level holdout
    pilot = []
    for p in sorted(glob.glob(os.path.join(PILOT, "*.xyz"))):
        pilot += load_frames(p, cfg, dtype=dt)
    _, b_valid, states = split_states(pilot, 0.2, 20261023)
    res["bulk_pilot_holdout"] = evaluate(model, pilot, b_valid, args.batch_size,
                                         device)
    res["bulk_pilot_holdout"]["n_states"] = len(states)
    res["bulk_pilot_holdout"]["states"] = [list(s) for s in states]
    print("bulk_pilot_holdout", json.dumps({k: v for k, v in
                                            res["bulk_pilot_holdout"].items()
                                            if k != "groups"}), flush=True)

    # bulk2 C2H4 stratified held-out
    b2 = "results/bulk2_C2H4_heldout.xyz"
    if os.path.exists(b2):
        frames = load_frames(b2, cfg, dtype=dt)
        res["bulk2_C2H4_holdout"] = evaluate(model, frames,
                                             list(range(len(frames))),
                                             args.batch_size, device)
        print("bulk2_C2H4_holdout", json.dumps({k: v for k, v in
                                                res["bulk2_C2H4_holdout"].items()
                                                if k != "groups"}), flush=True)

    # six-molecule smoke (12 molecules / box, never trained)
    if not args.skip_smoke:
        smoke = []
        for p in sorted(glob.glob("/workspace/group/shared/bza/symmetrix-polar/bench/"
                                  "path_b_six_molecule_smoke_2026-10-22/attempt-01/"
                                  "local_smoke/*.xyz")):
            smoke += load_frames(p, cfg, dtype=dt)
        res["six_molecule_smoke"] = evaluate(model, smoke,
                                             list(range(len(smoke))),
                                             args.batch_size, device)
        print("six_molecule_smoke", json.dumps({k: v for k, v in
                                               res["six_molecule_smoke"].items()
                                               if k != "groups"}), flush=True)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
