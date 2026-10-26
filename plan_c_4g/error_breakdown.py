#!/usr/bin/env python3
"""Per-density / per-temperature error breakdown for a Plan C 4G checkpoint.

Why: the aggregate bulk MAE hides where the error lives. Two hypotheses have
very different fixes:
  * if the error grows with density, the missing physics is long-range
    (our QEq is charge-only; the PolarMACE teacher has l=1 multipoles);
  * if it is flat across density, it is local capacity/descriptor coverage,
    which more layers and more frames can fix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plan_c_4g.data import load_frames
from plan_c_4g.descriptors import DescriptorConfig
from plan_c_4g.model import FourGStudent, ModelConfig


def group_eval(model, frames, indices, batch_size, device):
    groups = defaultdict(list)
    for i in indices:
        key = tuple(round(float(x), 5) for x in frames[i].cell.reshape(-1).tolist())
        groups[key].append(i)
    rows = []
    with torch.enable_grad():
        for gidx in groups.values():
            for start in range(0, len(gidx), batch_size):
                chunk = gidx[start:start + batch_size]
                from plan_c_4g.data import collate
                b = collate(frames, chunk, device)
                out = model.predict(b, compute_forces=True, create_graph=False)
                e_err = (out["E_graph"] - b["energy"]) / b["sizes"]
                f_err = out["F"] - b["forces"]
                for j, i in enumerate(chunk):
                    mask = b["batch"] == j
                    rows.append({
                        "species": frames[i].name.split("_")[0],
                        "density": float(frames[i].info.get("density_mol_A3", 0.0)),
                        "temperature": float(frames[i].info.get("temperature_K", 0.0)),
                        "e_err_meV_atom": float(e_err[j].abs()) * 1e3,
                        "f_mae_meV_A": float(f_err[mask].abs().mean()) * 1e3,
                    })
    return rows


def summarize(rows, key):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[key]].append(r)
    out = {}
    for k in sorted(buckets):
        grp = buckets[k]
        out[str(k)] = {
            "n": len(grp),
            "E_meV_atom": float(np.mean([g["e_err_meV_atom"] for g in grp])),
            "F_meV_A": float(np.mean([g["f_mae_meV_A"] for g in grp])),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    device = args.device if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = DescriptorConfig(**ck["descriptor_config"])
    model = FourGStudent(cfg, ModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    frames = load_frames(args.valid, cfg, verbose=False)
    rows = group_eval(model, frames, list(range(len(frames))), args.batch_size, device)
    all_e = np.array([r["e_err_meV_atom"] for r in rows])
    all_f = np.array([r["f_mae_meV_A"] for r in rows])
    report = {
        "checkpoint": args.ckpt,
        "step": ck.get("step"),
        "n_frames": len(rows),
        "overall": {"E_meV_atom": float(all_e.mean()),
                    "F_meV_A": float(all_f.mean())},
        "by_density": summarize(rows, "density"),
        "by_temperature": summarize(rows, "temperature"),
        "by_species": summarize(rows, "species"),
    }
    print(json.dumps(report["overall"], indent=1))
    print("density buckets:")
    for k, v in report["by_density"].items():
        print(f"  rho={k:>8} n={v['n']:4d}  E {v['E_meV_atom']:8.1f}  F {v['F_meV_A']:8.1f}")
    print("species buckets:")
    for k, v in report["by_species"].items():
        print(f"  {k:>6} n={v['n']:4d}  E {v['E_meV_atom']:8.1f}  F {v['F_meV_A']:8.1f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
