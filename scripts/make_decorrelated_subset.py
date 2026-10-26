#!/usr/bin/env python3
"""Build a decorrelated training subset from the bulk+gas state split.

Why: the 26,841 frames in data_20261025 are NOT 26,841 independent samples.
The ACF audit reports median tau_int(E) = 14.7 frames for bulk trajectories
and N_eff/N = 0.18-0.24 overall.  Training on every frame therefore pays
~5x for correlated duplicates and biases the loss toward whichever states
happen to have the most saved frames.

This script keeps one frame every `--stride` frames **inside each trajectory**
(the same unit the ACF was measured on), so retained frames are separated by
stride * tau_int/e.  Gas rattle configs are single-frame states and are kept.
MD gas trajectories are strided the same way.

Outputs: <outdir>/{bulk_train.xyz,bulk_valid.xyz,gas_train.xyz,gas_valid.xyz}
and subset_manifest.json with before/after counts and the implied N_eff.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import ase.io
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def traj_key(atoms):
    info = atoms.info
    if "traj_id" in info:
        return str(info["traj_id"])
    if "state_id" in info:
        return str(info["state_id"])
    return str(info.get("_source_file", "unknown"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(REPO / "data_20261025/train.xyz"))
    ap.add_argument("--valid", default=str(REPO / "data_20261025/valid.xyz"))
    ap.add_argument("--outdir", default=str(REPO / "data_20261025_subset"))
    ap.add_argument("--stride", type=int, default=15,
                    help="keep 1 frame per stride within each trajectory "
                         "(~median tau_int(E) = 14.7)")
    ap.add_argument("--gas-stride", type=int, default=5,
                    help="gas MD trajectories have tau ~ 2-3 frames")
    ap.add_argument("--max-bulk", type=int, default=0, help="0 = no cap")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {"stride": args.stride, "gas_stride": args.gas_stride,
              "splits": {}}

    for split in ("train", "valid"):
        path = args.train if split == "train" else args.valid
        atoms_list = ase.io.read(path, index=":")
        groups = defaultdict(list)
        for i, a in enumerate(atoms_list):
            groups[traj_key(a)].append(i)

        bulk_sel, gas_sel = [], []
        for key, idxs in groups.items():
            is_gas = str(atoms_list[idxs[0]].info.get("dataset", "")) == "gas"
            k = args.gas_stride if is_gas else args.stride
            keep = idxs[::max(1, k)]
            (gas_sel if is_gas else bulk_sel).extend(keep)

        bulk_sel.sort()
        gas_sel.sort()
        if args.max_bulk and len(bulk_sel) > args.max_bulk:
            # deterministic even spread over the already decorrelated list
            pick = np.linspace(0, len(bulk_sel) - 1, args.max_bulk).round().astype(int)
            bulk_sel = [bulk_sel[i] for i in sorted(set(pick.tolist()))]

        for name, sel in (("bulk", bulk_sel), ("gas", gas_sel)):
            if not sel:
                continue
            out = outdir / f"{name}_{split}.xyz"
            ase.io.write(out, [atoms_list[i] for i in sel], format="extxyz")
            report["splits"][f"{name}_{split}"] = {
                "input_frames": len(atoms_list) if name == "bulk" else None,
                "trajectories": len(groups),
                "selected_frames": len(sel),
            }

        report["splits"][f"{split}_total"] = {
            "input_frames": len(atoms_list),
            "n_trajectories": len(groups),
            "bulk_selected": len(bulk_sel),
            "gas_selected": len(gas_sel),
            "kept_fraction": (len(bulk_sel) + len(gas_sel)) / len(atoms_list),
        }
        print(f"[subset] {split}: {len(atoms_list)} -> {len(bulk_sel)} bulk + "
              f"{len(gas_sel)} gas ({len(groups)} trajectories)", flush=True)

    (outdir / "subset_manifest.json").write_text(json.dumps(report, indent=1))
    print("[subset] wrote", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
