#!/usr/bin/env python3
"""Build the 2026-10-28 combined fine-tune split.

Keeps the 23,376-frame bulk/gas corpus whole, adds the newly labelled
site-field / slab / dense frames, and carves a state-level holdout out of the
new frames (whole MD seeds, never individual frames) so that the fine-tune can
be scored on data it never saw.
"""
import json
from pathlib import Path

import ase.io

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
BASE = REPO / "data_20261025/train.xyz"
NEW = REPO / "data_20261028_new/labeled_new.xyz"
OUT = REPO / "data_20261028_new"
HOLDOUT_SEEDS = ("f106", "f202")


def main():
    base = ase.io.read(str(BASE), index=":")
    new = ase.io.read(str(NEW), index=":")
    print(f"base {len(base)}  new {len(new)}", flush=True)

    train_new, hold = [], []
    for a in new:
        tag = str(a.info.get("state_tag", ""))
        seed = tag.split("-")[-1]
        (hold if seed in HOLDOUT_SEEDS else train_new).append(a)

    def write(path, frames):
        if path.exists():
            path.unlink()
        for a in frames:
            ase.io.write(str(path), a, format="extxyz", append=path.exists())

    write(OUT / "train_ft3.xyz", base + train_new)
    write(OUT / "holdout_field.xyz", hold)
    meta = {
        "base_frames": len(base),
        "new_frames": len(new),
        "new_train_frames": len(train_new),
        "new_holdout_frames": len(hold),
        "train_ft3_frames": len(base) + len(train_new),
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "holdout_tags": sorted({str(a.info.get("state_tag")) for a in hold}),
    }
    (OUT / "split_ft3.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
