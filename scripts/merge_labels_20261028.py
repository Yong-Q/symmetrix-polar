#!/usr/bin/env python3
"""Merge every 2026-10-28 labelling pass into one deduplicated dataset.

Deduplicates on (state_tag, timestep), then rebuilds the whole-seed holdout
(f106, f202) from the merged pool so the two delta passes cannot leak into
the training side.
"""
import json
from pathlib import Path

import ase.io

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
DATA = REPO / "data_20261028_new"
SOURCES = ["labeled_new.xyz", "labeled_delta_20261028b.xyz",
           "labeled_delta_20261028c.xyz", "labeled_delta_20261028d.xyz"]
HOLDOUT_SEEDS = ("f106", "f202")


def main():
    merged, seen = [], set()
    for name in SOURCES:
        path = DATA / name
        if not path.exists():
            continue
        n = 0
        for a in ase.io.iread(str(path), format="extxyz"):
            # The unfielded slab / dense frames carry no state tag, so
            # (tag, timestep) alone would collapse genuinely different boxes.
            # REF_energy separates them.
            key = (str(a.info.get("state_tag", "")),
                   int(a.info.get("timestep", -1)),
                   len(a),
                   round(float(a.info["REF_energy"]), 6))
            if key in seen:
                continue
            seen.add(key)
            merged.append(a)
            n += 1
        print(f"{name}: +{n}", flush=True)

    train, hold = [], []
    for a in merged:
        seed = str(a.info.get("state_tag", "")).split("-")[-1]
        (hold if seed in HOLDOUT_SEEDS else train).append(a)

    def write(path, frames):
        if path.exists():
            path.unlink()
        for a in frames:
            ase.io.write(str(path), a, format="extxyz", append=path.exists())
        print(f"wrote {len(frames)} -> {path.name}", flush=True)

    write(DATA / "labeled_all_20261028.xyz", merged)
    write(DATA / "holdout_field_v2.xyz", hold)
    meta = {
        "total": len(merged), "train_side": len(train),
        "holdout_side": len(hold), "holdout_seeds": list(HOLDOUT_SEEDS),
        "sources": SOURCES,
        "holdout_tags": sorted({str(a.info.get("state_tag")) for a in hold}),
    }
    (DATA / "merge_20261028.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "holdout_tags"},
                     indent=1))


if __name__ == "__main__":
    main()
