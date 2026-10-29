#!/usr/bin/env python3
"""Density-matched split: field augmentation without a new density regime.

The 2026-10-28 combined fine-tune paid 0.12 meV/atom on the old energy gate,
and the cost traced to the 784 new frames that sit above the old corpus
density ceiling.  This split keeps only the new frames whose density is
already represented in the corpus, so the effect of the field dimension can be
separated from the effect of the density extension.
"""
import json
from pathlib import Path

import ase.io

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
BASE = REPO / "data_20261025/train.xyz"
NEW = REPO / "data_20261028_new/labeled_all_20261028.xyz"
OUT = REPO / "data_20261028_new"
CEILING = {"C2H4": 0.01, "C2H6": 0.005, "CH4": 0.01, "CO2": 0.02,
           "H2O": 0.035, "NH3": 0.02}
HOLDOUT_SEEDS = ("f106", "f202")


def main():
    base = ase.io.read(str(BASE), index=":")
    keep, dropped = [], []
    for a in ase.io.iread(str(NEW), format="extxyz"):
        tag = str(a.info.get("state_tag", ""))
        if tag.split("-")[-1] in HOLDOUT_SEEDS:
            continue
        sp = str(a.info.get("config_type", "")).split("_")[0]
        r = float(a.info.get("density_mol_A3", 0.0))
        (dropped if r > CEILING.get(sp, 0.0) + 1e-6 else keep).append(a)

    path = OUT / "train_ft6.xyz"
    if path.exists():
        path.unlink()
    for a in base + keep:
        ase.io.write(str(path), a, format="extxyz", append=path.exists())
    meta = {
        "base_frames": len(base),
        "new_train_side": len(keep) + len(dropped),
        "new_kept": len(keep),
        "new_dropped_above_ceiling": len(dropped),
        "train_frames": len(base) + len(keep),
        "ceiling": CEILING,
    }
    (OUT / "split_ft6.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
