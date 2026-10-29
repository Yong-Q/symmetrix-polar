#!/usr/bin/env python3
"""How much of the student error is inherited from MACE-OFF23?

MACE-Polar is the teacher and therefore the reference.  For the same frames we
score two approximations against it:

  * MACE-OFF23-medium  -- the functional form the linear student is built on
  * the distilled student

If the two errors are equal, the student has already extracted everything the
OFF23 architecture can express and more data will not help; a gap between them
is what distillation (and more data) can still close.  Forces need no baseline
correction.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import ase.io
import numpy as np

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
POLAR = ("/workspace/group/shared/bza/task3_sampling_database/mlip/models/"
         "mace-polar/MACE-POLAR-1-M.model")
OFF23 = ("/workspace/group/shared/bza/task3_sampling_database/mlip/models/"
         "MACE-OFF23_medium.model")

import sys
sys.path.insert(0, "/workspace/group/shared/bza/external-repos/"
                   "graph_electrostatics-0.4.0")
from mace.calculators import MACECalculator, mace_polar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--student", default=str(REPO / "bench/plan_d_bulk_2026-10-25/"
                                                "models/bulk/plan_d_bulk.model"))
    ap.add_argument("--filter", default="")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pol = mace_polar(model=POLAR, device="cuda", default_dtype="float64")
    off = MACECalculator(model_paths=OFF23, device="cuda", default_dtype="float64")
    stu = MACECalculator(model_paths=args.student, device="cuda",
                         default_dtype="float64")
    frames = [a for a in ase.io.read(args.xyz, index=":")
              if args.filter in str(a.info.get("state_tag", ""))
              or args.filter in str(a.info.get("config_type", ""))]
    frames = frames[::args.stride]
    print(f"[bench] {len(frames)} frames", flush=True)

    acc = defaultdict(lambda: {"n": 0, "nat": 0, "off_df": 0.0, "stu_df": 0.0,
                               "off_de": 0.0, "stu_de": 0.0})
    for i, a in enumerate(frames):
        a.calc = pol
        ep = float(a.get_potential_energy())
        fp = a.get_forces()
        a.calc = off
        eo = float(a.get_potential_energy())
        fo = a.get_forces()
        a.calc = stu
        es = float(a.get_potential_energy())
        fs = a.get_forces()
        n = len(a)
        key = str(a.info.get("state_tag", "")) or "?"
        df_off = float(np.abs(fo - fp).mean()) * 1e3
        df_stu = float(np.abs(fs - fp).mean()) * 1e3
        de_off = abs(eo - ep) / n * 1e3
        de_stu = abs(es - ep) / n * 1e3
        for k in ("ALL", key):
            d = acc[k]
            d["n"] += 1
            d["nat"] += n
            d["off_df"] += df_off * n
            d["stu_df"] += df_stu * n
            d["off_de"] += de_off * n
            d["stu_de"] += de_stu * n
        if (i + 1) % 10 == 0:
            print(f"[bench] {i+1}/{len(frames)}", flush=True)

    res = {}
    for k, v in acc.items():
        res[k] = {
            "n_frames": v["n"],
            "off23_vs_polar_f": v["off_df"] / v["nat"],
            "student_vs_polar_f": v["stu_df"] / v["nat"],
            "off23_vs_polar_e": v["off_de"] / v["nat"],
            "student_vs_polar_e": v["stu_de"] / v["nat"],
            "student_over_off_f": (v["stu_df"] / v["off_df"]) if v["off_df"] else None,
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f'{k:44s} n={v["n_frames"]:3d} '
              f'OFF={v["off23_vs_polar_f"]:7.2f}  STU={v["student_vs_polar_f"]:7.2f}  '
              f'ratio={v["student_over_off_f"]:.2f}', flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
