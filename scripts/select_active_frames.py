#!/usr/bin/env python3
"""Active selection: keep the frames the current student disagrees with most.

The coverage pools are sampled on the teacher's own PES, so every frame already
carries REF_energy / REF_forces.  Scoring them with the student turns that pool
into an acquisition set: per state, frames are ranked by the student-teacher
force difference and only the top `--keep-frac` survive.  A per-state floor
keeps every state represented even when one is uniformly easy.

Why per state: the disagreement is dominated by density, so a global top-k
would keep only the densest states and throw away the rest of the coverage.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import ase.io
import numpy as np
from mace.calculators import MACECalculator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="+", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--keep-frac", type=float, default=0.6)
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    calc = MACECalculator(model_paths=args.student, device=args.device,
                          default_dtype="float64")
    frames = []
    for p in args.pool:
        frames.extend(ase.io.read(p, index=":"))
    print(f"[select] {len(frames)} candidates", flush=True)

    scored = []
    for i, atoms in enumerate(frames):
        atoms.calc = calc
        f_stu = atoms.get_forces()
        f_ref = np.asarray(atoms.arrays["REF_forces"])
        e_stu = float(atoms.get_potential_energy())
        e_ref = float(atoms.info["REF_energy"])
        scored.append({
            "atoms": atoms,
            "tag": str(atoms.info.get("state_tag", "?")),
            "df": float(np.abs(f_stu - f_ref).mean()) * 1e3,
            "de": abs(e_stu - e_ref) / len(atoms) * 1e3,
        })
        if (i + 1) % 10 == 0:
            print(f"[select] {i+1}/{len(frames)}", flush=True)

    by_tag = defaultdict(list)
    for s in scored:
        by_tag[s["tag"]].append(s)

    keep = []
    report = {}
    for tag, rows in sorted(by_tag.items()):
        rows.sort(key=lambda r: -r["df"])
        n_keep = max(args.min_frames,
                     int(round(args.keep_frac * len(rows))))
        n_keep = min(n_keep, len(rows))
        keep.extend(rows[:n_keep])
        report[tag] = {
            "n": len(rows), "kept": n_keep,
            "df_all_median": float(np.median([r["df"] for r in rows])),
            "df_kept_min": float(min(r["df"] for r in rows[:n_keep])),
            "df_dropped_max": (float(max(r["df"] for r in rows[n_keep:]))
                               if len(rows) > n_keep else None),
            "de_all_mean": float(np.mean([r["de"] for r in rows])),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    # Detach the calculator and write in one call: append mode across frames of
    # different sizes reuses the first frame's column layout and breaks.
    selected = []
    for r in keep:
        a = r["atoms"]
        a.calc = None
        selected.append(a)
    ase.io.write(str(out), selected, format="extxyz")
    print(f"[select] kept {len(keep)}/{len(frames)} -> {out}", flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(
            {"n_candidates": len(frames), "n_kept": len(keep),
             "keep_frac": args.keep_frac, "per_state": report}, indent=1))
    print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()}
                      for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
