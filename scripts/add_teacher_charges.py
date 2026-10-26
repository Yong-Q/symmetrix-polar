#!/usr/bin/env python3
"""Add PolarMACE reference charges to an extxyz file.

The teacher exposes its l=0 density coefficients as ``charges``.  The
electrostatic kernel carries a 1/(4*pi) factor, so the physical (elementary
charge) values are ``4*pi * coefficient``; this is checked by comparing the
monopole Coulomb energy against the teacher's reported electrostatic energy
(ratio 1.26 on a dilute water frame; the remainder is the l=1 contribution).

Output is a new extxyz with an additional ``REF_charges`` array.  Existing
``REF_energy`` / ``REF_forces`` are preserved.

Example:
  PYTHONPATH=/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0:$PWD \
  python scripts/add_teacher_charges.py \
    --input data_20261025_subset/bulk_train.xyz \
    --output data_20261025_subset/bulk_train_charges.xyz \
    --stride 5 --limit 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import ase.io
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="/workspace/group/shared/bza/"
                    "task3_sampling_database/mlip/models/mace-polar/"
                    "MACE-POLAR-1-M.model")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scale", type=float, default=4.0 * math.pi)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--every", type=int, default=50)
    args = ap.parse_args()

    from mace.calculators import mace_polar

    frames = ase.io.read(args.input, index=":")
    if args.stride > 1:
        frames = frames[::args.stride]
    if args.limit:
        frames = frames[:args.limit]
    print(f"[charges] {len(frames)} frames from {args.input}", flush=True)

    calc = mace_polar(model=args.model, device=args.device,
                      default_dtype=args.dtype)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from io import StringIO
    q_abs = []
    n_written = 0
    # Write incrementally so a late failure does not discard all labels.
    with open(out_path, "w") as fh:
        for i, atoms in enumerate(frames):
            a = atoms.copy()
            a.calc = calc
            a.get_potential_energy()
            q = np.asarray(calc.results["charges"], dtype=float).reshape(-1)
            if q.size != len(a):
                if q.size < len(a):
                    raise RuntimeError(
                        f"teacher returned {q.size} charges for {len(a)} atoms")
                q = q[:len(a)]
            q = q * args.scale
            a.set_array("REF_charges", q)
            buf = StringIO()
            ase.io.write(buf, a, format="extxyz")
            fh.write(buf.getvalue())
            fh.flush()
            n_written += 1
            q_abs.append(np.abs(q).mean())
            if (i + 1) % args.every == 0:
                print(f"[charges] {i + 1}/{len(frames)} "
                      f"mean|q|={np.mean(q_abs[-args.every:]):.4f} e", flush=True)
    meta = {
        "input": str(Path(args.input).resolve()),
        "input_sha256": sha256(Path(args.input)),
        "teacher": str(Path(args.model).resolve()),
        "teacher_sha256": sha256(Path(args.model)),
        "scale": args.scale,
        "n_frames": n_written,
        "stride": args.stride,
        "limit": args.limit,
        "mean_abs_q": float(np.mean(q_abs)),
        "note": "REF_charges = 4*pi * PolarMACE l=0 density coefficient",
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"[charges] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)",
          flush=True)
    print(json.dumps(meta, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
