#!/usr/bin/env python3
"""Evaluate one MACE student checkpoint on an arbitrary labelled extxyz file.

Reports energy (meV/atom) and force (meV/A, mean absolute Cartesian
component) MAE, grouped by species and by state tag so that the source of a
regression can be localised.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import ase.io
import numpy as np
from mace.calculators import MACECalculator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--tag-key", default="state_tag")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    calc = MACECalculator(model_paths=args.model, device=args.device,
                          default_dtype=args.dtype)
    frames = ase.io.read(args.xyz, index=":")
    if args.max_frames:
        frames = frames[:args.max_frames]
    print(f"[eval] {len(frames)} frames from {Path(args.xyz).name}", flush=True)

    acc = defaultdict(lambda: {"n_frames": 0, "n_atoms": 0,
                               "se": 0.0, "sf": 0.0, "nf": 0})
    t0 = time.time()
    for i, atoms in enumerate(frames):
        atoms.calc = calc
        e_pred = float(atoms.get_potential_energy())
        f_pred = atoms.get_forces()
        e_ref = float(atoms.info["REF_energy"])
        f_ref = np.asarray(atoms.arrays["REF_forces"])
        de = abs(e_pred - e_ref) / len(atoms) * 1e3
        df = float(np.abs(f_pred - f_ref).mean()) * 1e3
        species = str(atoms.info.get("config_type", "")).split("_")[0] or "?"
        dataset = str(atoms.info.get("dataset", "")) or "?"
        tag = str(atoms.info.get(args.tag_key, "")) or "?"
        for key in ("ALL", f"dataset:{dataset}", f"species:{species}",
                    f"tag:{tag}"):
            a = acc[key]
            a["n_frames"] += 1
            a["n_atoms"] += len(atoms)
            a["se"] += de * len(atoms)
            a["sf"] += df * len(atoms)
            a["nf"] += len(atoms)
        if (i + 1) % 250 == 0:
            print(f"[eval] {i+1}/{len(frames)} {time.time()-t0:.0f}s", flush=True)

    res = {k: {"n_frames": a["n_frames"], "n_atoms": a["n_atoms"],
               "mae_e_meV_per_atom": a["se"] / a["n_atoms"],
               "mae_f_meV_per_A": a["sf"] / a["nf"]}
           for k, a in acc.items()}
    print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()} for k, v in res.items()},
                     indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
        print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
