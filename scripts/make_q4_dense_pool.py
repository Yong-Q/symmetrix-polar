#!/usr/bin/env python3
"""Convert q4 polar_md_q4 trajectories into a labelled REF_* pool.

polar_md_q4.py runs the MACE-Polar teacher itself, so every frame already
carries the teacher energy/forces in the plain `energy`/`forces` columns.
This script only renames them to REF_energy/REF_forces and attaches the
state metadata the evaluation script groups by.
"""
import argparse
import re
from pathlib import Path

import ase.io
import numpy as np


def parse_tag(name):
    m = re.match(r"traj_([A-Za-z0-9]+)_([0-9.]+)_(\d+)K_s(\d+)", name)
    if not m:
        return None
    species, rho, temp, seed = m.groups()
    return species, float(rho), int(temp), int(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = []
    for path in args.traj:
        p = Path(path)
        meta = parse_tag(p.stem)
        if meta is None:
            print(f"skip (no tag): {p}")
            continue
        species, rho, temp, seed = meta
        tag = f"{species}-64mol-r{rho:g}-{temp}K-s{seed}"
        for atoms in ase.io.read(p, index=":"):
            atoms.info["REF_energy"] = float(atoms.get_potential_energy())
            atoms.arrays["REF_forces"] = np.asarray(atoms.get_forces())
            atoms.info["config_type"] = f"{species}_field"
            atoms.info["dataset"] = "site_field"
            atoms.info["state_tag"] = tag
            atoms.info["density_mol_A3"] = rho
            atoms.info["temperature_K"] = temp
            frames.append(atoms)

    ase.io.write(args.out, frames, format="extxyz")
    tags = sorted({a.info["state_tag"] for a in frames})
    print(f"wrote {args.out}: {len(frames)} frames, {len(tags)} states")
    for t in tags:
        print("  ", t, sum(1 for a in frames if a.info["state_tag"] == t))


if __name__ == "__main__":
    main()
