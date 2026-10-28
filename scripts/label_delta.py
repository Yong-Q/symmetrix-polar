#!/usr/bin/env python3
"""Label only the frames that appeared after the last labelling pass.

Reads the already-labelled extxyz to learn, per state tag, the largest
timestep that has labels, then labels any later frame in the corresponding
site-field state directory with MACE-Polar and appends it to a new file.
"""
import argparse
import glob
import json
from pathlib import Path

import ase.io
import numpy as np
from ase import Atoms

GRAPH = "/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0"
import sys
if GRAPH not in sys.path:
    sys.path.insert(0, GRAPH)

from label_new_frames import read_cell, read_dump  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--site-glob", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seen = {}
    if Path(args.labelled).exists():
        for a in ase.io.iread(args.labelled, format="extxyz"):
            tag = str(a.info.get("state_tag", ""))
            ts = int(a.info.get("timestep", -1))
            seen[tag] = max(seen.get(tag, -1), ts)
    print(f"[delta] labels known for {len(seen)} tags", flush=True)

    from mace.calculators import mace_polar
    calc = mace_polar(model=args.model, device=args.device,
                      default_dtype="float64")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    written = 0
    for d in sorted(glob.glob(args.site_glob)):
        work = Path(d)
        traj = work / "traj.lammpstrj"
        data = work / "data.lammps"
        meta = work / "field.json"
        if not (traj.exists() and data.exists()):
            continue
        info = json.loads(meta.read_text()) if meta.exists() else {}
        tag = info.get("tag", work.name)
        elements = info.get("pair_elements", ["O", "H"])
        cell = read_cell(data)
        floor = seen.get(tag, -10 ** 9)
        for ts, types, pos in read_dump(traj):
            if ts <= floor:
                continue
            atoms = Atoms(symbols=[elements[t - 1] for t in types],
                          positions=pos, cell=cell, pbc=True)
            atoms.calc = calc
            atoms.info.update({
                "config_type": f"{info.get('species', '')}_field",
                "REF_energy": float(atoms.get_potential_energy()),
                "dataset": "site_field",
                "state_tag": tag,
                "temperature_K": info.get("T_K", 0.0),
                "density_mol_A3": info.get("density_mol_A3", 0.0),
                "timestep": ts,
            })
            atoms.arrays["REF_forces"] = atoms.get_forces()
            ase.io.write(str(out), atoms, format="extxyz",
                         append=out.exists())
            written += 1
            print(f"[delta] {tag} ts={ts}", flush=True)
    print(f"[delta] wrote {written} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
