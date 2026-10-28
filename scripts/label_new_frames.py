#!/usr/bin/env python3
"""Label the new site-field / slab / dense frames with MACE-Polar E and F.

Inputs:
  --site-glob  glob of site-field state dirs (each with data.lammps,
               field.json, traj.lammpstrj)
  --xyz        extra extxyz files (slabs, dense) to label
  --model      MACE-Polar checkpoint
  --out        output extxyz

Writes one extxyz with REF_energy, REF_forces and per-frame metadata.
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


def read_cell(data_path):
    cell = np.zeros(3)
    for line in Path(data_path).read_text().splitlines():
        p = line.split()
        if len(p) == 4 and p[-2:] == ["xlo", "xhi"]:
            cell[0] = float(p[1]) - float(p[0])
        if len(p) == 4 and p[-2:] == ["ylo", "yhi"]:
            cell[1] = float(p[1]) - float(p[0])
        if len(p) == 4 and p[-2:] == ["zlo", "zhi"]:
            cell[2] = float(p[1]) - float(p[0])
    return cell


def read_dump(path):
    text = Path(path).read_text()
    for chunk in text.split("ITEM: TIMESTEP")[1:]:
        body = "ITEM: TIMESTEP" + chunk
        if "ITEM: ATOMS" not in body:
            continue
        head, _, _ = body.partition("ITEM: ATOMS")
        try:
            n = int(head.split("ITEM: NUMBER OF ATOMS")[1].split()[0])
        except Exception:
            continue
        ts = int(chunk.strip().splitlines()[0])
        lines = body.split("ITEM: ATOMS", 1)[1].strip().splitlines()[1:]
        if len(lines) < n:
            continue
        arr = np.array([ln.split() for ln in lines[:n]], dtype=float)
        yield ts, arr[:, 1].astype(int), arr[:, 2:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-glob", default="")
    ap.add_argument("--xyz", nargs="*", default=[])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

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
        species = info.get("species", "")
        elements = info.get("pair_elements", ["O", "H"])
        cell = read_cell(data)
        for ts, types, pos in read_dump(traj):
            sym = [elements[t - 1] for t in types]
            atoms = Atoms(symbols=sym, positions=pos,
                          cell=cell, pbc=True)
            atoms.calc = calc
            e = float(atoms.get_potential_energy())
            f = atoms.get_forces()
            atoms.info.update({
                "config_type": f"{species}_field",
                "REF_energy": e,
                "dataset": "site_field",
                "state_tag": info.get("tag", work.name),
                "temperature_K": info.get("T_K", 0.0),
                "density_mol_A3": info.get("density_mol_A3", 0.0),
                "timestep": ts,
            })
            atoms.arrays["REF_forces"] = f
            ase.io.write(str(out), atoms, format="extxyz",
                         append=out.exists())
            written += 1
            if written % 50 == 0:
                print(f"[label] {written} frames", flush=True)

    for path in args.xyz:
        for atoms in ase.io.iread(path, format="extxyz"):
            atoms.calc = calc
            e = float(atoms.get_potential_energy())
            f = atoms.get_forces()
            atoms.info["REF_energy"] = e
            atoms.info.setdefault("dataset", "extra")
            atoms.arrays["REF_forces"] = f
            ase.io.write(str(out), atoms, format="extxyz",
                         append=out.exists())
            written += 1
            if written % 50 == 0:
                print(f"[label] {written} frames", flush=True)

    print(f"done {written} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
