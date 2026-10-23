#!/usr/bin/env python3
"""Build a ready-to-run LAMMPS case for a Symmetrix JSON model.

The output directory contains data.lammps, in.lammps, and a run command.  The
LAMMPS side uses the production pair style:

    pair_style symmetrix/mace/kk charge 0.0 spin 1.0 streamed_edges generic
    pair_coeff * * model.json <species order>

Example:
  python scripts/make_lammps_case.py \
    --structure frame.xyz --model-json model.json --outdir case_h2o \
    --species O H --steps 0 --dump
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import ase.io


def write_data(atoms, path: Path, species: list[str]) -> None:
    cell = np.asarray(atoms.cell, dtype=float)
    offdiag = max(abs(cell[0, 1]), abs(cell[0, 2]), abs(cell[1, 0]),
                  abs(cell[1, 2]), abs(cell[2, 0]), abs(cell[2, 1]))
    if offdiag > 1e-10:
        raise SystemExit(
            "This generator currently writes orthogonal cells only; use ASE "
            "lammps-data export or an orthogonal supercell.")
    numbers = np.asarray(atoms.get_atomic_numbers())
    symbols = atoms.get_chemical_symbols()
    if sorted(set(symbols)) != sorted(species):
        raise SystemExit(
            f"species mismatch: structure has {sorted(set(symbols))}, "
            f"requested {sorted(species)}")
    types = {s: i + 1 for i, s in enumerate(species)}
    masses = {s: atoms.get_masses()[symbols.index(s)] for s in species}
    lines = ["Symmetrix-Polar interface case", "",
             f"{len(atoms)} atoms", f"{len(species)} atom types", ""]
    lines += [f"0.0 {cell[0, 0]:.16f} xlo xhi",
              f"0.0 {cell[1, 1]:.16f} ylo yhi",
              f"0.0 {cell[2, 2]:.16f} zlo zhi", "", "Masses", ""]
    lines += [f"{types[s]} {masses[s]:.8f}" for s in species]
    lines += ["", "Atoms # atomic", ""]
    for i, (sym, pos) in enumerate(zip(symbols, atoms.get_positions()), 1):
        lines.append(f"{i} {types[sym]} {pos[0]:.16f} {pos[1]:.16f} {pos[2]:.16f}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True)
    ap.add_argument("--model-json", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--species", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--copy-model", action="store_true")
    args = ap.parse_args()

    atoms = ase.io.read(args.structure, index=0)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    write_data(atoms, out / "data.lammps", args.species)

    model = Path(args.model_json).resolve()
    if args.copy_model:
        target = out / "model.json"
        target.write_bytes(model.read_bytes())
        model = target
    coeff = " ".join(args.species)
    lines = [
        "units metal", "atom_style atomic/kk", "atom_modify map yes",
        "boundary p p p", "package kokkos neigh half newton on", "newton on",
        "read_data data.lammps",
        "pair_style symmetrix/mace/kk charge 0.0 spin 1.0 streamed_edges generic",
        f"pair_coeff * * {model} {coeff}",
        "neighbor 2.0 bin", "neigh_modify every 1 delay 0 check yes",
        "timestep 0.000001", "thermo 1", "thermo_style custom step pe",
        "thermo_modify format float %.17g", "run_style verlet/kk",
        "run 0",
    ]
    if args.steps:
        lines.append(f"run {args.steps}")
    if args.dump:
        lines.append("write_dump all custom forces.dump id type x y z fx fy fz "
                     "modify format float %.17g sort id")
    (out / "in.lammps").write_text("\n".join(lines) + "\n")
    cmd = (f"lmp_gemm -k on g 1 -in {out / 'in.lammps'} "
           f"-log {out / 'log.lammps'}")
    (out / "RUN.txt").write_text(cmd + "\n")
    print(f"wrote {out}/data.lammps, in.lammps, RUN.txt")
    print(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
