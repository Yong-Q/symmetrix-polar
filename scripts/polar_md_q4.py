#!/usr/bin/env python3
"""Long MACE-Polar MD trajectories for c1/c2 (q4 single-GPU jobs).

Builds a cubic box from an ASE molecule template, runs Langevin NVT with the
MACE-Polar teacher (FP64), and writes an extxyz trajectory plus a log.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
from ase import Atoms, units
from ase.build import molecule as ase_molecule
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

GRAPH = os.environ.get(
    "GRAPH_LONG_RANGE_PATH",
    "/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0")
if GRAPH not in sys.path:
    sys.path.insert(0, GRAPH)


def build_box(species: str, n_mol: int, rho: float, seed: int) -> Atoms:
    rng = np.random.default_rng(seed)
    mol = ase_molecule(species)
    base = mol.get_positions() - mol.get_positions().mean(axis=0)
    numbers = mol.numbers
    L = (n_mol / rho) ** (1.0 / 3.0)
    n_side = int(math.ceil(n_mol ** (1.0 / 3.0)))
    spacing = L / n_side
    sites = [((i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing)
             for i in range(n_side) for j in range(n_side) for k in range(n_side)]
    rng.shuffle(sites)
    pos = []
    for c in sites[:n_mol]:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        pos.append(base @ R.T + np.asarray(c) + rng.normal(scale=0.05,
                                                          size=base.shape))
    return Atoms(numbers=np.tile(numbers, n_mol),
                 positions=np.vstack(pos), cell=[L, L, L], pbc=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="H2O")
    ap.add_argument("--nmols", type=int, default=32)
    ap.add_argument("--rho", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--ps", type=float, default=100.0)
    ap.add_argument("--equil-ps", type=float, default=2.0)
    ap.add_argument("--frame-ps", type=float, default=0.5)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()

    from mace.calculators import mace_polar
    atoms = build_box(args.species, args.nmols, args.rho, args.seed)
    atoms.calc = mace_polar(model=args.model, device="cuda",
                            default_dtype="float64")
    from ase.optimize import FIRE
    FIRE(atoms, logfile=None).run(fmax=0.1, steps=300)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    dyn = Langevin(atoms, timestep=0.5 * units.fs,
                   temperature_K=args.temperature, friction=0.01)
    n_eq = int(args.equil_ps / 0.0005)
    n_prod = int(args.ps / 0.0005)
    every = int(args.frame_ps / 0.0005)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()

    def log(*_):
        e = atoms.get_potential_energy() / len(atoms)
        t = atoms.get_temperature()
        print(f"[md] step {step} E/atom {e:.4f} eV T {t:.1f} K "
              f"wall {time.time()-t0:.0f}s", flush=True)
        write(args.out, atoms, format="extxyz", append=os.path.exists(args.out))

    for i in range(n_eq):
        dyn.run(1)
        if (i + 1) % 2000 == 0:
            log()
    dyn.attach(log, interval=every)
    dyn.run(n_prod)
    print("done", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
