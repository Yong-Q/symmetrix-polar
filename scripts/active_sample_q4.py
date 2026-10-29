#!/usr/bin/env python3
"""Coverage-gap sampler for q4: run MACE-Polar MD in density windows the
training corpus never visited, and write the frames already labelled.

The teacher is the reference, so sampling on its own PES gives on-policy
configurations with free labels: every dumped frame carries REF_energy and
REF_forces.  Selection (which frames are worth keeping) happens afterwards on
the student-teacher disagreement, so this job only has to produce a candidate
pool that covers the gap.

States are given as a compact spec:
    SPECIES:NMOLS:RHO:TEMPERATURE[:SEED]
joined by semicolons, e.g. "CO2:64:0.03:500;CH4:64:0.02:500".
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from ase import units
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE

GRAPH = os.environ.get(
    "GRAPH_LONG_RANGE_PATH",
    "/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0")
if GRAPH not in sys.path:
    sys.path.insert(0, GRAPH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from polar_md_q4 import build_box  # noqa: E402


def parse_states(spec):
    out = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        species, nmols, rho, temp = (parts[0], int(parts[1]),
                                     float(parts[2]), float(parts[3]))
        seed = int(parts[4]) if len(parts) > 4 else 20261029
        out.append((species, nmols, rho, temp, seed))
    return out


def run_state(calc, species, nmols, rho, temp, seed, dt_fs, equil_ps, ps,
              frame_ps, out, tag):
    atoms = build_box(species, nmols, rho, seed)
    atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=0.1, steps=300)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
    dyn = Langevin(atoms, timestep=dt_fs * units.fs, temperature_K=temp,
                   friction=0.01)
    n_eq = int(round(equil_ps * 1000.0 / dt_fs))
    n_prod = int(round(ps * 1000.0 / dt_fs))
    every = int(round(frame_ps * 1000.0 / dt_fs))
    t0 = time.time()
    dyn.run(n_eq)
    n_written = [0]

    def dump(*_):
        e = float(atoms.get_potential_energy())
        f = atoms.get_forces()
        atoms.info.update({
            "config_type": f"{species}_dense",
            "REF_energy": e,
            "dataset": "active_cov",
            "state_tag": tag,
            "species": species,
            "density_mol_A3": rho,
            "temperature_K": temp,
            "md_seed": seed,
            "timestep": int(dyn.get_number_of_steps()),
        })
        atoms.arrays["REF_forces"] = f
        write(out, atoms, format="extxyz", append=os.path.exists(out))
        n_written[0] += 1
        print(f"[cov] {tag} step {dyn.get_number_of_steps()} "
              f"E/atom {e/len(atoms):.4f} eV T {atoms.get_temperature():.0f} K "
              f"wall {time.time()-t0:.0f}s", flush=True)

    dyn.attach(dump, interval=every)
    dyn.run(n_prod)
    return n_written[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt-fs", type=float, default=1.0)
    ap.add_argument("--equil-ps", type=float, default=1.0)
    ap.add_argument("--ps", type=float, default=6.0)
    ap.add_argument("--frame-ps", type=float, default=0.5)
    args = ap.parse_args()

    from mace.calculators import mace_polar
    calc = mace_polar(model=args.model, device="cuda", default_dtype="float64")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    manifest = {"states": [], "out": args.out,
                "dt_fs": args.dt_fs, "ps": args.ps}
    for species, nmols, rho, temp, seed in parse_states(args.states):
        tag = f"{species}-{nmols}mol-r{rho:g}-{int(temp)}K-f{seed % 1000}"
        t0 = time.time()
        n = run_state(calc, species, nmols, rho, temp, seed, args.dt_fs,
                      args.equil_ps, args.ps, args.frame_ps, args.out, tag)
        manifest["states"].append({
            "tag": tag, "species": species, "n_molecules": nmols,
            "rho": rho, "T": temp, "seed": seed, "frames": n,
            "wall_s": round(time.time() - t0, 1)})
        print(f"[cov] {tag} done frames={n} wall={time.time()-t0:.0f}s",
              flush=True)
    with open(args.out + ".manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("done", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
