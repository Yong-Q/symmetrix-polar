#!/usr/bin/env python3
"""Path B smoke: six-molecule single-point teacher labels with PolarMACE.

Builds low-density periodic boxes for H2O, NH3, C2H4, CH4, C2H6, CO2,
rattles them, and evaluates the FP64 PolarMACE teacher. Writes one extxyz per
species plus a summary JSON. This is a data/validation smoke, not a training
run.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import ase
import ase.io
from ase.build import molecule
from scipy.spatial.transform import Rotation

from mace.calculators import mace_polar


def build_box(name, nmol, seed, rattle, side=30.0):
    rng = np.random.default_rng(seed)
    base = molecule(name)
    grid = np.linspace(4.0, side - 4.0, 3)
    slots = [(x, y, z) for x in grid for y in grid for z in grid]
    rng.shuffle(slots)
    out = ase.Atoms(cell=[side, side, side], pbc=True)
    for i in range(nmol):
        m = base.copy()
        # random rigid rotation around the molecular centroid
        c = m.positions.mean(axis=0)
        R = Rotation.random(random_state=rng).as_matrix()
        m.positions = (m.positions - c) @ R.T + c
        # small bond/angle distortion, then place
        m.positions = c + (m.positions - c) * (1.0 + rng.uniform(-0.03, 0.03))
        m.positions += np.array(slots[i]) + rng.normal(0.0, 0.4, 3)
        m.positions += rng.normal(0.0, rattle, m.positions.shape)
        out.extend(m)
    out.cell = [side, side, side]
    out.pbc = True
    out.info["charge"] = 0
    out.info["spin"] = 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-configs", type=int, default=8)
    ap.add_argument("--n-molecules", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    species = ["H2O", "NH3", "C2H4", "CH4", "C2H6", "CO2"]
    calc = mace_polar(model=args.model, device="cuda", default_dtype="float64")
    summary = {}

    for si, name in enumerate(species):
        rows = []
        path = out / f"{name}_single_points.xyz"
        if path.exists():
            path.unlink()
        for i in range(args.n_configs):
            atoms = build_box(name, args.n_molecules, seed=20261022 + si * 100 + i,
                              rattle=0.02 + 0.01 * (i % 5))
            atoms.calc = calc
            t0 = time.perf_counter()
            e = float(atoms.get_potential_energy())
            f = np.asarray(atoms.get_forces(), dtype=np.float64)
            dt = time.perf_counter() - t0
            atoms.info["config_type"] = f"{name}_{i}"
            ase.io.write(path, atoms, format="extxyz", append=True)
            rows.append({
                "config": i,
                "atoms": len(atoms),
                "energy_eV": e,
                "f_rms_eV_A": float(np.sqrt((f ** 2).mean())),
                "f_max_eV_A": float(np.abs(f).max()),
                "wall_s": dt,
            })
            print(name, i, len(atoms), e, rows[-1]["f_max_eV_A"], round(dt, 3),
                  flush=True)
        energies = np.array([r["energy_eV"] for r in rows])
        summary[name] = {
            "n_configs": len(rows),
            "atoms_per_box": rows[0]["atoms"],
            "energy_mean_eV": float(energies.mean()),
            "energy_std_eV": float(energies.std()),
            "energy_min_eV": float(energies.min()),
            "energy_max_eV": float(energies.max()),
            "f_rms_mean_eV_A": float(np.mean([r["f_rms_eV_A"] for r in rows])),
            "f_max_max_eV_A": float(np.max([r["f_max_eV_A"] for r in rows])),
            "xyz": str(path),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("WROTE", out / "summary.json")


if __name__ == "__main__":
    main()
