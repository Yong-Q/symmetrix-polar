#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PolarMACE 静电层中间量验收夹具(P3 gate fixtures, 2026-10-15)。

对每个构型(NH3/CH4/N2/CO2 生产盒 × 2 帧)落盘:
  - 输入: symbols/positions/cell/molecule_index
  - 静电层出口: spin_charge_density(128,2,4)/density_coefficients(128,4)/
    charges/spins/dipole/total_charge
  - 分量能量: electron_energy / electrostatic_energy / interaction_energy / energy
  - 各多极层: charges_history(若暴露)
用途:引擎 C++ 每实现一个静电模块,即以其对应量在夹具上的逐位/近位对照为门。

用法: CUDA_VISIBLE_DEVICES=<gpu> <polar venv python> polar_intermediates_dump.py
"""
import json
import os
import sys

import numpy as np

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
import mace_md as mm  # noqa: E402

POLAR = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"
OUTDIR = f"{ROOT}/bench/polar_fixtures"
MOLS = [("NH3", 770600), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]


def main():
    from mace.calculators import mace_polar
    os.makedirs(OUTDIR, exist_ok=True)
    calc = mace_polar(model=POLAR, device="cuda", default_dtype="float64")

    index = {}
    for mol, seed in MOLS:
        base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
        for k in range(2):
            atoms = base.copy()
            if k == 1:
                atoms.rattle(stdev=0.05, seed=seed + 7)
            atoms.info["charge"] = 0
            atoms.info["spin"] = 1
            atoms.calc = calc
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            r = calc.results
            name = f"{mol.lower()}-f{k}"
            np.savez(
                f"{OUTDIR}/{name}.npz",
                symbols=np.asarray(atoms.get_chemical_symbols(), dtype="U2"),
                positions=atoms.get_positions(),
                cell=np.asarray(atoms.get_cell()),
                molecule_index=np.repeat(np.arange(32), len(atoms) // 32),
                spin_charge_density=np.asarray(r["spin_charge_density"]),
                density_coefficients=np.asarray(r["density_coefficients"]),
                charges=np.asarray(r["charges"]),
                spins=np.asarray(r["spins"]),
                dipole=np.asarray(r["dipole"]),
                forces=np.asarray(f),
            )
            comp = {
                "energy": float(e),
                "electron_energy": float(r["electron_energy"]),
                "electrostatic_energy": float(r["electrostatic_energy"]),
                "interaction_energy": float(r["interaction_energy"]),
                "total_charge": float(r.get("total_charge", 0.0)),
            }
            index[name] = comp
            print(f"  {name}: E={comp['energy']:.6f}  le={comp['electron_energy']:+.6f}  "
                  f"es={comp['electrostatic_energy']:+.6f}  int={comp['interaction_energy']:+.6f}", flush=True)
    with open(f"{OUTDIR}/index.json", "w") as fh:
        json.dump(index, fh, indent=1)
    print(f"FIXTURES-DONE -> {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
