#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P5 gate A: LAMMPS pair-style single-point energy/force parity.

Builds the same NH3 x32 frame used by the engine end-to-end parity gate, writes
a LAMMPS data file, runs one `run 0` with `pair_style symmetrix/mace/kk
no_domain_decomposition`, and compares LAMMPS energy/forces against the
Symmetrix Python engine on the identical ASE Atoms object.

The engine itself is already validated against PyTorch PolarMACE at
rel <= 4.2e-12 (energy) and <= 1.6e-7 eV/A (forces), so this gate isolates the
LAMMPS wiring: data-file mapping, neighbor graph, cell/positions, force sign,
and polar node-force addition.
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
import mace_md as mm  # noqa: E402

MODEL_JSON = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
LMP = "/workspace/group/shared/bza/external-repos/symmetrix-xl/symmetrix/build-lammps-aacd036df381800a/lmp"
OUTDIR = Path(f"{ROOT}/bench/lammps_parity")

MASSES = {1: 14.0067, 2: 1.008}  # N, H


def write_data(path, atoms):
    cell = np.asarray(atoms.cell, dtype=float)
    # LAMMPS triclinic convention: a=(lx,0,0), b=(xy,ly,0), c=(xz,yz,lz)
    lx = cell[0, 0]
    ly = cell[1, 1]
    lz = cell[2, 2]
    xy, xz, yz = cell[0, 1], cell[0, 2], cell[1, 2]
    offdiag = max(abs(cell[0, 1]), abs(cell[0, 2]), abs(cell[1, 0]),
                  abs(cell[1, 2]), abs(cell[2, 0]), abs(cell[2, 1]))
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    type_of = {7: 1, 1: 2}
    if set(np.unique(numbers)) - set(type_of):
        raise RuntimeError(f"unexpected elements: {np.unique(numbers)}")
    types = np.array([type_of[int(z)] for z in numbers], dtype=int)
    lines = ["PolarMACE LAMMPS parity data", ""]
    lines.append(f"{len(atoms)} atoms")
    lines.append("2 atom types")
    lines.append("")
    if offdiag > 1e-12:
        lines.append(f"0.0 {lx:.16f} xlo xhi")
        lines.append(f"0.0 {ly:.16f} ylo yhi")
        lines.append(f"0.0 {lz:.16f} zlo zhi")
        lines.append(f"{xy:.16f} {xz:.16f} {yz:.16f} xy xz yz")
    else:
        lines.append(f"0.0 {lx:.16f} xlo xhi")
        lines.append(f"0.0 {ly:.16f} ylo yhi")
        lines.append(f"0.0 {lz:.16f} zlo zhi")
    lines += ["", "Masses", "", "1 14.0067", "2 1.008", "", "Atoms # atomic", ""]
    for atom_id, (t, xyz) in enumerate(zip(types, positions), start=1):
        lines.append(
            f"{atom_id} {t} {xyz[0]:.16f} {xyz[1]:.16f} {xyz[2]:.16f}"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def write_input(path, data_path, model_path, dump_path):
    path.write_text(
        f"""units metal
atom_style atomic/kk
atom_modify map yes
boundary p p p
package kokkos neigh half newton on
newton on
read_data {data_path}
pair_style symmetrix/mace/kk no_domain_decomposition charge 0.0 spin 1.0
pair_coeff * * {model_path} N H
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo 1
thermo_style custom step pe
thermo_modify format float %.15g
run_style verlet/kk
run 0
write_dump all custom {dump_path} id type x y z fx fy fz modify format float %.15g
"""
    )


def parse_dump(path):
    lines = path.read_text().splitlines()
    i = lines.index("ITEM: ATOMS id type x y z fx fy fz")
    rows = [line.split() for line in lines[i + 1:] if line.strip()]
    data = np.array(rows, dtype=float)
    data = data[np.argsort(data[:, 0])]
    ids = data[:, 0].astype(int)
    return ids, data[:, 1].astype(int), data[:, 2:5], data[:, 5:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", default="NH3")
    ap.add_argument("--nmol", type=int, default=32)
    ap.add_argument("--density", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=770600)
    ap.add_argument("--rattle", type=float, default=0.0)
    ap.add_argument("--energy-abs", type=float, default=5e-5)
    ap.add_argument("--force-max", type=float, default=1e-4)
    ap.add_argument("--out", default=str(OUTDIR / "polar_lammps_parity"))
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    atoms, _, _ = mm.build_gas(args.mol, args.nmol, args.density, args.seed, 2.0)
    if args.rattle > 0:
        rng = np.random.default_rng(args.seed + 17)
        atoms.set_positions(
            atoms.get_positions()
            + rng.normal(scale=args.rattle, size=atoms.get_positions().shape)
        )
    atoms.info["charge"] = 0.0
    atoms.info["spin"] = 1.0

    data_path = OUTDIR / "polar_parity.data"
    input_path = OUTDIR / "in.polar_parity"
    dump_path = OUTDIR / "polar_parity_forces.dump"
    log_path = OUTDIR / "polar_parity.log"
    write_data(data_path, atoms)
    write_input(input_path, data_path.resolve(), MODEL_JSON, dump_path.resolve())

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["LD_LIBRARY_PATH"] = (
        "/opt/matos-tools/cuda/12.9/lib64:"
        "/opt/matos-tools/conda/envs/matos-compute/lib"
    )
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    print(f"LAMMPS: {LMP}")
    print(f"model : {MODEL_JSON}")
    print(f"system: {args.mol} x{args.nmol} ({len(atoms)} atoms), box={atoms.cell.lengths()}")
    proc = subprocess.run(
        [LMP, "-k", "on", "g", "1", "-in", str(input_path), "-log", str(log_path)],
        cwd=OUTDIR,
        env=env,
        text=True,
        capture_output=True,
    )
    (OUTDIR / "lammps.stdout").write_text(proc.stdout)
    (OUTDIR / "lammps.stderr").write_text(proc.stderr)
    print(proc.stdout[-2500:])
    if proc.returncode != 0 and not dump_path.exists():
        print(proc.stderr[-3000:])
        raise SystemExit(f"LAMMPS failed with rc={proc.returncode}")

    # LAMMPS total potential energy from the thermo table.
    pe = None
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].isdigit():
            try:
                pe = float(fields[1])
            except ValueError:
                pass
    if pe is None:
        raise RuntimeError("could not parse LAMMPS PotEng from stdout")

    dump_ids, dump_types, dump_pos, dump_forces = parse_dump(dump_path)
    if not np.array_equal(dump_ids, np.arange(1, len(atoms) + 1)):
        raise RuntimeError("dump atom ordering mismatch")

    # Reference: the same Symmetrix f64 engine through the ASE calculator.
    os.environ.setdefault("SYMMETRIX_BACKEND", "cuda12-sm80")
    os.environ.setdefault(
        "SYMMETRIX_BACKEND_DESCRIPTOR_PATH",
        f"{ROOT}/models/mace-polar/backend_cuda12_sm80.json",
    )
    os.environ["PATH"] = f"{ROOT}/bench/nvidia_smi_shim:" + os.environ["PATH"]
    os.environ["LD_LIBRARY_PATH"] = (
        "/opt/matos-tools/cuda/12.9/lib64:"
        "/opt/matos-tools/conda/envs/matos-compute/lib:"
        + os.environ.get("LD_LIBRARY_PATH", "")
    )
    from symmetrix.calculator import Symmetrix

    engine = Symmetrix(MODEL_JSON, dtype="float64")
    ref = engine.get_forces(atoms)
    ref_e = engine.get_potential_energy(atoms)

    d_e = pe - ref_e
    d_f = dump_forces - ref
    max_df = float(np.max(np.linalg.norm(d_f, axis=1)))
    mean_df = float(np.mean(np.linalg.norm(d_f, axis=1)))
    fmax = float(np.max(np.linalg.norm(ref, axis=1)))
    box_lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    position_delta = dump_pos - atoms.get_positions()
    position_delta -= np.round(position_delta / box_lengths) * box_lengths
    pos_drift = float(np.max(np.abs(position_delta)))

    result = {
        "gate": "P5-A",
        "system": {
            "mol": args.mol,
            "nmol": args.nmol,
            "atoms": len(atoms),
            "cell_lengths": atoms.cell.lengths().tolist(),
            "rattle": args.rattle,
        },
        "energy": {
            "lammps": pe,
            "engine": ref_e,
            "dE": d_e,
            "dE_abs": abs(d_e),
            "tol_abs": args.energy_abs,
            "pass": bool(abs(d_e) <= args.energy_abs),
        },
        "forces": {
            "max_dF": max_df,
            "mean_dF": mean_df,
            "max_engine_force": fmax,
            "tol_max": args.force_max,
            "pass": bool(max_df <= args.force_max),
        },
        "geometry": {"max_position_drift": pos_drift},
    }
    result["pass"] = bool(result["energy"]["pass"] and result["forces"]["pass"])
    out_json = Path(args.out + ".json")
    out_json.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].scatter(ref.ravel(), dump_forces.ravel(), s=8, alpha=0.7)
        lim = [min(ref.min(), dump_forces.min()), max(ref.max(), dump_forces.max())]
        axes[0].plot(lim, lim, "k--", lw=0.8)
        axes[0].set_xlabel("Engine force component (eV/Å)")
        axes[0].set_ylabel("LAMMPS force component (eV/Å)")
        axes[0].set_title(f"Force parity: max|ΔF|={max_df:.2e}")
        axes[1].bar(["energy", "forces"], [
            abs(d_e) / args.energy_abs,
            max_df / args.force_max,
        ], color=["C0", "C1"])
        axes[1].axhline(1.0, color="r", ls="--", lw=0.8)
        axes[1].set_ylabel("error / tolerance")
        axes[1].set_title("P5-A gate margins")
        fig.tight_layout()
        fig.savefig(args.out + ".png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"plot skipped: {exc}")

    print("GATE PASS" if result["pass"] else "GATE FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
