#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P5 gate B: short NVT MD stability smoke for the PolarMACE LAMMPS pair style.

Runs 50 steps (0.5 fs) of NVT on the same NH3 x32 system used by the parity
gate. Checks completion, finite thermodynamics, bounded temperature/energy,
and reports the per-step wall time.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/bench")
import mace_md as mm  # noqa: E402
from polar_lammps_parity import LMP, MODEL_JSON, OUTDIR, write_data  # noqa: E402

MD_DIR = OUTDIR / "md"
STEPS = 50
TIMESTEP_FS = 0.5
TEMPERATURE = 300.0
SEED = 4928459


def main():
    MD_DIR.mkdir(parents=True, exist_ok=True)
    atoms, _, _ = mm.build_gas("NH3", 32, 0.01, 770600, 2.0)
    atoms.info["charge"] = 0.0
    atoms.info["spin"] = 1.0
    data_path = MD_DIR / "polar_md.data"
    input_path = MD_DIR / "in.polar_md"
    log_path = MD_DIR / "polar_md.log"
    write_data(data_path, atoms)
    input_path.write_text(
        f"""units metal
atom_style atomic/kk
atom_modify map yes
boundary p p p
package kokkos neigh half newton on
newton on
read_data {data_path.resolve()}
pair_style symmetrix/mace/kk no_domain_decomposition charge 0.0 spin 1.0
pair_coeff * * {MODEL_JSON} N H
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
timestep {TIMESTEP_FS/1000.0}
velocity all create {TEMPERATURE} {SEED} mom yes rot yes dist gaussian
fix 1 all nvt temp {TEMPERATURE} {TEMPERATURE} 0.05
thermo 5
thermo_style custom step temp pe ke etotal press
thermo_modify format float %.15g
run_style verlet/kk
run {STEPS}
"""
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["LD_LIBRARY_PATH"] = (
        "/opt/matos-tools/cuda/12.9/lib64:"
        "/opt/matos-tools/conda/envs/matos-compute/lib"
    )
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    print(f"LAMMPS: {LMP}\nsystem: {len(atoms)} atoms, {STEPS} steps x {TIMESTEP_FS} fs")
    proc = subprocess.run(
        [LMP, "-k", "on", "g", "1", "-in", str(input_path), "-log", str(log_path)],
        cwd=MD_DIR,
        env=env,
        text=True,
        capture_output=True,
    )
    (MD_DIR / "lammps.stdout").write_text(proc.stdout)
    (MD_DIR / "lammps.stderr").write_text(proc.stderr)
    print(proc.stdout[-3000:])
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        raise SystemExit(f"LAMMPS MD failed with rc={proc.returncode}")

    # Parse the thermo table: Step Temp PotEng KinEng TotEng Press.
    rows = []
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) == 6 and fields[0].isdigit():
            try:
                rows.append([float(v) for v in fields])
            except ValueError:
                pass
    if not rows:
        raise RuntimeError("could not parse the LAMMPS thermo table")
    table = np.asarray(rows, dtype=float)
    steps, temp, pe, ke, etotal, press = table.T

    # Parse loop time (excludes model load).
    loop_time = None
    match = re.search(r"Loop time of ([0-9.eE+-]+) on", proc.stdout)
    if match:
        loop_time = float(match.group(1))

    lost = "Lost atoms" in proc.stdout or "Lost atoms" in proc.stderr
    errors = "ERROR" in proc.stdout or "ERROR" in proc.stderr
    finite = bool(np.all(np.isfinite(table)))
    temp_ok = bool(np.max(np.abs(temp)) < 1000.0)
    energy_span = float(np.max(np.abs(etotal - etotal[0])))
    energy_ok = energy_span < 100.0  # eV; 128 atoms, short NVT smoke
    result = {
        "gate": "P5-B",
        "system": {"atoms": len(atoms), "steps": STEPS, "timestep_fs": TIMESTEP_FS},
        "completed": bool(len(rows) >= 2 and steps[-1] >= STEPS),
        "thermo_rows": int(len(rows)),
        "temperature": {
            "target": TEMPERATURE,
            "min": float(temp.min()),
            "max": float(temp.max()),
            "mean": float(temp.mean()),
            "ok": temp_ok,
        },
        "energy": {
            "pe_first": float(pe[0]),
            "pe_last": float(pe[-1]),
            "etotal_first": float(etotal[0]),
            "etotal_last": float(etotal[-1]),
            "etotal_span": energy_span,
            "ok": energy_ok,
        },
        "pressure": {
            "min": float(press.min()),
            "max": float(press.max()),
            "last": float(press[-1]),
        },
        "lost_atoms": lost,
        "errors": errors,
        "finite": finite,
        "loop_time_s": loop_time,
        "ms_per_step": (
            None if loop_time is None else 1000.0 * loop_time / max(STEPS, 1)
        ),
    }
    result["pass"] = bool(
        result["completed"]
        and not lost
        and not errors
        and finite
        and temp_ok
        and energy_ok
    )
    out_json = MD_DIR / "polar_md.json"
    out_json.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].plot(steps, temp, "o-", ms=3)
        axes[0].axhline(TEMPERATURE, color="r", ls="--", lw=0.8)
        axes[0].set_xlabel("MD step")
        axes[0].set_ylabel("Temperature (K)")
        axes[0].set_title("NVT temperature")
        axes[1].plot(steps, pe, "o-", ms=3, label="Potential")
        axes[1].plot(steps, etotal, "s-", ms=3, label="Total")
        axes[1].set_xlabel("MD step")
        axes[1].set_ylabel("Energy (eV)")
        axes[1].set_title("Energy stability")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(MD_DIR / "polar_md.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"plot skipped: {exc}")

    print("GATE PASS" if result["pass"] else "GATE FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
