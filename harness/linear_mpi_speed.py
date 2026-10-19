#!/usr/bin/env python3
"""Measure the upstream linear-engine `mpi_message_passing` on 1 vs 2 ranks.

This is the proven Symmetrix MPI design (per-layer H1 exchange), tested on the
same 128-atom box and the same single A100 as the PolarMACE comparison.
"""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
OUT = Path(f"{ROOT}/bench/lammps_parity/linear_mpi")
LMP = "/workspace/group/shared/bza/external-repos/symmetrix-xl/symmetrix/build-lammps-aacd036df381800a/lmp"
MPIEXEC = "/workspace/caches/conda/envs/lammps-deepmd/bin/mpiexec"
MODEL = f"{ROOT}/models/symmetrix/mace-off23-medium.json"
DATA = f"{ROOT}/bench/lammps_parity/engine_compare/nh3_128.data"


def run(nproc):
    tag = f"n{nproc}"
    inp = OUT / f"in.{tag}"
    log = OUT / f"log.{tag}"
    inp.write_text(
        f"""units metal
atom_style atomic/kk
atom_modify map yes
boundary p p p
package kokkos neigh half newton on
newton on
read_data {DATA}
pair_style symmetrix/mace/kk mpi_message_passing
pair_coeff * * {MODEL} N H
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo 1
thermo_style custom step pe
run_style verlet/kk
run 1
"""
    )
    cmd = [LMP, "-k", "on", "g", "1", "-in", str(inp), "-log", str(log)]
    if nproc > 1:
        cmd = [MPIEXEC, "-np", str(nproc)] + cmd
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": (
            "/workspace/caches/conda/envs/lammps-deepmd/lib:"
            "/opt/matos-tools/cuda/12.9/lib64:"
            "/opt/matos-tools/conda/envs/matos-compute/lib"
        ),
        "CUDA_VISIBLE_DEVICES": "0",
    }
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(cmd, cwd=OUT, env=env, text=True, capture_output=True)
    (OUT / f"stdout.{tag}").write_text(proc.stdout)
    (OUT / f"stderr.{tag}").write_text(proc.stderr)
    match = re.search(r"Loop time of ([0-9.eE+-]+) on", proc.stdout)
    loop = float(match.group(1)) if match else None
    return {
        "nproc": nproc,
        "rc": proc.returncode,
        "loop_time_s": loop,
        "ms_per_eval": None if loop is None else loop * 1e3,
        "error": None if proc.returncode == 0 else proc.stderr[-1200:],
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = [run(1), run(2)]
    print(json.dumps(res, indent=1))
    (OUT / "linear_mpi.json").write_text(json.dumps(res, indent=1))
