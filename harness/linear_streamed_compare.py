#!/usr/bin/env python3
"""Isolate the execution-path effect on the same linear MACE model.

Runs the linear MACE-OFF23 medium model on the 128-atom box with
`streamed_edges auto` (resolves generic in this build) and with
`streamed_edges materialized`, one rank, and reports the loop time.
"""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
OUT = Path(f"{ROOT}/bench/lammps_parity/streamed_compare")
LMP = "/workspace/group/shared/bza/external-repos/symmetrix-xl/symmetrix/build-lammps-aacd036df381800a/lmp"
MODEL = f"{ROOT}/models/symmetrix/mace-off23-medium.json"
DATA = f"{ROOT}/bench/lammps_parity/engine_compare/nh3_128.data"


def run(tag, mode):
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
pair_style symmetrix/mace/kk streamed_edges {mode}
pair_coeff * * {MODEL} N H
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo 1
thermo_style custom step pe
run_style verlet/kk
run 1
"""
    )
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
    proc = subprocess.run(
        [LMP, "-k", "on", "g", "1", "-in", str(inp), "-log", str(log)],
        cwd=OUT, env=env, text=True, capture_output=True,
    )
    (OUT / f"stdout.{tag}").write_text(proc.stdout)
    match = re.search(r"Loop time of ([0-9.eE+-]+) on", proc.stdout)
    resolved = None
    for line in proc.stdout.splitlines():
        if "Symmetrix streamed_edges requested" in line:
            resolved = line.strip()
    return {
        "requested": mode,
        "resolved": resolved,
        "rc": proc.returncode,
        "ms_per_eval": None if not match else float(match.group(1)) * 1e3,
        "error": None if proc.returncode == 0 else proc.stderr[-1200:],
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    results = [run("generic", "auto"), run("materialized", "materialized")]
    print(json.dumps(results, indent=1))
    (OUT / "streamed_compare.json").write_text(json.dumps(results, indent=1))
