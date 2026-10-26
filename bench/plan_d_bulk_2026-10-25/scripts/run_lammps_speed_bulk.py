#!/usr/bin/env python3
"""Measure the Plan D bulk linear student on the Symmetrix FP32 path.

Same protocol as bench/plan_d_linear_2026-10-23/scripts/run_lammps_speed.py:
one warm-up `run 1`, ten measured `run 1` blocks, three fresh processes.
The JIT artifact is the existing 128-channel linear-student cubin; the bulk
student has the same architecture, only different weights.
"""
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path("/workspace/group/shared/bza/symmetrix-polar/bench/"
            "plan_d_bulk_2026-10-25")
LMP = Path("/workspace/group/shared/bza/external-repos/symmetrix-xl/"
           "symmetrix/build-kernel-r3/lmp")
DATA = Path("/workspace/group/shared/bza/symmetrix-polar/bench/"
            "paper_scale_2026-10-20/water_5184.data")
MODEL = HERE / "models/bulk/plan_d_bulk_symmetrix.json"
CUBIN = Path("/workspace/group/shared/bza/symmetrix-polar/bench/"
             "plan_d_linear_2026-10-23/jit_cache/artifacts/"
             "77e83555653584dac8c1bbc4914aba36e995e70e757b7a32324b7aaf69a4f305/"
             "factorized_cuda_module_gen11_"
             "src847380c859abdbe6d213af7f595ebc0cba0f1986c73e8bce6c2a05b98cffecb1.cubin")
OUT = HERE / "results"
LOOP_RE = re.compile(
    r"Loop time of ([0-9.eE+-]+) on .*? for (\d+) steps with (\d+) atoms")
MEASURED = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(tag):
    case = HERE / "speed" / tag
    case.mkdir(parents=True, exist_ok=True)
    lines = [
        "units metal", "atom_style atomic/kk", "atom_modify map yes",
        "boundary p p p", "package kokkos neigh half newton on",
        "newton on", f"read_data {DATA}",
        "pair_style symmetrix/mace/float32/kk streamed_edges direct "
        f"jit_device_artifact {CUBIN} "
        "jit_device_blocks_per_compute_unit 8",
        f"pair_coeff * * {MODEL} O H",
        "neighbor 2.0 bin",
        "neigh_modify every 100 delay 0 check no",
        "timestep 0.000001", "thermo 1", "thermo_style custom step pe",
        "run_style verlet/kk",
    ]
    lines += ["run 1"] * (1 + MEASURED)
    inp = case / f"in.{tag}"
    inp.write_text("\n".join(lines) + "\n")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "MATOS_POLAR_PROFILE"):
        env.pop(key, None)
    env["LD_LIBRARY_PATH"] = (
        "/workspace/caches/conda/envs/lammps-deepmd/lib:"
        "/opt/matos-tools/cuda/12.9/lib64:"
        "/opt/matos-tools/conda/envs/matos-compute/lib")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [str(LMP), "-k", "on", "g", "1", "-in", str(inp),
         "-log", str(case / "log")],
        cwd=case, env=env, text=True, capture_output=True, timeout=7200)
    (case / "stdout").write_text(proc.stdout)
    loops = [(float(m.group(1)) * 1e3, int(m.group(3)))
             for m in LOOP_RE.finditer(proc.stdout)]
    samples = [v for v, _ in loops[1:1 + MEASURED]]
    atoms = loops[0][1] if loops else None
    result = {
        "tag": tag, "rc": proc.returncode, "atoms": atoms,
        "samples_ms": samples,
        "median_ms": statistics.median(samples) if samples else None,
        "min_ms": min(samples) if samples else None,
        "median_us_per_atom_step": (statistics.median(samples) * 1e3 / atoms)
                                   if samples and atoms else None,
        "min_us_per_atom_step": (min(samples) * 1e3 / atoms)
                                if samples and atoms else None,
        "resolved": [line.strip() for line in proc.stdout.splitlines()
                     if "resolved=" in line][-1:] or None,
        "jit_lines": [line.strip() for line in proc.stdout.splitlines()
                      if "JIT device artifact" in line],
        "neighbor_builds": [line.strip() for line in proc.stdout.splitlines()
                            if "Neighbor list builds" in line],
        "stderr_tail": proc.stderr[-1600:] if proc.returncode else "",
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    OUT.mkdir(parents=True, exist_ok=True)
    if not MODEL.is_file():
        raise SystemExit(f"missing model JSON: {MODEL}")
    results = [run(f"plan_d_bulk_f32_trial{i}") for i in range(n_trials)]
    med = [r["median_us_per_atom_step"] for r in results
           if r["median_us_per_atom_step"] is not None]
    mins = [r["min_us_per_atom_step"] for r in results
            if r["min_us_per_atom_step"] is not None]
    summary = {
        "model": str(MODEL),
        "model_sha256": sha256(MODEL),
        "model_bytes": MODEL.stat().st_size,
        "data": str(DATA),
        "protocol": "1 warm-up + 10 measured run-1 blocks, 3 fresh processes",
        "median_us_per_atom_step_per_trial": med,
        "median_of_trial_medians_us_per_atom_step":
            statistics.median(med) if med else None,
        "min_us_per_atom_step_per_trial": mins,
        "best_min_us_per_atom_step": min(mins) if mins else None,
    }
    print("SUMMARY", json.dumps(summary), flush=True)
    (OUT / "speed_lammps_5184.json").write_text(
        json.dumps({"results": results, "summary": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
