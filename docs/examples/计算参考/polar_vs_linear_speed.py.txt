#!/usr/bin/env python3
"""Compare the linear MACE engine, the nonlinear backbone, and full PolarMACE.

All runs use the same NH3 x32 data file, one MPI rank, one A100, and `run 1`.
The LAMMPS `Loop time` excludes model loading and is reported as ms/eval.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/bench")
import mace_md as mm  # noqa: E402
from polar_lammps_parity import LMP, write_data  # noqa: E402

OUT = Path(f"{ROOT}/bench/lammps_parity/engine_compare")
MODELS = [
    (
        "linear MACE-OFF23 medium",
        f"{ROOT}/models/symmetrix/mace-off23-medium.json",
        "MACE (linear engine)",
    ),
    (
        "nonlinear backbone only",
        f"{ROOT}/models/mace-polar/polar_symmetrix_probe_polaroff.json",
        "MACE_Nonlinear, polar blocks off",
    ),
    (
        "full PolarMACE v29",
        f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json",
        "MACE_Nonlinear + polar",
    ),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    atoms, _, _ = mm.build_gas("NH3", 32, 0.01, 770600, 2.0)
    atoms.info["charge"] = 0.0
    atoms.info["spin"] = 1.0
    data_path = OUT / "nh3_128.data"
    write_data(data_path, atoms)

    env = {
        **__import__("os").environ,
        "LD_LIBRARY_PATH": (
            "/workspace/caches/conda/envs/lammps-deepmd/lib:"
            "/opt/matos-tools/cuda/12.9/lib64:"
            "/opt/matos-tools/conda/envs/matos-compute/lib"
        ),
        "CUDA_VISIBLE_DEVICES": "0",
    }
    env.pop("PYTHONPATH", None)

    results = []
    for label, model, note in MODELS:
        input_path = OUT / f"in.{label.replace(' ', '_')}"
        log_path = OUT / f"log.{label.replace(' ', '_')}"
        input_path.write_text(
            f"""units metal
atom_style atomic/kk
atom_modify map yes
boundary p p p
package kokkos neigh half newton on
newton on
read_data {data_path.resolve()}
pair_style symmetrix/mace/kk charge 0.0 spin 1.0
pair_coeff * * {model} N H
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo 1
thermo_style custom step pe
run_style verlet/kk
run 1
"""
        )
        proc = subprocess.run(
            [LMP, "-k", "on", "g", "1", "-in", str(input_path),
             "-log", str(log_path)],
            cwd=OUT, env=env, text=True, capture_output=True,
        )
        (OUT / f"stdout.{label.replace(' ', '_')}").write_text(proc.stdout)
        (OUT / f"stderr.{label.replace(' ', '_')}").write_text(proc.stderr)
        loop_time = None
        match = re.search(r"Loop time of ([0-9.eE+-]+) on", proc.stdout)
        if match:
            loop_time = float(match.group(1))
        pe = None
        for line in proc.stdout.splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0].isdigit():
                try:
                    pe = float(fields[1])
                except ValueError:
                    pass
        results.append({
            "label": label,
            "note": note,
            "model": model,
            "rc": proc.returncode,
            "loop_time_s": loop_time,
            "ms_per_eval": None if loop_time is None else loop_time * 1e3,
            "pe": pe,
            "error": None if proc.returncode == 0 else proc.stderr[-1500:],
        })
        print(json.dumps(results[-1], indent=1))

    (OUT / "engine_compare.json").write_text(json.dumps(results, indent=1))

    # simple bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
        ok = [r for r in results if r["ms_per_eval"] is not None]
        fig, ax = plt.subplots(figsize=(9, 4.6))
        y = np.arange(len(ok))[::-1]
        colors = ["#2da44e", "#bf8700", "#1f6feb"]
        ax.barh(y, [r["ms_per_eval"] for r in ok], color=colors[:len(ok)],
                height=0.55)
        for yi, r in zip(y, ok):
            ax.text(r["ms_per_eval"] * 1.05, yi,
                    f'{r["ms_per_eval"]:.1f} ms', va="center",
                    fontsize=10, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels([r["label"] for r in ok])
        ax.set_xlabel("LAMMPS pair evaluation, 1 rank (ms, log scale)")
        ax.set_title("Same 128-atom NH3 box: linear MACE vs nonlinear PolarMACE")
        fig.tight_layout()
        fig.savefig(OUT / "engine_compare.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print("plot skipped:", exc)


if __name__ == "__main__":
    main()
