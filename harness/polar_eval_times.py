#!/usr/bin/env python3
"""Print per-eval times (forced recompute) to expose bimodal/allocation drift."""
import os
import sys
import time

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
import mace_md as mm  # noqa: E402

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
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

from symmetrix.calculator import Symmetrix  # noqa: E402

atoms, _, _ = mm.build_gas("NH3", 32, 0.01, 770600, 2.0)
atoms.info["charge"] = 0.0
atoms.info["spin"] = 1.0
engine = Symmetrix(
    f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json", dtype="float64"
)
print("WARMUP")
for _ in range(3):
    engine.atoms = None
    engine.results = {}
    _ = engine.get_potential_energy(atoms)
    _ = engine.get_forces(atoms)
times = []
for i in range(6):
    engine.atoms = None
    engine.results = {}
    t0 = time.perf_counter()
    _ = engine.get_potential_energy(atoms)
    _ = engine.get_forces(atoms)
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"EVAL {i:2d}: {dt*1e3:9.1f} ms")
times = sorted(times)
print(f"MIN={times[0]*1e3:.1f}  P10={times[1]*1e3:.1f}  "
      f"MEDIAN={times[len(times)//2]*1e3:.1f}  MAX={times[-1]*1e3:.1f} ms")
