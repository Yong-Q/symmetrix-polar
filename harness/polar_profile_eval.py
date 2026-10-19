#!/usr/bin/env python3
"""Warm-up + repeated eval under MATOS_POLAR_PROFILE=1 for stage breakdown."""
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
engine.get_potential_energy(atoms)
engine.get_forces(atoms)
for i in range(8):
    engine.atoms = None
    engine.results = {}
    t0 = time.perf_counter()
    e = engine.get_potential_energy(atoms)
    f = engine.get_forces(atoms)
    print(f"EVAL {i}: E={e:.9f} eV  t={time.perf_counter()-t0:.3f} s")
print("PROFILE-DONE")
