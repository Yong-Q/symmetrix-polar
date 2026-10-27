#!/usr/bin/env python3
"""Light dense coverage batch for the states not finished by the first pass."""
import importlib.util
from pathlib import Path

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
BENCH = REPO / "bench/plan_d_bulk_2026-10-25"
OUT = REPO / "data_20261025_new/dense_remaining.xyz"

spec = importlib.util.spec_from_file_location("cds", BENCH / "collect_dense_slabs.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from mace.calculators import MACECalculator  # noqa: E402

calc = MACECalculator(model_paths=m.MODEL, device="cuda", default_dtype="float32")
states = []
for sp in ("NH3", "CH4", "C2H4", "C2H6", "CO2"):
    for T in (300, 400, 500):
        if sp == "NH3" and T == 300:
            continue
        states.append((sp, T))
if OUT.exists():
    OUT.unlink()
for i, (sp, T) in enumerate(states):
    atoms = m.build_box(sp, 64, 0.035, seed=20261030 + i * 11)
    atoms.info["rho"] = 0.035
    m.run_state(atoms, f"dense-{sp}-0.035-{T}K", T, steps=2500,
                dump_every=1000, out_path=OUT, calc=calc, eq_steps=500)
print("done", OUT, flush=True)
