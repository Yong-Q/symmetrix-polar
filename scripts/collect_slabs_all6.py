#!/usr/bin/env python3
"""Inhomogeneous slab sampling for all six molecules.

Uses the builder/MD helpers from collect_dense_slabs.py.  8 ps production
(2 ps equilibration + 6 ps sampled at 1 ps) per state.
"""
import importlib.util
from pathlib import Path

import ase.io

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
BENCH = REPO / "bench/plan_d_bulk_2026-10-25"
OUT = REPO / "data_20261025_new/slabs_all6.xyz"

spec = importlib.util.spec_from_file_location("cds", BENCH / "collect_dense_slabs.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from mace.calculators import MACECalculator  # noqa: E402

calc = MACECalculator(model_paths=m.MODEL, device="cuda", default_dtype="float32")
states = [
    ("H2O", 216, 300), ("H2O", 216, 400), ("H2O", 216, 500),
    ("NH3", 125, 300), ("NH3", 125, 400),
    ("CH4", 125, 300), ("CH4", 125, 400),
    ("C2H4", 125, 300), ("C2H6", 125, 300),
    ("CO2", 125, 300), ("CO2", 125, 400),
]
if OUT.exists():
    OUT.unlink()
for i, (sp, n, T) in enumerate(states):
    atoms = m.make_slab(sp, n, 0.035, seed=20261029 + i * 7)
    atoms.info["rho"] = 0.035
    m.run_state(atoms, f"slab-{sp}-0.035-{T}K", T, steps=8000,
                dump_every=1000, out_path=OUT, calc=calc,
                eq_steps=2000, minimize=True)
print("done", OUT, flush=True)
