#!/usr/bin/env python3
"""Add teacher l=0 charges and l=1 atomic dipoles to an extxyz file."""
import json
import math
import sys
from io import StringIO
from pathlib import Path

import ase.io
import numpy as np

INP = Path(sys.argv[1])
OUT = Path(sys.argv[2])
MODEL = ("/workspace/group/shared/bza/task3_sampling_database/mlip/models/"
         "mace-polar/MACE-POLAR-1-M.model")
SCALE = 4.0 * math.pi

from mace.calculators import mace_polar  # noqa: E402

frames = ase.io.read(str(INP), index=":")
print(f"[multipole] {len(frames)} frames", flush=True)
calc = mace_polar(model=MODEL, device="cuda", default_dtype="float64")
q_abs, mu_abs = [], []
with open(OUT, "w") as fh:
    for i, atoms in enumerate(frames):
        a = atoms.copy()
        a.calc = calc
        a.get_potential_energy()
        dc = np.asarray(calc.results["density_coefficients"], dtype=float)
        q = dc[:, 0] * SCALE
        mu = dc[:, 1:4] * SCALE
        a.set_array("REF_charges", q)
        a.set_array("REF_atomic_dipoles", mu)
        buf = StringIO()
        ase.io.write(buf, a, format="extxyz")
        fh.write(buf.getvalue())
        fh.flush()
        q_abs.append(np.abs(q).mean())
        mu_abs.append(np.abs(mu).mean())
        if (i + 1) % 100 == 0:
            print(f"[multipole] {i+1}/{len(frames)} q={np.mean(q_abs[-100:]):.4f} "
                  f"mu={np.mean(mu_abs[-100:]):.4f}", flush=True)
meta = {
    "input": str(INP), "n_frames": len(frames), "scale": SCALE,
    "mean_abs_q": float(np.mean(q_abs)), "mean_abs_mu": float(np.mean(mu_abs)),
    "note": "REF_charges=4*pi*dc[:,0]; REF_atomic_dipoles=4*pi*dc[:,1:4]",
}
OUT.with_suffix(".json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
