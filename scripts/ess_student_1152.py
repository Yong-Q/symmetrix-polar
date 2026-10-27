#!/usr/bin/env python3
"""Student->POLAR reweighting ESS on the 1,152-atom 500 ps frames."""
import io
import json
from pathlib import Path

import ase.io
import numpy as np

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
PKG = Path("/workspace/group/shared/bza/test/validation/"
           "off23_500ps_water1152_2026-10-23")
MODEL = (REPO / "bench/plan_d_bulk_2026-10-25/models/bulk/plan_d_bulk.model")
OUT = (REPO / "bench/plan_d_bulk_2026-10-25/results/"
       "ess_student_1152.json")
KB = 8.617333262e-5
T = 300.0


def read_dump(path):
    text = Path(path).read_text()
    for chunk in text.split("ITEM: TIMESTEP")[1:]:
        body = "ITEM: TIMESTEP" + chunk
        if "ITEM: ATOMS" not in body:
            continue
        head, _, _ = body.partition("ITEM: ATOMS")
        try:
            n_atoms = int(head.split("ITEM: NUMBER OF ATOMS")[1].split()[0])
        except Exception:
            continue
        ts = int(chunk.strip().splitlines()[0])
        atom_lines = body.split("ITEM: ATOMS", 1)[1].strip().splitlines()
        if len(atom_lines) < n_atoms:
            continue
        atoms = ase.io.read(io.StringIO(body), format="lammps-dump-text")
        z = np.asarray(atoms.numbers)
        atoms.set_chemical_symbols(["O" if int(t) == 1 else "H" for t in z])
        yield ts, atoms


def ess(dE, T=T):
    b = 1.0 / (KB * T)
    dE = np.asarray(dE, float)
    dE = dE - dE.mean()
    lw = -b * dE
    lw -= lw.max()
    w = np.exp(lw)
    w /= w.sum()
    return {"n": int(len(dE)), "sigma_beta": float(np.std(b * dE)),
            "neff": float(1.0 / np.sum(w * w)),
            "ess_fraction": float(1.0 / np.sum(w * w) / len(dE))}


from mace.calculators import MACECalculator  # noqa: E402

calc = MACECalculator(model_paths=str(MODEL), device="cuda",
                      default_dtype="float64")
ref = json.loads((PKG / "reweight_frames.json").read_text())
e_polar = np.array([f["e_target_eV"] for f in ref])

e_student = []
for ts, atoms in read_dump(PKG / "traj_500ps_1fs.lammpstrj"):
    atoms.calc = calc
    e_student.append(float(atoms.get_potential_energy()))
e_student = np.array(e_student)
print("frames", len(e_student), len(e_polar), flush=True)
n = min(len(e_student), len(e_polar))
dE = e_polar[:n] - e_student[:n]
res = {"all": ess(dE), "per_atom_std_meV": float(np.std(dE) / 1152 * 1e3),
       "per_atom_mean_meV": float(np.mean(dE) / 1152 * 1e3)}
for i in range(4):
    sl = slice(i * n // 4, (i + 1) * n // 4)
    res[f"block{i}"] = ess(dE[sl])
res["dE_std_eV"] = float(np.std(dE))
res["dE_mean_eV"] = float(np.mean(dE))
res["dE_per_frame_eV"] = [float(x) for x in dE]
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
