#!/usr/bin/env python3
"""Speed benchmark: MACELES (short-range MACE + Latent Ewald Summation) vs
plain MACE with the same architecture, on a 1,152-atom water box.

Both models are trained for one epoch on a tiny synthetic water set, only to
obtain a valid checkpoint; the timing measures the forward pass, not accuracy.

The MACELES forward includes the LES long-range electrostatics by
construction.  The MACE run is the same backbone without the `les` module, so
`t(MACELES) - t(MACE)` isolates the electrostatic overhead.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.build import molecule
from ase.io import iread, write

from mace.calculators import MACECalculator
from mace.cli.run_train import run as mace_run
from mace.tools.arg_parser import build_default_arg_parser

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
WATER_TRAJ = REPO / "bench/off23_500ps_water1152_2026-10-23/traj_500ps_1fs.lammpstrj"


def make_fit_set(path: Path, n=24):
    wat = molecule("H2O")
    rng = np.random.default_rng(5)
    cfgs = []
    # isolated atoms so MACE can derive E0s
    for z in (8, 1):
        iso = Atoms(numbers=[z], positions=[[0, 0, 0]], cell=[6, 6, 6],
                    pbc=True)
        iso.info["REF_energy"] = 0.0
        iso.info["config_type"] = "IsolatedAtom"
        cfgs.append(iso)
    for _ in range(n):
        c = wat.copy()
        c.positions += rng.normal(0.1, size=c.positions.shape)
        c.set_cell([4.0, 4.0, 4.0])
        c.set_pbc(True)
        c.info["REF_energy"] = float(rng.normal(0.1))
        c.new_array("REF_forces", rng.normal(0.1, size=c.positions.shape))
        cfgs.append(c)
    write(path, cfgs)


def train(model_kind: str, work: Path, device: str, epochs: int):
    fit = work / "fit.xyz"
    if not fit.exists():
        make_fit_set(fit)
    params = {
        "name": model_kind,
        "model": model_kind,
        "valid_fraction": 0.05,
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "stress_weight": 0.0,
        "hidden_irreps": "128x0e",
        "r_max": 5.0,
        "batch_size": 4,
        "max_num_epochs": epochs,
        "device": device,
        "seed": 5,
        "loss": "ef",
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "eval_interval": 1,
        "checkpoints_dir": str(work / model_kind),
        "model_dir": str(work / model_kind),
        "train_file": str(fit),
        "plot": False,
    }
    argv = [f"--{k}={v}" for k, v in params.items()]
    args = build_default_arg_parser().parse_args(argv)
    mace_run(args)
    return Path(work / model_kind / f"{model_kind}.model")


def water_box(n_atoms=1152):
    for atoms in iread(WATER_TRAJ, format="lammps-dump-text",
                       specorder=["O", "H"]):
        return atoms
    raise RuntimeError("empty water trajectory")


def bench(model_path: Path, atoms: Atoms, device: str, reps: int):
    # float64: LES builds its Ewald matrices in double precision, and a
    # float32 calculator trips an LU dtype mismatch inside the solve.
    calc = MACECalculator(model_paths=str(model_path), device=device,
                          default_dtype="float64")
    a = atoms.copy()
    a.calc = calc
    for _ in range(3):
        a.get_potential_energy()
        a.get_forces()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        calc.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        e = a.get_potential_energy()
        f = a.get_forces()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return {
        "n_atoms": len(a), "reps": reps,
        "ms_per_step_mean": float(ts.mean() * 1e3),
        "ms_per_step_std": float(ts.std() * 1e3),
        "us_per_atom_step": float(ts.mean() / len(a) * 1e6),
        "energy_eV": float(e),
        "force_max_eV_A": float(np.abs(f).max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="maceles_bench_",
                                 dir=str(REPO / ".tmp")))
    print(f"[bench] workdir {work}", flush=True)
    atoms = water_box()
    print(f"[bench] box {len(atoms)} atoms, "
          f"{atoms.cell.cellpar()[:3].round(2)} A", flush=True)

    out = {"workdir": str(work), "n_atoms": len(atoms)}
    for kind in ("MACE", "MACELES"):
        t0 = time.time()
        model = train(kind, work, args.device, args.epochs)
        print(f"[bench] trained {kind} in {time.time() - t0:.1f}s -> {model}",
              flush=True)
        r = bench(model, atoms, args.device, args.reps)
        r["train_seconds"] = time.time() - t0
        r["model_path"] = str(model)
        out[kind] = r
        print(f"[bench] {kind:8s} {r['ms_per_step_mean']:.2f} ms/step  "
              f"{r['us_per_atom_step']:.2f} us/atom/step", flush=True)

    out["les_overhead_us_per_atom_step"] = (
        out["MACELES"]["us_per_atom_step"] - out["MACE"]["us_per_atom_step"])
    out["les_overhead_fraction"] = (
        out["les_overhead_us_per_atom_step"] / out["MACE"]["us_per_atom_step"])
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
