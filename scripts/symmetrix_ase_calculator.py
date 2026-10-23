#!/usr/bin/env python3
"""ASE calculator backed by the Symmetrix LAMMPS engine.

This is the Python-software interface for an exported Symmetrix JSON model.
Each call writes a short LAMMPS case under a workspace directory, runs the
engine for a single point, and parses energy/forces.  It is intentionally
simple and correct rather than fast; for long MD use the native LAMMPS pair
style directly.

Example:
  from symmetrix_ase_calculator import SymmetrixPolarCalculator
  calc = SymmetrixPolarCalculator("model.json", engine="/path/lmp_gemm")
  atoms.calc = calc
  e = atoms.get_potential_energy(); f = atoms.get_forces()
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

DEFAULT_DESCRIPTOR = (
    "/workspace/group/shared/bza/task3_sampling_database/mlip/models/"
    "mace-polar/backend_cuda12_sm80.json")
DEFAULT_LD = (
    "/workspace/caches/conda/envs/lammps-deepmd/lib:"
    "/opt/matos-tools/cuda/12.9/lib64:"
    "/opt/matos-tools/cuda/12.9/targets/x86_64-linux/lib:"
    "/opt/matos-tools/conda/envs/matos-compute/lib")


class SymmetrixPolarCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model_json, engine, workdir=None, env=None,
                 descriptor=DEFAULT_DESCRIPTOR, **kwargs):
        super().__init__(**kwargs)
        self.model_json = str(Path(model_json).resolve())
        self.engine = str(Path(engine).resolve())
        self.workdir = Path(workdir or "bench/interfaces_test/ase_runs").resolve()
        self.env = dict(os.environ)
        self.env.update({
            "LD_LIBRARY_PATH": DEFAULT_LD,
            "CUDA_VISIBLE_DEVICES": self.env.get("CUDA_VISIBLE_DEVICES", "0"),
            "SYMMETRIX_BACKEND": "cuda12-sm80",
            "SYMMETRIX_BACKEND_DESCRIPTOR_PATH": descriptor,
            "OMP_NUM_THREADS": "1",
            "MATOS_REPLAY_CACHE": "1",
            "MATOS_MH1_TEAM_SIZE": "64",
            "MATOS_MH1_TEAM_CHANNEL_VEC": "1",
            "MATOS_POLAR_UVU_GEMM": "1",
        })
        if env:
            self.env.update(env)

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        symbols = list(self.atoms.get_chemical_symbols())
        species = sorted(set(symbols))
        self.workdir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="ase_", dir=self.workdir)).resolve()
        write(run_dir / "frame.xyz", self.atoms, format="extxyz")
        from make_lammps_case import write_data
        write_data(self.atoms, run_dir / "data.lammps", species)
        coeff = " ".join(species)
        (run_dir / "in.lammps").write_text("\n".join([
            "units metal", "atom_style atomic/kk", "atom_modify map yes",
            "boundary p p p", "package kokkos neigh half newton on", "newton on",
            "read_data data.lammps",
            "pair_style symmetrix/mace/kk charge 0.0 spin 1.0 streamed_edges generic",
            f"pair_coeff * * {self.model_json} {coeff}",
            "neighbor 2.0 bin", "neigh_modify every 1 delay 0 check yes",
            "timestep 0.000001", "thermo 1", "thermo_style custom step pe",
            "thermo_modify format float %.17g", "run_style verlet/kk",
            "run 0",
            "write_dump all custom forces.dump id type x y z fx fy fz "
            "modify format float %.17g sort id", ""]))
        log = run_dir / "log.lammps"
        with open(run_dir / "stdout", "w") as fh:
            proc = subprocess.run(
                [self.engine, "-k", "on", "g", "1", "-in",
                 str(run_dir / "in.lammps"), "-log", str(log)],
                cwd=run_dir, env=self.env, stdout=fh,
                stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Symmetrix engine failed ({proc.returncode}); see "
                f"{run_dir / 'stdout'}")
        text = log.read_text()
        match = re.search(r"^\s*0\s+(-?[0-9.eE+-]+)\s*$", text, re.M)
        if not match:
            raise RuntimeError(f"could not parse energy from {log}")
        self.results["energy"] = float(match.group(1))
        rows = []
        lines = (run_dir / "forces.dump").read_text().splitlines()
        header = "ITEM: ATOMS id type x y z fx fy fz"
        start = lines.index(header) + 1
        for line in lines[start:]:
            if line.strip():
                rows.append(line.split())
        data = np.array(rows, dtype=float)
        data = data[np.argsort(data[:, 0])]
        self.results["forces"] = data[:, 5:8]


if __name__ == "__main__":
    import ase.io
    calc = SymmetrixPolarCalculator(
        "bench/interfaces_test/student128L1_export.json",
        "bench/polar_uvu_gemm_engine_2026-10-21/bin/lmp_gemm")
    atoms = ase.io.read("bench/interfaces_test/h2o.xyz")
    atoms.calc = calc
    print("E", atoms.get_potential_energy())
    print("F", atoms.get_forces()[0])
