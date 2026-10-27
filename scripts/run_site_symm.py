#!/usr/bin/env python3
"""Inhomogeneous-site-field MD with the Symmetrix MACE-OFF23 engine.

The external potential is the JCTC-style four-sine scalar field applied to
every atom site (orientation-resolving, no COM projection):

    V(x)/kB = T * sum_n a_n sin(k_n x + phi_n),   k_n = 2*pi*n/L
    F_i,x   = -dV/dx(x_i)                        [eV/A]

MACE-OFF23 is only the *sampler*; MACE-Polar remains the label source.  The
engine runs the FP32 streamed-edges path used by the 500 ps audit.

Example:
    python scripts/run_site_symm.py --smoke
    python scripts/run_site_symm.py --temps 300 500 700 --seeds 101 202 \
        --steps-prod 20000 --dump-every 1000
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
OUT = REPO / "bench/plan_d_inhomog_2026-10-27"
DATA = REPO / "bench/off23_500ps_water1152_2026-10-23/data.lammps"
MODEL = Path("/workspace/group/shared/bza/task3_sampling_database/mlip/models/symmetrix/mace-off23-medium.json")
LMP = Path("/workspace/group/shared/bza/external-repos/symmetrix-xl/symmetrix/build-lammps-aacd036df381800a/lmp")
CUBIN = Path("/workspace/group/shared/bza/symmetrix-polar/bench/plan_d_linear_2026-10-23/jit_cache/artifacts/9831e6a1a234bfe74d76b597d598ff09d50f6b129aae2dbb913beff5a78961b5/factorized_cuda_module_gen11_src847380c859abdbe6d213af7f595ebc0cba0f1986c73e8bce6c2a05b98cffecb1.cubin")
KB_EV_PER_K = 8.617333262e-5
N_HARM = 4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def box_length_x() -> float:
    for line in DATA.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[-2:] == ["xlo", "xhi"]:
            return float(parts[1]) - float(parts[0])
    raise RuntimeError("xlo/xhi not found in data.lammps")


def make_field(T: float, seed: int, L: float):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, N_HARM)
    phi = rng.uniform(0.0, 2.0 * np.pi, N_HARM)
    n = np.arange(1, N_HARM + 1, dtype=float)
    k = 2.0 * np.pi * n / L
    A = KB_EV_PER_K * T * a          # eV
    return {"a": a, "phi": phi, "k": k, "A": A, "L": L, "T": T, "seed": seed}


def field_variables(f: dict) -> str:
    sin_terms = " + ".join(
        f"({A:.17g})*sin(({k:.17g})*x+({p:.17g}))"
        for A, k, p in zip(f["A"], f["k"], f["phi"]))
    cos_terms = " + ".join(
        f"({-A*k:.17g})*cos(({k:.17g})*x+({p:.17g}))"
        for A, k, p in zip(f["A"], f["k"], f["phi"]))
    return sin_terms, cos_terms


def write_input(work: Path, T: float, seed: int, steps_eq: int,
                steps_prod: int, dump_every: int, field: dict) -> Path:
    sin_terms, cos_terms = field_variables(field)
    jit = ""
    if CUBIN.exists():
        jit = (f" jit_device_artifact {CUBIN}"
               " jit_device_blocks_per_compute_unit 8")
    inp = work / "in.site"
    inp.write_text(f"""units metal
atom_style atomic/kk
atom_modify map yes
boundary p p p
package kokkos neigh half newton on
newton on
read_data data.lammps
pair_style symmetrix/mace/float32/kk streamed_edges direct profile speed{jit}
pair_coeff * * {MODEL} O H
neighbor 2.0 bin
neigh_modify every 100 delay 0 check no
timestep 0.001
velocity all create {T} {seed} mom yes rot no dist gaussian
fix 1 all nvt temp {T} {T} 0.1
variable field_energy atom "{sin_terms}"
variable field_force atom "{cos_terms}"
fix field all addforce v_field_force 0.0 0.0 energy v_field_energy
fix_modify field energy yes
thermo 200
thermo_style custom step time temp pe ke etotal
run_style verlet/kk
run {steps_eq}
reset_timestep 0
dump 2 all custom {dump_every} traj.lammpstrj id type x y z
dump_modify 2 sort id format float %.10g
run {steps_prod}
write_restart final.restart
""")
    return inp


def run_state(T: int, seed: int, args) -> dict:
    tag = f"site-H2O-1152-{T}K-f{seed}"
    work = OUT / tag
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA, work / "data.lammps")
    field = make_field(T, seed, box_length_x())
    inp = write_input(work, T, seed, args.steps_eq, args.steps_prod,
                      args.dump_every, field)
    (work / "field.json").write_text(json.dumps({
        "tag": tag, "T_K": T, "seed": seed,
        "a": field["a"].tolist(), "phi": field["phi"].tolist(),
        "k_invA": field["k"].tolist(), "A_eV": field["A"].tolist(),
        "k_convention": "2n_pi_over_L", "n_harm": N_HARM,
        "direction": "x", "L_A": field["L"],
        "V_eV": "sum_n A_n sin(k_n x + phi_n)",
        "F_eV_per_A": "-sum_n A_n k_n cos(k_n x + phi_n)",
        "sampler": "MACE-OFF23 medium via Symmetrix engine",
        "label_source": "MACE-Polar-1-M (separate teacher pass)",
    }, indent=1))
    env = {**os.environ,
           "LD_LIBRARY_PATH": (
               "/workspace/caches/conda/envs/lammps-deepmd/lib:"
               "/opt/matos-tools/cuda/12.9/lib64:"
               "/opt/matos-tools/conda/envs/matos-compute/lib"),
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [str(LMP), "-k", "on", "g", "1", "-in", inp.name,
         "-log", "log.site"], cwd=work, env=env, text=True,
        capture_output=True)
    (work / "stdout.site").write_text(proc.stdout)
    (work / "stderr.site").write_text(proc.stderr)
    ok = proc.returncode == 0 and (work / "traj.lammpstrj").exists()
    rec = {"tag": tag, "T_K": T, "seed": seed, "rc": proc.returncode,
           "ok": ok, "workdir": str(work),
           "traj": str(work / "traj.lammpstrj") if ok else None,
           "n_frames": 0}
    if ok:
        n = sum(1 for line in open(work / "traj.lammpstrj")
                if line.strip() == "ITEM: TIMESTEP")
        rec["n_frames"] = n
    else:
        rec["error"] = (proc.stderr or proc.stdout)[-1500:]
    print(json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", type=int, nargs="+", default=[300, 500, 700])
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 202])
    ap.add_argument("--steps-eq", type=int, default=2000)
    ap.add_argument("--steps-prod", type=int, default=20000)
    ap.add_argument("--dump-every", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.temps, args.seeds = [300], [101]
        args.steps_eq, args.steps_prod, args.dump_every = 100, 200, 100
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for T in args.temps:
        for seed in args.seeds:
            records.append(run_state(T, seed, args))
    manifest = {
        "engine": str(LMP), "engine_sha256": sha256(LMP),
        "model": str(MODEL), "model_sha256": sha256(MODEL),
        "data": str(DATA), "data_sha256": sha256(DATA),
        "cubin": str(CUBIN) if CUBIN.exists() else None,
        "states": records,
    }
    (OUT / "manifest_site.json").write_text(json.dumps(manifest, indent=1))
    bad = [r for r in records if not r["ok"]]
    print(f"done: {len(records) - len(bad)}/{len(records)} states ok")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
