#!/usr/bin/env python3
"""Inhomogeneous-site-field MD with the Symmetrix MACE-OFF23 engine.

The external potential is the JCTC-style four-sine scalar field applied to
every atom site (orientation-resolving, no COM projection):

    V(x)/kB = T * sum_n a_n sin(k_n x + phi_n),   k_n = 2*pi*n/L
    F_i,x   = -dV/dx(x_i)                        [eV/A]

MACE-OFF23 is only the *sampler*; MACE-Polar remains the label source.  The
engine runs the FP32 streamed-edges path used by the 500 ps audit.

Examples:
    # the 1,152-atom water box
    python scripts/run_site_symm.py --smoke
    python scripts/run_site_symm.py --temps 300 500 700 --seeds 101 202 \
        --steps-prod 20000 --dump-every 1000

    # 125-molecule CO2 boxes at two densities
    python scripts/run_site_symm.py --species CO2 --rhos 0.02 0.035 \
        --nmols 125 --temps 300 500 700 --seeds 101 \
        --steps-eq 2000 --steps-prod 3000 --dump-every 500
"""
import argparse
import math
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
TORCH_MODEL = Path("/workspace/group/shared/bza/task3_sampling_database/mlip/models/MACE-OFF23_medium.model")
LMP = Path("/workspace/group/shared/bza/external-repos/symmetrix-xl/symmetrix/build-lammps-aacd036df381800a/lmp")
CUBIN = Path("/workspace/group/shared/bza/symmetrix-polar/bench/plan_d_linear_2026-10-23/jit_cache/artifacts/9831e6a1a234bfe74d76b597d598ff09d50f6b129aae2dbb913beff5a78961b5/factorized_cuda_module_gen11_src847380c859abdbe6d213af7f595ebc0cba0f1986c73e8bce6c2a05b98cffecb1.cubin")
STUDENT_LMP = Path("/workspace/group/shared/bza/external-repos/symmetrix-xl/symmetrix/build-kernel-r3/lmp")
STUDENT_MODEL = REPO / "bench/plan_d_bulk_2026-10-25/models/bulk/plan_d_bulk_symmetrix.json"
STUDENT_CUBIN = Path("/workspace/group/shared/bza/symmetrix-polar/bench/plan_d_linear_2026-10-23/jit_cache/artifacts/77e83555653584dac8c1bbc4914aba36e995e70e757b7a32324b7aaf69a4f305/factorized_cuda_module_gen11_src847380c859abdbe6d213af7f595ebc0cba0f1986c73e8bce6c2a05b98cffecb1.cubin")

SAMPLERS = {
    "off23": {"lmp": LMP, "model": MODEL, "cubin": CUBIN,
              "tag": "site", "label": "MACE-OFF23 medium"},
    "student": {"lmp": STUDENT_LMP, "model": STUDENT_MODEL,
                "cubin": STUDENT_CUBIN, "tag": "student-site",
                "label": "distilled Plan D student (MACE-Polar surrogate)"},
}
KB_EV_PER_K = 8.617333262e-5
N_HARM = 4
PAIR_ELEMENTS = {"H2O": ["O", "H"], "CO2": ["C", "O"],
                 "NH3": ["N", "H"], "CH4": ["C", "H"],
                 "C2H4": ["C", "H"], "C2H6": ["C", "H"]}
MASS = {"H": 1.008, "C": 12.011, "O": 15.999, "N": 14.007}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def box_length_x(path: Path) -> float:
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[-2:] == ["xlo", "xhi"]:
            return float(parts[1]) - float(parts[0])
    raise RuntimeError(f"xlo/xhi not found in {path}")


def build_box(species: str, n_mol: int, rho: float, seed: int):
    """Grid placement with random orientations, as in the dense sampler."""
    from ase import Atoms
    from ase.build import molecule as ase_molecule
    rng = np.random.default_rng(seed)
    mol = ase_molecule(species)
    base = mol.get_positions() - mol.get_positions().mean(axis=0)
    numbers = mol.numbers
    L = (n_mol / rho) ** (1.0 / 3.0)
    n_side = int(math.ceil(n_mol ** (1.0 / 3.0)))
    spacing = L / n_side
    sites = [((i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing)
             for i in range(n_side) for j in range(n_side) for k in range(n_side)]
    rng.shuffle(sites)
    centers = [np.asarray(s) + rng.uniform(-0.05 * spacing, 0.05 * spacing, 3)
               for s in sites[:n_mol]]
    pos = []
    for c in centers:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        pos.append(base @ R.T + c + rng.normal(scale=0.05, size=base.shape))
    return Atoms(numbers=np.tile(numbers, n_mol),
                 positions=np.vstack(pos), cell=[L, L, L], pbc=True)


def write_lammps_data(atoms, path: Path, elements: list) -> None:
    L = atoms.cell.lengths()
    sym2type = {e: i for i, e in enumerate(elements, 1)}
    lines = [f"{atoms.get_chemical_formula()} site-field box", "",
             f"{len(atoms)} atoms", f"{len(elements)} atom types", "",
             f"0.0 {L[0]:.10f} xlo xhi",
             f"0.0 {L[1]:.10f} ylo yhi",
             f"0.0 {L[2]:.10f} zlo zhi", "", "Masses", ""]
    for i, e in enumerate(elements, 1):
        lines.append(f"{i} {MASS[e]:.6f}")
    lines += ["", "Atoms # atomic", ""]
    for i, (s, p) in enumerate(zip(atoms.get_chemical_symbols(),
                                   atoms.get_positions()), 1):
        lines.append(f"{i} {sym2type[s]} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}")
    path.write_text("\n".join(lines) + "\n")


def prepare_box(species: str, n_mol: int, rho: float, seed: int,
                work: Path, min_steps: int) -> Path:
    """Build, relax, and (for dense boxes) compress from a dilute packing.

    Random placement directly at 0.035 molecules/A^3 leaves ~0.7 A contacts
    that FIRE cannot remove; build at 0.02, relax, then compress to the
    target density and relax again.
    """
    rho_build = min(rho, 0.02) if species == "CO2" else rho
    atoms = build_box(species, n_mol, rho_build, seed)
    if min_steps > 0:
        from ase.optimize import FIRE
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=str(TORCH_MODEL), device="cuda",
                              default_dtype="float32")
        atoms.calc = calc
        FIRE(atoms, logfile=None).run(fmax=0.2, steps=min_steps)
        if rho > rho_build + 1e-12:
            stages = 5
            rho_cur = rho_build
            for k in range(1, stages + 1):
                rho_k = rho_build + (rho - rho_build) * k / stages
                scale = (rho_cur / rho_k) ** (1.0 / 3.0)
                atoms.set_cell(atoms.cell * scale, scale_atoms=True)
                FIRE(atoms, logfile=None).run(fmax=0.2, steps=300)
                rho_cur = rho_k
        calc = MACECalculator(model_paths=str(TORCH_MODEL), device="cuda",
                              default_dtype="float32")
        atoms.calc = calc
        FIRE(atoms, logfile=None).run(fmax=0.08, steps=max(1000, 2 * min_steps))
    atoms.wrap()
    path = work / "data.lammps"
    write_lammps_data(atoms, path, PAIR_ELEMENTS[species])
    return path


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
                steps_prod: int, dump_every: int, field: dict,
                elements: list, sampler: dict, timestep: float = 0.001) -> Path:
    sin_terms, cos_terms = field_variables(field)
    jit = ""
    if sampler["cubin"].exists():
        jit = (f" jit_device_artifact {sampler['cubin']}"
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
pair_coeff * * {sampler['model']} {' '.join(elements)}
neighbor 2.0 bin
neigh_modify every 100 delay 0 check no
timestep {timestep}
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


def wait_for_gpu_reserve(min_free_mib: int = 15000, poll_s: int = 30,
                         max_wait_s: int = 3600) -> None:
    """Do not start a new state if the GPU is close to full."""
    import time
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            free = min(int(x) for x in out.stdout.split())
        except Exception:
            return
        if free >= min_free_mib:
            return
        print(f"[gpu] free {free} MiB < {min_free_mib} MiB; waiting",
              flush=True)
        time.sleep(poll_s)


def run_state(species: str, rho, T: int, seed: int, args) -> dict:
    sampler = SAMPLERS[args.sampler]
    if rho is None:
        tag = f"{sampler['tag']}-{species}-1152-{T}K-f{seed}"
        data_src = "water_1152"
    else:
        tag = f"{sampler['tag']}-{species}-{args.nmols}mol-r{rho:g}-{T}K-f{seed}"
        data_src = "built"
    wait_for_gpu_reserve(args.min_free_gpu_mib)
    work = OUT / tag
    work.mkdir(parents=True, exist_ok=True)
    if rho is None:
        shutil.copy2(DATA, work / "data.lammps")
    else:
        prepare_box(species, args.nmols, rho, seed, work, args.min_steps)
    data_path = work / "data.lammps"
    elements = PAIR_ELEMENTS[species]
    field = make_field(T, seed, box_length_x(data_path))
    inp = write_input(work, T, seed, args.steps_eq, args.steps_prod,
                      args.dump_every, field, elements, sampler,
                      args.timestep)
    meta = {
        "tag": tag, "species": species, "T_K": T, "seed": seed,
        "density_mol_A3": rho, "n_molecules": args.nmols if rho else 384,
        "a": field["a"].tolist(), "phi": field["phi"].tolist(),
        "k_invA": field["k"].tolist(), "A_eV": field["A"].tolist(),
        "k_convention": "2n_pi_over_L", "n_harm": N_HARM,
        "direction": "x", "L_A": field["L"],
        "V_eV": "sum_n A_n sin(k_n x + phi_n)",
        "F_eV_per_A": "-sum_n A_n k_n cos(k_n x + phi_n)",
        "sampler": sampler["label"] + " via Symmetrix engine",
        "label_source": "MACE-Polar-1-M (separate teacher pass)",
        "data_source": data_src, "data_sha256": sha256(data_path),
        "pair_elements": elements,
    }
    (work / "field.json").write_text(json.dumps(meta, indent=1))
    env = {**os.environ,
           "LD_LIBRARY_PATH": (
               "/workspace/caches/conda/envs/lammps-deepmd/lib:"
               "/opt/matos-tools/cuda/12.9/lib64:"
               "/opt/matos-tools/conda/envs/matos-compute/lib"),
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [str(sampler["lmp"]), "-k", "on", "g", "1", "-in", inp.name,
         "-log", "log.site"], cwd=work, env=env, text=True,
        capture_output=True)
    (work / "stdout.site").write_text(proc.stdout)
    (work / "stderr.site").write_text(proc.stderr)
    ok = proc.returncode == 0 and (work / "traj.lammpstrj").exists()
    rec = {"tag": tag, "species": species, "T_K": T, "seed": seed,
           "density_mol_A3": rho, "rc": proc.returncode, "ok": ok,
           "workdir": str(work),
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
    ap.add_argument("--species", choices=sorted(PAIR_ELEMENTS), default="H2O")
    ap.add_argument("--sampler", choices=sorted(SAMPLERS), default="off23")
    ap.add_argument("--rhos", type=float, nargs="+", default=None,
                    help="densities in molecules/A^3; omit for the 1152-atom "
                         "water box")
    ap.add_argument("--nmols", type=int, default=125)
    ap.add_argument("--temps", type=int, nargs="+", default=[300, 500, 700])
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 202])
    ap.add_argument("--steps-eq", type=int, default=2000)
    ap.add_argument("--steps-prod", type=int, default=20000)
    ap.add_argument("--dump-every", type=int, default=1000)
    ap.add_argument("--min-steps", type=int, default=600)
    ap.add_argument("--min-free-gpu-mib", type=int, default=15000,
                    help="do not start a new state below this free memory")
    ap.add_argument("--timestep", type=float, default=0.001,
                    help="ps; use 0.0005 for dense/hot boxes")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.temps, args.seeds = [300], [101]
        args.steps_eq, args.steps_prod, args.dump_every = 100, 200, 100
        if args.rhos:
            args.nmols = min(args.nmols, 32)
    OUT.mkdir(parents=True, exist_ok=True)
    rhos = args.rhos if args.rhos else [None]
    records = []
    for rho in rhos:
        for T in args.temps:
            for seed in args.seeds:
                records.append(run_state(args.species, rho, T, seed, args))
    sampler = SAMPLERS[args.sampler]
    manifest = {
        "sampler": sampler["label"],
        "engine": str(sampler["lmp"]), "engine_sha256": sha256(sampler["lmp"]),
        "model": str(sampler["model"]),
        "model_sha256": sha256(sampler["model"]),
        "torch_model": str(TORCH_MODEL),
        "cubin": str(sampler["cubin"]) if sampler["cubin"].exists() else None,
        "states": records,
    }
    (OUT / f"manifest_site_{args.species}_{args.sampler}.json").write_text(
        json.dumps(manifest, indent=1))
    bad = [r for r in records if not r["ok"]]
    print(f"done: {len(records) - len(bad)}/{len(records)} states ok")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
