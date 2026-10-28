#!/usr/bin/env python3
"""Short unfielded OFF23 trajectories at 500/700 K for the model-error scale.

Runs the Symmetrix MACE-OFF23 engine on the 1,152-atom water box, dumps 10
frames over 10 ps, and leaves the frames for a MACE-Polar energy pass.
"""
import argparse
import json
import subprocess
from pathlib import Path

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
OUT = REPO / "bench/off_T_scale_20261028"
DATA = REPO / "bench/off23_500ps_water1152_2026-10-23/data.lammps"
LMP = Path("/workspace/group/shared/bza/external-repos/symmetrix-xl/symmetrix/"
           "build-lammps-aacd036df381800a/lmp")
MODEL = ("/workspace/group/shared/bza/task3_sampling_database/mlip/models/"
         "symmetrix/mace-off23-medium.json")
CUBIN = Path("/workspace/group/shared/bza/symmetrix-polar/bench/"
             "plan_d_linear_2026-10-23/jit_cache/artifacts/"
             "9831e6a1a234bfe74d76b597d598ff09d50f6b129aae2dbb913beff5a78961b5/"
             "factorized_cuda_module_gen11_"
             "src847380c859abdbe6d213af7f595ebc0cba0f1986c73e8bce6c2a05b98cffecb1.cubin")


def run(T: int, steps: int, dump_every: int):
    work = OUT / f"T{T}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "data.lammps").write_bytes(DATA.read_bytes())
    inp = work / "in.off"
    inp.write_text(f"""units metal
atom_style atomic/kk
atom_modify map yes
boundary p p p
package kokkos neigh half newton on
newton on
read_data data.lammps
pair_style symmetrix/mace/float32/kk streamed_edges direct profile speed jit_device_artifact {CUBIN} jit_device_blocks_per_compute_unit 8
pair_coeff * * {MODEL} O H
neighbor 2.0 bin
neigh_modify every 100 delay 0 check no
timestep 0.001
velocity all create {T} 98765 mom yes rot no dist gaussian
fix 1 all nvt temp {T} {T} 0.1
thermo 200
thermo_style custom step time temp pe ke etotal
run_style verlet/kk
run 2000
reset_timestep 0
dump 2 all custom {dump_every} traj.lammpstrj id type x y z
dump_modify 2 sort id format float %.10g
run {steps}
""")
    env = {"LD_LIBRARY_PATH": (
        "/workspace/caches/conda/envs/lammps-deepmd/lib:"
        "/opt/matos-tools/cuda/12.9/lib64:"
        "/opt/matos-tools/conda/envs/matos-compute/lib"),
        "CUDA_VISIBLE_DEVICES": "0"}
    proc = subprocess.run([str(LMP), "-k", "on", "g", "1", "-in", "in.off",
                           "-log", "log.off"], cwd=work, env=env, text=True,
                          capture_output=True)
    (work / "stdout").write_text(proc.stdout)
    print(json.dumps({"T": T, "rc": proc.returncode,
                      "traj": str(work / "traj.lammpstrj")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", type=int, nargs="+", default=[500, 700])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--dump-every", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for T in args.temps:
        run(T, args.steps, args.dump_every)
