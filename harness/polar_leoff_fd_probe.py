#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""le-off 隔离探针 (2026-10-16): ⑧ 残差归因用。

关掉 ⑦ le(polar_blocks 无 local_electron_energy ⇒ 能量不装配 le、le 链不跑):
  引擎总能量 = 骨架 + ⑥ es;引擎解析力 = 骨架(P2) + ⑥链反向(种子自动 bit1)。
  FD 自洽探针只测『骨架+⑥』这一段 —— 残差与总模型下残差对比即可归因:
    - 若此处 FD≈0 ⇒ ⑥ 链正确;总模型残差 ← le 相关(⑧-7 slice 或 le 位置语义)。
    - 若此处 FD 仍大 ⇒ ⑥ 链/骨架注入有错,先修这里。

与 ⑧ 门同环境运行(descriptor 后端 + nvidia-smi shim;见 module8 gate docstring)。
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/bench")

J = f"{ROOT}/models/mace-polar/polar_symmetrix_probe_leoff.json"
FIX = f"{ROOT}/bench/polar_fixtures"


def build_atoms_from_npz(d):
    from ase import Atoms
    return Atoms(symbols=[str(s) for s in d["symbols"]],
                 positions=np.asarray(d["positions"], dtype=np.float64),
                 cell=np.asarray(d["cell"], dtype=np.float64), pbc=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="nh3-f0,nh3-f1,n2-f0")
    ap.add_argument("--h", type=float, default=1e-3)
    ap.add_argument("--fd-atoms", type=int, default=2)
    args = ap.parse_args()

    from symmetrix.calculator import Symmetrix

    engine = Symmetrix(J, dtype="float64")
    ev = engine.evaluator
    print(f"terms_last={int(ev.polar_reverse_terms_last())} "
          f"(le-off ⇒ 期望 2=仅⑥位)", flush=True)

    for name in args.frames.split(","):
        d = np.load(f"{FIX}/{name}.npz", allow_pickle=False)
        atoms = build_atoms_from_npz(d)
        n = len(atoms)
        ids = list(range(0, n, max(1, n // args.fd_atoms)))[:args.fd_atoms]
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        atoms.calc = engine
        e_eng = float(atoms.get_potential_energy())
        f_eng = np.asarray(atoms.get_forces(), dtype=np.float64).reshape(-1, 3)
        pol = np.asarray(ev.polar_node_forces(), dtype=np.float64).reshape(-1, 3)
        fd_rows = []
        for ai in ids:
            for comp in range(3):
                vals = []
                for sign in (+1.0, -1.0):
                    x = atoms.copy()
                    p = x.get_positions()
                    p[ai, comp] += sign * args.h
                    x.set_positions(p)
                    x.calc = engine
                    vals.append(float(x.get_potential_energy()))
                fd = -(vals[0] - vals[1]) / (2.0 * args.h)
                fd_rows.append((ai, comp, fd, -f_eng[ai, comp]))
        diffs = np.array([abs(fd - an) for _, _, fd, an in fd_rows])
        print(f"  {name:8s} E={e_eng:.9f}  抽样原子={ids}", flush=True)
        for ai, comp, fd, an in fd_rows:
            print(f"      atom{ai} comp{comp}: FD={fd:+.8f}  analytic={an:+.8f}"
                  f"  |d|={abs(fd - an):.3e}", flush=True)
        print(f"      FD-res max={diffs.max():.3e}  |polar|max={abs(pol).max():.3e}"
              f"  ΣF={abs(f_eng.sum(axis=0)).max():.3e}", flush=True)
    print("LEOFF-PROBE-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
