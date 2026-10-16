#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""⑦ le 反向 stage 对账 (2026-10-16): 引擎 dbg 访问器 vs python 参照。

引擎: 一帧能量+力求值(触发反向) → polar_dbg_le_{qin,plast,nf}_bar 三探针。
参照: polar_fixtures/le_stage_ref_<frame>.npz(polar_le_stage_ref.py 生成)。
输出: 三探针 max|Δ|/rel + 最大偏差位置 + e_total/e_le 对照。
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/bench")

FIX = f"{ROOT}/bench/polar_fixtures"
J = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"


def load_atoms(name):
    from ase import Atoms
    d = np.load(f"{FIX}/{name}.npz", allow_pickle=False)
    return Atoms(symbols=[str(s) for s in d["symbols"]],
                 positions=np.asarray(d["positions"], dtype=np.float64),
                 cell=np.asarray(d["cell"], dtype=np.float64), pbc=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="nh3-f0,co2-f0")
    args = ap.parse_args()

    from symmetrix.calculator import Symmetrix

    engine = Symmetrix(J, dtype="float64")
    ev = engine.evaluator
    for probe in ("polar_dbg_le_qin_bar", "polar_dbg_le_plast_bar",
                  "polar_dbg_le_nf_bar"):
        if not hasattr(ev, probe):
            print(f"DUMP-ACCESSORS-MISSING: {probe}(先重建 v22d)", flush=True)
            return 2

    for fname in args.frames.split(","):
        ref = np.load(f"{FIX}/le_stage_ref_{fname}.npz")
        atoms = load_atoms(fname)
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        atoms.calc = engine
        e_eng = float(atoms.get_potential_energy())
        _ = atoms.get_forces()
        n = len(atoms)
        eng = {
            "qin_bar": np.asarray(ev.polar_dbg_le_qin_bar(),
                                  dtype=np.float64).reshape(n, 8),
            "plast_bar": np.asarray(ev.polar_dbg_le_plast_bar(),
                                    dtype=np.float64).reshape(n, 16),
            "nf_bar": np.asarray(ev.polar_dbg_le_nf_bar(),
                                 dtype=np.float64).reshape(n, -1),
        }
        print(f"\n== {fname} (n={n}) ==", flush=True)
        print(f"  E: engine={e_eng:.6f}  ref={float(ref['e_total'][0]):.6f}"
              f"  Δ={e_eng - float(ref['e_total'][0]):.3e}", flush=True)
        for key in ("qin_bar", "plast_bar", "nf_bar"):
            a = eng[key]
            b = np.asarray(ref[key], dtype=np.float64).reshape(a.shape)
            d = np.abs(a - b)
            denom = max(float(np.abs(b).max()), 1e-30)
            idx = np.unravel_index(np.argmax(d), d.shape)
            print(f"  {key:10s} shape={a.shape} |ref|max={np.abs(b).max():.4e} "
                  f"|e|max={np.abs(a).max():.4e} max|Δ|={d.max():.4e} "
                  f"rel={d.max() / denom:.3e} @node{idx[0]},col{idx[1]}"
                  f"  (e={a[idx]:+.4e} ref={b[idx]:+.4e})", flush=True)
    print("\nLE-STAGE-DUMP-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
