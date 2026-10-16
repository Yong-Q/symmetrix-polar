#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Symmetrix 引擎 × PolarMACE backbone JSON — 冒烟 + 数值核验(P2, 2026-10-14)。

阶段 1(本脚本当前):引擎能加载并求值 → 能量/力量级与有限性。
阶段 2(后续追加):与 Python 分量对账(electron_energy+interaction_energy 等)+
   力有限差分;再对接 mace-torch 全模型对照(四关方法论)。

用法:
  PYTHONPATH=<symmetrix-xl>/symmetrix/source \
  LD_LIBRARY_PATH=/opt/matos-tools/cuda/12.9/lib64 \
  CUDA_VISIBLE_DEVICES=0 <python> symmetrix_polar_check.py
"""
import sys

import numpy as np

sys.path.insert(0, "/workspace/group/shared/bza/task3_sampling_database/mlip/scripts")
import mace_md as mm  # noqa: E402

J = "/workspace/group/shared/bza/task3_sampling_database/mlip/models/mace-polar/polar_symmetrix_probe.json"


def main():
    from symmetrix.calculator import Symmetrix
    atoms, _, _ = mm.build_gas("NH3", 32, 0.01, 770600, 2.0)
    print(f"atoms: {len(atoms)}  cell: {float(np.asarray(atoms.cell)[0,0]):.4f} A", flush=True)

    calc = Symmetrix(J, dtype="float64")
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    fmax = float(np.abs(f).max())
    print(f"ENGINE-BACKBONE  E = {e:.6f} eV   fmax = {fmax:.6f} eV/A", flush=True)
    print(f"finite: E={np.isfinite(e)}  F={bool(np.isfinite(f).all())}", flush=True)

    # 力有限差分抽样(引擎内部一致性):3 原子 × 1 轴
    rng = np.random.default_rng(3)
    errs = []
    for idx in rng.choice(len(atoms), size=3, replace=False):
        h = 1e-4
        pos = atoms.get_positions()
        pos[idx, 0] += h
        ap = atoms.copy(); ap.set_positions(pos); ap.calc = calc
        ep = ap.get_potential_energy()
        pos = atoms.get_positions()
        pos[idx, 0] -= h
        am = atoms.copy(); am.set_positions(pos); am.calc = calc
        em = am.get_potential_energy()
        f_num = -(ep - em) / (2 * h)
        errs.append(abs(f_num - f[idx, 0]))
    print(f"FD check (3 atoms, h=1e-4): max|F_an - F_num| = {max(errs):.3e} eV/A", flush=True)
    print("SYMMETRIX-POLAR-CHECK-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
