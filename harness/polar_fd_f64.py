#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MACE-POLAR float64 有限差分力精确校验（Gate-1 强版本, 2026-10-14）。

float32 下的 FD 误差 (~0.2 eV/Å) 是能量量化噪声主导;本脚本用 float64
权重/运算复核解析力与 -dE/dx 的一致性(预期 ≲1e-4 eV/Å),并对单原子做
h 收敛扫描。生产盒子 NH3-32(seed 770500)。

用法: CUDA_VISIBLE_DEVICES=0 <polar venv python> bench/polar_fd_f64.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/workspace/group/shared/bza/task3_sampling_database/mlip/scripts")
import mace_md as mm  # noqa: E402

POLAR = "/workspace/group/shared/bza/task3_sampling_database/mlip/models/mace-polar/MACE-POLAR-1-M.model"


def main():
    from mace.calculators import mace_polar
    t0 = time.time()
    calc = mace_polar(model=POLAR, device="cuda", default_dtype="float64")
    atoms, _, _ = mm.build_gas("NH3", 32, 0.01, 770500, 2.0)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = calc
    print(f"load {time.time()-t0:.1f}s; atoms={len(atoms)}", flush=True)

    e0 = atoms.get_potential_energy()
    f_an = atoms.get_forces()
    print(f"PE = {e0:.8f} eV", flush=True)

    rng = np.random.default_rng(11)
    idxs = rng.choice(len(atoms), size=3, replace=False)
    errs = []
    for idx in idxs:
        for ax in (0, 1):
            h = 1e-4
            pos = atoms.get_positions()
            pos[idx, ax] += h
            ap = atoms.copy(); ap.set_positions(pos); ap.info.update(atoms.info)
            ap.calc = calc
            ep = ap.get_potential_energy()
            pos = atoms.get_positions()
            pos[idx, ax] -= h
            am = atoms.copy(); am.set_positions(pos); am.info.update(atoms.info)
            am.calc = calc
            em = am.get_potential_energy()
            f_num = -(ep - em) / (2 * h)
            errs.append(abs(f_num - f_an[idx, ax]))
    print(f"[float64] max |F_analytic - F_numeric| = {max(errs):.2e} eV/Å  (3 原子 × 2 轴, h=1e-4)",
          flush=True)

    idx = int(idxs[0]); ax = 0
    for h in (1e-3, 1e-4, 1e-5):
        pos = atoms.get_positions()
        pos[idx, ax] += h
        ap = atoms.copy(); ap.set_positions(pos); ap.info.update(atoms.info)
        ap.calc = calc
        ep = ap.get_potential_energy()
        pos = atoms.get_positions()
        pos[idx, ax] -= h
        am = atoms.copy(); am.set_positions(pos); am.info.update(atoms.info)
        am.calc = calc
        em = am.get_potential_energy()
        f_num = -(ep - em) / (2 * h)
        print(f"  h={h:.0e}: |F_an-F_num| = {abs(f_num-f_an[idx,ax]):.2e} eV/Å", flush=True)
    print("FD64-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
