#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MACE-POLAR-1-M 正确性四关（用户要求：确保计算没有问题）。

Gate-1 有限差分力校验：数值 -dE/dx vs 解析力（抽样若干原子，h=1e-4 Å）；
Gate-2 NVE 能量守恒：1000 步（0.5 ps，dt=0.5fs）VelocityVerlet，报告总能量漂移；
Gate-3 合理性对照：与 MACE-OFF23 同盒子的能量/力尺度对比；
Gate-4 长程演示：CO2 二聚体间距扫描（5–12 Å）——POLAR 在截断外应有非零相互作用，
        MACE-OFF23（r_cut≈5Å）以外应为 0。
"""
import sys, time
import numpy as np
import torch

sys.path.insert(0, "/workspace/group/shared/bza/task3_sampling_database/mlip/scripts")
import mace_md as mm  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.md.verlet import VelocityVerlet  # noqa: E402
from ase import units  # noqa: E402

POLAR = "/workspace/group/shared/bza/task3_sampling_database/mlip/models/mace-polar/MACE-POLAR-1-M.model"
MACEOFF = "/workspace/group/shared/bza/task3_sampling_database/mlip/models/MACE-OFF23_medium.model"


_CACHE = {}


def polar_calc():
    if "polar" not in _CACHE:
        from mace.calculators import mace_polar
        _CACHE["polar"] = mace_polar(model=POLAR, device="cuda", default_dtype="float32")
    return _CACHE["polar"]


def maceoff_calc():
    if "off23" not in _CACHE:
        from mace.calculators import MACECalculator
        _CACHE["off23"] = MACECalculator(model_paths=MACEOFF, device="cuda", default_dtype="float32")
    return _CACHE["off23"]


def main():
    atoms0, _, _ = mm.build_gas("NH3", 32, 0.01, 770500, 2.0)
    atoms0.info["charge"] = 0; atoms0.info["spin"] = 1
    n = len(atoms0)

    # ---------- Gate 1: finite-difference forces ----------
    atoms = atoms0.copy(); atoms.calc = polar_calc()
    e0 = atoms.get_potential_energy(); f_an = atoms.get_forces()
    print(f"Gate-1 finite-difference forces (h=1e-4 Å, sampled atoms)")
    rng = np.random.default_rng(7)
    errs = []
    for idx in rng.choice(n, size=4, replace=False):
        for ax in ([0, 2] if idx % 2 else [0, 1]):
            pos = atoms.get_positions(); pos[idx, ax] += 1e-4
            ap = atoms.copy(); ap.set_positions(pos); ap.info.update(atoms.info); ap.calc = atoms.calc
            ep = ap.get_potential_energy()
            pos = atoms.get_positions(); pos[idx, ax] -= 1e-4
            am = atoms.copy(); am.set_positions(pos); am.info.update(atoms.info); am.calc = atoms.calc
            em = am.get_potential_energy()
            f_num = -(ep - em) / 2e-4
            errs.append(abs(f_num - f_an[idx, ax]))
    print(f"   max |F_analytic - F_numeric| = {max(errs):.2e} eV/Å  (期望 <1e-2，float32 下 <1e-1)")

    # ---------- Gate 2: NVE energy conservation ----------
    atoms = atoms0.copy(); atoms.calc = polar_calc()
    atoms.rattle(1e-3, seed=11)
    atoms.info.update(atoms0.info)
    from ase.md import MDLogger  # noqa: F401
    dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
    e_start = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    for chunk in range(5):
        dyn.run(200)
        e_now = atoms.get_potential_energy() + atoms.get_kinetic_energy()
        print(f"  Gate-2 NVE {200*(chunk+1)}/1000 steps: dE_tot = {abs(e_now-e_start):.4e} eV",
              flush=True)
    e_end = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    kbT = len(atoms) * 8.617333262e-5 * 400.0
    print(f"Gate-2 NVE 1000 steps (0.5 ps): dE_tot = {abs(e_end-e_start):.4e} eV "
          f"({abs(e_end-e_start)/kbT:.2e} 相对 3NkBT)——期望 ≲1e-3 量级")

    # ---------- Gate 3: plausibility vs MACE-OFF23 ----------
    a1 = atoms0.copy(); a1.calc = polar_calc()
    a2 = atoms0.copy(); a2.calc = maceoff_calc()
    print(f"Gate-3 尺度对照: POLAR PE={a1.get_potential_energy():.1f} eV fmax={np.abs(a1.get_forces()).max():.2f} | "
          f"OFF23 PE={a2.get_potential_energy():.1f} eV fmax={np.abs(a2.get_forces()).max():.2f}")

    # ---------- Gate 4: long-range dimer scan ----------
    print("Gate-4 CO2 二聚体间距扫描（相互作用能，eV; 单分子减除）")
    mono, _, _ = mm.build_gas("CO2", 1, 0.001, 4242, 2.0)
    mono.info["charge"] = 0; mono.info["spin"] = 1
    for name, mk in [("POLAR", polar_calc), ("OFF23", maceoff_calc)]:
        calc = mk()
        mono.calc = calc
        e_mono = mono.get_potential_energy()
        row = []
        for d in [4.0, 5.0, 6.0, 8.0, 10.0, 12.0]:
            box = 30.0
            pos = mono.get_positions()
            pos2 = pos + np.array([d, 0.0, 0.0])
            dimer = Atoms(symbols=list(mono.symbols) * 2, positions=np.vstack([pos, pos2]),
                          cell=[box, box, box], pbc=True)
            dimer.info["charge"] = 0; dimer.info["spin"] = 1
            dimer.calc = calc
            e_di = dimer.get_potential_energy()
            row.append(f"{d:.0f}Å:{e_di - 2*e_mono:+.4f}")
        print(f"  {name:6s}", " ".join(row))
    print("说明：OFF23 在 r_cut（≈5Å）外应为 ~0；POLAR 应在 6–12Å 显示非零长程相互作用。")


if __name__ == "__main__":
    main()
