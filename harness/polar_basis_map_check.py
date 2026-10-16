#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基变换矩阵普适性复核(2026-10-15): 多构型上拟合 acc vs python ref 的最佳带符号置换。

预期(单构型已得): vec 块上 acc = −(c1_v2, c1_v0, c1_v1)，即矩阵 −P(P=(2,0,1) 循环)。
本脚本在 4 个独立构型(含扰动)上复核该映射恒定, 排除构型依附巧合。

用法: CUDA_VISIBLE_DEVICES=<gpu> <polar venv python> polar_basis_map_check.py
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import sys  # noqa: E402
from itertools import permutations, product  # noqa: E402

import numpy as np  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
import mace_md as mm  # noqa: E402

J = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
POLAR = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"


def main():
    from symmetrix.calculator import Symmetrix
    from mace.calculators import mace_polar

    engine = Symmetrix(J, dtype="float64")
    ref = mace_polar(model=POLAR, device="cuda", default_dtype="float64")
    model = ref.models[0]
    caps = {}
    hs = []
    for i, m in enumerate(model.lr_source_maps):
        hs.append(m.register_forward_hook(
            (lambda idx: lambda mod, inp, out: caps.__setitem__(
                idx, out.detach().cpu().numpy()))(i)))

    cases = [("NH3", 770600, 0), ("NH3", 770600, 7), ("CH4", 770700, 11),
             ("CO2", 770500, 15)]
    for mol, seed, rat in cases:
        base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
        a = base.copy()
        if rat:
            a.rattle(stdev=0.05, seed=seed + rat)
        a.info["charge"] = 0
        a.info["spin"] = 1
        caps.clear()
        a.calc = ref
        a.get_potential_energy()
        c1 = caps[0][:, 0, :] + caps[1][:, 0, :]
        b = a.copy()
        b.calc = engine
        b.get_potential_energy()
        acc = np.asarray(engine.evaluator.polar_spin_charge_density(),
                         dtype=np.float64).reshape(len(b), 8)
        best = []
        for vs in ([1, 2, 3], [5, 6, 7]):
            A, R = acc[:, vs], c1[:, vs]
            for perm in permutations(range(3)):
                for signs in product([1, -1], repeat=3):
                    cand = np.stack([signs[k] * R[:, perm[k]] for k in range(3)], 1)
                    best.append((np.abs(A - cand).max(), tuple(vs), perm, signs))
        best.sort()
        print(f"{mol}-r{rat}: map perm={best[0][2]} signs={best[0][3]}  "
              f"max|Δ|={best[0][0]:.3e}  (2nd best {best[1][0]:.3e})", flush=True)
    for h in hs:
        h.remove()
    print("BASIS-MAP-CHECK-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
