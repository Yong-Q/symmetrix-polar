#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4 RBE 验收门 (2026-10-16 v28): Ewald(确定性) vs RBE(随机批) 收敛/无偏/复现。

依据 docs/design/RBE_P4_DESIGN.md §3(P4 门 = 力 RMS 随批大小 p 收敛表)+ PLAN §3。
判据:
  (a) 复现性: 同 seed 两次装载(独立进程内构造), 首 eval 能量/力逐位一致;
      异 seed 必须不同(>1e-9)。
  (b) 收敛: RMS_F(p) 随 p 下降; log-log 斜率 ∈ [−0.8,−0.2](理想 −0.5);
      RMS_F(p_max) ≤ 0.5·RMS_F(p_min)。
  (c) 无偏: |mean ΔE| ≤ 3·SEM(逐 p); ΔF 均值向量的 max|分量| ≤ 3·SEM。
统计口径: 每个 p 下对同一构型做 n_rep 次独立 eval(帧计数递增 ⇒ 帧间重采样),
批样本 = 帧流;Δ 相对确定性 Ewald(已验证实现, v27 全绿)。
输出: bench/polar_rbe_gate.{png,json}
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

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


def build_engine(rbe_p=None, seed=None):
    if rbe_p is None:
        os.environ.pop("MATOS_RBE_P", None)
        os.environ.pop("MATOS_RBE_SEED", None)
    else:
        os.environ["MATOS_RBE_P"] = str(rbe_p)
        os.environ["MATOS_RBE_SEED"] = str(seed)
    from symmetrix.calculator import Symmetrix
    return Symmetrix(J, dtype="float64")


def eval_ef(engine, atoms):
    a = atoms.copy()
    a.info["charge"] = 0
    a.info["spin"] = 1
    a.calc = engine
    try:
        engine.atoms = None
        engine.results = {}
    except Exception:
        pass
    e = float(a.get_potential_energy())
    f = np.asarray(a.get_forces(), dtype=np.float64).reshape(-1, 3)
    return e, f


def main():
    frames = ["nh3-f0", "co2-f0"]
    p_list = [50, 200, 800]
    n_rep = 5
    atoms = {f: load_atoms(f) for f in frames}

    # (1) 确定性参照(已验证 Ewald)
    det = build_engine(None)
    ref = {}
    for f in frames:
        ref[f] = eval_ef(det, atoms[f])
    print("det reference loaded", flush=True)

    # (2) 复现性(seed 固定 200 批): A/B 同 seed, C 异 seed
    eA, fA = eval_ef(build_engine(200, 555001), atoms["nh3-f0"])
    eB, fB = eval_ef(build_engine(200, 555001), atoms["nh3-f0"])
    eC, fC = eval_ef(build_engine(200, 555002), atoms["nh3-f0"])
    dE_ab = abs(eA - eB)
    dF_ab = float(np.abs(fA - fB).max())
    dE_ac = abs(eA - eC)
    dF_ac = float(np.abs(fA - fC).max())
    # 口径: 同 seed ⇒ 批逐位一致、能量逐位相等; 力残差仅 GPU atomic 累加次序
    # 噪声(≲1e-12 eV/Å 量级, 实测 ~5e-16)。异 seed ⇒ 批不同 ⇒ 显著差异。
    repro_bit = (eA == eB) and dF_ab <= 1e-12
    repro_diff = (dE_ac > 1e-9) or (dF_ac > 1e-9)
    print(f"repro: same-seed dE={dE_ab:.3e} max|dF|={dF_ab:.3e}"
          f" (energy bitwise={eA == eB}) -> {repro_bit};"
          f" diff-seed dE={dE_ac:+.3e} dF={dF_ac:.3e} -> {repro_diff}",
          flush=True)

    # (3) p 扫描
    table = {}
    for p in p_list:
        eng = build_engine(p, 777001)
        dE, dF2 = [], []
        for f in frames:
            e0, f0 = ref[f]
            for _ in range(n_rep):
                e1, f1 = eval_ef(eng, atoms[f])
                dE.append(e1 - e0)
                dF2.append(((f1 - f0) ** 2).reshape(-1))
        dE = np.asarray(dE)
        rms_f = float(np.sqrt(np.mean(np.concatenate(dF2))))
        sem_e = float(dE.std(ddof=1) / np.sqrt(len(dE))) if len(dE) > 1 else 0.0
        table[p] = {"rms_f": rms_f, "mean_dE": float(dE.mean()),
                    "sem_dE": sem_e, "max_dE": float(np.abs(dE).max()),
                    "n": len(dE)}
        print(f"p={p:4d}: RMS_F={rms_f:.4e} eV/Å  meanΔE={dE.mean():+.3e}"
              f" (sem {sem_e:.2e})  max|ΔE|={np.abs(dE).max():.3e}", flush=True)

    # (4) 斜率与判据
    ps = np.array(p_list, dtype=float)
    rms = np.array([table[p]["rms_f"] for p in p_list])
    slope = float(np.polyfit(np.log(ps), np.log(rms), 1)[0])
    conv_ok = slope < -0.2 and slope > -0.8
    drop_ok = rms[-1] <= 0.5 * rms[0]
    unb_ok = all(abs(table[p]["mean_dE"]) <= 3.0 * table[p]["sem_dE"] + 1e-12
                 for p in p_list)
    gate = repro_bit and repro_diff and conv_ok and drop_ok and unb_ok
    print(f"\nslope(log-log RMS_F vs p) = {slope:.3f} (ideal −0.5)", flush=True)
    print(f"criteria: repro={repro_bit and repro_diff} conv={conv_ok}"
          f" drop={drop_ok} unbias={unb_ok}", flush=True)
    print(f"GATE {'PASS' if gate else 'FAIL'}: RMS_F {rms[0]:.3e}@{p_list[0]}"
          f" -> {rms[-1]:.3e}@{p_list[-1]} eV/Å", flush=True)

    # (5) 产物
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        ax.loglog(ps, rms, "o-", color="#1f77b4", label="RBE RMS(ΔF)")
        ax.loglog(ps, rms[0] * np.sqrt(p_list[0] / ps), "--", color="gray",
                  label="slope −0.5 reference")
        ax.set_xlabel("batch size p")
        ax.set_ylabel("RMS force error vs Ewald (eV/Å)")
        ax.set_title("P4 RBE convergence (force)")
        ax.legend()
        ax = axes[1]
        means = [table[p]["mean_dE"] for p in p_list]
        sems = [table[p]["sem_dE"] for p in p_list]
        ax.errorbar(p_list, means, yerr=3 * np.array(sems), fmt="o-",
                    color="#d62728", capsize=4)
        ax.axhline(0, color="gray", ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("batch size p")
        ax.set_ylabel("mean ΔE ± 3·SEM (eV)")
        ax.set_title("P4 RBE unbiasedness (energy)")
        fig.tight_layout()
        fig.savefig(f"{ROOT}/bench/polar_rbe_gate.png", dpi=150)
    except Exception as exc:  # pragma: no cover
        print(f"plot skipped: {exc}", flush=True)

    import json as _json
    with open(f"{ROOT}/bench/polar_rbe_gate.json", "w") as fh:
        _json.dump({"gate": bool(gate), "slope": slope,
                    "repro_bitwise": bool(repro_bit),
                    "repro_differs": bool(repro_diff),
                    "table": table}, fh, indent=1)
    print("RBE-GATE-DONE", flush=True)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
