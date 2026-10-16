#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 L1 里程碑门:引擎(⑥coulomb + ⑦le 落地后)× Python 全模型 端到端 能量/力总账(2026-10-15)。

口径(关键):引擎 `get_potential_energy()` 必须已含 ⑥⑦ 能量装配
    E_engine = e0 + inter_e(scale/shift 后) + le_total + coulomb_energy   (extensions.py L866-929)
本脚本对账 = 引擎总能量/总力 vs Python `mace_polar` 全模型总能量/总力(同一构型、同一 ASE
Atoms 对象、同一 f64 精度)。**不**做任何分量替换或重组(分量分解仅作诊断列打印)。

前置(必须):
  1) 引擎补丁 ⑥(docs/module6_patch_final.md)+ ⑦(docs/module7_patch_final.md)已应用并重建;
     判定 = evaluator 具 `polar_coulomb_energy` 与 `polar_le_energy` 访问器,且首次对账帧
     的 E_engine ≠ backbone-only(e0+scale·inter_e+shift;内联装配探针)。
     任一不满足 → 打印 `ENDTOEND-*` 提示并以退出码 **2** 优雅退出(区分"未就绪"与"数值 FAIL")。
  2) JSON 为 E1/E2 后重生成版(models/mace-polar/polar_symmetrix_probe.json)。

判据(标度感知;参照 polar_module6_gate.py 判据风格):
  能量:**rel ≤ 1e-6 且 abs ≤ 5e-6 eV**(逐帧两列都打印;总量 |E|~1e4-1e5 eV 时 abs 项为主,
        rel 项防小值放行)。分布(逐帧 |ΔE| 的 p50/p90/max)留档。
  力:  逐帧 max|ΔF| 与 mean|ΔF|(每原子范数均值)分布留档。
       - 默认:力差参与判定,阈值 `--force-tol`(默认 1e-4 eV/Å;⑧ 落地后按门收紧)。
       - `--forces-expect-absent`:力**只打印分布、不判 fail**(⑧ 未落地前力必差——⑦ 只接线
         能量;位置直接依赖为零的特殊几何除外)。当前阶段推荐默认带此旗标运行。

产物:bench/polar_endtoend_parity.{png,json}(json 含全部逐帧明细 rows + 分布 summary);
      退出码 0=PASS / 1=FAIL / 2=前置未就绪。
      注:本环境下引擎解释器退出阶段有已留档的 `cudaStreamSynchronize … driver shutting
      down` abort,会把**进程** exit code 覆盖为 134(实测,endtoend 前置缺失一次);
      判定以 stdout 的 `ENDTOEND-*`(前置)与 `GATE PASS/FAIL`(数值)行为准。

用法:
  cd /workspace/group/shared/bza/task3_sampling_database/mlip && \
  CUDA_VISIBLE_DEVICES=<gpu> /workspace/caches/conda/envs/polar/bin/python \
    bench/polar_endtoend_parity.py [--frames 8] [--rattle 0.05] [--forces-expect-absent]
运行 env(与既有门脚本一致;本机实测可通过 .so 依赖解析的配方,若主线程已有
  run_polar_env.sh 以其为准):
  export PATH="${ROOT}/bench/nvidia_smi_shim:$PATH"        # 静态回答 compute_cap 8.0
  export LD_LIBRARY_PATH="/opt/matos-tools/cuda/12.9/lib64:/opt/matos-tools/conda/envs/matos-mlip-py311/lib"
        # ↑ 后者提供 libopenblas.so.0(.so 的 DT_NEEDED;polar venv 专属 ldd 实测已无 not-found)
  export SYMMETRIX_BACKEND=cuda12-sm80
  export SYMMETRIX_BACKEND_DESCRIPTOR_PATH="${ROOT}/models/mace-polar/backend_cuda12_sm80.json"
  CUDA_VISIBLE_DEVICES=<gpu> ${ROOT 的 polar venv}/bin/python bench/polar_endtoend_parity.py …
polar venv(`/workspace/caches/conda/envs/polar/bin/python`,sys.path 已含 matos-mlip-py312
  site-packages → mace/torch 可用)。退出阶段 `cudaStreamSynchronize/driver shutting down`
  属已留档无害项(v11 起)。

几何自证 / info 口径(与既有 parity 脚本一致):
  - 构型 = scripts/mace_md.py::build_gas(mol, 32, 0.01, seed, 2.0)(与
    symmetrix_polar_batch_parity.py 同参同种子);frame 0 与已锁定夹具
    bench/polar_fixtures/le_hooks/{mol}-f0.npz 的 positions/cell **逐位断言相等**;
  - `atoms.info["charge"]=0`、`atoms.info["spin"]=1`(每帧、每次换计算器前都设);
  - 引擎/参考用同一 Atoms 对象;对账后重验位置未被计算器改动。
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
import mace_md as mm  # noqa: E402

J = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
POLAR = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"
HOOKS = f"{ROOT}/bench/polar_fixtures/le_hooks"
INDEX = f"{ROOT}/bench/polar_fixtures/index.json"
MOLS = [("NH3", 770600), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]
NMOL = 32          # 与 batch parity 同盒(ρ=0.01, min_dist=2.0)
DENSITY = 0.01
MINDIST = 2.0
E_ABS = 5e-6       # eV
E_REL = 1e-6
ASSEMBLY_PROBE_ABS = 1e-3   # eV:引擎 E 与 backbone-only 之差 <此值 ⇒ ⑥⑦ 装配缺失
                            # (backbone 复现噪声实测 ~1e-6 eV;P3 分量 O(1-10) eV,裕度 ≥1e3)
P3_NEGLIGIBLE = 1e-2        # eV:Python P3 分量(abs le+electro)可忽略时不做装配探针


def positions_of(atoms):
    return np.asarray(atoms.get_positions(), dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--rattle", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force-tol", type=float, default=1e-4,
                    help="默认力判据 max|dF| ≤ 该值(eV/Å);--forces-expect-absent 时忽略")
    ap.add_argument("--forces-expect-absent", action="store_true",
                    help="⑧ 未落地:力差只打印分布、不判 fail")
    ap.add_argument("--out", default=f"{ROOT}/bench/polar_endtoend_parity")
    args = ap.parse_args()

    from symmetrix.calculator import Symmetrix

    engine = Symmetrix(J, dtype="float64")
    ev = engine.evaluator
    required = ("polar_coulomb_energy", "polar_le_energy")
    missing = [n for n in required if not hasattr(ev, n)]
    if missing:
        print(f"ENDTOEND-ACCESSORS-MISSING: {missing} —— 引擎未含 ⑥⑦ 访问器;"
              "先应用 docs/merge_preflight_67.md 的步骤 1-13 并重建", flush=True)
        return 2

    from mace.calculators import mace_polar

    ref_calc = mace_polar(model=POLAR, device=args.device, default_dtype="float64")

    with open(J) as fh:
        model_json = json.load(fh)
    scale = float(model_json["scale_shift"]["scale"]["values"][0])
    shift = float(model_json["scale_shift"]["shift"]["values"][0])
    zmap = model_json["atomic_numbers"]
    e0_table = model_json["atomic_energies"]["values"]
    index = json.load(open(INDEX)) if os.path.exists(INDEX) else {}

    print(f"engine: {J}\nref   : {POLAR}\n帧: {args.frames}/分子 x {len(MOLS)} 分子"
          f" | 盒: {NMOL} 分子 | forces_expect_absent={args.forces_expect_absent}",
          flush=True)

    rows = []
    assembly_probed = False

    for mol, seed in MOLS:
        base, _, _ = mm.build_gas(mol, NMOL, DENSITY, seed, MINDIST)
        assert bool(np.all(base.pbc)), f"{mol}: 盒必须全周期(⑥ 能量口径 TTT)"

        # ---- 几何自证:frame0 与锁定夹具逐位一致(与既有门/parity 同几何) ----
        fix_path = f"{HOOKS}/{mol.lower()}-f0.npz"
        if os.path.exists(fix_path):
            d = np.load(fix_path, allow_pickle=False)
            assert np.allclose(positions_of(base), d["positions"], atol=1e-10), \
                f"{mol}: build_gas 与夹具 positions 不一致"
            assert np.allclose(np.asarray(base.cell), d["cell"], atol=1e-10), \
                f"{mol}: build_gas 与夹具 cell 不一致"
            fixture_ok = True
        else:
            fixture_ok = False
            print(f"  [warn] 缺夹具 {fix_path},跳过几何自证", flush=True)
        zs = np.asarray(base.get_atomic_numbers())
        sum_e0 = float(np.sum([e0_table[zmap.index(int(z))] for z in zs]))

        for k in range(args.frames):
            atoms = base.copy()
            if k > 0:
                atoms.rattle(stdev=args.rattle * (k / max(1, args.frames - 1)) * 0.5,
                             seed=seed + 1000 * k)
            pos0 = positions_of(atoms)

            # ---- 引擎(Symmetrix(J) 读总能量/力) ----
            atoms.info["charge"] = 0
            atoms.info["spin"] = 1
            atoms.calc = engine
            e_eng = float(atoms.get_potential_energy())
            f_eng = np.asarray(atoms.get_forces(), dtype=np.float64)

            # ---- Python mace_polar 全模型 ----
            atoms.info["charge"] = 0
            atoms.info["spin"] = 1
            atoms.calc = ref_calc
            e_py = float(atoms.get_potential_energy())
            f_py = np.asarray(atoms.get_forces(), dtype=np.float64)
            r = ref_calc.results
            le_py = float(r.get("electron_energy", float("nan")))
            es_py = float(r.get("electrostatic_energy", float("nan")))
            inter_py = float(r.get("interaction_energy", float("nan")))
            backbone = scale * inter_py + sum_e0 + shift
            p3_py = le_py + es_py

            # ---- 位置未被计算器改动的自证 ----
            drift = float(np.abs(positions_of(atoms) - pos0).max())

            # ---- 装配前置探针(首帧一次):E_eng==backbone 且 P3 非零 ⇒ ⑥⑦ 装配缺失 ----
            if not assembly_probed:
                if abs(p3_py) > P3_NEGLIGIBLE and abs(e_eng - backbone) < ASSEMBLY_PROBE_ABS:
                    print(f"ENDTOEND-ENERGY-ASSEMBLY-MISSING: 引擎 E={e_eng:.9f} 与 "
                          f"backbone-only={backbone:.9f} 相等(Δ={e_eng - backbone:.2e} eV),"
                          f"但 Python P3 分量 le+electro={p3_py:+.6f} eV —— ⑥⑦ 能量装配行未生效;"
                          "检查 module6 §3.3(补丁 C2)/ module7 §4.4", flush=True)
                    return 2
                assembly_probed = True

            dE = e_eng - e_py
            rel = abs(dE) / max(abs(e_py), 1e-30)
            df = f_eng - f_py
            df_max = float(np.abs(df).max()) if df.size else 0.0
            df_mean = float(np.linalg.norm(df, axis=1).mean()) if df.size else 0.0
            dP3 = (e_eng - backbone) - p3_py     # 引擎 P3 增量 vs Python P3 分量(诊断)

            ok_e = bool(abs(dE) <= E_ABS and rel <= E_REL)
            ok_f = True if args.forces_expect_absent else bool(df_max <= args.force_tol)
            ok = ok_e and ok_f
            fix_e2e = None
            if k == 0 and fixture_ok and f"{mol.lower()}-f0" in index:
                fix_e2e = abs(e_py - float(index[f"{mol.lower()}-f0"]["energy"]))

            rows.append({
                "mol": mol, "frame": k,
                "e_engine": e_eng, "e_py_total": e_py,
                "dE": float(dE), "dE_abs": float(abs(dE)), "dE_rel": float(rel),
                "backbone_ref": backbone, "le_py": le_py, "electro_py": es_py,
                "p3_py": p3_py, "e_engine_minus_backbone": float(e_eng - backbone),
                "dP3_vs_python": float(dP3), "py_vs_fixture_f0": fix_e2e,
                "df_max": df_max, "df_mean": df_mean,
                "fmax_engine": float(np.abs(f_eng).max()), "fmax_py": float(np.abs(f_py).max()),
                "pos_drift": drift,
                "pass_energy": ok_e, "pass_forces": ok_f, "pass": ok,
            })
            print(f"  {mol:4s} f{k}: ΔE={dE:+.3e} eV (rel {rel:.1e})  "
                  f"ΔP3={dP3:+.3e}  ΔFmax={df_max:.3e}  ΔFmean={df_mean:.3e}"
                  f"  {'PASS' if ok else ('FAIL-F' if ok_e and not ok_f else 'FAIL')}",
                  flush=True)

    # ---------------- 汇总 + 分布 ----------------
    dE = np.array([r["dE_abs"] for r in rows])
    dFr = np.array([r["df_max"] for r in rows])
    dFm = np.array([r["df_mean"] for r in rows])
    drift = max(r["pos_drift"] for r in rows)
    per_mol = {}
    for mol, _seed in MOLS:
        sel = [r for r in rows if r["mol"] == mol]
        per_mol[mol] = {
            "n": len(sel),
            "dE_abs_mean": float(np.mean([r["dE_abs"] for r in sel])),
            "dE_abs_max": float(np.max([r["dE_abs"] for r in sel])),
            "df_max_mean": float(np.mean([r["df_max"] for r in sel])),
            "df_max_max": float(np.max([r["df_max"] for r in sel])),
        }
    def pct(a):
        return {p: float(np.percentile(a, p)) for p in (10, 50, 90, 100)}
    ok = bool(all(r["pass"] for r in rows))
    verdict = "PASS" if ok else "FAIL"

    print("\n== 逐分子 ==")
    for mol, v in per_mol.items():
        print(f"  {mol:4s} n={v['n']}  |ΔE| mean={v['dE_abs_mean']:.2e} max={v['dE_abs_max']:.2e} eV"
              f"  ΔFmax mean={v['df_max_mean']:.3e} max={v['df_max_max']:.3e} eV/Å")
    print("== 分布(逐帧) ==")
    print(f"  |ΔE|  p10/p50/p90/max: {pct(dE)}")
    print(f"  ΔFmax p10/p50/p90/max: {pct(dFr)}")
    print(f"  ΔFmean p10/p50/p90/max: {pct(dFm)}")
    print(f"  pos_drift max = {drift:.2e} Å (期望 0)")
    mode = "forces-expect-absent(仅留档)" if args.forces_expect_absent else \
        f"forces gated ≤{args.force_tol:.0e} eV/Å"
    print(f"\nGATE {verdict}: {len(rows)} 帧 ({sum(r['pass'] for r in rows)} PASS)  "
          f"判据 E: rel≤{E_REL:.0e} & abs≤{E_ABS:.0e} eV; {mode}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    a = np.array([r["e_py_total"] for r in rows])
    b = np.array([r["e_engine"] for r in rows])
    axes[0].scatter(a, b, s=18)
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set_xlabel("Python mace_polar total E (eV)")
    axes[0].set_ylabel("Symmetrix engine E (eV)")
    axes[0].set_title(f"End-to-end energy parity ({len(rows)} frames)")
    axes[0].grid(alpha=0.3)
    axes[1].hist(np.array([r["dE"] for r in rows]), bins=24, log=True)
    axes[1].set_xlabel("dE = engine - python (eV)")
    axes[1].set_title(f"dE: p50={np.percentile(dE, 50):.1e} max={dE.max():.1e} eV"
                      f" (crit rel≤{E_REL:.0e} & abs≤{E_ABS:.0e})")
    axes[1].grid(alpha=0.3)
    axes[2].hist(dFr, bins=24, log=True)
    axes[2].set_xlabel("max|dF| per frame (eV/Å)")
    axes[2].set_title("Force diff (⑧ absent -> expected nonzero)"
                      if args.forces_expect_absent else "Force diff (gated)")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)

    summary = {
        "n_frames": len(rows),
        "n_pass": int(sum(r["pass"] for r in rows)),
        "verdict": verdict,
        "dE_abs": pct(dE), "df_max": pct(dFr), "df_mean": pct(dFm),
        "per_molecule": per_mol,
        "pos_drift_max": drift,
    }
    with open(f"{args.out}.json", "w") as fh:
        json.dump({"meta": {
            "json": J, "python_model": POLAR, "frames": args.frames,
            "rattle": args.rattle, "nmol": NMOL, "density": DENSITY,
            "forces_expect_absent": args.forces_expect_absent,
        }, "gate": {"e_rel": E_REL, "e_abs": E_ABS,
                    "force_tol": args.force_tol,
                    "forces_gated": not args.forces_expect_absent},
            "rows": rows, "summary": summary}, fh, indent=1)
    print(f"saved: {args.out}.png , {args.out}.json", flush=True)
    print("ENDTOEND-PARITY-DONE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
