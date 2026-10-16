#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 模块②验收门: Fukui 初始化校正(2026-10-15)。

模块② = fukui_source_map + 每图归一 + l=0 通道校正 (mace extensions.py L737-766):
  fukui_sources = fukui_source_map(node_feats[-1])          # (N,2)
  fukui_norm_g  = scatter_sum(fukui_sources, batch)         # 每图每通道; 0→1
  src_n         = fukui_sources / fukui_norm_g[batch]
  pred_0        = scatter_sum(density[:, :, 0], batch)      # 每图 l=0 总量 (2 通道)
  density[:,0,0] += src_n[:,0] * ((Q+S...)/2 - pred_0[:,0])   # Q_p_S=Q+(S-1)
  density[:,1,0] += src_n[:,1] * ((Q-S...)/2 - pred_0[:,1])   # Q_m_S=Q-(S-1)
出口(引擎新增访问器):
  polar_fukui_sources              (N,2) 未归一源
  polar_field_independent_density  (N,8) 校正后密度(= 后续 SCF 起点; mace 变量名
                                        field_independent_spin_charge_density)

参考双源(与模块①门同构):
  A) 夹具 lr_src_0+1 + fukui_src → numpy 逐式重建 ref_corrected(N,8);
  B) 活体: mace_polar + hooks(lr_source_maps×2, fukui_source_map) 现取现重建。

门: 两项 max|Δ| ≤ 5e-6。
用法: CUDA_VISIBLE_DEVICES=<gpu> <polar venv python> polar_module2_gate.py [--live 2]
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
FIXDIR = f"{ROOT}/bench/polar_fixtures/scf"
MOLS = [("NH3", 770600), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]
GATE = 5e-6


def ref_fukui_correct(init_flat, fukui_src, charge, spin):
    """numpy 逐式重建(依据 extensions.py L742-766; 单图=整盒求和)。"""
    n = init_flat.shape[0]
    den = init_flat.reshape(n, 2, 4).copy()
    src = np.asarray(fukui_src, dtype=np.float64)  # (N,2)
    norm = src.sum(axis=0)                          # (2,)
    norm = np.where(norm == 0.0, 1.0, norm)
    src_n = src / norm
    pred_0 = den[:, :, 0].sum(axis=0)               # (2,)
    qp = charge + (spin - 1.0)
    qm = charge - (spin - 1.0)
    den[:, 0, 0] += src_n[:, 0] * (qp / 2.0 - pred_0[0])
    den[:, 1, 0] += src_n[:, 1] * (qm / 2.0 - pred_0[1])
    return den.reshape(n, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=int, default=2)
    ap.add_argument("--out", default=f"{ROOT}/bench/polar_module2_gate")
    args = ap.parse_args()

    from symmetrix.calculator import Symmetrix
    from mace.calculators import mace_polar

    engine = Symmetrix(J, dtype="float64")
    ev = engine.evaluator
    if not hasattr(ev, "polar_fukui_sources") or not hasattr(
            ev, "polar_field_independent_density"):
        print("MODULE2-ACCESSORS-MISSING: 引擎未含模块②访问器(先编译 v14)", flush=True)
        return 2

    caps = {}
    handles = []
    if args.live > 0:
        ref_calc = mace_polar(model=POLAR, device="cuda", default_dtype="float64")
        model = ref_calc.models[0]
        for i, m in enumerate(model.lr_source_maps):
            handles.append(m.register_forward_hook(
                (lambda idx: lambda mod, inp, out: caps.setdefault(
                    f"lrs{idx}", []).append(out.detach().cpu().numpy()))(i)))
        handles.append(model.fukui_source_map.register_forward_hook(
            lambda mod, inp, out: caps.setdefault("fukui", []).append(
                out.detach().cpu().numpy())))

    rows = []

    def run_case(tag, atoms, lrs, fukui_ref, charge, spin):
        atoms.calc = engine
        atoms.get_potential_energy()
        n = len(atoms)
        f_fukui = np.asarray(ev.polar_fukui_sources(), dtype=np.float64).reshape(n, 2)
        f_fid = np.asarray(ev.polar_field_independent_density(),
                           dtype=np.float64).reshape(n, 8)
        init = lrs[0] + lrs[1] if lrs.ndim == 3 else lrs
        ref_fid = ref_fukui_correct(init, fukui_ref, charge, spin)
        df = np.abs(f_fukui - fukui_ref).max()
        scale = max(float(np.abs(fukui_ref).max()), 1.0)
        df_rel = float(df / scale)  # 标度感知: fukui 原始值 O(10-500), 用相对判据
        # 归一后源(物理路径上实际进入校正的量)绝对对照
        rn = fukui_ref / np.where(fukui_ref.sum(axis=0) == 0, 1.0,
                                  fukui_ref.sum(axis=0))
        en = f_fukui / np.where(f_fukui.sum(axis=0) == 0, 1.0,
                                f_fukui.sum(axis=0))
        dn = np.abs(en - rn).max()
        dd = np.abs(f_fid - ref_fid).max()
        rows.append({"case": tag, "fukui_max": float(df), "fukui_rel": df_rel,
                     "fukui_refmax": scale, "norm_max": float(dn),
                     "density_max": float(dd)})
        print(f"  {tag:14s} |Δfukui|={df:.3e} (rel {df_rel:.2e}, refmax {scale:.1f})"
              f"   |Δnorm|={dn:.3e}   |Δdensity|={dd:.3e}", flush=True)

    print("== 源 A: 夹具(scf/*.npz) ==", flush=True)
    for mol, seed in MOLS:
        name = f"{mol.lower()}-f0"
        d = np.load(f"{FIXDIR}/{name}.npz", allow_pickle=False)
        base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
        assert np.allclose(d["positions"], base.get_positions(), atol=1e-10)
        atoms = base.copy()
        lrs = np.stack([np.asarray(d["lr_src_0"])[:, 0, :],
                        np.asarray(d["lr_src_1"])[:, 0, :]])
        run_case(name, atoms, lrs, np.asarray(d["fukui_src"]), 0.0, 1.0)

    if args.live > 0:
        print(f"== 源 B: 活体重算(--live {args.live}/分子) ==", flush=True)
        for mol, seed in MOLS:
            base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
            for k in range(args.live):
                atoms = base.copy()
                atoms.rattle(stdev=0.05, seed=seed + 7 + 4 * k)
                atoms.info["charge"] = 0
                atoms.info["spin"] = 1
                caps.clear()
                atoms.calc = ref_calc
                atoms.get_potential_energy()
                assert len(caps["lrs0"]) == 1 and len(caps["lrs1"]) == 1 \
                    and len(caps["fukui"]) == 1, "hook capture count off"
                lrs = np.stack([caps["lrs0"][0][:, 0, :], caps["lrs1"][0][:, 0, :]])
                fukui_ref = caps["fukui"][0]
                run_case(f"{mol.lower()}-live{k}", atoms, lrs, fukui_ref, 0.0, 1.0)

    for h in handles:
        h.remove()

    df = np.array([r["fukui_max"] for r in rows])
    drel = np.array([r["fukui_rel"] for r in rows])
    dn = np.array([r["norm_max"] for r in rows])
    dd = np.array([r["density_max"] for r in rows])
    # 判据(标度感知, 依据 stageprobe: 链实现逐级 2e-13<精确>,实跑差=骨干 f64 漂移 ~1e-8 相对):
    #  reL(fukui 原始) ≤ 1e-6  且  归一源 ≤ 5e-6  且  校正后密度 ≤ 5e-6
    ok = bool(drel.max() <= 1e-6 and dn.max() <= GATE and dd.max() <= GATE)
    verdict = "PASS" if ok else "FAIL"
    print(f"\nGATE {verdict}: configs={len(rows)}  fukui rel max={drel.max():.2e} "
          f"(abs {df.max():.3e})  norm max={dn.max():.3e}  density max={dd.max():.3e}  "
          f"(criteria rel<=1e-6, abs<=5e-6)", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = range(len(rows))
    axes[0].bar(x, df, label="|Δfukui|")
    axes[0].bar(x, dd, bottom=df, alpha=0.6, label="|Δdensity|")
    axes[0].axhline(GATE, color="r", ls="--", lw=1, label=f"gate={GATE:.0e}")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([r["case"] for r in rows], rotation=60, fontsize=7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("max|Δ|")
    axes[0].set_title(f"Module-2 Fukui parity ({len(rows)} configs)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].text(0.5, 0.5, "Module 2:\nfukui_source_map + per-graph norm\n"
                 "+ l=0 channel correction\nvs numpy reconstruction\n"
                 f"verdict: {verdict}", ha="center", va="center", fontsize=12)
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    with open(f"{args.out}.json", "w") as fh:
        json.dump({"rows": rows, "gate": GATE, "verdict": verdict}, fh, indent=1)
    print(f"saved: {args.out}.png , {args.out}.json", flush=True)
    print("MODULE2-GATE-DONE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
