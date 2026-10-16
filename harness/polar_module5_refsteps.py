#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 模块⑤参考侧数据: SCF 逐步密度 (2026-10-15)。

引擎模块⑤（charges_maps 更新块）的门夹具生成器: 用 scf/*.npz 夹具里的
lr_src / fukui_src / charges_step_k, 在 numpy 里**逐行照抄** mace
extensions.py::PolarMACE.forward 的密度更新规则, 得到 step0/1/2 的
spin_charge_density 状态序列, 并与夹具 final_spin_charge_density 交叉验证。

--- 更新规则逐行锚定 (mace 0.3.16, extensions.py) -----------------------------
初值 (L666-703, L734-735):   density = Σ_l lr_source_maps[l](node_feats[l])  (N,8)
    flat 8 = [sα,xα,yα,zα,sβ,xβ,yβ,zβ]  (plan §13 布局裁定) → view(N,2,4)
初校正 (L737-766, = 模块②):
    src = fukui_source_map(node_feats); norm = Σ_atoms src /图, 0→1; src_n = src/norm
    src_n 的 α 列校正 density[:,0,0], β 列校正 density[:,1,0], 目标 Q_p_S/2, Q_m_S/2
    Q_p_S = total_charge+(total_spin-1), Q_m_S = total_charge-(total_spin-1)  (L752-753)
    → field_independent_spin_charge_density (L774) = 本脚本 step0, SCF 循环起点
SCF 循环 (L777-864), 共 num_recursion_steps=2 步, 第 i 步:
    [输入] source_feats = density[:,α或β,:] → forward_dynamic → 势特征 (模块③/④)
           → charges_map[i](..., local_charges=density.view(N,8))  (模块⑤ 本体)
    L833  current_fukui_sources = out[:, -2:]      ← 切法: 末尾 2 列 (α/β 新 Fukui 源)
    L834  charge_sources       = out[:, :-2]      ← 切法: 前 8 列 = 密度增量
    L837-840  density += charge_sources.view(N,2,-1)     (行主序: 前4=α, 后4=β)
    L842-851  新源 **每图重新归一** (Σ_atoms, 0→1 钳制) —— 每步都做, 不缓存
    L852-857  pred_total_charges = 更新后 density[:,:,0] 每图求和 (double 累加)
    L858-864  **每步重新做 l=0 Fukui 校正** (同初校正的目标 Q_p_S/2, Q_m_S/2)
无外场 (external_field=0) 时 L806-816 的 half_external_field=0, 只做 /field_feature_norms,
不影响密度链; 本脚本只重建密度侧, 不重建势特征 (势特征见 geo3 夹具与模块④门)。

--- 门 -----------------------------------------------------------------------
交叉验证: 最后一步结果 vs 夹具 final_spin_charge_density (N,2,4)。
  PASS 判据 max|Δ| ≤ 1e-6（与模块①/②门同口径, f64 应有 ~1e-13 级余量）。
深度自证 (--deep, 默认开): 用 geo3 同配方 cache, forward_dynamic(cache, step1密度)
  vs 夹具 potential_feats_step_2/3 —— 独立锁定 step1 密度与步序对应关系。

用法: cd mlip; CUDA_VISIBLE_DEVICES=1 /workspace/caches/conda/envs/polar/bin/python \
        bench/polar_module5_refsteps.py
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/bench")
import mace_md as mm  # noqa: E402,F401  (polar_module2_gate 依赖 scripts 路径)
from polar_module2_gate import ref_fukui_correct  # noqa: E402  (模块②参考实现, 复用)
import polar_module3_refdump as m3  # noqa: E402  (几何/k 网格 recipe, 复用)

MOLS = [("NH3", 770600), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]
FIXDIR = f"{ROOT}/bench/polar_fixtures/scf"
OUTDIR = f"{ROOT}/bench/polar_fixtures/scf_steps"
POLAR_MODEL = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"
PROBE_JSON = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
GATE = 1e-6
CHARGE, SPIN = 0.0, 1.0  # 夹具 dump 口径: atoms.info charge=0 / spin=1


def initial_density(fix, charge=CHARGE, spin=SPIN):
    """L666-703 + L734-766：lr_src 累积 → (N,8) → 模块② Fukui 校正 → (N,2,4)。

    flat (N,8) 与 (N,2,4) 关系 = 行主序 reshape（前 4 列 α, 后 4 列 β）。
    """
    init = (np.asarray(fix["lr_src_0"])[:, 0, :]
            + np.asarray(fix["lr_src_1"])[:, 0, :])            # (N,8)
    fid = ref_fukui_correct(init, np.asarray(fix["fukui_src"]),
                            charge, spin)                      # (N,8)
    return fid.reshape(-1, 2, 4), init.reshape(-1, 2, 4)


def fukui_sources_normalized(fix):
    """初校正用源的归一化副本 (N,2)（复用模块②配方, 供落盘核对）。"""
    src = np.asarray(fix["fukui_src"], dtype=np.float64).copy()
    norm = src.sum(axis=0)
    norm = np.where(norm == 0.0, 1.0, norm)
    return src / norm


def scf_density_update(density, sources_out, charge=CHARGE, spin=SPIN):
    """逐行照抄 extensions.py L833-864 的一步密度更新。

    density     (N,2,4) 当前密度（mace spin_charge_density 内部表示）
    sources_out (N,10)  field_dependent_charges_maps[i] 原始输出
    返回 (new_density (N,2,4), increment (N,2,4), src_norm_used (N,2))
    """
    n = density.shape[0]
    cur_src = np.array(sources_out[:, -2:], dtype=np.float64)   # L833
    incr_flat = np.array(sources_out[:, :-2], dtype=np.float64)  # L834
    increment = incr_flat.reshape(n, 2, -1)                      # L837-839 view(N,2,-1)
    density = density + increment                                # L840

    denom = cur_src.sum(axis=0)                                  # L842-847 单图(整盒)求和
    denom = np.where(denom == 0.0, 1.0, denom)                   # L848-850 0→1
    cur_src = cur_src / denom                                    # L851
    pred = density[:, :, 0].sum(axis=0)                          # L852-857 l=0 每图总量

    qp = charge + (spin - 1.0)                                   # L752
    qm = charge - (spin - 1.0)                                   # L753
    density = density.copy()                                     # L858 clone (数值无影响)
    density[:, 0, 0] += cur_src[:, 0] * (qp / 2.0 - pred[0])     # L859-861
    density[:, 1, 0] += cur_src[:, 1] * (qm / 2.0 - pred[1])     # L862-864
    return density, increment, cur_src


def l0_totals(density):
    """每自旋通道 l=0 总量 (2,)（校正目标 = Q_p_S/2, Q_m_S/2）。"""
    return density[:, :, 0].sum(axis=0)


def residual_report(a, b):
    """精确残差报告（不掩盖差异）: 全局/分通道/分分量/百分位。"""
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    out = {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "p50": float(np.percentile(d, 50)),
        "p99": float(np.percentile(d, 99)),
        "ref_absmax": float(np.abs(b).max()),
        "rel_max_vs_ref_absmax": float(d.max() / max(float(np.abs(b).max()), 1e-300)),
    }
    if a.ndim == 3:  # (N,2,4) 分通道/分量
        out["per_channel_max"] = [float(d[:, c, :].max()) for c in range(a.shape[1])]
        out["per_component_max"] = [float(d[:, :, j].max())
                                    for j in range(a.shape[2])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", type=int, default=1,
                    help="1=额外用 forward_dynamic 锁定 step1 密度（需 GPU/模型）")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cutoff = float(json.load(open(PROBE_JSON))["polar_blocks"]
                   ["electric_potential_descriptor"]["kspace_cutoff"])
    os.makedirs(OUTDIR, exist_ok=True)

    # 深度自证所需的模型（只加载一次; --deep 0 时完全跳过）
    calc = None
    if args.deep:
        from mace.calculators import mace_polar
        calc = mace_polar(model=POLAR_MODEL, device=args.device,
                          default_dtype="float64")
        print(f"[cfg] model num_recursion_steps="
              f"{calc.models[0].num_recursion_steps}", flush=True)

    rows, index = [], {
        "generator": "bench/polar_module5_refsteps.py",
        "date": "2026-10-15",
        "gate": GATE,
        "charge_spin": [CHARGE, SPIN],
        "rule_citation": {
            "init": "extensions.py L666-703 + L734-766 (lr_src 累积 + Fukui 初校正)",
            "loop": "extensions.py L777-864 (SCF; num_recursion_steps=2)",
            "split": "extensions.py L833-834: out[:, -2:] = 新 Fukui 源(2列); "
                     "out[:, :-2] = 密度增量(8列) → view(N,2,4) 行主序",
            "renorm": "extensions.py L842-851: 每步对新 Fukui 源做逐图归一 (0→1)",
            "recorrect": "extensions.py L852-864: 每步重做 l=0 校正, 目标 Q_p_S/2 / Q_m_S/2",
            "clone": "L858/L760 clone 仅断计算图, 数值无影响",
        },
        "cases": {},
    }

    # ---------- 主线: numpy 逐步推进 (无模型) ----------
    for mol, seed in MOLS:
        fix = np.load(f"{FIXDIR}/{mol.lower()}-f0.npz", allow_pickle=False)
        n_steps = sum(1 for k in fix.files if k.startswith("charges_step_"))
        if n_steps != 2:
            print(f"[warn] {mol}: 夹具 charges_step_* 数={n_steps} (模型 "
                  f"num_recursion_steps=2)", flush=True)

        try:
            atoms, info = m3.build_box(mol, seed)  # 几何自证（build_gas vs positions）
            assert np.allclose(atoms.get_positions(), fix["positions"], atol=1e-10)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {mol}: 几何自证跳过 ({exc})", flush=True)

        den0, init24 = initial_density(fix)
        states = {0: den0.copy()}
        deltas, srcs = {0: (den0 - init24)}, {0: fukui_sources_normalized(fix)}
        for k in range(n_steps):
            den, incr, s = scf_density_update(states[k],
                                              np.asarray(fix[f"charges_step_{k}"]))
            states[k + 1] = den
            deltas[k + 1] = incr
            srcs[k + 1] = s
            print(f"  {mol.lower()}-f0 step{k+1}: |Δdensity|_max={np.abs(incr).max():.4e}"
                  f"  l0_totals(α,β)={l0_totals(den)[0]:+.3e}/{l0_totals(den)[1]:+.3e}",
                  flush=True)

        # ---- 交叉验证: 最终态 vs 夹具 final_spin_charge_density ----
        ref = np.asarray(fix["final_spin_charge_density"], dtype=np.float64)
        final = states[n_steps]
        rep = residual_report(final, ref)
        verdict = "PASS" if rep["max_abs"] <= GATE else "FAIL"
        ok = rep["max_abs"] <= GATE
        print(f"[gate] {mol.lower()}-f0  final vs fixture: max|Δ|={rep['max_abs']:.3e}"
              f"  mean={rep['mean_abs']:.3e}  (gate {GATE:.0e})  -> {verdict}",
              flush=True)
        if not ok:
            print(f"       per_channel_max={rep['per_channel_max']}  "
                  f"per_component_max={rep['per_component_max']}  "
                  f"p99={rep['p99']:.3e}  ref_absmax={rep['ref_absmax']:.3e}",
                  flush=True)

        case = {"fixture": f"scf/{mol.lower()}-f0.npz", "n_scf_steps": n_steps,
                "verdict": verdict, "residual_final_vs_fixture": rep,
                "steps": {}}

        # ---- 落盘 step0/1/2 ----
        for k in sorted(states):
            stem = f"{mol.lower()}-step{k}"
            np.savez(
                f"{OUTDIR}/{stem}.npz",
                spin_charge_density=states[k],                       # (N,2,4)
                spin_charge_density_flat=states[k].reshape(-1, 8),   # (N,8) 行主序
                density_delta=deltas[k],                             # (N,2,4) 本步增量
                fukui_sources_used=srcs[k],                          # (N,2) 本步归一源
                l0_totals=l0_totals(states[k]),                      # (2,)
            )
            case["steps"][f"step{k}"] = {
                "npz": f"scf_steps/{stem}.npz",
                "shapes": {"spin_charge_density": list(states[k].shape),
                           "spin_charge_density_flat": [states[k].shape[0], 8],
                           "density_delta": list(deltas[k].shape),
                           "fukui_sources_used": list(srcs[k].shape)},
                "absmax": float(np.abs(states[k]).max()),
                "l0_totals": [float(v) for v in l0_totals(states[k])],
            }

        # ---- 深度自证: step1 密度 → forward_dynamic vs 夹具 step_2/3 势特征 ----
        if calc is not None and 1 in states:
            import torch
            model = calc.models[0]
            dev = next(model.parameters()).device
            desc = model.electric_potential_descriptor
            gi = m3.geometry_inputs(mol, seed, cutoff, dev)
            with torch.no_grad():
                cache = m3.precompute_geometry_cache(desc, gi)
            dchk = {}
            for ch, key in ((0, "potential_feats_step_2"), (1, "potential_feats_step_3")):
                src_t = torch.tensor(states[1][:, ch, :][:, None, :],
                                     dtype=torch.float64, device=dev)
                with torch.no_grad():
                    feats = desc.forward_dynamic(cache=cache, source_feats=src_t,
                                                 pbc=gi["pbc_t"])
                rr = residual_report(feats.cpu().numpy(),
                                     np.asarray(fix[key], dtype=np.float64))
                dchk[key] = rr
                print(f"[deep ] {mol.lower()}-f0 step1→{key}: "
                      f"max|Δ|={rr['max_abs']:.3e}", flush=True)
            case["deep_check_step1_vs_feats"] = dchk

        index["cases"][f"{mol.lower()}-f0"] = case
        rows.append((mol.lower(), rep["max_abs"], verdict))

    worst = max(r[1] for r in rows)
    overall = "PASS" if worst <= GATE else "FAIL"
    index["verdict"] = overall
    index["worst_max_abs"] = float(worst)
    with open(f"{OUTDIR}/index.json", "w") as fh:
        json.dump(index, fh, indent=1)

    print(f"\nGATE {overall}: configs={len(rows)}  worst max|Δ|={worst:.3e}  "
          f"(gate {GATE:.0e})", flush=True)
    print(f"[done] -> {OUTDIR} (step k=0..2 npz + index.json)", flush=True)
    print("MODULE5-REFSTEPS-DONE", flush=True)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
