#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 模块③④验收门: k 网格/GTO 基函数/势特征(2026-10-15)。

模块③ = k 网格(半空间枚举+滤波) + GTO 几何预计算(cos/sin、density/feature basis)
模块④ = forward_dynamic(初始密度的势特征 α|β,含逐通道归一)

参考:
  A) ③: bench/polar_fixtures/geo3/{mol}-f0-kvecs.npz + -pregeom.npz
     (智能体B 直调 graph_longrange 的参考 dump;探针公式已 numpy 预检逐位吻合)
  B) ④: bench/polar_fixtures/scf/{mol}-f0.npz 的 potential_feats_step_0(α)/
     step_1(β) —— SCF 首步即初始(校正后)密度的 forward_dynamic 输出。

判据: ③ 结构量采用"逐位级"绝对门(1e-9);④ 势特征 O(10) 量级用相对 1e-6
(与模块①/②同口径;骨干/算术序漂移 ~1e-8 相对)。
用法: CUDA_VISIBLE_DEVICES=<gpu> <polar venv python> polar_module34_gate.py
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
import mace_md as mm  # noqa: E402

J = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
GEODIR = f"{ROOT}/bench/polar_fixtures/geo3"
SCFDIR = f"{ROOT}/bench/polar_fixtures/scf"
MOLS = [("NH3", 770600), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]


def main():
    import json as _json_cfg_module
    global _json_cfg
    _json_cfg = _json_cfg_module.load(open(J))

    from symmetrix.calculator import Symmetrix

    engine = Symmetrix(J, dtype="float64")
    ev = engine.evaluator
    needed = ("polar_k_vectors", "polar_k_norm2", "polar_k0_mask",
              "polar_cosines", "polar_density_basis_re",
              "polar_density_basis_im", "polar_feature_basis_re",
              "polar_feature_basis_im", "polar_potential_features")
    if not all(hasattr(ev, n) for n in needed):
        print("M34-ACCESSORS-MISSING", flush=True)
        return 2

    rows = []
    for mol, seed in MOLS:
        name = f"{mol.lower()}-f0"
        base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
        atoms = base.copy()
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        atoms.calc = engine
        atoms.get_potential_energy()

        n = len(atoms)
        dk = np.load(f"{GEODIR}/{name}-kvecs.npz")
        dg = np.load(f"{GEODIR}/{name}-pregeom.npz")
        df = np.load(f"{SCFDIR}/{name}.npz")
        nk = len(dk["k_vectors"])

        kv = np.asarray(ev.polar_k_vectors(), dtype=np.float64).reshape(nk, 3)
        kn2 = np.asarray(ev.polar_k_norm2(), dtype=np.float64)
        k0m = np.asarray(ev.polar_k0_mask(), dtype=np.float64)
        cos_e = np.asarray(ev.polar_cosines(), dtype=np.float64).reshape(nk, n)
        dbr = np.asarray(ev.polar_density_basis_re(), dtype=np.float64).reshape(nk, 4)
        dbi = np.asarray(ev.polar_density_basis_im(), dtype=np.float64).reshape(nk, 4)
        fbr = np.asarray(ev.polar_feature_basis_re(), dtype=np.float64).reshape(nk, 8)
        fbi = np.asarray(ev.polar_feature_basis_im(), dtype=np.float64).reshape(nk, 8)
        pf = np.asarray(ev.polar_potential_features(), dtype=np.float64).reshape(n, 16)

        checks = {
            "k_vectors": np.abs(kv - dk["k_vectors"]).max(),
            "k_norm2": np.abs(kn2 - dk["k_norm2"]).max(),
            "k0_mask": np.abs(k0m - dk["k0_mask"]).max(),
            "cosines": np.abs(cos_e - dg["cosines"]).max(),
            "dens_basis_re": np.abs(dbr - dg["density_basis_fs_re"][:, 0, :]).max(),
            "dens_basis_im": np.abs(dbi - dg["density_basis_fs_im"][:, 0, :]).max(),
            "feat_basis_re": np.abs(fbr - dg["feature_basis_fs_re"].reshape(nk, 8)).max(),
            "feat_basis_im": np.abs(fbi - dg["feature_basis_fs_im"].reshape(nk, 8)).max(),
        }
        alpha_ref = np.asarray(df["potential_feats_step_0"], dtype=np.float64)
        beta_ref = np.asarray(df["potential_feats_step_1"], dtype=np.float64)
        # 夹具钩子在 forward_dynamic 输出端(归一前);引擎输出含 /norms → 乘回
        norms = np.asarray(
            _json_cfg["polar_blocks"]["configs"]["field_feature_norms"]["values"],
            dtype=np.float64)
        pf_alpha = pf[:, :8] * norms[None, :]
        pf_beta = pf[:, 8:] * norms[None, :]
        refmax = max(float(np.abs(alpha_ref).max()), float(np.abs(beta_ref).max()), 1.0)
        rel_a = float(np.abs(pf_alpha - alpha_ref).max()) / refmax
        rel_b = float(np.abs(pf_beta - beta_ref).max()) / refmax
        checks["potfeat_alpha_rel"] = rel_a
        checks["potfeat_beta_rel"] = rel_b

        row = {"case": name, **{k: float(v) for k, v in checks.items()}}
        rows.append(row)
        print(f"  {name}: k {checks['k_vectors']:.1e} | cos {checks['cosines']:.1e}"
              f" | basis {max(checks['dens_basis_re'], checks['feat_basis_re'], checks['dens_basis_im'], checks['feat_basis_im']):.1e}"
              f" | potfeat rel α {rel_a:.2e} β {rel_b:.2e}", flush=True)

    def col(key):
        return np.array([r[key] for r in rows])

    struct_max = max(col(k).max() for k in
                     ("k_vectors", "k_norm2", "k0_mask", "cosines",
                      "dens_basis_re", "dens_basis_im",
                      "feat_basis_re", "feat_basis_im"))
    pf_rel_max = max(col("potfeat_alpha_rel").max(), col("potfeat_beta_rel").max())
    ok = bool(struct_max <= 1e-9 and pf_rel_max <= 1e-6)
    verdict = "PASS" if ok else "FAIL"
    print(f"\nGATE {verdict}: configs={len(rows)}  struct max={struct_max:.3e} "
          f"(gate 1e-9)  potfeat rel max={pf_rel_max:.2e} (gate 1e-6)", flush=True)

    import json as _json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    keys = ["k_vectors", "k_norm2", "k0_mask", "cosines", "dens_basis_re",
            "dens_basis_im", "feat_basis_re", "feat_basis_im"]
    axes[0].bar(range(len(keys)), [max(float(col(k).max()), 1e-18) for k in keys])
    axes[0].set_xticks(range(len(keys)))
    axes[0].set_xticklabels(keys, rotation=60, fontsize=7)
    axes[0].axhline(1e-9, color="r", ls="--", lw=1)
    axes[0].set_yscale("log")
    axes[0].set_title("Module-3 structural quantities max|Δ|")
    axes[0].grid(alpha=0.3)
    axes[1].bar(range(len(rows)), np.maximum(
        np.maximum(col("potfeat_alpha_rel"), col("potfeat_beta_rel")), 1e-18))
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels([r["case"] for r in rows], rotation=60, fontsize=7)
    axes[1].axhline(1e-6, color="r", ls="--", lw=1)
    axes[1].set_yscale("log")
    axes[1].set_title("Module-4 potential features rel err")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/bench/polar_module34_gate.png", dpi=150)
    with open(f"{ROOT}/bench/polar_module34_gate.json", "w") as fh:
        _json.dump({"rows": rows, "verdict": verdict}, fh, indent=1)
    print("MODULE34-GATE-DONE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
