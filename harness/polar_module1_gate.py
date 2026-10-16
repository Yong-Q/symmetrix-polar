#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 模块①验收门: 引擎 polar_spin_charge_density vs Python 参考(2026-10-15)。

模块① = Σ_ℓ lr_source_maps[ℓ](node_feats[ℓ])  (mace extensions.py L666-703)
出口 = 引擎 polar_spin_charge_density (N×8 平坦序 [sα, xα, yα, zα, sβ, xβ, yβ, zβ])

参考双源:
  A) 既有 SCF 钩子夹具 bench/polar_fixtures/scf/*.npz 的 lr_src_0 + lr_src_1
     (32 分子盒 × 4 分子; 与引擎同几何, f64);
  B) 同进程活体重算: mace_polar + forward hook, 对每分子 --live 个附加确定性扰动帧
     现取 lr_src 和(独立于夹具的第二源, 检验非夹具构型)。

门: max|Δ| ≤ 5e-6(绝对; 与 P2 3.6e-7 同口径, 留一个量级给邻居表实现差异)。

用法: CUDA_VISIBLE_DEVICES=<gpu> <polar venv python> polar_module1_gate.py [--live 2]
环境: run env(DESCRIPTOR/backend/LD_LIBRARY_PATH/nvidia-smi shim); unset PYTHONPATH。
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


def engine_accumulator(engine, atoms):
    atoms.calc = engine
    atoms.get_potential_energy()
    acc = np.asarray(engine.evaluator.polar_spin_charge_density(),
                     dtype=np.float64)
    return acc.reshape(len(atoms), 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=int, default=2,
                    help="每分子附加活体扰动帧数(参考=同进程 mace_polar 重算)")
    ap.add_argument("--out", default=f"{ROOT}/bench/polar_module1_gate")
    args = ap.parse_args()

    from symmetrix.calculator import Symmetrix
    from mace.calculators import mace_polar

    engine = Symmetrix(J, dtype="float64")

    caps = {}
    handles = []
    if args.live > 0:
        ref_calc = mace_polar(model=POLAR, device="cuda", default_dtype="float64")
        model = ref_calc.models[0]
        for i, m in enumerate(model.lr_source_maps):
            handles.append(m.register_forward_hook(
                (lambda idx: lambda mod, inp, out: caps.setdefault(
                    idx, []).append(out.detach().cpu().numpy()))(i)))

    rows = []

    def compare(tag, atoms, ref_sum, acc):
        d = acc - ref_sum
        rows.append({
            "case": tag, "n_nodes": int(len(atoms)),
            "max_abs": float(np.abs(d).max()),
            "mae": float(np.abs(d).mean()),
            "ref_absmax": float(np.abs(ref_sum).max()),
            "per_comp_max": [float(np.abs(d[:, k]).max()) for k in range(8)],
        })
        r = rows[-1]
        print(f"  {tag:14s} max|Δ|={r['max_abs']:.3e}  mae={r['mae']:.3e}  "
              f"(ref absmax={r['ref_absmax']:.3f})", flush=True)

    # ---- 源 A: SCF 钩子夹具 ----
    print("== 源 A: scf 钩子夹具 ==", flush=True)
    for mol, seed in MOLS:
        name = f"{mol.lower()}-f0"
        d = np.load(f"{FIXDIR}/{name}.npz", allow_pickle=False)
        # 夹具未存 cell → 用确定性 build_gas 重建盒,并以夹具 positions 逐位核对
        base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
        assert [str(s) for s in d["symbols"]] == base.get_chemical_symbols(), \
            f"{name}: symbol order mismatch vs fixture"
        assert np.allclose(d["positions"], base.get_positions(), atol=1e-10), \
            f"{name}: rebuilt positions diverge from fixture"
        atoms = base.copy()
        acc = engine_accumulator(engine, atoms)
        ref = np.asarray(d["lr_src_0"], dtype=np.float64)[:, 0, :] \
            + np.asarray(d["lr_src_1"], dtype=np.float64)[:, 0, :]
        compare(name, atoms, ref, acc)

    # ---- 源 B: 活体重算(附加扰动帧) ----
    if args.live > 0:
        print(f"== 源 B: 活体重算(--live {args.live}/分子) ==", flush=True)
        for mol, seed in MOLS:
            base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
            for k in range(args.live):
                atoms = base.copy()
                atoms.rattle(stdev=0.05, seed=seed + 7 + 4 * k)
                atoms.info["charge"] = 0
                atoms.info["spin"] = 1
                acc = engine_accumulator(engine, atoms)
                caps.clear()
                atoms.calc = ref_calc
                atoms.get_potential_energy()
                assert all(len(caps[i]) == 1 for i in (0, 1)), \
                    f"hook capture count off: {[len(caps.get(i, [])) for i in (0, 1)]}"
                ref = caps[0][0][:, 0, :] + caps[1][0][:, 0, :]
                compare(f"{mol.lower()}-live{k}", atoms, ref, acc)

    if handles:
        for h in handles:
            h.remove()

    dm = np.array([r["max_abs"] for r in rows])
    ok = bool(dm.max() <= GATE)
    verdict = "PASS" if ok else "FAIL"
    print(f"\nGATE {verdict}: configs={len(rows)}  max|Δ|={dm.max():.3e}  "
          f"(gate {GATE:.0e})", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = {"sα": 0, "xα": 1, "yα": 2, "zα": 3, "sβ": 4, "xβ": 5,
              "yβ": 6, "zβ": 7}
    allmax = np.array([r["max_abs"] for r in rows])
    axes[0].bar(range(len(rows)), allmax)
    axes[0].axhline(GATE, color="r", ls="--", lw=1, label=f"gate={GATE:.0e}")
    axes[0].set_xticks(range(len(rows)))
    axes[0].set_xticklabels([r["case"] for r in rows], rotation=60, fontsize=7)
    axes[0].set_ylabel("max|Δ| (engine - python)")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Module-1 accumulator parity ({len(rows)} configs)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    pc = np.array([r["per_comp_max"] for r in rows]).max(axis=0)
    axes[1].bar(list(colors.keys()), pc)
    axes[1].set_ylabel("max|Δ| per component")
    axes[1].set_yscale("log")
    axes[1].set_title("Per-component (flat order [sα,xyzα,sβ,xyzβ])")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    with open(f"{args.out}.json", "w") as fh:
        json.dump({"rows": rows, "gate": GATE, "verdict": verdict}, fh, indent=1)
    print(f"saved: {args.out}.png , {args.out}.json", flush=True)
    print("MODULE1-GATE-DONE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
