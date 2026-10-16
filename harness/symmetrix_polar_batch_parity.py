#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Symmetrix 引擎 × PolarMACE backbone — 批量能量/力 parity 核验(P2, 2026-10-15)。

批 = NH3/CH4/N2/CO2 生产盒子(ρ=0.01)× N 个确定性扰动帧(默认 8/分子 → 32 帧;
--frames 25 → 100 帧)。

参考组合(经 NH3-32 校准,吻合至 1e-6 eV):
    ref_E = scale * interaction_energy + Σ E0(per-atom) + shift
    (scale=1, shift=0 for POLAR-M;electron_energy 与 electrostatic_energy 属
     P3 静电极性层,不在 backbone 对账范围——力差同口径解释)

用法: CUDA_VISIBLE_DEVICES=<gpu> <polar venv python> symmetrix_polar_batch_parity.py [--frames 8]
环境: SYMMETRIX_BACKEND_DESCRIPTOR_PATH / SYMMETRIX_BACKEND / LD_LIBRARY_PATH 见 run_polar_env.sh
"""
import argparse
import json
import sys

import numpy as np

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
import mace_md as mm  # noqa: E402

J = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
POLAR = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"
MOLS = [("NH3", 770600), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--rattle", type=float, default=0.05)
    ap.add_argument("--out", default=f"{ROOT}/bench/polar_parity")
    args = ap.parse_args()

    from symmetrix.calculator import Symmetrix
    from mace.calculators import mace_polar

    model_json = json.load(open(J))
    scale = float(model_json["scale_shift"]["scale"]["values"][0])
    shift = float(model_json["scale_shift"]["shift"]["values"][0])
    zmap = model_json["atomic_numbers"]
    e0_table = model_json["atomic_energies"]["values"]

    engine = Symmetrix(J, dtype="float64")
    ref_calc = mace_polar(model=POLAR, device="cuda", default_dtype="float64")

    rows, fe, fr = [], [], []
    for mol, seed in MOLS:
        base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
        zs = np.asarray(base.get_atomic_numbers())
        sum_e0 = float(np.sum([e0_table[zmap.index(int(z))] for z in zs]))
        for k in range(args.frames):
            atoms = base.copy()
            if k > 0:
                atoms.rattle(stdev=args.rattle * (k / max(1, args.frames - 1)) * 0.5,
                             seed=seed + 1000 * k)
            atoms.info["charge"] = 0
            atoms.info["spin"] = 1

            atoms.calc = engine
            e_eng = atoms.get_potential_energy()
            f_eng = atoms.get_forces()

            atoms.calc = ref_calc
            e_ref_total = atoms.get_potential_energy()
            f_ref = atoms.get_forces()
            r = ref_calc.results
            e_ref = scale * float(r["interaction_energy"]) + sum_e0 + shift

            rows.append({
                "mol": mol, "frame": k,
                "e_engine": float(e_eng),
                "e_ref": e_ref,
                "dE": float(e_eng - e_ref),
                "e_ref_total": float(e_ref_total),
                "dE_vs_total": float(e_eng - e_ref_total),
                "dfmax": float(np.abs(f_eng - f_ref).max()),
                "fmax_engine": float(np.abs(f_eng).max()),
                "fmax_ref": float(np.abs(f_ref).max()),
            })
            fe.append(float(e_eng)); fr.append(e_ref)
            print(f"  {mol:4s} f{k}: dE={rows[-1]['dE']:+.6f} eV | dFmax={rows[-1]['dfmax']:.4f} "
                  f"(E_eng={e_eng:.6f})", flush=True)

    d = np.array([r["dE"] for r in rows])
    dT = np.array([r["dE_vs_total"] for r in rows])
    dF = np.array([r["dfmax"] for r in rows])
    print(f"\nframes: {len(rows)}")
    print(f"dE(engine vs ref=scale*interaction+ΣE0+shift): MAE {np.abs(d).mean():.2e}  max {np.abs(d).max():.2e} eV")
    print(f"dE(engine vs python TOTAL,含 P3 静电层→应≈-9.22 附近): mean {dT.mean():.4f} eV")
    print(f"dFmax(engine vs python total forces, 缺 P3 静电力): mean {dF.mean():.4f}  max {dF.max():.4f} eV/Å")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    a = np.asarray(fe); b = np.asarray(fr)
    axes[0].scatter(b, a, s=18)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set_xlabel("Python ref: scale*interaction + ΣE0 + shift (eV)")
    axes[0].set_ylabel("Symmetrix engine E (eV)")
    axes[0].set_title(f"Backbone energy parity ({len(rows)} frames)")
    axes[0].grid(alpha=0.3)
    axes[1].hist(d, bins=20)
    axes[1].set_xlabel("dE = engine - ref (eV)")
    axes[1].set_title(f"dE: MAE {np.abs(d).mean():.1e} eV, max {np.abs(d).max():.1e} eV")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    png = f"{args.out}.png"
    fig.savefig(png, dpi=150)
    with open(f"{args.out}.json", "w") as fh:
        json.dump({"rows": rows, "summary": {
            "n_frames": len(rows),
            "dE_MAE": float(np.abs(d).mean()), "dE_max": float(np.abs(d).max()),
            "dE_vs_total_mean": float(dT.mean()),
            "dFmax_mean": float(dF.mean()), "dFmax_max": float(dF.max())}}, fh, indent=1)
    print(f"saved: {png} , {args.out}.json")
    print("PARITY-DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
