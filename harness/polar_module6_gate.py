#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 模块⑥验收门: coulomb_energy(全 k 空间 GTO-Ewald)(2026-10-15)。

模块⑥ = 全 k GTO-Ewald 前向标量。引擎侧访问器(补丁 = docs/module6_patch_final.md):
  polar_coulomb_energy()    (1,) eV   —— 能量标量
  polar_coulomb_rho_re/im() (Nk)      —— ρ̂(k) 诊断(未乘 (2π)³/V)

参考侧 = bench/polar_module6_preflight.py 的逐式函数(import 复用,该脚本对
index.json 双源 rel ≤1.2e-15 已锁定):
  rebuild_density_basis / assemble_density / self_interaction_energy /
  pbc_correction_energy;能量链 E = 0.5·V·Σ_k 2·(d·p)/(2π)^6, p = d·kfac·FIELD_CONSTANT。

口径(重要): ⑥ 的能量取自 **SCF 末态密度** —— 门内先 atoms.get_potential_energy()
跑完整前向(⑤ 已落地时含 SCF 循环),之后才读 polar_scf_final_density()/
polar_coulomb_energy();禁止在 SCF 之前的中间态取 ⑥ 标量。

密度模式(--density-mode):
  auto(默认) 探测 polar_scf_final_density 访问器且非空 → scf-final,否则 post-fukui
  scf-final  引擎 SCF 末态(终态门;端到端目标 = index.json electrostatic_energy)
  post-fukui 过渡模式:② 出口(polar_field_independent_density,= SCF 起点),
             端到端目标 = 预检的 E_es(step0) 过渡值, 判据 abs ≤1e-6 eV
             (module6_patch_draft §3.4;适合 ⑤ 未落地/回退时)

判据(参照模块①②门):
  ① ρ̂(k) 中间量: rel(对 |ρ̂|max) ≤ 1e-6
  ② 隔离能量: |E_engine − E_numpy(引擎密度)| rel ≤1e-6 且 abs ≤5e-6 eV
  ③ 夹具端到端: scf-final 同 ②(对 index.json electrostatic_energy);
                post-fukui abs ≤1e-6 eV(对 E_es(step0))
  活体帧(可选 --live k,扰动帧):仅 ①②(无夹具目标)。

诊断列: SI 反事实(include_self_interaction=true ⇒ 引擎**不得**减 0.5·Σ<q,self>,
反事实值应远大于 ② 的 Δ)、TTT pbc 修正选择器(is_molecule/is_slab 应为 False、
修正恒 0)、两种密度链的 ΔE 并列(供识别密度源接错)。

用法: cd mlip && CUDA_VISIBLE_DEVICES=<gpu> \
      /workspace/caches/conda/envs/polar/bin/python bench/polar_module6_gate.py \
      [--live 1] [--density-mode auto] [--out bench/polar_module6_gate]
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
import mace_md as mm  # noqa: E402
import polar_module6_preflight as pf  # noqa: E402  (参考实现复用)

J = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
FIXDIR = f"{ROOT}/bench/polar_fixtures"
GEODIR = f"{FIXDIR}/geo3"
SCFDIR = f"{FIXDIR}/scf"
STEPDIR = f"{FIXDIR}/scf_steps"
MOLS = [("NH3", 770600), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]
GATE_ABS = 5e-6        # module①② 绝对门
GATE_REL = 1e-6        # module①② 相对门
GATE_INTERIM_ABS = 1e-6  # 过渡门(post-fukui)绝对门(§3.4)


def fast_energy(spin_sum, cosines, sines, basis_re, basis_im, volume,
                kfac):
    """E = 0.5·V·Σ_k 2·(d·p)/(2π)^6 逐式(preflight §4 同链);
    返回 (E, rho=[(2π)³/V·ρ̂] 合并实虚部前的复数密度 (Nk,2))。"""
    rho, _, _ = pf.assemble_density(spin_sum, cosines, sines,
                                    basis_re, basis_im, volume)
    potential = rho * kfac[:, None] * pf.FIELD_CONSTANT
    per_k = 2.0 * (rho * potential).sum(-1)
    energy = 0.5 * volume * per_k.sum() / pf.TWO_PI ** 6
    return float(energy), rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=int, default=1,
                    help="每分子活体扰动帧数(0=仅夹具)")
    ap.add_argument("--density-mode", default="auto",
                    choices=["auto", "scf-final", "post-fukui"])
    ap.add_argument("--out", default=f"{ROOT}/bench/polar_module6_gate")
    args = ap.parse_args()

    with open(J) as fh:
        full = json.load(fh)
    pb = full["polar_blocks"]
    ce = pb["coulomb_energy"]
    index = json.load(open(f"{FIXDIR}/index.json"))
    print(f"include_self_interaction={ce['include_self_interaction']} "
          f"include_pbc_corrections={ce['include_pbc_corrections']} "
          f"kspace_cutoff={ce['kspace_cutoff']!r}", flush=True)

    from symmetrix.calculator import Symmetrix

    engine = Symmetrix(J, dtype="float64")
    ev = engine.evaluator
    required = ("polar_coulomb_energy", "polar_coulomb_rho_re",
                "polar_coulomb_rho_im")
    if not all(hasattr(ev, n) for n in required):
        print("MODULE6-ACCESSORS-MISSING: 引擎未含模块⑥访问器"
              "(先应用 docs/module6_patch_final.md 并重建)", flush=True)
        return 2
    if not hasattr(ev, "polar_field_independent_density"):
        print("MODULE6-BASE-ACCESSOR-MISSING: 缺 polar_field_independent_density",
              flush=True)
        return 2

    mode = None if args.density_mode == "auto" else args.density_mode
    rows = []

    def read_engine(atoms):
        """跑完前向后读取引擎侧全部量(SCF 已在前向内完成)。"""
        n = len(atoms)
        energies = np.asarray(ev.polar_coulomb_energy(), dtype=np.float64)
        rho_re = np.asarray(ev.polar_coulomb_rho_re(), dtype=np.float64)
        rho_im = np.asarray(ev.polar_coulomb_rho_im(), dtype=np.float64)
        scf_final = None
        if hasattr(ev, "polar_scf_final_density"):
            scf_final = np.asarray(ev.polar_scf_final_density(),
                                   dtype=np.float64)
        post_fukui = np.asarray(ev.polar_field_independent_density(),
                                dtype=np.float64)
        return n, energies, rho_re, rho_im, scf_final, post_fukui

    def run_case(tag, atoms, mol, case_kind, fixture_name=None):
        nonlocal mode
        atoms.calc = engine
        atoms.get_potential_energy()          # 完整前向(含 SCF)后再读数
        n, energies, rho_re, rho_im, scf_final, post_fukui = read_engine(atoms)
        if energies.size == 0 or rho_re.size == 0:
            print(f"  {tag}: 引擎访问器为空(polar 未启用?)", flush=True)
            return False
        e_engine = float(energies[0])

        if mode is None:  # auto 解析(需一次完整前向)
            if scf_final is not None and scf_final.size > 0:
                mode = "scf-final"
            else:
                mode = "post-fukui"
            print(f"  [auto] density-mode → {mode}", flush=True)
        if mode == "scf-final" and (scf_final is None or scf_final.size == 0):
            print(f"  {tag}: MODULE6-SCF-FINAL-EMPTY —— ⑤ 未产出末态密度,"
                  "请改用 --density-mode post-fukui(过渡门)", flush=True)
            return False
        dens_flat = (scf_final if mode == "scf-final" else post_fukui)
        dens = np.asarray(dens_flat, dtype=np.float64).reshape(n, 8)
        spin_sum = dens.reshape(n, 2, 4).sum(axis=1)     # charge_density(自旋和)

        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        # 引擎 k 网格(隔离门:③ 已由 module34 门锁定;此处直接用引擎几何 + JSON 表)
        kv = np.asarray(ev.polar_k_vectors(), dtype=np.float64).reshape(-1, 3)
        kn2 = np.asarray(ev.polar_k_norm2(), dtype=np.float64)
        k0m = np.asarray(ev.polar_k0_mask(), dtype=np.float64)
        nk = kv.shape[0]
        if rho_re.size != nk:
            print(f"  {tag}: rho 尺寸 {rho_re.size} != Nk {nk}", flush=True)
            return False
        cosines = np.cos(kv @ positions.T)
        sines = np.sin(kv @ positions.T)
        basis_re, basis_im, _ = pf.rebuild_density_basis(ce, kv, kn2, k0m)
        kfac = np.zeros_like(kn2)
        nz = k0m <= 0.0
        kfac[nz] = 1.0 / kn2[nz]                          # energy.py L141-143
        # 盒体积: 夹具与活体帧同盒(rattle 不改 cell),取该分子夹具记录值
        volume = float(np.load(f"{GEODIR}/{mol}-f0-pregeom.npz",
                               allow_pickle=False)["volumes"][0])
        e_np, rho_np = fast_energy(spin_sum, cosines, sines,
                                   basis_re, basis_im, volume, kfac)
        rho_np_raw = rho_np * (volume / pf.TWO_PI ** 3)   # 引擎 ρ̂ 口径(未乘 (2π)³/V)

        # ① ρ̂ 对照(实部/虚部分别;引擎 ρ̂ = [re, im] 两访问器)
        d_rho_re = float(np.abs(rho_re - rho_np_raw[:, 0]).max())
        d_rho_im = float(np.abs(rho_im - rho_np_raw[:, 1]).max())
        d_rho = max(d_rho_re, d_rho_im)
        rho_ref = max(float(np.abs(rho_np_raw).max()),
                      float(np.abs(rho_re).max()), 1e-30)
        rho_rel = d_rho / rho_ref
        # ② 隔离能量
        d_e = abs(e_engine - e_np)
        e_ref = max(abs(e_np), abs(e_engine), 1e-30)
        rel_e = d_e / e_ref

        # ③ 端到端(夹具)
        e2e, e2e_crit, target = None, None, None
        e_step0 = None
        if case_kind == "fixture":
            geo = np.load(f"{GEODIR}/{fixture_name}-pregeom.npz",
                          allow_pickle=False)
            dk = np.load(f"{GEODIR}/{fixture_name}-kvecs.npz",
                         allow_pickle=False)
            bf_re, bf_im, _ = pf.rebuild_density_basis(
                ce, dk["k_vectors"], dk["k_norm2"], dk["k0_mask"])
            kfac_f = np.zeros_like(dk["k_norm2"])
            nz_f = dk["k0_mask"] <= 0.0
            kfac_f[nz_f] = 1.0 / dk["k_norm2"][nz_f]
            step0 = np.load(f"{STEPDIR}/{mol}-step0.npz",
                            allow_pickle=False)["spin_charge_density"]
            e_step0, _ = fast_energy(step0.sum(axis=1), geo["cosines"],
                                     geo["sines"], bf_re, bf_im,
                                     float(geo["volumes"][0]), kfac_f)
            if mode == "scf-final":
                target = float(index[fixture_name]["electrostatic_energy"])
                e2e = abs(e_engine - target)
                e2e_crit = f"rel<={GATE_REL:.0e} & abs<={GATE_ABS:.0e}"
                e2e_ok = bool(e2e / max(abs(target), 1e-30) <= GATE_REL
                              and e2e <= GATE_ABS)
            else:  # post-fukui 过渡
                target = e_step0
                e2e = abs(e_engine - target)
                e2e_crit = f"abs<={GATE_INTERIM_ABS:.0e}"
                e2e_ok = bool(e2e <= GATE_INTERIM_ABS)
        else:
            e2e_ok = True

        # 诊断: SI 反事实 + TTT 选择器
        si_sum, _ = pf.self_interaction_energy(ce, spin_sum)
        mol_c, slab_c, is_mol, is_slab = pf.pbc_correction_energy(
            ce, spin_sum, positions, volume, np.array([True, True, True]))
        ok = bool(rho_rel <= GATE_REL and rel_e <= GATE_REL
                  and d_e <= GATE_ABS and e2e_ok)
        rows.append({
            "case": tag, "kind": case_kind, "mode": mode,
            "E_engine": e_engine, "E_numpy": e_np, "dE_abs": d_e,
            "dE_rel": rel_e, "rho_rel": rho_rel, "rho_abs": d_rho,
            "target_e2e": target, "e2e_abs": e2e, "e2e_crit": e2e_crit,
            "E_step0_interim": e_step0,
            "si_counterfactual": 0.5 * si_sum,
            "pbc_is_molecule": is_mol, "pbc_is_slab": is_slab,
            "pbc_mol_correction": mol_c, "pbc_slab_correction": slab_c,
            "pass": ok})
        print(f"  {tag:16s} ρ̂rel={rho_rel:.2e}  ΔE={d_e:.3e} (rel {rel_e:.2e})"
              f"  e2e={'-' if e2e is None else f'{e2e:.3e}'}  "
              f"SIc={0.5 * si_sum:+.4f}  TTT(mol/slab)={is_mol}/{is_slab}"
              f"  {'PASS' if ok else 'FAIL'}", flush=True)
        return ok

    print(f"== 源 A: 4 夹具(终态门;以 --density-mode {args.density_mode} 解析) ==",
          flush=True)
    for mol, seed in MOLS:
        name = f"{mol.lower()}-f0"
        base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
        d_fix = np.load(f"{SCFDIR}/{name}.npz", allow_pickle=False)
        assert np.allclose(d_fix["positions"], base.get_positions(),
                           atol=1e-10), f"{name} 盒与夹具不一致"
        atoms = base.copy()
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        run_case(name, atoms, mol.lower(), "fixture", name)

    if args.live > 0:
        print(f"== 源 B: 活体扰动帧(--live {args.live}/分子; 仅 ①②) ==",
              flush=True)
        for mol, seed in MOLS:
            base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
            for k in range(args.live):
                atoms = base.copy()
                atoms.rattle(stdev=0.05, seed=seed + 7 + 4 * k)
                atoms.info["charge"] = 0
                atoms.info["spin"] = 1
                run_case(f"{mol.lower()}-live{k}", atoms, mol.lower(), "live")

    d_e = np.array([r["dE_abs"] for r in rows])
    rel_e = np.array([r["dE_rel"] for r in rows])
    rho_rel = np.array([r["rho_rel"] for r in rows])
    e2e = np.array([r["e2e_abs"] for r in rows
                    if r["e2e_abs"] is not None])
    ok = bool(all(r["pass"] for r in rows))
    verdict = "PASS" if ok else "FAIL"
    print(f"\nGATE {verdict}: configs={len(rows)} (fixtures "
          f"{sum(1 for r in rows if r['kind'] == 'fixture')})  "
          f"ρ̂ rel max={rho_rel.max():.2e}  "
          f"ΔE rel max={rel_e.max():.2e} abs max={d_e.max():.3e} eV  "
          f"e2e max={e2e.max() if e2e.size else float('nan'):.3e} eV  "
          f"(criteria: rel<={GATE_REL:.0e} & abs<={GATE_ABS:.0e} "
          f"eV{'' if mode == 'scf-final' else '; interim abs<=1e-6'})",
          flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = range(len(rows))
    axes[0].bar(x, np.maximum(rho_rel, 1e-18), label="rho rel")
    axes[0].bar(x, np.maximum(rel_e, 1e-18), alpha=0.6, label="E rel")
    axes[0].axhline(GATE_REL, color="r", ls="--", lw=1, label=f"gate={GATE_REL:.0e}")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([r["case"] for r in rows], rotation=60, fontsize=7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("rel error")
    axes[0].set_title(f"Module-6 coulomb energy parity ({len(rows)} configs, "
                      f"mode={mode})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].text(0.5, 0.5,
                 f"Module 6: full-k GTO-Ewald\n"
                 f"energy from SCF-final density ({mode})\n"
                 f"rho rel max {rho_rel.max():.2e}\n"
                 f"dE abs max {d_e.max():.2e} eV\n"
                 f"verdict: {verdict}", ha="center", va="center", fontsize=12)
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    with open(f"{args.out}.json", "w") as fh:
        json.dump({"rows": rows, "mode": mode, "gate_rel": GATE_REL,
                   "gate_abs": GATE_ABS, "verdict": verdict}, fh, indent=1)
    print(f"saved: {args.out}.png , {args.out}.json", flush=True)
    print("MODULE6-GATE-DONE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
