#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 模块⑧验收门: 力/全链反向 (2026-10-15; 只写不跑, 由集成方执行)。

配套提案: docs/module8_patch_final.md (§9)。
模块⑧ = 极性链全链反向: ⑥/⑦ 能量种子 → SCF 逐步(replay→Fukui 校正逆→更新块逆→描述符逆)
        → 初始化 Fukui 逆 → lr_src/⑤ mixer 逆 → ⑧-A → polar_node_forces(3N, −∂E/∂r)。

判据(两阶段; 由 ev.polar_reverse_terms_last() 自动分档):
  pre  (mask==0: 引擎尚无极化能量项消费方 ⇒ 零种子早退):
    - 引擎自身 FD 自洽(FD_PRE=1e-5 eV/Å; P2 自洽实测 3.6e-7,作为回归上界);
    - polar_node_forces ≡ 0(全节点;且与骨架力的 P2 自洽不变)。
  post (mask==3: ⑥/⑦ 已落码):
    - FD 自洽(FD_POST=1e-3 eV/Å, 抽样原子, h=1e-3 Å, f64 引擎);
    - python 总力 parity(FORCE_GATE=1e-3 eV/Å; 先看分布 mean/p50/p90/p99/max,
      P2 遗留基线 mean 1.24 / max 4.15 eV/Å 打印对照);
    - Σ_n F_n ≈ 0(平移不变)。

用法(参照 module5/7 门; descriptor 环境 + unset PYTHONPATH + nvidia-smi shim):
  cd mlip && env -u PYTHONPATH \
    SYMMETRIX_BACKEND_DESCRIPTOR_PATH=$PWD/models/mace-polar/backend_cuda12_sm80.json \
    SYMMETRIX_BACKEND=cuda12-sm80 \
    PATH=$PWD/bench/nvidia_smi_shim:$PATH \
    LD_LIBRARY_PATH=/opt/matos-tools/cuda/12.9/lib64:/opt/matos-tools/conda/envs/matos-compute/lib \
    CUDA_VISIBLE_DEVICES=<gpu> /workspace/caches/conda/envs/polar/bin/python \
    bench/polar_module8_gate.py [--fd-atoms 4] [--h 1e-3] [--live 1] [--stage auto]
退出码: 0=PASS, 1=FAIL, 2=访问器缺失/装载失败。
注: 解释器退出阶段 cudaStreamSynchronize/driver shutting down 为已知无害噪音。
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

J = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
POLAR = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"
FIX = f"{ROOT}/bench/polar_fixtures"
MOLS = [("NH3", 770600), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]
FD_PRE = 1e-5        # eV/Å(pre 阶段: 无极性项回归; P2 自洽 3.6e-7)
FD_POST = 1e-3       # eV/Å(post 阶段: 全量解析力自洽)
FORCE_GATE = 1e-3    # eV/Å(post 阶段: 与 python 总力 parity)
P2_LEGACY = {"mean": 1.24, "max": 4.15}   # 缺 P3 静电层时的历史力差(eV/Å),留档对照
ACCESSORS = ("polar_node_forces", "polar_reverse_terms_last",
             "polar_blocks_enabled")


def dist(a, b):
    d = np.abs(np.asarray(a, dtype=np.float64).reshape(-1)
               - np.asarray(b, dtype=np.float64).reshape(-1))
    return {"mean": float(d.mean()), "p50": float(np.percentile(d, 50)),
            "p90": float(np.percentile(d, 90)),
            "p99": float(np.percentile(d, 99)), "max": float(d.max()),
            "n": int(d.size)}


def build_atoms_from_npz(d):
    from ase import Atoms
    return Atoms(symbols=[str(s) for s in d["symbols"]],
                 positions=np.asarray(d["positions"], dtype=np.float64),
                 cell=np.asarray(d["cell"], dtype=np.float64), pbc=True)


def engine_energy_forces(engine, ev, atoms):
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = engine
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64).reshape(-1, 3)
    n = len(atoms)
    polar = np.asarray(ev.polar_node_forces(), dtype=np.float64).reshape(-1, 3)
    if polar.shape[0] != n:
        raise RuntimeError(f"polar_node_forces shape {polar.shape} != ({n},3)")
    terms = int(ev.polar_reverse_terms_last())
    return energy, forces, polar, terms


def fd_check(engine, atoms, atom_ids, h):
    """中央差分: F_x ≈ −(E(+h)−E(−h))/(2h); 返回 {(ai,comp): fd}。"""
    fd = {}
    for ai in atom_ids:
        for comp in range(3):
            vals = []
            for sign in (+1.0, -1.0):
                a = atoms.copy()
                p = a.get_positions()
                p[ai, comp] += sign * h
                a.set_positions(p)
                a.calc = engine
                vals.append(float(a.get_potential_energy()))
            fd[(ai, comp)] = -(vals[0] - vals[1]) / (2.0 * h)
    return fd


def run_frame(engine, ev, atoms, atom_ids, h, py_calc, want_python):
    energy, forces, polar, terms = engine_energy_forces(engine, ev, atoms)
    fd = fd_check(engine, atoms, atom_ids, h)
    keys = list(fd.keys())
    fd_vals = np.array([fd[k] for k in keys], dtype=np.float64)
    # [修正 2026-10-16] 引擎/ASE 力符号 = 物理力(与 python 同号, 已证): FD≈+F;
    # 原 an_vals=-forces 为符号错误(测出 2|F| 伪影, 详见 le 调查日志)。
    an_vals = np.array([forces[k[0], k[1]] for k in keys], dtype=np.float64)
    out = {"energy": energy, "terms": terms, "fd": dist(fd_vals, an_vals),
           "polar_max": float(np.abs(polar).max()),
           "force_balance": float(np.abs(forces.sum(axis=0)).max())}
    if want_python and py_calc is not None:
        a = atoms.copy()
        a.info["charge"] = 0
        a.info["spin"] = 1
        a.calc = py_calc
        py_forces = np.asarray(a.get_forces(), dtype=np.float64).reshape(-1, 3)
        out["parity"] = dist(forces, py_forces)
        out["python_force_max"] = float(np.abs(py_forces).max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd-atoms", type=int, default=4, help="每帧 FD 抽样原子数")
    ap.add_argument("--h", type=float, default=1e-3, help="FD 步长 Å")
    ap.add_argument("--live", type=int, default=1, help="每分子活体帧数(0=只用夹具)")
    ap.add_argument("--stage", choices=("auto", "pre", "post"), default="auto")
    ap.add_argument("--out", default=f"{ROOT}/bench/polar_module8_gate")
    args = ap.parse_args()

    from symmetrix.calculator import Symmetrix

    engine = Symmetrix(J, dtype="float64")
    ev = engine.evaluator
    missing = [n for n in ACCESSORS if not hasattr(ev, n)]
    if missing:
        print(f"MODULE8-ACCESSORS-MISSING: {missing} 引擎未含模块⑧访问器"
              f"(先应用 docs/module8_patch_final.md 并重建)", flush=True)
        return 2
    if not bool(ev.polar_blocks_enabled()):
        print("MODULE8: polar_blocks 未启用(JSON/装载问题)", flush=True)
        return 2

    py_calc = None
    if args.stage != "pre":
        from mace.calculators import mace_polar
        py_calc = mace_polar(model=POLAR, device="cuda", default_dtype="float64")

    rows = []
    stage_seen = None

    # ---------------- 源 A: 夹具(4 分子 × 2 帧) ----------------
    print("== 源 A: 夹具 polar_fixtures/*.npz(8 构型) ==", flush=True)
    for mol, _seed in MOLS:
        d0 = np.load(f"{FIX}/{mol.lower()}-f0.npz", allow_pickle=False)
        n = len(d0["symbols"])
        atom_ids = list(range(0, n, max(1, n // args.fd_atoms)))[:args.fd_atoms]
        for k in (0, 1):
            d = np.load(f"{FIX}/{mol.lower()}-f{k}.npz", allow_pickle=False)
            atoms = build_atoms_from_npz(d)
            row = {"case": f"{mol.lower()}-f{k}"}
            row.update(run_frame(engine, ev, atoms, atom_ids, args.h,
                                 py_calc, args.stage != "pre"))
            stage_seen = row["terms"] if stage_seen is None else stage_seen
            rows.append(row)
            par = row.get("parity", {}).get("max", float("nan"))
            print(f"  {row['case']:10s} terms={row['terms']} "
                  f"FD max={row['fd']['max']:.3e}  parity max={par:.3e}  "
                  f"|polar|max={row['polar_max']:.3e}  "
                  f"ΣF={row['force_balance']:.3e}", flush=True)

    # ---------------- 源 B: 活体帧 ----------------
    if args.live > 0:
        print(f"== 源 B: 活体重建(--live {args.live}/分子) ==", flush=True)
        for mol, seed in MOLS:
            base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
            for k in range(args.live):
                atoms = base.copy()
                atoms.rattle(stdev=0.05, seed=seed + 7 + 4 * k)
                n = len(atoms)
                atom_ids = list(range(0, n,
                                      max(1, n // args.fd_atoms)))[:args.fd_atoms]
                row = {"case": f"{mol.lower()}-live{k}"}
                row.update(run_frame(engine, ev, atoms, atom_ids, args.h,
                                     py_calc, args.stage != "pre"))
                rows.append(row)
                par = row.get("parity", {}).get("max", float("nan"))
                print(f"  {row['case']:12s} terms={row['terms']} "
                      f"FD max={row['fd']['max']:.3e}  parity max={par:.3e}",
                      flush=True)

    stage = args.stage
    if stage == "auto":
        stage = "pre" if stage_seen == 0 else "post"
    fd_gate = FD_PRE if stage == "pre" else FD_POST
    ok_fd = all(r["fd"]["max"] <= fd_gate for r in rows)
    ok_zero = (all(r["polar_max"] == 0.0 for r in rows)
               if stage == "pre" else True)
    ok_parity = True
    if stage == "post":
        ok_parity = all(("parity" in r) and (r["parity"]["max"] <= FORCE_GATE)
                        for r in rows)
    ok_balance = all(r["force_balance"] <= 1e-8 * max(
        1.0, r.get("python_force_max", 1.0)) for r in rows)
    ok = bool(ok_fd and ok_zero and ok_parity and ok_balance)
    verdict = "PASS" if ok else "FAIL"

    fd_max = max(r["fd"]["max"] for r in rows)
    par_max = max((r.get("parity", {}).get("max", 0.0)) for r in rows)
    print(f"\nGATE {verdict}: stage={stage} configs={len(rows)}  "
          f"FD max={fd_max:.3e} (gate {fd_gate:.0e})  "
          f"parity max={par_max:.3e} (gate {FORCE_GATE:.0e}; "
          f"P2 legacy mean/max={P2_LEGACY['mean']}/{P2_LEGACY['max']})",
          flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(rows))
    axes[0].bar(x, np.maximum([r["fd"]["max"] for r in rows], 1e-18))
    axes[0].axhline(fd_gate, color="r", ls="--", lw=1,
                    label=f"FD gate={fd_gate:.0e}")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Engine FD self-consistency ({stage})")
    axes[1].bar(x, np.maximum([r.get("parity", {}).get("max", 0.0)
                               for r in rows], 1e-18), label="|ΔF| max")
    axes[1].axhline(FORCE_GATE, color="r", ls="--", lw=1,
                    label=f"parity gate={FORCE_GATE:.0e}")
    axes[1].axhline(P2_LEGACY["max"], color="gray", ls=":", lw=1,
                    label="P2 legacy max=4.15")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].set_title("python total-force parity (max per frame)")
    axes[2].text(0.5, 0.5,
                 f"Module 8 (forces / full reverse)\nstage: {stage}\n"
                 f"terms mask: {stage_seen}\nverdict: {verdict}",
                 ha="center", va="center", fontsize=12)
    axes[2].axis("off")
    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels([r["case"] for r in rows], rotation=60, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    with open(f"{args.out}.json", "w") as fh:
        json.dump({"rows": rows, "stage": stage,
                   "gates": {"fd": fd_gate, "parity": FORCE_GATE,
                             "p2_legacy": P2_LEGACY},
                   "verdict": verdict}, fh, indent=1)
    print(f"saved: {args.out}.png , {args.out}.json", flush=True)
    print("MODULE8-GATE-DONE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
