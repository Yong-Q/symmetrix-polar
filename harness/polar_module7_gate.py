#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 模块⑦验收门: local_electron_energy(2026-10-15; 只写不跑, 由集成方执行)。

模块⑦ = OneBodyMLPFieldReadout(mace field_blocks.py L652-752):
  q_in  = charges_induced(SCF 末态) + charges_0(初 Fukui 校正后)   # L729
  q_up  = linear_up_q(q_in)  (1x0e+1x1o+1x0e+1x1o -> 512x0e+512x1o)
  v_up  = linear_up_v(field_feats) (2x0e+2x1o+2x0e+2x1o -> 同上)
  invar = cat(dot_products_q(node_feats,q_up), dot_products_v(node_feats,v_up))
  local_q_e = mlp(invar).squeeze(-1);  le_total = 图内 Σ local_q_e(无缩放)
  (extensions.py L867-885; 能量 = e0+inter_e(scale/shift 后) + le_total)

引擎侧访问器(补丁 docs/module7_patch_final.md §4.5):
  polar_le_q_in / polar_le_q_up / polar_le_v_up / polar_le_dot_q / polar_le_dot_v
  polar_le_invar / polar_le_mlp_out / polar_local_electron_energy(N,) / polar_le_energy

参考:
  A) 夹具 bench/polar_fixtures/le_hooks/{mol}-f{0,1}.npz(8 构型) —— 输入全量 +
     逐级中间量 + local_q_e + le_total + fixture_electron_energy。
     生成: bench/polar_module7_preflight.py; 自身对活体 numpy 重放 worst 9.44e-16,
     对 fixtures/index.json worst 8.88e-15(均 PASS, logs/polar_module7_preflight.log)。
  B) 可选活体(--live K>0): mace_polar + forward hooks 现取同一批逐级量(独立性交叉源)。

判据(参照①②门, 标度感知):
  中间级 q_in/q_up/v_up/dot_q/dot_v/invar/mlp_out: abs ≤ 5e-6 或 rel ≤ 1e-6;
  出口 local_q_e / le_total(vs npz.le_total 及 npz.fixture_electron_energy):
  abs ≤ 1e-6(rel ≤ 1e-6 兜底)。上游①-⑤骨干 f64 漂移 ~1e-8 rel, 预期 ≥10× 裕度。

用法: CUDA_VISIBLE_DEVICES=<gpu> <polar venv python> bench/polar_module7_gate.py [--live 2]
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
MOLS = [("NH3", 770600), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]
STAGES = ("q_in", "q_up", "v_up", "dot_q", "dot_v", "invar")
STAGE_ABS = 5e-6    # 中间级绝对(≤5e-6)
STAGE_REL = 1e-6    # 中间级相对(或)
OUT_ABS = 1e-6      # 出口(local_q_e / le_total)绝对
ACCESSORS = ("polar_le_q_in", "polar_le_q_up", "polar_le_v_up",
             "polar_le_dot_q", "polar_le_dot_v", "polar_le_invar",
             "polar_le_mlp_out", "polar_local_electron_energy", "polar_le_energy")


def residual(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    d = float(np.abs(a - b).max())
    ref = float(np.abs(b).max()) if b.size else 0.0
    return d, ref, d / max(ref, 1e-300)


def build_atoms_from_npz(d):
    from ase import Atoms
    return Atoms(symbols=[str(s) for s in d["symbols"]],
                 positions=np.asarray(d["positions"], dtype=np.float64),
                 cell=np.asarray(d["cell"], dtype=np.float64), pbc=True)


def run_engine(engine, ev, atoms):
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = engine
    atoms.get_potential_energy()
    n = len(atoms)
    return {
        "q_in": np.asarray(ev.polar_le_q_in(), dtype=np.float64).reshape(n, 8),
        "q_up": np.asarray(ev.polar_le_q_up(), dtype=np.float64).reshape(n, -1),
        "v_up": np.asarray(ev.polar_le_v_up(), dtype=np.float64).reshape(n, -1),
        "dot_q": np.asarray(ev.polar_le_dot_q(), dtype=np.float64).reshape(n, -1),
        "dot_v": np.asarray(ev.polar_le_dot_v(), dtype=np.float64).reshape(n, -1),
        "invar": np.asarray(ev.polar_le_invar(), dtype=np.float64).reshape(n, -1),
        "mlp_out": np.asarray(ev.polar_le_mlp_out(), dtype=np.float64).reshape(-1),
        "local_q_e": np.asarray(ev.polar_local_electron_energy(),
                                dtype=np.float64).reshape(n),
        "le_total": float(ev.polar_le_energy()),
    }


def compare(tag, eng, ref, rows):
    """eng/ref = 同键 dict; ref 含 STAGES + mlp_out + local_q_e + le_total_ref。"""
    stages, out = {}, {}
    for st in STAGES:
        d, r, rel = residual(eng[st], ref[st])
        stages[st] = {"abs": d, "ref_absmax": r, "rel": rel}
    d, r, rel = residual(eng["mlp_out"], ref["mlp_out"])
    stages["mlp_out"] = {"abs": d, "ref_absmax": r, "rel": rel}
    d, _, _ = residual(eng["local_q_e"], ref["local_q_e"])
    out["local_q_e_abs"] = d
    d, _, _ = residual([eng["le_total"]], [ref["le_total_ref"]])
    out["le_total_abs"] = d
    ok = all(v["abs"] <= STAGE_ABS or v["rel"] <= STAGE_REL
             for v in stages.values())
    ok = ok and all(v <= OUT_ABS for v in out.values())
    row = {"case": tag, "stages": stages, "out": out,
           "stage_max_abs": max(v["abs"] for v in stages.values()),
           "stage_worst": max(stages.items(), key=lambda kv: kv[1]["abs"])[0],
           "verdict": "PASS" if ok else "FAIL"}
    rows.append(row)
    print(f"  {tag:14s} stage max={row['stage_max_abs']:.3e} "
          f"(worst {row['stage_worst']})  |Δlocal_q_e|={out['local_q_e_abs']:.3e}  "
          f"|Δle_total|={out['le_total_abs']:.3e}  -> {row['verdict']}", flush=True)
    return ok


def live_capture(model, atoms, ref_calc, caps, handles):
    """活体源: preflight 同构 hooks(le kwargs + 4 子模块输出), 返回逐级参考 dict。"""
    if not handles:
        le = model.local_electron_energy

        def hook_kwargs(module, hargs, kwargs, output):
            caps["le_kw"] = {k: (v.detach().cpu().numpy().copy()
                                 if hasattr(v, "detach") else None)
                             for k, v in kwargs.items()}
            out = output.detach().cpu().numpy().copy()
            caps["le_out"] = out
            return None

        def hook_out(name):
            def fn(module, hargs, output):
                caps[name] = output.detach().cpu().numpy().copy()
                return None
            return fn

        handles.append(le.register_forward_hook(hook_kwargs, with_kwargs=True))
        for sub in ("linear_up_q", "linear_up_v", "dot_products_q",
                    "dot_products_v"):
            handles.append(getattr(le, sub).register_forward_hook(hook_out(sub)))
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = ref_calc
    atoms.get_potential_energy()
    kw = caps["le_kw"]
    ref = {
        "q_in": kw["charges_induced"] + kw["charges_0"],
        "q_up": caps["linear_up_q"],
        "v_up": caps["linear_up_v"],
        "dot_q": caps["dot_products_q"],
        "dot_v": caps["dot_products_v"],
        "invar": np.concatenate([caps["dot_products_q"],
                                 caps["dot_products_v"]], axis=-1),
        "mlp_out": caps["le_out"],
        "local_q_e": caps["le_out"],
        "le_total_ref": float(caps["le_out"].sum()),
    }
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=int, default=0,
                    help="每分子额外活体重算帧数(默认 0=只用夹具)")
    ap.add_argument("--out", default=f"{ROOT}/bench/polar_module7_gate")
    args = ap.parse_args()

    from symmetrix.calculator import Symmetrix

    engine = Symmetrix(J, dtype="float64")
    ev = engine.evaluator
    missing = [n for n in ACCESSORS if not hasattr(ev, n)]
    if missing:
        print(f"MODULE7-ACCESSORS-MISSING: {missing} 引擎未含模块⑦访问器"
              f"(先应用 docs/module7_patch_final.md 并重建)", flush=True)
        return 2

    rows = []

    # ---------------- 源 A: 夹具(8 构型) ----------------
    print("== 源 A: 夹具 le_hooks/*.npz(8 构型) ==", flush=True)
    for mol, _seed in MOLS:
        for k in range(2):
            name = f"{mol.lower()}-f{k}"
            d = np.load(f"{HOOKS}/{name}.npz", allow_pickle=False)
            atoms = build_atoms_from_npz(d)
            eng = run_engine(engine, ev, atoms)
            ref = {st: np.asarray(d[st], dtype=np.float64) for st in STAGES}
            ref["mlp_out"] = np.asarray(d["mlp_h9"], dtype=np.float64).reshape(-1)
            ref["local_q_e"] = np.asarray(d["local_q_e"], dtype=np.float64)
            ref["le_total_ref"] = float(d["fixture_electron_energy"])
            compare(name, eng, ref, rows)

    # ---------------- 源 B: 活体重算(可选) ----------------
    handles = []
    if args.live > 0:
        from mace.calculators import mace_polar
        ref_calc = mace_polar(model=POLAR, device="cuda", default_dtype="float64")
        model = ref_calc.models[0]
        caps = {}
        print(f"== 源 B: 活体重算(--live {args.live}/分子) ==", flush=True)
        for mol, seed in MOLS:
            base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
            for k in range(args.live):
                atoms = base.copy()
                atoms.rattle(stdev=0.05, seed=seed + 7 + 4 * k)
                ref = live_capture(model, atoms, ref_calc, caps, handles)
                eng = run_engine(engine, ev, atoms)
                compare(f"{mol.lower()}-live{k}", eng, ref, rows)
        for h in handles:
            h.remove()

    stage_max = max(r["stage_max_abs"] for r in rows)
    out_max = max(max(r["out"].values()) for r in rows)
    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    ok = n_pass == len(rows)
    verdict = "PASS" if ok else "FAIL"
    print(f"\nGATE {verdict}: configs={len(rows)} ({n_pass} PASS)  "
          f"stage max={stage_max:.3e} (判据 abs≤{STAGE_ABS:.0e} 或 rel≤{STAGE_REL:.0e})  "
          f"out max={out_max:.3e} (abs≤{OUT_ABS:.0e})", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(rows))
    axes[0].bar(x, [max(r["stage_max_abs"], 1e-18) for r in rows],
                label="stage max|Δ|")
    axes[0].bar(x, [max(max(r["out"].values()), 1e-18) for r in rows],
                alpha=0.6, label="out max|Δ|")
    axes[0].axhline(STAGE_ABS, color="r", ls="--", lw=1,
                    label=f"gate={STAGE_ABS:.0e}")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([r["case"] for r in rows], rotation=60, fontsize=7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("max|Δ|")
    axes[0].set_title(f"Module-7 local_electron_energy parity ({len(rows)} cfg)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].text(0.5, 0.5,
                 "Module 7:\nOneBodyMLPFieldReadout\n(q_in→linear_up→uvu 内积→"
                 "concat→RadialMLP)\nvs fixture le_hooks / live hooks\n"
                 f"verdict: {verdict}", ha="center", va="center", fontsize=12)
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    with open(f"{args.out}.json", "w") as fh:
        json.dump({"rows": rows, "gate": {"stage_abs": STAGE_ABS,
                                          "stage_rel": STAGE_REL,
                                          "out_abs": OUT_ABS},
                   "verdict": verdict}, fh, indent=1)
    print(f"saved: {args.out}.png , {args.out}.json", flush=True)
    print("MODULE7-GATE-DONE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
