#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 模块⑤验收门: SCF 密度更新链 (charges_maps / 更新块) + stage 级链 (2026-10-15)。

模块⑤ = `field_dependent_charges_maps[i]`(AgnosticEmbeddedOneBodyVariableUpdate) 的
前向链 + extensions.py L833-864 的密度更新/再归一/l=0 再校正, num_recursion_steps=2。

出口(引擎访问器, 命名已定):
  polar_scf_step_densities (S=2,N,8)  每个 SCF 步【更新后】的密度;
                                      step index 0/1 = "第0/1步更新后" ↔ scf_steps step1/step2
  polar_scf_final_density  (N,8)      末态 (= 最后一步更新后)
  stage 类(均为【最后一次迭代】值, ↔ hooks nh3-step1):
    polar_features_mixed(N,2048) polar_pe_out(N,2048) polar_dp_out(N,512)
    polar_se_out(N,512) polar_nl_out(N,512) polar_tp_out(N,2048)
    polar_ro_l1_out(N,160) polar_gate1_out(N,128) polar_ro_lmid_out(N,160)
    polar_gate2_out(N,128) polar_update_out(N,10)

参考夹具:
  A) 密度演化: bench/polar_fixtures/scf_steps/{mol}-step{0,1,2}.npz
     (0=初校正后=SCF 起点, 1/2=第0/1步更新后; 生成器 bench/polar_module5_refsteps.py,
      规则逐行锚定 extensions.py L833-864; 其自身对 scf 夹具 final 逐位 PASS)
     + bench/polar_fixtures/scf/{mol}-f0.npz (positions 几何自证 / lr_src / charges_step_k)
  B) stage: bench/polar_fixtures/charges_hooks/nh3-step1.npz (42 张量, 块1=最后迭代)
  C) 参考侧自证(本脚本内, --ref-selfcheck): mace_polar 跑一次 NH3 全前向, 钩子捕获
     两块 local_charges(更新前密度)/out(增量), numpy 重放更新规则 → 证实
     "引擎 step index i ↔ scf_steps step{i+1}" 的配对关系(见 §2(b) 设计)。

判据:
  - 密度演化: abs ≤ 5e-6 且 rel ≤ 1e-6 (rel = max|Δ| / max|ref|);
  - stage:    标度感知, rel ≤ 1e-6 为主判据(abs 打印留档——各 stage 量级不同);
  - 配对自证: 逐行取 rel 最小者, 期望 [step0, step1, step2, step2]。

用法(参照 module34 门; descriptor 环境 + unset PYTHONPATH + nvidia-smi shim):
  cd mlip && env -u PYTHONPATH \
    SYMMETRIX_BACKEND_DESCRIPTOR_PATH=$PWD/models/mace-polar/backend_cuda12_sm80.json \
    SYMMETRIX_BACKEND=cuda12-sm80 \
    PATH=$PWD/bench/nvidia_smi_shim:$PATH \
    LD_LIBRARY_PATH=/opt/matos-tools/cuda/12.9/lib64:/opt/matos-tools/conda/envs/matos-compute/lib \
    CUDA_VISIBLE_DEVICES=<gpu> /workspace/caches/conda/envs/polar/bin/python \
    bench/polar_module5_gate.py
  (--ref-selfcheck 0 跳过参考侧自证; --ref-device cuda 指定参考设备)
退出码: 0=PASS, 1=FAIL, 2=访问器缺失/引擎装载失败(优雅退出)。
注: 解释器退出阶段 cudaStreamSynchronize/driver shutting down 为已知无害噪音。
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/bench")
import mace_md as mm  # noqa: E402
from polar_module5_refsteps import scf_density_update  # noqa: E402  (L833-864 参考实现)

J = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
POLAR = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"
SCFDIR = f"{ROOT}/bench/polar_fixtures/scf"
STEPDIR = f"{ROOT}/bench/polar_fixtures/scf_steps"
HOOKDIR = f"{ROOT}/bench/polar_fixtures/charges_hooks"
MOLS = [("NH3", 770600), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]
CHARGE, SPIN = 0.0, 1.0                     # 夹具口径: atoms.info charge=0 / spin=1

DENS_ABS_GATE = 5e-6                        # 密度演化(逐位级口径)
DENS_REL_GATE = 1e-6
STAGE_REL_GATE = 1e-6                       # stage 标度感知(rel 主判据)

STEP_ACCESSORS = ("polar_scf_step_densities", "polar_scf_final_density")
STAGE_MAP = (
    ("polar_features_mixed", "in_node_feats"),   # = features_mixed
    ("polar_pe_out", "pe_out"),
    ("polar_dp_out", "dp_out"),
    ("polar_se_out", "se_out"),
    ("polar_nl_out", "nl_out"),
    ("polar_tp_out", "tp_out"),
    ("polar_ro_l1_out", "ro_l1_out"),
    ("polar_gate1_out", "gate_out"),
    ("polar_ro_lmid_out", "ro_lmid_out"),
    ("polar_gate2_out", "gate_out2"),
    ("polar_update_out", "out"),
)
# 期望配对(2(b) 设计): 引擎 step0/step1/final → scf_steps step1/step2/step2
EXPECTED_PAIRING = ["step1", "step2", "step2"]


def max_abs(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64)
                        - np.asarray(b, dtype=np.float64)).max())


def rel_err(dmax, ref):
    r = float(np.abs(np.asarray(ref, dtype=np.float64)).max())
    if r > 0.0:
        return dmax / r
    return 0.0 if dmax == 0.0 else float("inf")


# ---------------------------------------------------------------------------
# 参考侧自证: mace_polar 全前向 + 钩子捕获 → 重放更新规则, 证实 step index 配对
# ---------------------------------------------------------------------------
def reference_selfcheck(device="cuda"):
    """返回证据 dict(全部数值为 f64 实测量)。"""
    import torch
    from mace.calculators import mace_polar

    mol, seed = "NH3", 770600
    fix = np.load(f"{SCFDIR}/nh3-f0.npz", allow_pickle=False)
    steps = {k: np.asarray(np.load(f"{STEPDIR}/nh3-step{k}.npz",
                                   allow_pickle=False)["spin_charge_density_flat"],
                           dtype=np.float64) for k in (0, 1, 2)}

    base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
    geom_ok = bool(np.allclose(base.get_positions(), fix["positions"], atol=1e-10))

    calc = mace_polar(model=POLAR, device=device, default_dtype="float64")
    model = calc.models[0]
    n_steps = int(model.num_recursion_steps)
    caps = {"blk_in": {}, "blk_out": {}, "lr": {}}
    handles = []

    def pre_hook(idx):
        def fn(module, args, kwargs):
            lc = kwargs.get("local_charges")
            if lc is None and len(args) >= 6:
                lc = args[5]
            assert lc is not None, f"block{idx}: local_charges not captured"
            caps["blk_in"][idx] = lc.detach().cpu().numpy().astype(np.float64).copy()
        return fn

    def post_hook(store, idx):
        def fn(module, inp, out):
            caps[store][idx] = out.detach().cpu().numpy().astype(np.float64).copy()
        return fn

    for i, m in enumerate(model.field_dependent_charges_maps):
        handles.append(m.register_forward_pre_hook(pre_hook(i), with_kwargs=True))
        handles.append(m.register_forward_hook(post_hook("blk_out", i)))
    for i, m in enumerate(model.lr_source_maps):
        handles.append(m.register_forward_hook(post_hook("lr", i)))

    atoms = base.copy()
    atoms.info["charge"] = CHARGE
    atoms.info["spin"] = SPIN
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    live_final = np.asarray(calc.results["spin_charge_density"],
                            dtype=np.float64).reshape(-1, 8)
    for h in handles:
        h.remove()

    n = len(atoms)
    n_blk = len(model.field_dependent_charges_maps)
    # ---- numpy 重放(参考实现 = polar_module5_refsteps.scf_density_update) ----
    den0 = steps[0].reshape(n, 2, 4)
    den1, incr0, _ = scf_density_update(den0, caps["blk_out"][0])
    den2, incr1, _ = scf_density_update(den1, caps["blk_out"][1])

    lr_sum = caps["lr"][0].reshape(n, 8) + caps["lr"][1].reshape(n, 8)
    lr_fix = (np.asarray(fix["lr_src_0"], dtype=np.float64)[:, 0, :]
              + np.asarray(fix["lr_src_1"], dtype=np.float64)[:, 0, :])

    chk = {
        "geometry_rebuild_vs_scf_fixture_positions": geom_ok,
        "num_recursion_steps": n_steps,
        "hook_capture_counts_blk_in_out": [len(caps["blk_in"]), len(caps["blk_out"])],
        "hook_capture_counts_lr": len(caps["lr"]),
        # 状态1: 原始 lr_src 累积(未校正) ↔ scf 夹具 lr_src 和
        "state1_raw_lr_sum": max_abs(lr_sum, lr_fix),
        # 状态2: 块0 输入 local_charges = 初校正(SCF 起点) ↔ scf_steps step0
        "state2_block0_in_vs_step0": max_abs(caps["blk_in"][0], steps[0]),
        # 状态3: 块1 输入 local_charges = 第0步更新后 ↔ scf_steps step1
        "state3_block1_in_vs_step1": max_abs(caps["blk_in"][1], steps[1]),
        # 状态3': 由 step0 + 实捕 out0 重放 → scf_steps step1
        "state3_replay_update0_vs_step1": max_abs(den1.reshape(-1, 8), steps[1]),
        # 状态4: 由 step1 + 实捕 out1 重放 → scf_steps step2
        "state4_replay_update1_vs_step2": max_abs(den2.reshape(-1, 8), steps[2]),
        # 状态5: 模型实跑末态(spin_charge_density) ↔ scf_steps step2
        "state5_live_final_vs_step2": max_abs(live_final, steps[2]),
        # 状态4 ≡ 状态5(重放末态 ≡ 实跑末态)
        "state4_replay_vs_state5_live": max_abs(den2, live_final.reshape(n, 2, 4)),
        # 实捕块输出 ↔ scf 夹具 charges_step_k(增量锚)
        "blk_out0_vs_fixture_charges_step_0": max_abs(
            caps["blk_out"][0], np.asarray(fix["charges_step_0"], dtype=np.float64)),
        "blk_out1_vs_fixture_charges_step_1": max_abs(
            caps["blk_out"][1], np.asarray(fix["charges_step_1"], dtype=np.float64)),
    }
    # 引擎 5 个密度态 → scf_steps 配对账本(引擎待产出, 参考侧证据)
    ledger = [
        ("polar_spin_charge_density (原始 lr_src 累积)",
         "scf/{mol}-f0.npz: lr_src_0+lr_src_1", chk["state1_raw_lr_sum"]),
        ("polar_field_independent_density (Fukui 初校正后 = SCF 起点)",
         "scf_steps step0 (= 块0 local_charges 实捕)", chk["state2_block0_in_vs_step0"]),
        ("polar_scf_step_densities[0] (第0步更新后)",
         "scf_steps step1 (= 块1 local_charges 实捕)", chk["state3_block1_in_vs_step1"]),
        ("polar_scf_step_densities[1] (第1步更新后)",
         "scf_steps step2", chk["state4_replay_update1_vs_step2"]),
        ("polar_scf_final_density (末态 ≡ 最后一步更新后)",
         "scf_steps step2 (= 实跑 spin_charge_density)", chk["state5_live_final_vs_step2"]),
    ]
    return {"device": device, "energy_eV": energy, "checks": chk,
            "ledger": [{"engine_state": a, "reference": b, "max_abs": c}
                       for a, b, c in ledger],
            "pairing_rule": "engine polar_scf_step_densities[i] ↔ scf_steps step{i+1}; "
                            "polar_scf_final_density ↔ scf_steps step2 (S=2)"}


# ---------------------------------------------------------------------------
# 主门
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-selfcheck", type=int, default=1,
                    help="1=跑 mace 参考全前向自证(证实 step index 配对)")
    ap.add_argument("--ref-device", default="cuda")
    ap.add_argument("--out", default=f"{ROOT}/bench/polar_module5_gate")
    args = ap.parse_args()

    payload = {"module": 5, "date": "2026-10-15",
               "gates": {"dens_abs": DENS_ABS_GATE, "dens_rel": DENS_REL_GATE,
                         "stage_rel": STAGE_REL_GATE},
               "expected_pairing": EXPECTED_PAIRING}

    # ---- (5) 参考侧自证(先跑; 独立于引擎) ----
    if args.ref_selfcheck:
        print(f"== 参考侧自证: mace_polar NH3 全前向 ({args.ref_device}) ==", flush=True)
        try:
            ref = reference_selfcheck(args.ref_device)
            payload["reference_selfcheck"] = ref
            c = ref["checks"]
            print(f"  [ref] num_recursion_steps={c['num_recursion_steps']}"
                  f"  geometry_rebuild_vs_fixture={c['geometry_rebuild_vs_scf_fixture_positions']}"
                  f"  E={ref['energy_eV']:.6f} eV", flush=True)
            print(f"  [ref] 状态1 原始 lr_src 和      ↔ scf 夹具: max|Δ|={c['state1_raw_lr_sum']:.3e}", flush=True)
            print(f"  [ref] 状态2 块0 输入(更新前)    ↔ scf_steps step0: {c['state2_block0_in_vs_step0']:.3e}", flush=True)
            print(f"  [ref] 状态3 块1 输入(第0步后)   ↔ scf_steps step1: {c['state3_block1_in_vs_step1']:.3e}", flush=True)
            print(f"  [ref] 状态3' 重放(step0+out0)   ↔ scf_steps step1: {c['state3_replay_update0_vs_step1']:.3e}", flush=True)
            print(f"  [ref] 状态4 重放(step1+out1)    ↔ scf_steps step2: {c['state4_replay_update1_vs_step2']:.3e}", flush=True)
            print(f"  [ref] 状态5 实跑末态            ↔ scf_steps step2: {c['state5_live_final_vs_step2']:.3e}", flush=True)
            print(f"  [ref] 状态4 重放末态            ↔ 状态5 实跑末态: {c['state4_replay_vs_state5_live']:.3e}", flush=True)
            print(f"  [ref] 实捕 out0/out1           ↔ scf charges_step_0/1: "
                  f"{c['blk_out0_vs_fixture_charges_step_0']:.3e} / "
                  f"{c['blk_out1_vs_fixture_charges_step_1']:.3e}", flush=True)
            print(f"  [ref] 配对规则(数据自证): {ref['pairing_rule']}", flush=True)
            for e in ref["ledger"]:
                print(f"  [ref]   引擎态 {e['engine_state']:58s} ↔ {e['reference']}"
                      f"  (max|Δ|={e['max_abs']:.2e})", flush=True)
        except Exception:  # noqa: BLE001
            payload["reference_selfcheck"] = {"error": traceback.format_exc()[-2000:]}
            print("  [ref] SELFCHECK-FAILED (见 JSON error 段)", flush=True)
    else:
        print("== 参考侧自证: 跳过 (--ref-selfcheck 0) ==", flush=True)

    # ---- 引擎装载 + 访问器门 ----
    try:
        from symmetrix.calculator import Symmetrix
        engine = Symmetrix(J, dtype="float64")
        ev = engine.evaluator
    except Exception as exc:  # noqa: BLE001
        print(f"M5-ENGINE-LOAD-FAILED: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        payload["verdict"] = "ENGINE-LOAD-FAILED"
        with open(f"{args.out}.json", "w") as fh:
            json.dump(payload, fh, indent=1, default=float)
        return 2

    missing = [n for n in STEP_ACCESSORS + tuple(a for a, _ in STAGE_MAP)
               if not hasattr(ev, n)]
    if missing:
        print(f"M5-ACCESSORS-MISSING: 引擎缺 {len(missing)} 个访问器: "
              f"{', '.join(missing)}", flush=True)
        print("  (引擎侧实现后重跑本门; 参考侧自证证据见上/JSON)", flush=True)
        payload["verdict"] = "ACCESSORS-MISSING"
        payload["missing_accessors"] = missing
        with open(f"{args.out}.json", "w") as fh:
            json.dump(payload, fh, indent=1, default=float)
        return 2

    # ---- (1)(2) 密度演化门: 4 分子盒 ----
    rows, align_rows = [], []
    for mol, seed in MOLS:
        name = f"{mol.lower()}-f0"
        base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
        fix = np.load(f"{SCFDIR}/{name}.npz", allow_pickle=False)
        geom_ok = bool(np.allclose(base.get_positions(), fix["positions"], atol=1e-10))
        assert geom_ok, f"{name}: rebuilt geometry diverges from fixture"

        atoms = base.copy()
        atoms.info["charge"] = CHARGE
        atoms.info["spin"] = SPIN
        atoms.calc = engine
        atoms.get_potential_energy()
        n = len(atoms)

        sd = np.asarray(ev.polar_scf_step_densities(), dtype=np.float64)
        n_s = sd.size // (n * 8)
        sd = sd.reshape(n_s, n, 8)
        assert n_s == 2, f"{name}: polar_scf_step_densities S={n_s} (expected 2)"
        final = np.asarray(ev.polar_scf_final_density(), dtype=np.float64).reshape(n, 8)
        anchor = None
        if hasattr(ev, "polar_field_independent_density"):
            anchor = np.asarray(ev.polar_field_independent_density(),
                                dtype=np.float64).reshape(n, 8)

        stem = mol.lower()          # scf_steps/{stem}-step{k}.npz (无 -f0 后缀)
        fixsteps = {k: np.asarray(np.load(f"{STEPDIR}/{stem}-step{k}.npz",
                                          allow_pickle=False)["spin_charge_density_flat"],
                                  dtype=np.float64) for k in (0, 1, 2)}

        eng_states = [("polar_field_independent_density (锚: SCF 起点)", anchor),
                      ("polar_scf_step_densities[0]", sd[0]),
                      ("polar_scf_step_densities[1]", sd[1]),
                      ("polar_scf_final_density", final)]
        # 对齐矩阵(rows=engine, cols=scf_steps): rel = max|Δ|/max|ref|
        print(f"  [align] {name} rel 矩阵 (rows=engine states, cols=scf_steps step0/1/2):",
              flush=True)
        best = []
        for label, arr in eng_states:
            if arr is None:
                print(f"      {label:46s}: (访问器缺失)", flush=True)
                best.append(None)
                continue
            rels = [rel_err(max_abs(arr, fixsteps[k]), fixsteps[k]) for k in (0, 1, 2)]
            best.append(int(np.argmin(rels)))
            print(f"      {label:46s}: "
                  + "  ".join(f"{v:9.2e}" for v in rels), flush=True)
        got = [f"step{b}" if b is not None else "?" for b in best[1:]]
        align_ok = bool(best[1:] == [1, 2, 2])
        print(f"  [align] {name} best-match(step densities+final)={got} "
              f"expected={EXPECTED_PAIRING} -> {'OK' if align_ok else 'UNEXPECTED'}",
              flush=True)

        pairs = [("step_dens[0]↔step1", sd[0], fixsteps[1]),
                 ("step_dens[1]↔step2", sd[1], fixsteps[2]),
                 ("final↔step2", final, fixsteps[2])]
        row = {"case": name, "geometry_ok": geom_ok, "align_ok": align_ok}
        for tag, got_a, ref_a in pairs:
            dmax = max_abs(got_a, ref_a)
            r = rel_err(dmax, ref_a)
            ok = bool(dmax <= DENS_ABS_GATE and r <= DENS_REL_GATE)
            row[tag] = {"abs": dmax, "rel": r, "ref_absmax": float(np.abs(ref_a).max()),
                        "pass": ok}
            print(f"  [dens] {name} {tag}: abs={dmax:.3e} (gate {DENS_ABS_GATE:.0e})  "
                  f"rel={r:.2e} (gate {DENS_REL_GATE:.0e})  "
                  f"-> {'PASS' if ok else 'FAIL'}", flush=True)
        # 锚(模块②已门): 仅打印留档, 不计入本门判定
        if anchor is not None:
            da = max_abs(anchor, fixsteps[0])
            print(f"  [dens] {name} anchor field_indep↔step0: abs={da:.3e} "
                  f"rel={rel_err(da, fixsteps[0]):.2e} (模块②口径, 仅留档)", flush=True)
        rows.append(row)
        align_rows.append({"case": name, "best": got, "expected": EXPECTED_PAIRING,
                           "ok": align_ok})

    # ---- (3) stage 门: NH3 vs charges_hooks/nh3-step1.npz ----
    # 先重跑 NH3 盒(前述循环最后停在 CO2=96 原子盒, 须刷新引擎状态为 NH3=128)
    base_stage, _, _ = mm.build_gas("NH3", 32, 0.01, 770600, 2.0)
    atoms_stage = base_stage.copy()
    atoms_stage.info["charge"] = 0
    atoms_stage.info["spin"] = 1
    atoms_stage.calc = engine
    atoms_stage.get_potential_energy()
    hk = np.load(f"{HOOKDIR}/nh3-step1.npz", allow_pickle=False)
    stage_rows = []
    print(f"== stage 门: nh3-step1.npz (最后迭代) ==", flush=True)
    for acc, ref_key in STAGE_MAP:
        ref = np.asarray(hk[ref_key], dtype=np.float64)
        raw = np.asarray(getattr(ev, acc)(), dtype=np.float64)
        if raw.size != ref.size:
            stage_rows.append({"acc": acc, "ref": ref_key, "shape_mismatch":
                               [list(raw.shape), list(ref.shape)], "pass": False})
            print(f"  [stage] {acc} ↔ {ref_key}: SHAPE-MISMATCH "
                  f"{list(raw.shape)} vs {list(ref.shape)} -> FAIL", flush=True)
            continue
        a = raw.reshape(ref.shape)
        dmax = max_abs(a, ref)
        r = rel_err(dmax, ref)
        refmax = float(np.abs(ref).max())
        ok = bool(r <= STAGE_REL_GATE)
        stage_rows.append({"acc": acc, "ref": ref_key, "abs": dmax, "rel": r,
                           "ref_absmax": refmax, "pass": ok})
        print(f"  [stage] {acc} ↔ {ref_key}: rel={r:.2e} (gate {STAGE_REL_GATE:.0e})  "
              f"abs={dmax:.3e} (refmax={refmax:.3f}) -> {'PASS' if ok else 'FAIL'}",
              flush=True)

    # ---- 汇总 ----
    dens_items = [v for row in rows for v in row.values()
                  if isinstance(v, dict) and "pass" in v]
    dens_ok = all(v["pass"] for v in dens_items)
    stage_ok = all(r.get("pass") for r in stage_rows)
    align_ok_all = all(r["ok"] for r in align_rows)
    ok = bool(dens_ok and stage_ok and align_ok_all)
    verdict = "PASS" if ok else "FAIL"
    print(f"\nGATE {verdict}: dens items={len(dens_items)} (abs≤{DENS_ABS_GATE:.0e} & "
          f"rel≤{DENS_REL_GATE:.0e})  stage items={len(stage_rows)} "
          f"(rel≤{STAGE_REL_GATE:.0e})  pairing={'OK' if align_ok_all else 'UNEXPECTED'}",
          flush=True)

    payload.update({"verdict": verdict, "density_rows": rows, "align_rows": align_rows,
                    "stage_rows": stage_rows})

    # ---- 输出 png/json ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pair_tags = ["step_dens[0]↔step1", "step_dens[1]↔step2", "final↔step2"]
    x = np.arange(len(rows))
    w = 0.26
    for j, tag in enumerate(pair_tags):
        vals = [max(r[tag]["abs"], 1e-18) for r in rows]
        axes[0].bar(x + (j - 1) * w, vals, width=w, label=tag)
    axes[0].axhline(DENS_ABS_GATE, color="r", ls="--", lw=1, label=f"gate={DENS_ABS_GATE:.0e}")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r["case"] for r in rows], fontsize=8)
    axes[0].set_ylabel("max|Δ| (engine - scf_steps)")
    axes[0].set_title("Module-5 density evolution (4 boxes)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    sv = [max(r.get("rel", np.inf), 1e-18) for r in stage_rows]
    axes[1].bar(range(len(stage_rows)), sv)
    axes[1].axhline(STAGE_REL_GATE, color="r", ls="--", lw=1,
                    label=f"gate={STAGE_REL_GATE:.0e}")
    axes[1].set_xticks(range(len(stage_rows)))
    axes[1].set_xticklabels([f"{r['acc'].replace('polar_','')}\n↔{r['ref']}"
                             for r in stage_rows], rotation=70, fontsize=6)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("rel = max|Δ| / max|ref|")
    axes[1].set_title("Module-5 stage chain (nh3-step1 hooks)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    with open(f"{args.out}.json", "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print(f"saved: {args.out}.png , {args.out}.json", flush=True)
    print("MODULE5-GATE-DONE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
