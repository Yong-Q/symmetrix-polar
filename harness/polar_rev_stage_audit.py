#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""⑧ 反向链 stage 审计 v2 (2026-10-16 v24): 引擎检查点 vs python autograd 全模型 + le 拆分。

v2 变更:
  - 引擎 out_last/out_first/fmix 检查点逐行 reshape(引擎访问器返回扁平向量);
  - python 侧改用 autograd.grad(E_le)/grad(E_total) 双目标 ⇒ 可拆:
      ∂E_le/∂t(=sd 候选) 与 ∂E_coul/∂t(=∂(E−E_le), bone 不消费这些张量);
  - 同时装载 full + leoff 两引擎:leoff 的 dfinal 检查点 = 纯 ⑥ q̄,
      与 full 的差 = sd 投递 —— 三方对账(engine_full / engine_leoff / python×3 目标)。
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/bench")

FIX = f"{ROOT}/bench/polar_fixtures"
J_FULL = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
J_LEOFF = f"{ROOT}/models/mace-polar/polar_symmetrix_probe_leoff.json"
MODEL = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"


def load_atoms(name):
    from ase import Atoms
    d = np.load(f"{FIX}/{name}.npz", allow_pickle=False)
    return Atoms(symbols=[str(s) for s in d["symbols"]],
                 positions=np.asarray(d["positions"], dtype=np.float64),
                 cell=np.asarray(d["cell"], dtype=np.float64), pbc=True)


def report(label, eng, ref, n=None, cols=None):
    e = np.asarray(eng, dtype=np.float64).reshape(-1)
    r = np.asarray(ref, dtype=np.float64).reshape(-1)
    if n is not None and cols is not None:
        e = e.reshape(n, cols)
        r = r.reshape(n, cols)
    if e.shape != r.shape:
        print(f"  {label:14s} SHAPE-MISMATCH engine={e.shape} ref={r.shape}",
              flush=True)
        return
    d = np.abs(e - r)
    denom = max(float(np.abs(r).max()), 1e-30)
    idx = np.unravel_index(np.argmax(d), d.shape)
    flat = d.reshape(-1)
    top = np.argsort(flat)[-3:][::-1]
    tops = [f"{int(i)}:{flat[i]:.2e}" for i in top]
    print(f"  {label:14s} shape={e.shape} |ref|max={np.abs(r).max():.4e} "
          f"|e|max={np.abs(e).max():.4e} max|d|={d.max():.4e} "
          f"rel={d.max()/denom:.3e} @{idx} top3={tops}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="nh3-f0,co2-f0")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from mace.calculators import mace_polar

    calc = mace_polar(model=MODEL, device=args.device, default_dtype="float64")
    model = calc.models[0]
    le = model.local_electron_energy
    maps = model.field_dependent_charges_maps
    nsteps = len(maps)
    last_idx = nsteps - 1
    print(f"model: nsteps={nsteps} le={type(le).__name__}", flush=True)

    from symmetrix.calculator import Symmetrix

    os.environ["POLAR_DBG_CKPT"] = "1"
    engines = {"full": Symmetrix(J_FULL, dtype="float64"),
               "leoff": Symmetrix(J_LEOFF, dtype="float64")}
    probes = ("polar_dbg_rev_dfinal", "polar_dbg_rev_out_last",
              "polar_dbg_rev_out_first", "polar_dbg_rev_dpost",
              "polar_dbg_rev_dinit", "polar_dbg_rev_fmix")
    for p in probes:
        if not hasattr(engines["full"].evaluator, p):
            print(f"ACCESSORS-MISSING: {p}", flush=True)
            return 2

    for fname in args.frames.split(","):
        atoms = load_atoms(fname)
        n = len(atoms)
        cap = {}

        def _base_of(t):
            # 视图(.view()/切片)梯度只含经该视图的路径;引擎侧是张量级全路径
            # 伴随 ⇒ python 对账必须用父张量(其梯度自动含所有视图分支)。
            b = getattr(t, "_base", None)
            return b if b is not None else t

        def make_pre(idx):
            def pre_map(mod, a, kwargs):
                t = kwargs.get("node_feats")
                if t is not None:
                    t.retain_grad()
                    cap["fmix"] = t
                t = kwargs.get("potential_features")
                if t is not None:
                    t.retain_grad()
                    cap[f"plast{idx}"] = t
                t = kwargs.get("local_charges")
                if t is not None:
                    t = _base_of(t)
                    t.retain_grad()
                    cap[f"dpre{idx}"] = t
            return pre_map

        def make_post(idx):
            def post_map(mod, a, out):
                out.retain_grad()
                cap[f"out{idx}"] = out
            return post_map

        def pre_le(mod, a, kw):
            for key, name in (("charges_0", "dinit"),
                              ("charges_induced", "dfinal"),
                              ("field_feats", "plast_le")):
                t = kw.get(key)
                if t is not None:
                    t = _base_of(t)
                    t.retain_grad()
                    cap[name] = t

        hooks = [le.register_forward_pre_hook(pre_le, with_kwargs=True)]
        for i in range(nsteps):
            hooks.append(maps[i].register_forward_pre_hook(
                make_pre(i), with_kwargs=True))
            hooks.append(maps[i].register_forward_hook(make_post(i)))
        # 捕获 descriptor 几何缓存(相位伴随对账用): monkeypatch precompute_geometry。
        desc = model.electric_potential_descriptor
        _orig_pre = desc.precompute_geometry

        def _cap_pre(*a, **k):
            c = _orig_pre(*a, **k)
            cap["cache"] = c
            return c

        desc.precompute_geometry = _cap_pre

        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        batch = calc._clone_batch(calc._atoms_to_batch(atoms))
        bd = batch.to_dict()
        pos = bd["positions"]
        pos.requires_grad_(True)
        res = model(bd, compute_force=False, training=False)
        for h in hooks:
            h.remove()
        desc.precompute_geometry = _orig_pre
        E_all = res["energy"].sum()
        E_le = res["electron_energy"].sum()
        E_coul = res["electrostatic_energy"].sum()

        tkeys = ["dfinal", "dinit", f"dpre{last_idx}", f"out{last_idx}",
                 "out0", "fmix", "dpre0"]
        tl = [cap[k] for k in tkeys]

        gl = torch.autograd.grad(E_le, tl, retain_graph=True)
        ga = torch.autograd.grad(E_all, tl, retain_graph=True)
        py_le = {}
        py_all = {}
        for k, v1, v2 in zip(tkeys, gl, ga):
            py_le[k] = np.asarray(v1.detach().cpu().numpy(),
                                  dtype=np.float64).reshape(n, -1)
            py_all[k] = np.asarray(v2.detach().cpu().numpy(),
                                   dtype=np.float64).reshape(n, -1)
        py_coul = {k: py_all[k] - py_le[k] for k in tkeys}

        # 相位伴随真值: descriptor 缓存 cos/sin 路径 + coulomb 自身位置依赖。
        cos_t = cap["cache"]["cosines"]
        sin_t = cap["cache"]["sines"]
        g_cs = torch.autograd.grad(E_all, [cos_t, sin_t],
                                   retain_graph=True, allow_unused=True)
        g_cs = [g if g is not None else torch.zeros_like(t)
                for g, t in zip(g_cs, (cos_t, sin_t))]
        coul_pos = torch.autograd.grad(E_coul, pos, retain_graph=True)[0]
        phase_pos = torch.autograd.grad(
            [cos_t, sin_t], pos, grad_outputs=g_cs, retain_graph=True,
            allow_unused=True)[0]
        phase_pos = np.asarray(
            phase_pos.detach().cpu().numpy(), dtype=np.float64).reshape(n, 3)
        coul_pos = np.asarray(
            coul_pos.detach().cpu().numpy(), dtype=np.float64).reshape(n, 3)

        # -- 引擎: full + leoff
        engvals = {}
        for name, engine in engines.items():
            a2 = atoms.copy()
            a2.info["charge"] = 0
            a2.info["spin"] = 1
            a2.calc = engine
            _ = float(a2.get_potential_energy())
            _ = a2.get_forces()
            ev = engine.evaluator
            engvals[name] = {
                "dfinal": np.asarray(ev.polar_dbg_rev_dfinal(),
                                     dtype=np.float64).reshape(n, 8),
                "out_last": np.asarray(ev.polar_dbg_rev_out_last(),
                                       dtype=np.float64).reshape(n, -1),
                "out_first": np.asarray(ev.polar_dbg_rev_out_first(),
                                        dtype=np.float64).reshape(n, -1),
                "dpost": np.asarray(ev.polar_dbg_rev_dpost(),
                                    dtype=np.float64).reshape(n, 8),
                "dinit": np.asarray(ev.polar_dbg_rev_dinit(),
                                    dtype=np.float64).reshape(n, 8),
                "fmix": np.asarray(ev.polar_dbg_rev_fmix(),
                                   dtype=np.float64).reshape(n, -1),
                "pos_coul": np.asarray(ev.polar_dbg_rev_pos_coul(),
                                       dtype=np.float64).reshape(n, 3),
                "pos_all": np.asarray(ev.polar_dbg_rev_pos_all(),
                                      dtype=np.float64).reshape(n, 3),
            }
        full = engvals["full"]
        leoff = engvals["leoff"]

        print(f"\n== {fname} (n={n}) ==", flush=True)
        print("  [dfinal 拆分: T=full vs py_all | ⑥=leoff vs py_coul | ⑦=full−leoff vs py_le]",
              flush=True)
        report("dfinal T", full["dfinal"], py_all["dfinal"], n, 8)
        report("dfinal ⑥", leoff["dfinal"], py_coul["dfinal"], n, 8)
        sd_eng = full["dfinal"] - leoff["dfinal"]
        report("dfinal ⑦", sd_eng, py_le["dfinal"], n, 8)
        print("  [其余检查点: full vs py_all]", flush=True)
        report("out_last", full["out_last"], py_all[f"out{last_idx}"])
        report("out_first", full["out_first"], py_all["out0"])
        report("dpost", full["dpost"], py_all[f"dpre{last_idx}"], n, 8)
        report("dinit", full["dinit"], py_all["dinit"], n, 8)
        report("fmix", full["fmix"], py_all["fmix"])
        # -- dinit 链拆分 v2: 引擎链贡献 = dinit − sd0(sd0 == qin_bar == py_le dfinal);
        #    python 链真值 = ∂E/∂O0(循环前对象) = py_all["dpre0"]
        #    (python 中 le 走独立 clone,链条路径在 O0 上; 引擎单张量 = 两者和)。
        chain_eng = full["dinit"] - py_le["dfinal"]
        report("dinit 链v2", chain_eng, py_all["dpre0"], n, 8)
        # -- 相位伴随三方对账(pos 量纲 eV/Å; 引擎 pos_adj = +∂E/∂r)。
        report("pos ⑥", full["pos_coul"], coul_pos, n, 3)
        report("pos desc", full["pos_all"] - full["pos_coul"], phase_pos, n, 3)
        report("pos all", full["pos_all"], coul_pos + phase_pos, n, 3)
        np.savez(f"{FIX}/dbg_dinit_{fname}.npz",
                 eng_dinit=full["dinit"], py_all_dinit=py_all["dinit"],
                 py_le_dinit=py_le["dfinal"],
                 eng_dpost=full["dpost"], py_all_dpost=py_all[f"dpre{last_idx}"],
                 eng_dfinal=full["dfinal"], py_all_dfinal=py_all["dfinal"],
                 eng_pos_all=full["pos_all"], py_pos_all=coul_pos + phase_pos,
                 eng_pos_coul=full["pos_coul"], py_pos_coul=coul_pos,
                 py_phase=phase_pos)
    print("\nREV-STAGE-AUDIT-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
