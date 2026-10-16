#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""⑦ le 反向 stage 参照 (2026-10-16): python autograd 期望伴随值。

对每帧: 全模型前向(hooks 捕 le 块输入/中间、retain_grad) → E_le=Σmlp_out →
  backward → 导出:
    q_in   = linear_up_q 输入   → qin_bar  = ∂E_le/∂q_in
    P_last = linear_up_v 输入   → plast_bar= ∂E_le/∂P_last
    nf     = dot_products_x1   → nf_bar   = ∂E_le/∂nf
    q_up   = linear_up_q 输出   → q_up_bar = ∂E_le/∂q_up
    v_up   = linear_up_v 输出   → v_up_bar = ∂E_le/∂v_up
    dot_q/v= uvu 输出           → dot_q_bar/dot_v_bar
    invar  = mlp 输入           → invar_bar
落盘 bench/polar_fixtures/le_stage_ref_<frame>.npz。
用途: 引擎 polar_dbg_le_* 访问器逐级对账(引擎 e3nn 基 ≡ python 基)。
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
FIX = f"{ROOT}/bench/polar_fixtures"


def load_atoms(name):
    from ase import Atoms
    d = np.load(f"{FIX}/{name}.npz", allow_pickle=False)
    return Atoms(symbols=[str(s) for s in d["symbols"]],
                 positions=np.asarray(d["positions"], dtype=np.float64),
                 cell=np.asarray(d["cell"], dtype=np.float64), pbc=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="nh3-f0,co2-f0")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from mace.calculators import mace_polar

    calc = mace_polar(model=f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model",
                      device=args.device, default_dtype="float64")
    model = calc.models[0]
    le = model.local_electron_energy
    print(f"le block: {type(le).__name__}, "
          f"add_local_electron_energy={model.add_local_electron_energy}",
          flush=True)

    for fname in args.frames.split(","):
        cap = {}

        def keep_pre(key, idx=0):
            def hook(mod, inp):
                t = inp[idx]
                if isinstance(t, torch.Tensor):
                    if t.requires_grad:
                        t.retain_grad()
                    cap[key] = t
            return hook

        def post(key):
            def hook(mod, inp, out):
                t = out
                if isinstance(t, torch.Tensor):
                    if t.requires_grad:
                        t.retain_grad()
                    cap[key] = t
            return hook

        handles = [
            le.linear_up_q.register_forward_pre_hook(keep_pre("q_in")),
            le.linear_up_q.register_forward_hook(post("q_up")),
            le.linear_up_v.register_forward_pre_hook(keep_pre("P_last")),
            le.linear_up_v.register_forward_hook(post("v_up")),
            le.dot_products_q.register_forward_pre_hook(keep_pre("nf")),
            le.dot_products_q.register_forward_pre_hook(keep_pre("q_up_in", 1)),
            le.dot_products_q.register_forward_hook(post("dot_q")),
            le.dot_products_v.register_forward_pre_hook(keep_pre("v_up_in", 1)),
            le.dot_products_v.register_forward_hook(post("dot_v")),
            le.mlp.register_forward_pre_hook(keep_pre("invar")),
            le.mlp.register_forward_hook(post("mlp_out")),
        ]
        atoms = load_atoms(fname)
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        atoms.calc = calc
        e_total = float(atoms.get_potential_energy())
        for h in handles:
            h.remove()
        assert "mlp_out" in cap, sorted(cap)

        # calculator 内部已做过力反向(图释放)。改用捕获值重建 le 子图
        # (q_in / P_last / nf → Lin/uvu/mlp → Σ) 再 autograd:
        #  前向先与捕获中间量逐一对账(cat 顺序/接线验证), 再取全部伴随。
        def leaf(t):
            return t.detach().clone().requires_grad_(True)

        q_in = leaf(cap["q_in"])
        P_last = leaf(cap["P_last"])
        nf = leaf(cap["nf"])
        q_up = le.linear_up_q(q_in)
        v_up = le.linear_up_v(P_last)
        dot_q = le.dot_products_q(nf, q_up)
        dot_v = le.dot_products_v(nf, v_up)
        invar = torch.cat([dot_q, dot_v], dim=-1)
        mlp_out = le.mlp(invar)
        # 前向自证: 重建链 vs 捕获中间量
        for name, rebuilt, captured in (
                ("q_up", q_up, cap["q_up"]), ("v_up", v_up, cap["v_up"]),
                ("dot_q", dot_q, cap["dot_q"]), ("dot_v", dot_v, cap["dot_v"]),
                ("invar", invar, cap["invar"]),
                ("mlp_out", mlp_out, cap["mlp_out"])):
            d = float((rebuilt.detach() - captured.detach()).abs().max())
            print(f"  [rebuilt-vs-captured] {name:8s} max|Δ|={d:.3e}",
                  flush=True)
        e_le = float(mlp_out.sum())
        targets = {"qin_bar": q_in, "plast_bar": P_last, "nf_bar": nf,
                   "q_up_bar": q_up, "v_up_bar": v_up,
                   "dot_q_bar": dot_q, "dot_v_bar": dot_v,
                   "invar_bar": invar}
        grads = torch.autograd.grad(mlp_out.sum(), list(targets.values()),
                                    retain_graph=False, allow_unused=True)
        out = {"e_total": np.array([e_total]), "e_le": np.array([e_le])}
        for arrname, g in zip(targets.keys(), grads):
            if g is None:
                print(f"  [warn] grad None: {arrname}", flush=True)
                continue
            out[arrname] = g.detach().cpu().numpy()
        np.savez(f"{FIX}/le_stage_ref_{fname}.npz", **out)
        qin = out.get("qin_bar")
        print(f"== {fname}: e_total={e_total:.6f} e_le={e_le:.6f} "
              f"qin_bar|max={np.abs(qin).max() if qin is not None else float('nan'):.4e} "
              f"nf_bar|max={np.abs(out.get('nf_bar', np.zeros(1))).max():.4e} "
              f"plast_bar|max={np.abs(out.get('plast_bar', np.zeros(1))).max():.4e}",
              flush=True)
        print(f"   saved: le_stage_ref_{fname}.npz keys={sorted(out)}",
              flush=True)
    print("LE-STAGE-REF-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
