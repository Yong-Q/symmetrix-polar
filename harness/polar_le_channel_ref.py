#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""⑦ le 反向逐通道真值 (2026-10-16 v3): 块边界偏导 × 全图 VJP。

依据: E_le = F(c0, ci, ff, nf)(块=函数于其参数)。
  d 通道真值 = −[VJP_full(c0; ∂F/∂c0) + VJP_full(ci; ∂F/∂ci)]
  P 通道真值 = −VJP_full(ff; ∂F/∂ff)
  nf 通道真值 = −VJP_full(nf; ∂F/∂nf)
其中 ∂F/∂a 用"块重调(全 detach 克隆)+autograd"取(全图梯度对非反链割的输入
会含跨路径项, 不可用作种子——见 2026-10-16 调查注)。
注: 三通道和 ≠ F_le(路径双计), 逐通道各自与引擎 POLAR_DBG_SKIP 有线差分对账。

产物: bench/polar_fixtures/le_channel_ref_<frame>.npz:
    F_le, A_d, A_P, A_nf (+e_le)。F_le 由全图 autograd 取得(与 FD 逐位一致)。
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
FIX = f"{ROOT}/bench/polar_fixtures"
KEYS = ["node_attrs", "node_feats", "edge_attrs", "edge_feats", "edge_index",
        "field_feats", "charges_0", "charges_induced"]


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

    for fname in args.frames.split(","):
        cap = {}

        def block_pre(mod, args_, kwargs):
            for i, k in enumerate(KEYS):
                if k in kwargs:
                    cap[k] = kwargs[k]
                if i < len(args_):
                    cap[k] = args_[i]

        h1 = le.register_forward_pre_hook(block_pre, with_kwargs=True)
        h2 = le.mlp.register_forward_hook(
            lambda m, i, o: cap.__setitem__("mlp_out", o))

        atoms = load_atoms(fname)
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        batch = calc._clone_batch(calc._atoms_to_batch(atoms))
        bd = batch.to_dict()
        pos = bd["positions"]
        pos.requires_grad_(True)
        out = model(bd, compute_force=False, training=False)
        h1.remove()
        h2.remove()
        mlp_out = cap["mlp_out"]
        e_le = float(mlp_out.sum())

        # (1) 全图: F_le 与 VJP 所需的图保持
        g_pos = torch.autograd.grad(mlp_out.sum(), pos,
                                    retain_graph=True)[0]
        f_le = (-g_pos).detach()

        # (2) 块重调: 全 detach 克隆 → 块自身偏导 ∂F/∂a
        tg = [k for k in ("charges_0", "charges_induced", "field_feats",
                          "node_feats")
              if k in cap and isinstance(cap[k], torch.Tensor)]
        argsd = {k: cap[k].detach().clone().requires_grad_(True) for k in tg}
        out2 = le(node_attrs=cap["node_attrs"], node_feats=argsd["node_feats"],
                  edge_attrs=cap["edge_attrs"], edge_feats=cap["edge_feats"],
                  edge_index=cap["edge_index"], field_feats=argsd["field_feats"],
                  charges_0=argsd["charges_0"],
                  charges_induced=argsd["charges_induced"])
        dcheck = float((out2.detach() - mlp_out.detach()).abs().max())
        g_blk = torch.autograd.grad(out2.sum(), [argsd[k] for k in tg])

        # (3) 全图 VJP(以块偏导为种子): 逐通道真值
        chan = {}
        for k, gk in zip(tg, g_blk):
            vjp = torch.autograd.grad(cap[k], pos, grad_outputs=gk.detach(),
                                      retain_graph=True)[0]
            chan[k] = (-vjp).detach()

        a_d = chan["charges_0"] + chan["charges_induced"]
        a_p = chan["field_feats"]
        a_nf = chan["node_feats"]
        np.savez(f"{FIX}/le_channel_ref_{fname}.npz",
                 F_le=f_le.cpu().numpy(), A_d=a_d.cpu().numpy(),
                 A_P=a_p.cpu().numpy(), A_nf=a_nf.cpu().numpy(),
                 e_le=np.array([e_le]))
        print(f"== {fname}: e_le={e_le:.6f} blkcheck={dcheck:.2e} "
              f"F_le|max={float(f_le.abs().max()):.4e} | "
              f"A_d|max={float(a_d.abs().max()):.4e} "
              f"A_P|max={float(a_p.abs().max()):.4e} "
              f"A_nf|max={float(a_nf.abs().max()):.4e}", flush=True)
        print(f"   F_le[0]={f_le[0].cpu().numpy().round(6)} "
              f"A_d[0]={a_d[0].cpu().numpy().round(6)} "
              f"A_P[0]={a_p[0].cpu().numpy().round(6)} "
              f"A_nf[0]={a_nf[0].cpu().numpy().round(6)}", flush=True)
        print(f"   saved: le_channel_ref_{fname}.npz", flush=True)
    print("LE-CHANNEL-REF-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
