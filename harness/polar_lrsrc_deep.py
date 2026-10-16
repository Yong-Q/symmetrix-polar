#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模块①失配深度诊断(2026-10-15)。

已知: 引擎 E3Linear 在冻结 python node_feats 上逐位吻合(2.5e-16)。
未知: 引擎内部 state.output 与本几何的 python node_feats 差在何处。
方法: 用 python 钩子取 in0/in1(逐层 node_feats), 再对候选变换 T 组出假设和
      c_T = lr0(T(in0)) + lr1(T(in1)), 与引擎 accumulator 对照 —— 找出 T。

候选 T: identity / ir_mul 重排(l=1 块) / 逐层串位(in0↔in1) / 整体缩放拟合。

用法: CUDA_VISIBLE_DEVICES=<gpu> <polar venv python> polar_lrsrc_deep.py
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
import mace_md as mm  # noqa: E402

J = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
POLAR = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"


def to_ir_mul(x, mul=512):
    """mul_ir (N, 512*4) -> ir_mul: l=1 块按分量主序重排(0e 块不动)。"""
    n = x.shape[0]
    out = np.empty_like(x)
    out[:, :mul] = x[:, :mul]
    v = x[:, mul:].reshape(n, mul, 3)
    out[:, mul:] = np.transpose(v, (0, 2, 1)).reshape(n, mul * 3)
    return out


def main():
    import symmetrix
    if not symmetrix._kokkos_is_initialized():
        symmetrix._init_kokkos()
    from symmetrix.calculator import Symmetrix
    from mace.calculators import mace_polar

    d = json.load(open(J))
    lins = [symmetrix.E3LinearKokkos(json.dumps(d["polar_blocks"]["lr_source_maps"][i]))
            for i in (0, 1)]

    def lr(i, x):
        n = x.shape[0]
        return np.asarray(lins[i].evaluate_batch(x.reshape(-1).tolist(), n),
                          dtype=np.float64).reshape(n, 8)

    ref = mace_polar(model=POLAR, device="cuda", default_dtype="float64")
    model = ref.models[0]
    caps = {}
    hs = []
    for i, m in enumerate(model.lr_source_maps):
        hs.append(m.register_forward_hook(
            (lambda idx: lambda mod, inp, out: caps.__setitem__(
                idx, inp[0].detach().cpu().numpy()))(i)))

    base, _, _ = mm.build_gas("NH3", 32, 0.01, 770600, 2.0)
    atoms = base.copy()
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = ref
    atoms.get_potential_energy()
    for h in hs:
        h.remove()
    in0, in1 = caps[0], caps[1]
    print(f"in0 {in0.shape} in1 {in1.shape}", flush=True)

    engine = Symmetrix(J, dtype="float64")
    atoms2 = base.copy()
    atoms2.calc = engine
    atoms2.get_potential_energy()
    acc = np.asarray(engine.evaluator.polar_spin_charge_density(),
                     dtype=np.float64).reshape(len(atoms2), 8)

    def rep(name, cand):
        dmax = float(np.abs(acc - cand).max())
        rms = float(np.sqrt(np.mean((acc - cand) ** 2)))
        print(f"  {name:34s} max|Δ|={dmax:.3e}  rms={rms:.3e}", flush=True)

    base_ok = lr(0, in0) + lr(1, in1)
    rep("c1: lr0(in0)+lr1(in1) [期望]", base_ok)
    rep("c2: lr0(in0)+lr1(in0)  [层1串位]", lr(0, in0) + lr(1, in0))
    rep("c3: lr0(in1)+lr1(in1)  [层0串位]", lr(0, in1) + lr(1, in1))
    rep("c4: lr0(Tir)+lr1(Tir)  [ir_mul]", lr(0, to_ir_mul(in0)) + lr(1, to_ir_mul(in1)))
    # 缩放拟合 acc ≈ a * c1
    a = float(np.dot(acc.reshape(-1), base_ok.reshape(-1))
              / np.dot(base_ok.reshape(-1), base_ok.reshape(-1)))
    rep(f"c5: a*c1, a={a:.6f}", a * base_ok)
    # 每列独立缩放
    a_col = (acc * base_ok).sum(0) / (base_ok ** 2).sum(0)
    rep(f"c6: colwise a*c1, a={np.round(a_col,4).tolist()}", base_ok * a_col)
    # 相关性
    for i, cand in enumerate([base_ok]):
        pass
    print(f"\n  acc absmax={np.abs(acc).max():.4f}  c1 absmax={np.abs(base_ok).max():.4f}",
          flush=True)
    print(f"  corr(acc, c1) = {np.corrcoef(acc.reshape(-1), base_ok.reshape(-1))[0,1]:.5f}",
          flush=True)
    np.savez(f"{ROOT}/bench/polar_lrsrc_deep.npz", acc=acc, c1=base_ok, in0=in0, in1=in1)
    print("DEEP-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
