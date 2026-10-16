#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lr_source_maps 隔离诊断(2026-10-15, 模块①门失败的二分定位)。

把 Python 前向钩子捕获的【冻结 node_feats】直接喂给引擎 E3LinearKokkos
(经 pybind `symmetrix.E3LinearKokkos(json).evaluate_batch`), 对照 Python 的
lr_src 输出:
  - 若失配 → 序列化/引擎层实现问题(与 state.output 无关);
  - 若吻合 → 问题在引擎 state.output(输入张量语义)。

用法: CUDA_VISIBLE_DEVICES=<gpu> <polar venv python> polar_lrsrc_isolate.py
环境: 同 run env。
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


def main():
    import symmetrix
    if not symmetrix._kokkos_is_initialized():
        symmetrix._init_kokkos()
    from mace.calculators import mace_polar

    ref = mace_polar(model=POLAR, device="cuda", default_dtype="float64")
    model = ref.models[0]
    caps = {}
    hs = []
    for i, m in enumerate(model.lr_source_maps):
        hs.append(m.register_forward_hook(
            (lambda idx: lambda mod, inp, out: caps.__setitem__(
                idx, (inp[0].detach().cpu().numpy(),
                      out.detach().cpu().numpy())))(i)))

    base, _, _ = mm.build_gas("NH3", 32, 0.01, 770600, 2.0)
    atoms = base.copy()
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = ref
    atoms.get_potential_energy()
    for h in hs:
        h.remove()

    d = json.load(open(J))
    ok = True
    for i in (0, 1):
        lin = symmetrix.E3LinearKokkos(json.dumps(d["polar_blocks"]["lr_source_maps"][i]))
        inp0, out0 = caps[i]
        N = inp0.shape[0]
        print(f"layer {i}: engine lin dims {lin.input_dimension}->{lin.output_dimension}; "
              f"py in {inp0.shape} out {out0.shape}", flush=True)
        got = np.asarray(lin.evaluate_batch(inp0.reshape(-1).tolist(), N),
                         dtype=np.float64).reshape(N, 8)
        ref_out = out0[:, 0, :]
        dmax = float(np.abs(got - ref_out).max())
        print(f"  max|Δ(engine lin vs python lr_src)| = {dmax:.3e}   "
              f"(ref absmax {np.abs(ref_out).max():.3f})", flush=True)
        ok = ok and dmax <= 5e-6
        # 附加: 若失配, 打印第一行 8 分量的对照
        if dmax > 5e-6:
            print("  engine[0]:", np.round(got[0], 6).tolist(), flush=True)
            print("  python[0]:", np.round(ref_out[0], 6).tolist(), flush=True)

    # 组合检查: 引擎两层输出之和 vs python 两层之和(冻结输入口径)
    tot_e = np.zeros_like(caps[0][1][:, 0, :])
    for i in (0, 1):
        lin = symmetrix.E3LinearKokkos(json.dumps(d["polar_blocks"]["lr_source_maps"][i]))
        inp0, out0 = caps[i]
        N = inp0.shape[0]
        tot_e += np.asarray(lin.evaluate_batch(inp0.reshape(-1).tolist(), N),
                            dtype=np.float64).reshape(N, 8)
    tot_py = caps[0][1][:, 0, :] + caps[1][1][:, 0, :]
    print(f"\nsum: max|Δ| = {np.abs(tot_e - tot_py).max():.3e}", flush=True)
    print("ISOLATE-DONE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
