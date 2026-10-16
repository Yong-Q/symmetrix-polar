#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fukui_src ~1e-6 共模偏差定位(2026-10-15, 模块②门边界失配诊断)。

已知: |Δfukui| ~ 2-8e-6(超门 5e-6),但 |Δdensity| ~ 1e-8(归一后抵消) ⇒ 共模乘性微差。
本探针逐级对照: linear_1 → silu → linear_mid → silu → linear_2:
  - 引擎侧: 用 pybind E3LinearKokkos 逐级前向(与引擎 run 完全同代码);
  - 参考侧: mace 子模块 forward hook 捕获每级输入/输出;
  - 首先校验 δ=eng-py 的比例结构(乘性 vs 加性),再逐级定位首个失配级。

用法: CUDA_VISIBLE_DEVICES=<gpu> <polar venv python> polar_fukui_stageprobe.py
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

    data = json.load(open(J))
    fj = data["polar_blocks"]["fukui_source_map"]
    cst = float(fj["activation_constants"][0])
    print(f"silu cst = {cst!r}", flush=True)

    lins = {k: symmetrix.E3LinearKokkos(json.dumps(fj[k]))
            for k in ("linear_1", "linear_mid", "linear_2")}
    for k, v in lins.items():
        print(f"{k}: in {v.input_dimension} out {v.output_dimension}", flush=True)

    from mace.calculators import mace_polar

    ref = mace_polar(model=POLAR, device="cuda", default_dtype="float64")
    model = ref.models[0]
    fs = model.fukui_source_map
    caps = {}
    hs = []

    def hook(name):
        def fn(mod, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            caps.setdefault(name, []).append(
                (inp[0].detach().cpu().numpy(),
                 out.detach().cpu().numpy()))
        return fn

    for name, mod in (("lin1", fs.linear_1), ("silu1", fs.non_linearity),
                      ("linmid", fs.linear_mid), ("silu2", fs.non_linearity),
                      ("lin2", fs.linear_2), ("mod", fs)):
        hs.append(mod.register_forward_hook(hook(name)))

    base, _, _ = mm.build_gas("NH3", 32, 0.01, 770600, 2.0)
    atoms = base.copy()
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = ref
    atoms.get_potential_energy()
    for h in hs:
        h.remove()

    def eng_eval(key, x):
        n = x.shape[0]
        return np.asarray(lins[key].evaluate_batch(x.reshape(-1).tolist(), n),
                          dtype=np.float64).reshape(n, -1)

    def rep(tag, a, b):
        d = a - b
        print(f"  {tag:22s} max|Δ|={np.abs(d).max():.3e}  "
              f"(|ref|max={np.abs(b).max():.3f})", flush=True)

    # fukui 的输入 = 末层 node_feats(即 lin1 的输入)
    x_in, _ = caps["lin1"][0]
    print(f"fukui input: {x_in.shape}", flush=True)

    # ---- 分级对照 ----
    e1 = eng_eval("linear_1", x_in)
    _, py1 = caps["lin1"][0]
    rep("lin1 engine-vs-py", e1, py1)

    a1_e = cst * e1 / (1.0 + np.exp(-e1))
    # non_linearity 模块被前向用两次(两级 silu),两个 hook 都收到 2 次调用:
    #   caps[..][0] = 第一级输出, caps[..][1] = 第二级输出。
    _, pyA1 = caps["silu1"][0]
    rep("silu1 numpy(engine)", a1_e, pyA1)

    em = eng_eval("linear_mid", a1_e)
    _, pyM = caps["linmid"][0]
    rep("linmid engine-vs-py", em, pyM)

    a2_e = cst * em / (1.0 + np.exp(-em))
    _, pyA2 = caps["silu2"][1]
    rep("silu2 numpy(engine)", a2_e, pyA2)

    e2 = eng_eval("linear_2", a2_e)
    _, py2 = caps["lin2"][0]
    rep("lin2 engine-vs-py", e2, py2.reshape(e2.shape))

    _, py_mod = caps["mod"][0]
    py_mod = py_mod.reshape(e2.shape)
    rep("module out eng-chain", e2, py_mod)

    # ---- 与引擎实跑访问器对照 ----
    from symmetrix.calculator import Symmetrix
    engine = Symmetrix(J, dtype="float64")
    b2 = base.copy()
    b2.calc = engine
    b2.get_potential_energy()
    live = np.asarray(engine.evaluator.polar_fukui_sources(),
                      dtype=np.float64).reshape(len(b2), 2)
    rep("engine-live vs py module", live, py_mod)

    # 比例结构: Δ = live - py_mod
    d = live - py_mod
    for c in range(2):
        alpha = float(np.dot(d[:, c], py_mod[:, c])
                      / np.dot(py_mod[:, c], py_mod[:, c]))
        resid = d[:, c] - alpha * py_mod[:, c]
        print(f"  col{c}: Δ ≈ {alpha:.3e}·ref ;  残差 max {np.abs(resid).max():.3e}",
              flush=True)
    # 与内部基差异关联检查: Δ 是否 ∝ silu 导数放大? 打印几行
    print("  sample live[0..3]:", live[:4].round(7).tolist(), flush=True)
    print("  sample   py[0..3]:", py_mod[:4].round(7).tolist(), flush=True)
    print("FUKUI-STAGE-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
