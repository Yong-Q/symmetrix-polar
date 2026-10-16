#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 模块⑤(charges_maps 更新块) stage 级参考夹具: 前向钩子捕获输入/输出 (2026-10-15)。

在 NH3 夹具盒 (mm.build_gas("NH3",32,0.01,770600,2.0), 几何先与
bench/polar_fixtures/scf/nh3-f0.npz 的 positions np.allclose 自证) 上跑 mace_polar
参考模型。SCF 循环 (extensions.py L777-864) 每一步调用 field_dependent_charges_maps[i]
(i=0,1, num_recursion_steps=2 ⇒ 每个块恰被调用一次), 对两个块均注册 forward hook
(with_kwargs), 捕获:
  [块级] 输入 kwargs: node_attrs / node_feats(=features_mixed) / edge_attrs /
         edge_feats / edge_index / potential_features / local_charges + 输出 (N,10)
  [子模块] potential_embedding(+3 分量 linear)、dot_products、source_embedding、
         nonlinearity、tp_out、readout.linear_1 / equivariant_nonlin(gate, 前向中
         被调用两次→ gate1_* / gate2_*)/ linear_mid / linear_2 的输入与输出
落盘 bench/polar_fixtures/charges_hooks/nh3-step{k}.npz (k=0←块0, k=1←块1) + index.json。

对账(同代码路径夹具):
  - local_charges  vs scf_steps/nh3-step{k}.npz:spin_charge_density_flat;
  - blk{k}_out    vs scf/nh3-f0.npz:charges_step_{k};
  - potential_features[:, ch*8:(ch+1)*8] * field_feature_norms
                  vs scf/nh3-f0.npz:potential_feats_step_{2k+ch} (归一前原始势特征)。

用法: cd mlip && CUDA_VISIBLE_DEVICES=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
      /workspace/caches/conda/envs/polar/bin/python bench/polar_module5_hooks.py
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
import mace_md as mm  # noqa: E402

POLAR = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"
SCFDIR = f"{ROOT}/bench/polar_fixtures/scf"
STEPDIR = f"{ROOT}/bench/polar_fixtures/scf_steps"
OUTDIR = f"{ROOT}/bench/polar_fixtures/charges_hooks"
MOL, SEED = "NH3", 770600
BLOCK_IN = ("node_attrs", "node_feats", "edge_attrs", "edge_feats", "edge_index",
            "potential_features", "local_charges")

# 子模块落盘映射: module name -> (输入名元组, 输出名元组)
SUB_CAP = {
    "potential_embedding": (("pe_in_potential", "pe_in_node_feats", "pe_in_node_attrs",
                             "pe_in_local_charges"), ("pe_out",)),
    "potential_embedding.potential_linear": (("pe_pot_lin_in",), ("pe_pot_lin_out",)),
    "potential_embedding.node_feats_linear": (("pe_nf_lin_in",), ("pe_nf_lin_out",)),
    "potential_embedding.charge_embedding": (("pe_q_lin_in",), ("pe_q_lin_out",)),
    "dot_products": (("dp_in_node_feats", "dp_in_mixed_feats"), ("dp_out",)),
    "source_embedding": (("se_in_node_attrs",), ("se_out",)),
    "nonlinearity": (("nl_in",), ("nl_out",)),
    "tp_out": (("tp_in_node_feats", "tp_in_nonlin"), ("tp_out",)),
    "readout.linear_1": (("ro_l1_in",), ("ro_l1_out",)),
    "readout.equivariant_nonlin": (("gate_in",), ("gate_out",)),  # 第2次调用→ gate2_*
    "readout.linear_mid": (("ro_lmid_in",), ("ro_lmid_out",)),
    "readout.linear_2": (("ro_l2_in",), ("ro_l2_out",)),
}


def to_np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    if isinstance(t, (list, tuple)):
        return np.asarray([to_np(x) for x in t])
    return np.asarray(t)


class Recorder:
    """共享 step 游标：块级 pre-hook 推进 cur, 子模块 hook 写入 captures[cur]。"""

    def __init__(self):
        self.cur = -1
        self.captures = {}
        self.call_count = {}

    def put(self, name, arr):
        self.captures.setdefault(self.cur, {})[name] = arr

    def bump(self, key):
        """同名子模块多次调用计数 (如 gate 前向两次)。"""
        n = self.call_count.get((self.cur, key), 0)
        self.call_count[(self.cur, key)] = n + 1
        return n


def register_block_hooks(rec, blk, bi):
    def pre(module, args, kwargs):
        rec.cur = bi
        for an in BLOCK_IN:
            if an in kwargs:
                rec.put(f"in_{an}", to_np(kwargs[an]))
            else:
                # 位置参数兜底
                i = BLOCK_IN.index(an)
                if i < len(args):
                    rec.put(f"in_{an}", to_np(args[i]))

    def post(module, args, kwargs, output):
        rec.put("out", to_np(output))

    return [blk.register_forward_pre_hook(pre, with_kwargs=True),
            blk.register_forward_hook(post, with_kwargs=True)]


def register_sub_hooks(rec, blk):
    handles = []
    name2mod = dict(blk.named_modules())
    for mname, (in_names, out_names) in SUB_CAP.items():
        mod = name2mod[mname]

        def mk_pre(ins, key):
            def fn(module, args, kwargs):
                n = rec.bump(key)  # 0=首次; gate 第二次调用 → n=1
                suffix = "" if n == 0 else str(n + 1)
                for an, av in zip(ins, args if args else []):
                    rec.put(f"{an}{suffix}", to_np(av))
                for an in ins:
                    if an in (kwargs or {}):
                        rec.put(f"{an}{suffix}", to_np(kwargs[an]))
            return fn

        def mk_post(outs, key):
            def fn(module, args, kwargs, output):
                n = rec.call_count.get((rec.cur, key), 1) - 1
                suffix = "" if n == 0 else str(n + 1)
                for an in outs:
                    rec.put(f"{an}{suffix}", to_np(output))
            return fn

        handles.append(mod.register_forward_pre_hook(mk_pre(list(in_names), mname),
                                                     with_kwargs=True))
        handles.append(mod.register_forward_hook(mk_post(list(out_names), mname),
                                                 with_kwargs=True))
    return handles


def main():
    from mace.calculators import mace_polar

    os.makedirs(OUTDIR, exist_ok=True)
    calc = mace_polar(model=POLAR, device="cuda", default_dtype="float64")
    model = calc.models[0]

    # ---- 几何自证: build_gas 与 scf/nh3-f0.npz 同构 ----
    atoms, _, _ = mm.build_gas(MOL, 32, 0.01, SEED, 2.0)
    ref = np.load(f"{SCFDIR}/nh3-f0.npz", allow_pickle=False)
    geo_ok = bool(np.allclose(atoms.get_positions(), ref["positions"], atol=1e-10))
    print(f"[geo ] build_gas vs scf/nh3-f0.npz positions: allclose={geo_ok}", flush=True)
    assert geo_ok, "geometry self-check failed"

    # ---- 钩子: 两个块共用 Recorder ----
    rec = Recorder()
    handles = []
    for bi in range(len(model.field_dependent_charges_maps)):
        handles += register_block_hooks(rec, model.field_dependent_charges_maps[bi], bi)
        handles += register_sub_hooks(rec, model.field_dependent_charges_maps[bi])

    # ---- 单点 ----
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = calc
    e = atoms.get_potential_energy()
    print(f"[run ] E = {e:.10f} eV; captured steps={sorted(rec.captures)}", flush=True)

    # ---- 落盘 + 对账 ----
    norms = model.field_feature_norms.detach().cpu().numpy()
    index = {
        "generator": "bench/polar_module5_hooks.py",
        "date": "2026-10-15",
        "model": "models/mace-polar/MACE-POLAR-1-M.model",
        "fixture": {
            "molecule": MOL, "n_molecules": 32, "density": 0.01, "seed": SEED,
            "min_distance": 2.0,
            "geometry_selfcheck": {"vs": "polar_fixtures/scf/nh3-f0.npz:positions",
                                   "allclose": geo_ok},
            "charge": 0, "spin": 1, "energy_eV": float(e),
        },
        "semantics": {
            "step0": "SCF 迭代 0: field_dependent_charges_maps[0] 的调用 "
                     "(local_charges = 初校正后密度)",
            "step1": "SCF 迭代 1: field_dependent_charges_maps[1] 的调用 "
                     "(local_charges = 第0步更新后密度)",
            "call_site": "mace/modules/extensions.py L777-864 (kwargs 调用, "
                         "每块每 forward 恰一次)",
            "node_feats/in_node_feats": "= features_mixed = "
                                        "layer_feature_mixer(stack(node_feats_list))",
            "potential_features": "cat(α(8),β(8)); 每通道 2x0e+2x1o; 已 /field_feature_norms; "
                                  "无外场(half_external_field=0)",
            "local_charges": "spin_charge_density.view(N,8) = [sα,xα,yα,zα,sβ,xβ,yβ,zβ]",
            "out": "(N,10) = [8 density sources | 2 fukui sources] "
                   "(extensions.py L833-834: out[:,-2:]=fukui, out[:,:-2]=density 增量)",
            "gate1/gate2": "readout.equivariant_nonlin 前向中调用两次: gate1=作用在 "
                           "linear_1 输出 (silu/sigmoid, cst 见 polar_module5_audit.md), "
                           "gate2=作用在 linear_mid 输出",
        },
        "tensors": {},
    }

    for k in sorted(rec.captures):
        caps = rec.captures[k]
        stem = f"nh3-step{k}"
        payload = {"positions": atoms.get_positions(),
                   "symbols": np.asarray(atoms.get_chemical_symbols(), dtype="U2"),
                   "energy": np.float64(e)}
        summary = {}
        for name, arr in sorted(caps.items()):
            payload[name] = arr
            summary[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                             "absmax": float(np.abs(arr).max()),
                             "sum": float(np.sum(arr))}
        np.savez(f"{OUTDIR}/{stem}.npz", **payload)
        case = {"block_index": k, "npz": f"{stem}.npz", "tensors": summary,
                "checks": {}}
        print(f"[dump] {stem}: {len(summary)} tensors (block {k})", flush=True)

        st_path = f"{STEPDIR}/{stem}.npz"
        if os.path.exists(st_path) and "in_local_charges" in caps:
            st = np.load(st_path, allow_pickle=False)
            dd = float(np.abs(caps["in_local_charges"]
                              - st["spin_charge_density_flat"]).max())
            case["checks"]["local_charges_vs_scf_steps"] = dd
            print(f"       in_local_charges vs scf_steps/{stem}: max|Δ|={dd:.3e}", flush=True)
        if f"charges_step_{k}" in ref.files and "out" in caps:
            dd = float(np.abs(caps["out"] - ref[f"charges_step_{k}"]).max())
            case["checks"]["out_vs_scf_charges_step"] = dd
            print(f"       out vs scf charges_step_{k}: max|Δ|={dd:.3e}", flush=True)
        pf = caps.get("in_potential_features")
        if pf is not None:
            for ch, key in ((0, f"potential_feats_step_{2*k}"),
                            (1, f"potential_feats_step_{2*k+1}")):
                if key in ref.files:
                    dd = float(np.abs(pf[:, ch * 8:(ch + 1) * 8] * norms
                                      - ref[key]).max())
                    case["checks"][f"potfeat_{key}_x_norms"] = dd
                    print(f"       in_potential_features[{ch}]×norms vs scf {key}: "
                          f"max|Δ|={dd:.3e}", flush=True)
        index["tensors"][stem] = case

    with open(f"{OUTDIR}/index.json", "w") as fh:
        json.dump(index, fh, indent=1)
    print(f"[done] -> {OUTDIR}/index.json", flush=True)
    print("MODULE5-HOOKS-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
