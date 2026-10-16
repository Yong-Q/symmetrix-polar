#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 模块⑤(polar_blocks.field_dependent_charges_maps)序列化完备性探针 (2026-10-15)。

对照运行时模块 (checkpoint 实载) 的全部前向量与
models/mace-polar/polar_symmetrix_probe.json 已序列化内容, 逐叶子打印:
  1. 模块树 (class/param/buffer);
  2. o3.Linear 指令表 (含 path_weight/has_weight);
  3. SparseUvuTensorProduct 指令/权重/掩码/_path_meta (前向真正消费的参数);
  4. readout Gate (e3nn.nn.Gate) 子结构: Activation 常数 cst / mul 指令;
  5. nonlinearity (RadialMLP) 逐层类型 + LayerNorm eps;
  6. JSON leaf ⇄ live state_dict 对账 (missing/extra)。

用法: cd mlip && CUDA_VISIBLE_DEVICES=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
      /workspace/caches/conda/envs/polar/bin/python bench/polar_module5_audit_probe.py
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import json  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
POLAR = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"
J = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"


def leaf_paths(obj, prefix=""):
    """JSON 树叶子路径 (dict 递归; list 用 [i])。"""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out |= leaf_paths(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out |= leaf_paths(v, f"{prefix}[{i}].")
    else:
        out.add(prefix[:-1])
    return out


def main():
    import torch
    from mace.calculators import mace_polar

    calc = mace_polar(model=POLAR, device="cpu", default_dtype="float64")
    model = calc.models[0]
    b = model.field_dependent_charges_maps[0]
    jb = json.load(open(J))["polar_blocks"]["field_dependent_charges_maps"][0]

    print("=" * 78)
    print("[1] module tree (field_dependent_charges_maps[0])")
    for name, mod in b.named_modules():
        if name == "":
            continue
        cname = type(mod).__module__ + "." + type(mod).__name__
        extra = ""
        if hasattr(mod, "irreps_in") and isinstance(getattr(mod, "irreps_in"), object):
            try:
                extra = f" irreps_in={mod.irreps_in} irreps_out={getattr(mod, 'irreps_out', None)}"
            except Exception:
                pass
        print(f"  {name:52s} {cname}{extra}")
    print("  -- params/buffers --")
    for k, v in b.state_dict().items():
        print(f"    {k:60s} {tuple(v.shape)}")

    print("=" * 78)
    print("[2] o3.Linear instructions (path_weight/has_weight)")
    for lin_name, lin in [("potential_embedding.potential_linear",
                           b.potential_embedding.potential_linear),
                          ("potential_embedding.node_feats_linear",
                           b.potential_embedding.node_feats_linear),
                          ("potential_embedding.charge_embedding",
                           b.potential_embedding.charge_embedding),
                          ("source_embedding", b.source_embedding),
                          ("readout.linear_1", b.readout.linear_1),
                          ("readout.linear_mid", b.readout.linear_mid),
                          ("readout.linear_2", b.readout.linear_2)]:
        print(f"  {lin_name}: {lin.irreps_in} -> {lin.irreps_out}")
        for ins in lin.instructions:
            fields = {a: getattr(ins, a) for a in
                      ("i_in", "i_out", "path_weight", "path_shape", "has_weight",
                       "connection_mode") if hasattr(ins, a)}
            print(f"      {fields}")

    print("=" * 78)
    print("[3] SparseUvuTensorProduct")
    for sp_name, sp in [("dot_products", b.dot_products), ("tp_out", b.tp_out)]:
        print(f"  {sp_name}: layout={sp.layout} in1={sp.irreps_in1} in2={sp.irreps_in2} out={sp.irreps_out}")
        print(f"      weight {tuple(sp.weight.shape)} output_mask {tuple(sp.output_mask.shape)} "
              f"weight_numel={sp.weight_numel}")
        for ins in sp.instructions:
            fields = {a: getattr(ins, a) for a in
                      ("i_in1", "i_in2", "i_out", "connection_mode", "has_weight",
                       "path_weight", "path_shape") if hasattr(ins, a)}
            print(f"      {fields}")
        print(f"      _path_meta: {getattr(sp, '_path_meta', None)}")

    print("=" * 78)
    print("[4] readout gate (e3nn.nn.Gate)")
    gate = b.readout.equivariant_nonlin
    print(f"  class={type(gate).__module__}.{type(gate).__name__}")
    print(f"  irreps_scalars={gate.irreps_scalars} irreps_gates={gate.irreps_gates} "
          f"irreps_gated={gate.irreps_gated}")
    print(f"  irreps_in={gate.irreps_in} irreps_out={gate.irreps_out}")
    print(f"  state_dict={ {k: tuple(v.shape) for k, v in gate.state_dict().items()} }")
    print(f"  children={[(n, type(c).__name__) for n, c in gate.named_children()]}")
    sc = gate.sc
    print(f"  sc(_Sortcut): irreps_outs={sc.irreps_outs} cut.instr={sc.cut.instructions}")
    for aname, act in [("act_scalars", gate.act_scalars), ("act_gates", gate.act_gates)]:
        acts = [getattr(a, "__name__", type(a).__name__) for a in act.acts]
        csts = [float(getattr(a, "cst", 1.0)) for a in act.acts]
        idflags = [bool(getattr(a, "_is_id", True)) for a in act.acts]
        print(f"  {aname}: acts={acts} cst={csts} is_id={idflags} irreps={act.irreps_in}->{act.irreps_out}")
    mul = gate.mul
    print(f"  mul: {type(mul).__name__} in1={mul.irreps_in1} in2={mul.irreps_in2} out={mul.irreps_out}")
    print(f"      weight {tuple(mul.weight.shape)} output_mask={tuple(mul.output_mask.shape)} "
          f"weight_numel={mul.weight_numel}")
    for ins in mul.instructions:
        fields = {a: getattr(ins, a) for a in
                  ("i_in1", "i_in2", "i_out", "connection_mode", "has_weight",
                   "path_weight", "path_shape") if hasattr(ins, a)}
        print(f"      {fields}")

    print("=" * 78)
    print("[5] nonlinearity (RadialMLP net) 逐层")
    net = b.nonlinearity.net
    for i, layer in enumerate(net):
        info = f"  [{i}] {type(layer).__name__}"
        if hasattr(layer, "weight"):
            info += f" weight={tuple(layer.weight.shape)} bias={None if layer.bias is None else tuple(layer.bias.shape)}"
        if isinstance(layer, torch.nn.LayerNorm):
            info += (f" eps={layer.eps} normalized_shape={tuple(layer.normalized_shape)}"
                     f" elementwise_affine={layer.elementwise_affine}")
        print(info)
    print(f"  hs={b.nonlinearity.hs}")

    print("=" * 78)
    print("[6] JSON leaf <-> live module 对账")
    jleaves = leaf_paths(jb)
    # live: state_dict keys mapped to JSON-ish paths
    live = {}
    for k, v in b.state_dict().items():
        live[k] = tuple(v.shape)
    print(f"  live state_dict keys: {sorted(live.keys())}")
    print(f"  JSON top-level keys: {sorted(jb.keys())}")
    # 非 state_dict 前向依赖 (live 独有):
    print("  live-only (非 state_dict 前向量):")
    print("    potential_irreps =", b.potential_irreps)
    print("    charges_irreps   =", b.charges_irreps)
    print("    node_feats_irreps=", b.node_feats_irreps)
    print("    node_attrs_irreps=", b.node_attrs_irreps)
    print("    invar_irreps (nonlin 输入半宽) =", b.source_embedding.irreps_out)
    print("    field_norm_factor =", float(b.field_norm_factor))
    print("    readout.irreps_nonlin =", str(b.readout.irreps_nonlin))
    print("    readout.hidden_irreps =", str(b.readout.hidden_irreps))
    print("    readout.irreps_out =", str(b.readout.irreps_out))
    print("    layout wrappers:",
          "_potential_to_mul_ir", getattr(b.potential_embedding, "_potential_to_mul_ir", "MISSING"),
          "| _node_feats_to_mul_ir", getattr(b.potential_embedding, "_node_feats_to_mul_ir", "MISSING"),
          "| _charges_to_mul_ir", getattr(b.potential_embedding, "_charges_to_mul_ir", "MISSING"),
          "| _node_feats_from_mul_ir", getattr(b.potential_embedding, "_node_feats_from_mul_ir", "MISSING"))
    print("    readout layout wrappers:",
          "_readout_to_mul_ir", getattr(b, "_readout_to_mul_ir", "MISSING"),
          "| _readout_from_mul_ir", getattr(b, "_readout_from_mul_ir", "MISSING"))
    print("    cueq_config =", getattr(b, "cueq_config", "MISSING"))
    print("    radial_MLP attr =", getattr(b, "radial_MLP", "MISSING"))
    print("    instance __dict__ keys =", sorted(vars(b).keys()))
    print("  JSON leaves (前 120):")
    for p in sorted(jleaves)[:120]:
        print("    ", p)

    print("=" * 78)
    print("[7] JSON 记录的 nonlinearity layers vs live net")
    for i, (jl, ll) in enumerate(zip(jb["nonlinearity"]["layers"], net)):
        print(f"  [{i}] json={jl.get('type')} live={type(ll).__name__}"
              + (f"  raw_keys={list(jl.get('raw_state_dict', {}).keys())}" if "raw_state_dict" in jl else ""))

    print("=" * 78)
    print("[8] field_norm_factor / edge 输入使用检查 (AST 函数体, 不含签名)")
    import ast
    import inspect
    import textwrap

    import mace.modules.field_blocks as fb

    def body_identifiers(fn):
        fn = inspect.unwrap(fn)
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = tree.body[0]
        node.args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                                  kw_defaults=[], defaults=[])
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
        return names

    ids = body_identifiers(fb.AgnosticEmbeddedOneBodyVariableUpdate.forward)
    for tok in ("field_norm_factor", "edge_attrs", "edge_feats", "edge_index",
                "avg_num_neighbors", "node_attrs", "local_charges",
                "potential_features"):
        print(f"  forward body uses '{tok}':", tok in ids)
    pe_ids = body_identifiers(fb.AgnosticChargeBiasedLinearPotentialEmbedding.forward)
    print("  potential_embedding.forward body uses 'node_attrs':", "node_attrs" in pe_ids)

    print("=" * 78)
    print("[9] JSON 摘要")
    print("  field_norm_factor =", jb["field_norm_factor"])
    print("  readout keys =", sorted(jb["readout"].keys()))
    print("  readout.activation =", jb["readout"]["activation"],
          "activation_constants =", jb["readout"]["activation_constants"])
    print("  readout_gate keys =", {k: (list(v.keys()) if isinstance(v, dict) else v)
                                    for k, v in jb["readout_gate"].items()})
    print("  dot_products instr[0] json =", jb["dot_products"]["instructions"][0])
    print("  tp_out instr[0] json =", jb["tp_out"]["instructions"][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
