#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 模块⑦ local_electron_energy 公式预检 (2026-10-15)。

双层复现:
  A) 1e-12 级: 以 JSON polar_blocks.local_electron_energy 为唯一参数源, 纯 numpy
     重放 OneBodyMLPFieldReadout 前向链, 对照活体 mace_polar 前向 hook 捕获的
     逐级中间量(q_up / v_up / dot_q / dot_v / invar / mlp 各层 / local_q_e)。
  B) fixture 一致性: le_total(每图求和) vs bench/polar_fixtures/index.json 的
     electron_energy(逐构型)。

语义锚定(mace 0.3.16 安装副本 /opt/matos-tools/.../mace, 行号实测):
  - 调用点 extensions.py L867-878: local_electron_energy(node_attrs, node_feats,
    edge_attrs[:, :from_ell_max_field_update], edge_feats, edge_index,
    field_feats=potential_features, charges_0=field_independent...view(N,-1),
    charges_induced=spin_charge_density.view(N,-1));
  - le_total = scatter_sum(local_q_e, batch) L879-881; 能量计入受
    configs.add_local_electron_energy 门控 L882-885;
  - **无按原子数/分子数缩放**: 总量 = 逐原子 readout 的图内和;
    calculators/mace.py L689 只做 energy_units_to_eV 单位映射(=1.0, 已实测);
  - 输入语义(重要):
      charges_0      = field_independent_spin_charge_density (extensions L774,
                       初 Fukui 校正后、SCF 起点; 非末态);
      charges_induced = SCF 末态 spin_charge_density (L777-864 原地更新两次);
      field_feats    = potential_features = 最后一次 SCF 步的势特征
                       (L818-820, cat(α,β); 由"进入该步的密度"算出, 不是从
                       末态电荷重算);
  - field_feats = (field_σ + half_external_field) / field_feature_norms
    (L811-816; 生产 external field=0 ⇒ half_external_field=0, 只剩归一);
  - 模块本体 field_blocks.py L652-752 OneBodyMLPFieldReadout:
      q_in = charges_induced + charges_0                       (L729)
      q_up = linear_up_q(q_in), v_up = linear_up_v(field_feats) (L735/L742)
      invar = cat(dot_products_q(node_feats, q_up),
                  dot_products_v(node_feats, v_up))            (L745-751)
      return mlp(invar).squeeze(-1)                             (L752)
    RadialMLP = Linear→LayerNorm→SiLU→…→Linear (radial.py L361-384);
    e3nn o3.Linear: bias 仅标量输出块 (_linear.py L177-186), path_weight=
    alpha^{-1/2}(element 归一, L119-144); SparseUvuTensorProduct 见 L294-466。

落盘: bench/polar_fixtures/le_hooks/{mol}-f{k}.npz + index.json(引擎逐级门夹具)。

用法: cd mlip; CUDA_VISIBLE_DEVICES=1 /workspace/caches/conda/envs/polar/bin/python \
        bench/polar_module7_preflight.py [--json-audit 1] [--no-save 0]
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
import mace_md as mm  # noqa: E402

POLAR_MODEL = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"
PROBE_JSON = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
FIXROOT = f"{ROOT}/bench/polar_fixtures"
OUTDIR = f"{FIXROOT}/le_hooks"
MOLS = [("NH3", 770600), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]
REPLAY_GATE = 1e-12     # numpy 重放 vs 活体: 模块输出(逐原子 local_q_e/le_total)绝对
REPLAY_REL_GATE = 1e-14  # 中间级标度感知(高幅值中间量允许 f64 算术序漂移; 实测 ≤1.7e-15)
FIXTURE_GATE = 1e-9     # 活体值 vs 既有 fixtures/index.json(同模型同构型)
DEEP_GATE = 1e-12       # 与 scf/scf_steps 夹具的交叉核对

MLP_LAYER_NAMES = [f"mlp_h{i}" for i in range(10)]


# --------------------------------------------------------------------------
# 极简 irreps 解析 + numpy 原语(与 e3nn/mace 语义逐式对应)
# --------------------------------------------------------------------------
def parse_irreps(spec):
    """'512x0e+512x1o' -> [(512,0,'e'), (512,1,'o')]"""
    blocks = []
    for tok in spec.split("+"):
        tok = tok.strip()
        if "x" in tok:
            mul_s, rest = tok.split("x")
            mul = int(mul_s)
        else:
            mul, rest = 1, tok
        l = int(rest[:-1])
        blocks.append((mul, l, rest[-1]))
    return blocks


def irrep_slices(blocks):
    """[(mul,l,parity)] -> [(start, stop, mul, dim)]"""
    out, off = [], 0
    for mul, l, _p in blocks:
        dim = 2 * l + 1
        out.append((off, off + mul * dim, mul, dim))
        off += mul * dim
    return out


def np_e3nn_linear_json(x, lin):
    """重放 e3nn o3.Linear(依据 _linear.py::_codegen_linear L345-470)。

    - 每条 i_in>=0 指令: w=path_shape �reshape(mul_in,mul_out);
      out[b,w,i] += path_weight·Σ_u W[u,w]·x[b,u,i](L431-437);
    - bias 指令(i_in==-1): path_weight·b 广播加到对应输出块(L421-424);
      仅标量输出块有 bias(构造期 L177-186)。
    """
    blocks_in = parse_irreps(lin["irreps_in"])
    blocks_out = parse_irreps(lin["irreps_out"])
    sl_in = irrep_slices(blocks_in)
    sl_out = irrep_slices(blocks_out)
    weight = np.asarray(lin["weight"]["values"], dtype=np.float64)
    bias = np.asarray(lin["bias"]["values"], dtype=np.float64) if lin["bias"] else np.zeros(0)
    out_blocks = [np.zeros((x.shape[0], mul, dim), dtype=np.float64)
                  for (_a, _b, mul, dim) in sl_out]
    w_off = b_off = 0
    for ins in lin["instructions"]:
        i_in = ins["i_in"]
        i_out = ins["i_out"]
        pw = float(ins["path_weight"])
        pshape = list(ins["path_shape"])
        if i_in == -1:
            b = bias[b_off:b_off + pshape[0]]
            b_off += pshape[0]
            out_blocks[i_out] += (pw * b).reshape(1, pshape[0], 1)
            continue
        mul_in, dim_in = sl_in[i_in][2], sl_in[i_in][3]
        mul_out, dim_out = sl_out[i_out][2], sl_out[i_out][3]
        n_w = pshape[0] * pshape[1]
        W = weight[w_off:w_off + n_w].reshape(pshape[0], pshape[1])
        w_off += n_w
        xb = x[:, sl_in[i_in][0]:sl_in[i_in][1]].reshape(-1, mul_in, dim_in)
        # out[b,w,i] = pw · Σ_u W[u,w]·x[b,u,i]
        out_blocks[i_out] += pw * np.einsum("uw,bui->bwi", W, xb)
    return np.concatenate([b.reshape(x.shape[0], -1) for b in out_blocks], axis=-1)


def np_sparse_uvu_json(x1, x2, tp, path_weights):
    """重放 SparseUvuTensorProduct.forward(field_blocks.py L405-466)。

    mode0 (l×l→0): pair[b,u,v] = Σ_d x1[b,u,d]x2[b,v,d]/√d1;
                   mixed[b,u]  = Σ_v pair·W[u,v]; out += path_weight·mixed (L446-453);
    mode1 (l×0→l): scalars=x2; mixed=Σ_v scalars·W[u,v];
                   out += path_weight·x1·mixed/√d1                     (L454-464);
    最后 out *= output_mask (L466)。
    path_weights: JSON 尚未序列化 → 由活体模块提供(缺口 G2, 见审计)。
    """
    b1 = parse_irreps(tp["irreps_in1"])
    b2 = parse_irreps(tp["irreps_in2"])
    bo = parse_irreps(tp["irreps_out"])
    s1, s2, so = irrep_slices(b1), irrep_slices(b2), irrep_slices(bo)
    weight = np.asarray(tp["weight"]["values"], dtype=np.float64)
    mask = np.asarray(tp["output_mask"]["values"], dtype=np.float64)
    out = np.zeros((x1.shape[0], sum(mul * dim for _a, _b, mul, dim in so)),
                   dtype=np.float64)
    w_off = 0
    for idx, ins in enumerate(tp["instructions"]):
        i1, i2, io = ins["i_in1"], ins["i_in2"], ins["i_out"]
        pshape = list(ins["path_shape"])
        pw = float(path_weights[idx])
        mul1, d1 = s1[i1][2], s1[i1][3]
        mul2, d2 = s2[i2][2], s2[i2][3]
        _o0, _o1, mul_o, d_out = so[io]
        n_w = pshape[0] * pshape[1]
        W = weight[w_off:w_off + n_w].reshape(pshape[0], pshape[1])
        w_off += n_w
        x1v = x1[:, s1[i1][0]:s1[i1][1]].reshape(-1, mul1, d1)
        x2v = x2[:, s2[i2][0]:s2[i2][1]].reshape(-1, mul2, d2)
        if d_out == 1 and d1 == d2 and d1 > 0:  # mode 0
            pair = np.einsum("bud,bvd->buv", x1v, x2v) / math.sqrt(float(d1))
            mixed = np.einsum("buv,uv->bu", pair, W)
            out[:, so[io][0]:so[io][1]] += pw * mixed
        elif d2 == 1 and d_out == d1:           # mode 1
            scalars = x2v.reshape(x2.shape[0], mul2)
            mixed = np.einsum("bv,uv->bu", scalars, W)
            contrib = pw * (x1v * mixed[:, :, None] / math.sqrt(float(d1)))
            out[:, so[io][0]:so[io][1]] += contrib.reshape(x1.shape[0], mul_o * d_out)
        else:
            raise RuntimeError(f"numpy replay: unsupported uvu path {ins}")
    return out * mask


def np_radial_mlp_json(x, layers_json, linenorm_eps):
    """重放 RadialMLP(torch Sequential: Linear/LayerNorm/SiLU; radial.py L361-384)。"""
    h = x
    stages = []
    for layer in layers_json:
        t = layer["type"]
        if t == "linear":
            W = np.asarray(layer["weight"]["values"], dtype=np.float64).reshape(
                layer["weight"]["shape"])
            b = np.asarray(layer["bias"]["values"], dtype=np.float64)
            h = h @ W.T + b
        elif t in ("SiLU", "silu"):
            h = h / (1.0 + np.exp(-h))
        elif t == "LayerNorm":  # JSON raw 缺口 G1: weight/bias 在 raw_state_dict
            sd = layer["raw_state_dict"]
            g = np.asarray(sd["weight"]["values"], dtype=np.float64)
            b = np.asarray(sd["bias"]["values"], dtype=np.float64)
            mu = h.mean(axis=-1, keepdims=True)
            var = h.var(axis=-1, keepdims=True)  # torch LayerNorm = 有偏方差
            h = (h - mu) / np.sqrt(var + linenorm_eps) * g + b
        else:
            raise RuntimeError(f"numpy replay: unsupported mlp layer {t}")
        stages.append(h.copy())
    return h, stages


# --------------------------------------------------------------------------
# 残差报告 + 序列化审计
# --------------------------------------------------------------------------
def residual(a, b):
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    ref_absmax = float(np.abs(np.asarray(b, dtype=np.float64)).max())
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "ref_absmax": ref_absmax,
        "rel_max_vs_ref_absmax": float(d.max() / max(ref_absmax, 1e-300)),
        "shape": list(np.shape(a)),
    }


def audit_serialization(le_json, model):
    """JSON polar_blocks.local_electron_energy vs 参考模块 state_dict/前向需求。

    返回 {"gaps": [ {id,json_path,现状,引擎后果,补丁} ], "tensor_checks": [...],
          "derived": {...}}; tensor_checks 里 max_abs=0 表示 JSON 与活体逐位一致。
    """
    sd = {k: v.detach().cpu().numpy() for k, v in
          model.local_electron_energy.state_dict().items()}
    checks, gaps = [], []

    def cmp_tensor(json_path, jtensor, ref_name):
        if jtensor is None:
            checks.append({"json_path": json_path, "status": "absent"})
            return
        v = np.asarray(jtensor["values"], dtype=np.float64).reshape(
            jtensor["shape"] or ())
        ref = sd[ref_name]
        checks.append({
            "json_path": json_path,
            "shape_json": list(jtensor["shape"]), "shape_ref": list(ref.shape),
            "max_abs": float(np.abs(v - ref).max()) if v.shape == ref.shape else None,
        })

    for lin in ("linear_up_q", "linear_up_v"):
        jpath = f"polar_blocks.local_electron_energy.{lin}"
        cmp_tensor(f"{jpath}.weight", le_json[lin]["weight"], f"{lin}.weight")
        cmp_tensor(f"{jpath}.bias", le_json[lin]["bias"], f"{lin}.bias")
        cmp_tensor(f"{jpath}.output_mask", le_json[lin]["output_mask"],
                   f"{lin}.output_mask")
        # G4: e3nn Linear bias 仅标量输出块(512), 引擎 E3Linear 要求
        # bias.size()==output.dimension()(2048) → 现状会装载即抛。
        lout_dim = irrep_slices(parse_irreps(le_json[lin]["irreps_out"]))[-1][1]
        if le_json[lin]["bias"] and le_json[lin]["bias"]["shape"] != [lout_dim]:
            gaps.append({
                "id": "G4",
                "json_path": f"{jpath}.bias",
                "现状": f"bias shape={le_json[lin]['bias']['shape']} (仅标量输出块, "
                        f"e3nn _linear.py L177-186); 输出维度={lout_dim}",
                "引擎后果": "mace_nonlinear e3nn.cpp::E3Linear ctor "
                            "validate(bias.size()==output.dimension()) 直接抛 "
                            "'linear bias has an invalid size'",
                "补丁": "extract_mace_nonlinear._linear(): 若 bias numel<输出维度, "
                        "按 bias 指令(i_in==-1)的 i_out 块偏移补齐为满维(其余置 0)再序列化。",
            })
    for tp in ("dot_products_q", "dot_products_v"):
        jpath = f"polar_blocks.local_electron_energy.{tp}"
        cmp_tensor(f"{jpath}.weight", le_json[tp]["weight"], f"{tp}.weight")
        cmp_tensor(f"{jpath}.output_mask", le_json[tp]["output_mask"],
                   f"{tp}.output_mask")
        ins = le_json[tp]["instructions"]
        if any("path_weight" not in i for i in ins):
            live_pw = [float(m[8]) for m in
                       getattr(model.local_electron_energy, tp)._path_meta]
            gaps.append({
                "id": "G2",
                "json_path": f"{jpath}.instructions[*].path_weight",
                "现状": f"missing; 活体真值={live_pw} (=sqrt(1/1024)=1/32, e3nn "
                        f"_tensor_product.py L287-317 element 归一)",
                "引擎后果": "e3nn.cpp::parse_instruction L33 value('path_weight',1.0) "
                            "默认 1.0 → 内积路径整体 32× 标度错误",
                "补丁": "extract_polar_blocks._sparse_uvu(): 每条指令增加 "
                        "\"path_weight\": float(ins.path_weight)。",
            })
        if any("wigner_3j" not in i for i in ins):
            gaps.append({
                "id": "G3",
                "json_path": f"{jpath}.instructions[*].wigner_3j",
                "现状": "missing",
                "引擎后果": "e3nn.cpp::parse_instruction L40-41 缺 wigner_3j 直接抛 "
                            "('tensor-product instruction is missing wigner_3j')",
                "补丁": "extract_polar_blocks._sparse_uvu(): 对每条指令补 "
                        "o3.wigner_3j(l1,l2,l3) 张量(照抄 _tensor_product() L122-133)。"
                        "已验证: 3j(0,0,0)=[[[1]]]; 3j(1,1,0)=δ_ab/√3, 与 python "
                        "SparseUvu 的 x1·x2/√d1 收缩逐位等价。",
            })
    # G1: mlp LayerNorm raw
    mlp_layers = le_json["mlp"]["layers"]
    raw_ln = [i for i, l in enumerate(mlp_layers) if l.get("type") == "LayerNorm"]
    for i in raw_ln:
        sdv = mlp_layers[i]["raw_state_dict"]
        cmp_tensor("polar_blocks.local_electron_energy.mlp.layers[%d]"
                   ".raw_state_dict.weight" % i, sdv["weight"],
                   "mlp.net.%d.weight" % i)
        checks[-1]["note"] = "raw_state_dict(LayerNorm)"
        cmp_tensor("polar_blocks.local_electron_energy.mlp.layers[%d]"
                   ".raw_state_dict.bias" % i, sdv["bias"], "mlp.net.%d.bias" % i)
    if raw_ln:
        gaps.append({
            "id": "G1",
            "json_path": "polar_blocks.local_electron_energy.mlp.layers[%d]"
                         % raw_ln[0],
            "现状": f"LayerNorm(layers {raw_ln}) 以 {{'type':'LayerNorm', "
                    f"'raw_state_dict':{{weight,bias}}}} 落盘 —— eps 未序列化",
            "引擎后果": "AffineMLP(Kokkos) 只认 {{\"type\":\"layer_norm\", "
                        "\"normalized_shape\",\"eps\",\"weight\",\"bias\"}}; "
                        "现 schema 直接抛 'unsupported layer type LayerNorm'",
            "补丁": "extract_polar_blocks._torch_mlp_net(): 增加 LayerNorm 分支, "
                    "输出 AffineMLP 结构化 schema(normalized_shape=layer."
                    "normalized_shape, eps=float(layer.eps), weight/bias 张量); "
                    "与 extract_mace_nonlinear._affine_mlp() L148-157 同格式。",
        })
    for i, l in enumerate(mlp_layers):
        t = l.get("type")
        if t == "linear":
            cmp_tensor("polar_blocks.local_electron_energy.mlp.layers[%d].weight" % i,
                       l["weight"], "mlp.net.%d.weight" % i)
            cmp_tensor("polar_blocks.local_electron_energy.mlp.layers[%d].bias" % i,
                       l["bias"], "mlp.net.%d.bias" % i)
    derived = {
        "ln_eps_live": [float(model.local_electron_energy.mlp.net[i].eps)
                        for i in raw_ln],
        "dot_path_weight_live": [
            [float(m[8]) for m in getattr(model.local_electron_energy, tp)._path_meta]
            for tp in ("dot_products_q", "dot_products_v")],
        "engine_bias_requirement": "bias.size()==output.dimension()",
        "mlp_keys": sorted(le_json["mlp"].keys()),
    }
    return {"gaps": gaps, "tensor_checks": checks, "derived": derived}


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json-audit", type=int, default=1)
    ap.add_argument("--no-save", type=int, default=0)
    ap.add_argument("--cases", default="", help="逗号分隔子集(默认全部 8 构型)")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    with open(f"{FIXROOT}/index.json") as fh:
        fixtures = json.load(fh)

    print("[load] probe JSON (polar_blocks 契约)...", flush=True)
    probe = json.load(open(PROBE_JSON))
    pb = probe["polar_blocks"]
    le_json = pb["local_electron_energy"]
    norms = np.asarray(pb["configs"]["field_feature_norms"]["values"],
                       dtype=np.float64)
    add_le = bool(pb["configs"]["add_local_electron_energy"])
    print(f"[load] add_local_electron_energy={add_le}  "
          f"field_feature_norms={norms.tolist()}  ({time.time()-t0:.0f}s)", flush=True)

    from mace.calculators import mace_polar
    calc = mace_polar(model=POLAR_MODEL, device=args.device,
                      default_dtype="float64")
    model = calc.models[0]
    le = model.local_electron_energy
    print(f"[model] {type(model).__name__} le={type(le).__name__}  "
          f"params={sum(p.numel() for p in le.parameters())}  "
          f"energy_units_to_eV={calc.energy_units_to_eV}", flush=True)

    audit = None
    if args.json_audit:
        audit = audit_serialization(le_json, model)
        print(f"[audit] tensor_checks={len(audit['tensor_checks'])}  "
              f"gaps={[g['id'] for g in audit['gaps']]}", flush=True)
        for c in audit["tensor_checks"]:
            if c.get("status") == "absent" or c.get("max_abs") is None:
                print(f"    MISSING {c['json_path']}", flush=True)
            elif c["max_abs"] != 0.0:
                print(f"    Δ={c['max_abs']:.3e}  {c['json_path']}", flush=True)
        for g in audit["gaps"]:
            print(f"    GAP {g['id']}: {g['json_path']}", flush=True)

    # ---------------- 活体捕获 ----------------
    caps = {}
    handles = []

    def hook_kwargs(name):
        def fn(module, hook_args, kwargs, output):
            caps[name] = {k: (v.detach().cpu().numpy().copy()
                              if torch.is_tensor(v) else None)
                          for k, v in kwargs.items()}
            caps[name + "_out"] = (output.detach().cpu().numpy().copy()
                                   if torch.is_tensor(output) else output)
            # 必须返回 None(返回非 None 会顶替模块输出!)
        return fn

    def hook_out(name):
        def fn(module, hook_args, output):
            if torch.is_tensor(output):
                arr = output.detach().cpu().numpy().copy()
            elif isinstance(output, tuple) and torch.is_tensor(output[0]):
                arr = output[0].detach().cpu().numpy().copy()
            else:
                arr = np.asarray(output)
            caps[name] = arr
            # 返回 None(勿顶替)
        return fn

    handles.append(le.register_forward_hook(hook_kwargs("le"), with_kwargs=True))
    for sub in ("linear_up_q", "linear_up_v", "dot_products_q", "dot_products_v"):
        handles.append(getattr(le, sub).register_forward_hook(hook_out(sub)))
    for i in range(len(MLP_LAYER_NAMES)):
        handles.append(le.mlp.net[i].register_forward_hook(hook_out(MLP_LAYER_NAMES[i])))

    cases = {}
    rows = []
    subset = set(args.cases.split(",")) if args.cases else None
    worst = {"replay": 0.0, "out": 0.0, "fixture": 0.0, "deep": 0.0}

    for mol, seed in MOLS:
        base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
        for k in range(2):
            name = f"{mol.lower()}-f{k}"
            if subset and name not in subset:
                continue
            atoms = base.copy()
            if k == 1:
                atoms.rattle(stdev=0.05, seed=seed + 7)
            atoms.info["charge"] = 0
            atoms.info["spin"] = 1
            atoms.calc = calc
            caps.clear()
            e = atoms.get_potential_energy()
            r = calc.results
            assert "le" in caps and "le_out" in caps, "le hook did not fire"
            for stag in ("linear_up_q", "linear_up_v", "dot_products_q",
                         "dot_products_v", *MLP_LAYER_NAMES):
                assert stag in caps, f"stage hook missing: {stag}"
            kw = caps["le"]
            (node_attrs, node_feats, edge_attrs, edge_feats, edge_index,
             field_feats, charges_0, charges_induced) = (
                kw["node_attrs"], kw["node_feats"], kw["edge_attrs"],
                kw["edge_feats"], kw["edge_index"], kw["field_feats"],
                kw["charges_0"], kw["charges_induced"])
            le_live = caps["le_out"]

            # ---------- numpy 重放(JSON 为唯一参数源) ----------
            q_in = charges_induced + charges_0
            q_up = np_e3nn_linear_json(q_in, le_json["linear_up_q"])
            v_up = np_e3nn_linear_json(field_feats, le_json["linear_up_v"])
            pw_q = [float(m[8]) for m in le.dot_products_q._path_meta]
            pw_v = [float(m[8]) for m in le.dot_products_v._path_meta]
            dot_q = np_sparse_uvu_json(node_feats, q_up, le_json["dot_products_q"], pw_q)
            dot_v = np_sparse_uvu_json(node_feats, v_up, le_json["dot_products_v"], pw_v)
            invar = np.concatenate([dot_q, dot_v], axis=-1)
            ln_eps = float(le.mlp.net[1].eps)
            le_np, mlp_stages = np_radial_mlp_json(invar, le_json["mlp"]["layers"],
                                                   ln_eps)
            le_np_atom = le_np[:, 0] if le_np.ndim == 2 else le_np

            stage_cmp = {
                "q_up": residual(q_up, caps["linear_up_q"]),
                "v_up": residual(v_up, caps["linear_up_v"]),
                "dot_q": residual(dot_q, caps["dot_products_q"]),
                "dot_v": residual(dot_v, caps["dot_products_v"]),
                "invar": residual(invar, np.concatenate(
                    [caps["dot_products_q"], caps["dot_products_v"]], axis=-1)),
                **{MLP_LAYER_NAMES[i]: residual(mlp_stages[i], caps[MLP_LAYER_NAMES[i]])
                   for i in range(len(MLP_LAYER_NAMES))},
                "local_q_e": residual(le_np_atom, le_live),
            }
            replay_max = max(v["max_abs"] for v in stage_cmp.values())
            # 判据: 模块输出(逐原子) 绝对 ≤1e-12; 中间级 abs ≤ max(1e-12, 1e-14·ref_absmax)
            # (高幅值中间量 ~2.6e4 处 1e-12 绝对 = 4e-17 相对, 低于 f64 eps; 实测
            #  各中间级相对 ≤1.7e-15, 为 cublas/BLAS 累加次序噪声)
            replay_ok = stage_cmp["local_q_e"]["max_abs"] <= REPLAY_GATE and all(
                v["max_abs"] <= max(REPLAY_GATE, REPLAY_REL_GATE * v["ref_absmax"])
                for v in stage_cmp.values())
            worst_stage = max(stage_cmp.items(), key=lambda kv: kv[1]["max_abs"])[0]

            # ---------- fixture 对照 ----------
            fix_val = float(fixtures[name]["electron_energy"])
            live_val = float(r["electron_energy"])
            le_total_np = float(le_np_atom.sum())
            fixture_res = {
                "live_vs_fixture_abs": abs(live_val - fix_val),
                "numpy_np_vs_live": abs(le_total_np - live_val),
                "numpy_vs_fixture_abs": abs(le_total_np - fix_val),
            }
            fx_max = max(fixture_res.values())

            # ---------- 与 scf/scf_steps 夹具交叉核对(f0) ----------
            deep = {}
            try:
                step0 = np.load(f"{FIXROOT}/scf_steps/{name}-step0.npz",
                                allow_pickle=False)
                deep["charges_0_vs_scfstep0"] = residual(
                    charges_0, np.asarray(step0["spin_charge_density_flat"]))
                step2 = np.load(f"{FIXROOT}/scf_steps/{name}-step2.npz",
                                allow_pickle=False)
                deep["charges_induced_vs_scfstep2"] = residual(
                    charges_induced, np.asarray(step2["spin_charge_density_flat"]))
            except FileNotFoundError:
                pass
            try:
                fix = np.load(f"{FIXROOT}/scf/{name}.npz", allow_pickle=False)
                raw_last = np.concatenate(
                    [np.asarray(fix["potential_feats_step_2"]),
                     np.asarray(fix["potential_feats_step_3"])], axis=-1)
                norms16 = np.tile(norms, 2)[None, :]  # α/β 各套同一组 norms
                deep["field_feats_vs_step2_3_over_norms"] = residual(
                    field_feats, raw_last / norms16)
            except FileNotFoundError:
                pass
            deep_max = max((v["max_abs"] for v in deep.values()), default=0.0)

            verdict = ("PASS" if replay_ok and fx_max <= FIXTURE_GATE
                       and deep_max <= DEEP_GATE else "FAIL")
            rows.append((name, replay_max, fx_max, deep_max, verdict))
            print(f"  {name:9s} replay max|Δ|={replay_max:.3e} (worst {worst_stage}: "
                  f"{stage_cmp[worst_stage]['max_abs']:.2e} abs / "
                  f"{stage_cmp[worst_stage]['rel_max_vs_ref_absmax']:.2e} rel; "
                  f"out {stage_cmp['local_q_e']['max_abs']:.2e})  "
                  f"fixture |Δ|={fx_max:.3e} (live {live_val:+.9f} vs "
                  f"fix {fix_val:+.9f})  deep={deep_max:.3e}  -> {verdict}",
                  flush=True)

            cases[name] = {
                "npz": f"le_hooks/{name}.npz",
                "inputs": {
                    "node_attrs": list(node_attrs.shape),
                    "node_feats": list(node_feats.shape),
                    "edge_attrs": list(edge_attrs.shape),
                    "edge_feats": list(edge_feats.shape),
                    "edge_index": list(edge_index.shape),
                    "field_feats": list(field_feats.shape),
                    "charges_0": list(charges_0.shape),
                    "charges_induced": list(charges_induced.shape),
                },
                "fixture_electron_energy": fix_val,
                "live_electron_energy": live_val,
                "numpy_le_total": le_total_np,
                "fixture_residual": fixture_res,
                "replay": {k: v for k, v in stage_cmp.items()},
                "replay_max_abs": replay_max,
                "replay_worst_stage": worst_stage,
                "replay_ok": bool(replay_ok),
                "deep_cross_check": deep,
                "verdict": verdict,
            }
            worst["replay"] = max(worst["replay"], replay_max)
            worst["out"] = max(worst["out"],
                               stage_cmp["local_q_e"]["max_abs"])
            worst["fixture"] = max(worst["fixture"], fx_max)
            worst["deep"] = max(worst["deep"], deep_max)

            if not args.no_save:
                payload = dict(
                    symbols=np.asarray(atoms.get_chemical_symbols(), dtype="U2"),
                    positions=atoms.get_positions(),
                    cell=np.asarray(atoms.get_cell()),
                    molecule_index=np.repeat(np.arange(32), len(atoms) // 32),
                    node_attrs=node_attrs.astype(np.float64),
                    node_feats=node_feats.astype(np.float64),
                    edge_attrs=edge_attrs.astype(np.float64),
                    edge_feats=edge_feats.astype(np.float64),
                    edge_index=edge_index.astype(np.int64),
                    field_feats=field_feats.astype(np.float64),
                    charges_0=charges_0.astype(np.float64),
                    charges_induced=charges_induced.astype(np.float64),
                    q_in=q_in.astype(np.float64),
                    q_up=q_up.astype(np.float64),
                    v_up=v_up.astype(np.float64),
                    dot_q=dot_q.astype(np.float64),
                    dot_v=dot_v.astype(np.float64),
                    invar=invar.astype(np.float64),
                    local_q_e=le_np_atom.astype(np.float64),
                    le_total=np.float64(le_total_np),
                    energy=np.float64(e),
                    fixture_electron_energy=np.float64(fix_val),
                )
                for i, nm in enumerate(MLP_LAYER_NAMES):
                    payload[nm] = mlp_stages[i].astype(np.float64)
                np.savez(f"{OUTDIR}/{name}.npz", **payload)

    for h in handles:
        h.remove()

    overall = "PASS" if all(r[4] == "PASS" for r in rows) else "FAIL"
    index = {
        "generator": "bench/polar_module7_preflight.py",
        "date": "2026-10-15",
        "model": POLAR_MODEL,
        "device": args.device,
        "dtype": "float64",
        "gates": {"replay_module_out_abs": REPLAY_GATE,
                  "replay_stage_abs_or_rel": [REPLAY_GATE, REPLAY_REL_GATE],
                  "live_vs_fixture": FIXTURE_GATE,
                  "deep_cross_check": DEEP_GATE},
        "semantics": {
            "module": "OneBodyMLPFieldReadout (mace field_blocks.py L652-752)",
            "call_site": "extensions.py L867-878 (PolarMACE.forward)",
            "le_total": "scatter_sum(local_q_e, batch) L879-881; 每图求和, "
                        "**无原子数/分子数缩放**; 生产计入总能量受 "
                        "configs.add_local_electron_energy 门控(L882-885)",
            "units": f"calc.results['electron_energy'] = le_total × "
                     f"energy_units_to_eV({calc.energy_units_to_eV}) "
                     f"(calculators/mace.py L689)",
            "charges_0": "field_independent_spin_charge_density (L774; 初 Fukui "
                         "校正后, 非末态)",
            "charges_induced": "SCF 末态 spin_charge_density (L777-864 原地更新)",
            "field_feats": "potential_features = 最后一次 SCF 步的势特征 (L818-820); "
                           "= (field_σ + half_ext)/field_feature_norms (L811-816), "
                           "生产 external field=0",
            "q_in": "charges_induced + charges_0 (field_blocks.py L729)",
            "forward": "q_up=linear_up_q(q_in); v_up=linear_up_v(field_feats); "
                       "invar=cat(dot_products_q(node_feats,q_up),"
                       "dot_products_v(node_feats,v_up)); return mlp(invar).squeeze(-1)",
            "flat8": "(N,8) = view(N,2,4) 行主序 [sα,xα,yα,zα,sβ,xβ,yβ,zβ]; "
                     "irreps '1x0e+1x1o+1x0e+1x1o'",
            "field_feats16": "cat(field_α(8), field_β(8)); irreps '2x0e+2x1o+2x0e+2x1o'",
            "numpy_replay_source": "JSON polar_blocks.local_electron_energy "
                                   "(dot path_weight 暂取活体 _path_meta → 缺口 G2)",
        },
        "serialization_audit": audit,
        "worst": worst,
        "cases": cases,
        "verdict": overall,
        "wall_s": round(time.time() - t0, 1),
    }
    if not args.no_save:
        with open(f"{OUTDIR}/index.json", "w") as fh:
            json.dump(index, fh, indent=1, ensure_ascii=False)

    print(f"\nGATE {overall}: cases={len(rows)}  module-out worst={worst['out']:.3e} "
          f"(gate {REPLAY_GATE:.0e})  replay-stage worst abs={worst['replay']:.3e} "
          f"(判据 max({REPLAY_GATE:.0e}, {REPLAY_REL_GATE:.0e}·ref_absmax))  "
          f"fixture worst={worst['fixture']:.3e} (gate {FIXTURE_GATE:.0e})  "
          f"deep worst={worst['deep']:.3e}", flush=True)
    if not args.no_save:
        print(f"[done] -> {OUTDIR} (npz × {len(rows)} + index.json)", flush=True)
    print("MODULE7-PREFLIGHT-DONE", flush=True)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
