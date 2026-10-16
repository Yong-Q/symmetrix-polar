# 模块⑤ `field_dependent_charges_maps` 序列化完备性审计 (2026-10-15)

对象：`polar_blocks.field_dependent_charges_maps[i]`（`AgnosticEmbeddedOneBodyVariableUpdate`，
i=0,1；两实例 JSON 键结构逐一致，数值不同）。

方法（全部实机核对，非纸面）：

- 运行时模块 = checkpoint 实载（`mace_polar(MACE-POLAR-1-M.model)` → `calc.models[0]
  .field_dependent_charges_maps[0]`），探针 `bench/polar_module5_audit_probe.py`，
  日志 `logs/polar_module5_audit_probe.log`；
- JSON = `models/mace-polar/polar_symmetrix_probe.json`（由
  `symmetrix/source/symmetrix/extract_polar_blocks.py::_field_update_block` 产出，2026-10-15 重生成）；
- 数值保真：41 个 state_dict 叶子逐一张量比对（`_tensor` values 反序列化 vs live），
  **41/41 逐位相等（max|Δ| = 0.0）** ⇒ 已序列化内容无腐化，问题只在"缺什么"；
- 前向消费点逐一回源（`field_blocks.py` / `blocks.py` / `radial.py` / e3nn `_gate.py`）。

## 前向结构（行号锚定）

```
field_dependent_charges_maps[i](node_attrs, node_feats=features_mixed,
    edge_attrs[..9], edge_feats, edge_index, potential_features, local_charges)
   [extensions.py L821-831; i=SCF 迭代序号, num_recursion_steps=2 → 每块每次 forward 恰调用 1 次]
forward (field_blocks.py L563-594) =
   mixed = potential_embedding(potential_features, node_feats, node_attrs, local_charges)  # L574-579
           = potential_linear(potential) + node_feats_linear(node_feats) + charge_embedding(charges)  # L195-203
   inv   = dot_products(node_feats, mixed)            # L580  (uvu 0e×0e→0e, 1o×1o→0e)
   se    = source_embedding(node_attrs)               # L581
   h     = nonlinearity(cat([inv, se]))               # L582-585 (RadialMLP 1024→64→64→64→512)
   f     = tp_out(node_feats, h)                      # L586  (uvu 0e×0e→0e, 1o×0e→1o)
   out   = readout(f)                                 # L591  (GeneralNonLinearBiasReadoutBlock, blocks.py L309-355)
           = linear_2( gate( linear_mid( gate( linear_1(f) ) ) ) )   # gate = e3nn.nn.Gate 调用两次
输出 (N,10) = [8 density sources | 2 fukui sources]，extensions.py L833-834 切法。
```

## 审计表（叶子 → 来源 → JSON 状态 → 结论）

| # | 前向量（叶子） | 来源 (file:line) | JSON 状态 | 结论 |
|---|---|---|---|---|
| 1 | potential_linear.weight (4096,) | field_blocks.py L124-129 | ✓ 逐位 | OK |
| 2 | potential_linear.output_mask (2048,) | 同上 | ✓ | OK |
| 3 | potential_linear.instructions ×4（path_weight=0.5，path_shape(2,512)） | e3nn o3.Linear | ✓ 含 path_weight | OK |
| 4 | potential_linear.bias (0,)（biases=False） | 同上 | ✓ shape[0] | OK |
| 5-8 | node_feats_linear.{weight(524288),mask(2048),instructions(path_weight=0.044194…),bias(0)} | L130-135 | ✓ 逐位 | OK |
| 9-12 | charge_embedding.{weight(2048),mask(2048),instructions×4(path_weight=0.70710678…),bias(0)} | L137-142 | ✓ 逐位 | OK |
| 13 | PE 布局转置 `_potential/_node_feats/_charges_to_mul_ir` = **None**（cueq 关） | wrapper_ops.py L413-434 | 无需字段 | OK（仅 cueq 开启的 ckpt 才需要 layout 元数据） |
| 14 | PE 前向 = 3 线性求和（顺序无关） | L195-203 | 不在 JSON | 引擎硬编码（同类先例：fukui 读出头） |
| 15 | dot_products.weight (524288,) | L331 | ✓ 逐位 | OK |
| 16 | dot_products.output_mask (512,) | L343 | ✓ | OK |
| 17 | dot_products.instructions（i_in1/i_in2/i_out/connection_mode='uvu'/has_weight/path_shape） | L345 | ✓（除 path_weight） | 见 #18 |
| 18 | **dot_products 指令 path_weight = 0.03125 ×2** | **前向消费：L396 → L452-453/L460-462** | **✗ 缺** | **缺口 G1** |
| 19 | dot_products.layout='mul_ir'；numerics 常数 /√d（d=1,3 由 irreps 导出） | L446-464 | ✓ / 可导出 | OK |
| 20 | source_embedding.{weight(42496),bias(0),mask(512),instructions(path_weight=0.10976425998969035)} | L491-497 | ✓ 逐位 | OK |
| 21 | nonlinearity.net.{0,3,6,9} Linear weight/bias | radial.py L361-381 | ✓ 逐位 | OK |
| 22 | net.{2,5,8} SiLU | 同上 | ✓（"silu"） | OK |
| 23 | net.{1,4,7} LayerNorm.weight / .bias | torch.nn.LayerNorm | ✓（raw_state_dict） | OK |
| 24 | **LayerNorm.eps = 1e-5（torch 默认，不在 state_dict）** | torch LayerNorm attr | **✗ 缺** | **缺口 G3** |
| 25 | tp_out.weight (524288,) / output_mask (2048,) | L521-531 | ✓ 逐位 | OK |
| 26 | **tp_out 指令 path_weight = {0.04419417382415922, 0.07654655446197431}** | **前向消费：L396 → L446-464** | **✗ 缺** | **缺口 G1** |
| 27 | readout.linear_1.{w(49152),b(0),mask(160),instr(path_weight=0.04419…)} | blocks.py L342-344 | ✓ 逐位 | OK |
| 28 | readout.linear_mid.{w(3072),b(64),mask(160),instr(含 i_in=-1 偏置路径, path_weight 0.17678…/1.0)} | L345-347 | ✓ 逐位 | OK |
| 29 | readout.linear_2.{w(192),b(4),mask(10),instr(含偏置路径)}，irreps_out=8 源+2 fukui | L348-350 | ✓ 逐位 | OK |
| 30 | gate.mul.weight (0,) / output_mask (96,) | e3nn Gate L107 | ✓（raw_state_dict） | OK |
| 31 | **gate.act_scalars = normalize2mom(silu)，cst=1.6791767923989418**（×32 标量分支） | `_gate.py` L131 + `_normalize_activation.py` L36,46-50 | **✗ 缺** | **缺口 G2** |
| 32 | **gate.act_gates = normalize2mom(sigmoid)，cst=1.8467055342154763** | `_gate.py` L133 | **✗ 缺** | **缺口 G2** |
| 33 | **gate.mul 指令：(0,0,0,'uuu',has_weight=False,path_weight=1.7320508075688772,path_shape(32,))** | `_gate.py` L107,115-138 | **✗ 缺** | **缺口 G2** |
| 34 | gate 拆装（sc/_Sortcut：64x0e+32x1o → scalars32,gates32,gated96）与输出拼接 scalars32+gated96=128 | `_gate.py` L97-138 | 可从 irreps 导出但未显式 | G2 建议结构化 |
| 35 | **readout.activation / activation_constants = None**（提取器只认 `readout.non_linearity`，GeneralNonLinearBiasReadoutBlock 无该属性） | extract_mace_nonlinear.py L91-96 | **✗ 派生失败** | **缺口 G2** |
| 36 | field_norm_factor buffer=1.0 | field_blocks.py L258 | ✓ float | OK（**forward 体内 0 处使用**=dead buffer，引擎可忽略；模型级同名属性 extensions.py L380=1.0 亦未被前向消费，未入 configs，非缺口） |
| 37 | edge_attrs / edge_feats / edge_index | 签名 L567-569 | 前向体 0 处使用 | 无需序列化（夹具仍落盘存档） |
| 38 | node_attrs | 仅 source_embedding 消费 | ✓ 83x0e 在 irreps 链中 | OK |
| 39 | 输出切法 (N,10)→[8 增量 + 2 fukui] 与密度更新 | extensions.py L833-864 | JSON 有 linear_2 irreps_out；切法为运行时语义 | 引擎硬编码（夹具已锚定，见下） |
| 40 | 输入语义：local_charges=[sα,xα,yα,zα,sβ,xβ,yβ,zβ]；potential_features=cat(α8,β8) | extensions.py L734/L818-831 | 不在 JSON | G4 语义登记 |
| 41 | configs: num_recursion_steps=2 / field_feature_norms[20,20,0.5×6] / from_ell_max_field_update=9 | extensions.py L777/L813/L824 | ✓（§16 已补） | OK |

## 缺口清单（4 项）

### G1（数值硬缺口）sparse_uvu 指令缺 `path_weight`
`SparseUvuTensorProduct` 前向对**每条路径**乘 `float(ins.path_weight)`（field_blocks.py L396 →
`_path_meta`；L452-453 模式 0、L460-462 模式 1）。JSON `_sparse_uvu` 只存了
i_in1/i_in2/i_out/connection_mode/has_weight/path_shape，**无 path_weight**。
实测值：dot_products {0.03125, 0.03125}；tp_out {0.04419417382415922, 0.07654655446197431}。
（可解析重算：`sqrt(dim(ir_out) / Σ_{同 i_out 路径} num_elements)`，irrep_normalization='component'
+ path_normalization='element'；dot_products: √(1/1024)；tp_out: √(1/512)、√(3/512)。但重算要求引擎
精确复刻 e3nn 指令枚举，不如显式序列化稳。）

**补丁（不在本任务施行，只给指令）**：`symmetrix/source/symmetrix/extract_polar_blocks.py`
→ `_sparse_uvu()`，指令循环内每条加 `"path_weight": float(ins.path_weight)`（或直接改用
`extract_mace_nonlinear._instructions(tp)`——该 helper 含 path_weight/has_weight/path_shape/connection_mode）。

### G2（数值硬缺口）readout gate（e3nn.nn.Gate）序列化不完整
`readout_gate` 目前只有 `raw_state_dict = {mul.weight(0,), mul.output_mask(96,)}`；前向还消费：
- `act_scalars`：normalize2mom(silu) **cst=1.6791767923989418**（同 fukui 读出头已序列化值）；
- `act_gates`：normalize2mom(sigmoid) **cst=1.8467055342154763**（**新常数，当前 JSON 无任何记载**）；
- `mul`（ElementwiseTensorProduct, uuu）指令：has_weight=False, path_weight=1.7320508075688772, path_shape=(32,)；
- 结构元数据：irreps_scalars=32x0e / irreps_gates=32x0e / irreps_gated=32x1o / in=64x0e+32x1o / out=32x0e+32x1o；
- 同时修 `readout.activation/activation_constants`（现 None）：对 GeneralNonLinearBiasReadoutBlock 从
  `equivariant_nonlin.act_scalars/act_gates` 回退派生（名称 + cst 列表）。

**补丁**：`extract_polar_blocks.py::_field_update_block()`，替换 readout_gate 段为结构化
（需在文件头 import 里补 `_instructions`，来自 `.extract_mace_nonlinear`）：
```python
g = b.readout.equivariant_nonlin
out["readout_gate"] = {
    "kind": type(g).__name__,
    "irreps_scalars": str(g.irreps_scalars), "irreps_gates": str(g.irreps_gates),
    "irreps_gated": str(g.irreps_gated),
    "irreps_in": str(g.irreps_in), "irreps_out": str(g.irreps_out),
    "sc_instructions": [list(i) for i in g.sc.cut.instructions],
    "act_scalars": [{"name": getattr(a, "__name__", type(a).__name__),
                     "cst": float(getattr(a, "cst", 1.0))} for a in g.act_scalars.acts],
    "act_gates":   [ ... 同上 ... ],
    "mul": {"kind": type(g.mul).__name__, "irreps_in1": str(g.mul.irreps_in1),
            "irreps_in2": str(g.mul.irreps_in2), "irreps_out": str(g.mul.irreps_out),
            "weight": _tensor(g.mul.weight), "output_mask": _tensor(g.mul.output_mask),
            "instructions": _instructions(g.mul)},   # 含 path_weight/has_weight
}
```
（act 名：e3nn 的 acts 是 `normalize2mom` 包装，故 name 直接取 `__name__` = 'normalize2mom'，
建议同时存 `"act": "silu"/"sigmoid"` 需读 `g.act_scalars.acts[i].f` 的 repr 或从 readout ctor 常量；
最省事=存 cst + "act_hint" 常量 silu/sigmoid——gate 语义在类内固定。）

### G3（数值硬缺口）LayerNorm.eps 未序列化
`nonlinearity`（RadialMLP）内 net.{1,4,7} 为 torch LayerNorm，raw_state_dict 只有 weight/bias；
**eps=1e-5 是 Python 属性，不在 state_dict**（同类"字段缺失"教训 = field_feature_norms 事件）。

**补丁**：`extract_polar_blocks.py::_torch_mlp_net()` 增加显式分支：
```python
elif cname == "LayerNorm":
    layers.append({"type": "layernorm",
                   "normalized_shape": list(layer.normalized_shape),
                   "eps": float(layer.eps),
                   "elementwise_affine": bool(layer.elementwise_affine),
                   "weight": _tensor(layer.weight), "bias": _tensor(layer.bias)})
```

### G4（语义登记，非张量）输入/输出布局约定
JSON 不（也难）表达：`local_charges` 8 维块序 [sα,xα,yα,zα | sβ,xβ,yβ,zβ]、`potential_features`
= cat(α8,β8)（每通道 2x0e+2x1o）、输出 (N,10) = [前 8 密度增量 | 后 2 Fukui 源] 及 extensions.py
L833-864 的密度更新切法。建议（可选）在 `configs` 增 `charges_irreps_layout` 字符串注明；当前以
`bench/polar_fixtures/charges_hooks/index.json` 的 `semantics` 段 + 本文件为登记处。

## Task B：stage 级参考夹具（已落盘）

生成器 `bench/polar_module5_hooks.py`，日志 `logs/polar_module5_hooks.log`。
几何自证：`mm.build_gas("NH3",32,0.01,770600,2.0)` vs `scf/nh3-f0.npz:positions`
**np.allclose=True**；E = -49242.4564154931 eV（与 scf 夹具一致）。

- `bench/polar_fixtures/charges_hooks/nh3-step0.npz`（SCF 迭代 0 ← `charges_maps[0]` 调用）
- `bench/polar_fixtures/charges_hooks/nh3-step1.npz`（SCF 迭代 1 ← `charges_maps[1]` 调用）
- `bench/polar_fixtures/charges_hooks/index.json`（张量名/形状/来源/语义/对账）

每文件 42 张量（f64）：块级输入 `in_node_attrs(128,83)`、**`in_node_feats(128,2048)`=features_mixed**、
`in_edge_attrs(4652,9)`、`in_edge_feats(4652,8)`、`in_edge_index(2,4652)`、
**`in_potential_features(128,16)`**、**`in_local_charges(128,8)`**；块输出 `out(128,10)`；
子模块链：`pe_in_*`/`pe_out(128,2048)`、`pe_pot/nf/q_lin_*(128,2048)`、`dp_in_*(2048)`/`dp_out(512)`、
`se_out(512)`、`nl_in(1024)`/`nl_out(512)`、`tp_in_*`/`tp_out(2048)`、`ro_l1_in/out(2048/160)`、
`gate_in(160)/gate_out(128)`、`gate_in2/gate_out2`、`ro_lmid_in/out(128/160)`、`ro_l2_in(10)`。

**输入完整性**：7 个入参全部经 `register_forward_pre_hook(with_kwargs=True)` 从 kwargs 完整取出
（运行时为 eager 模式、无 torch.compile 包装），**无缺失项**；`gate` 因前向调用两次额外落盘
`gate_in2/gate_out2`。

对账（f64 噪声级，全 PASS）：
- `in_local_charges` vs `scf_steps/nh3-step{k}.npz:spin_charge_density_flat`：2.5e-16 / 4.7e-16；
- `out` vs `scf/nh3-f0.npz:charges_step_{k}`：1.3e-14 / 4.3e-15；
- `in_potential_features[:,ch*8:(ch+1)*8] × field_feature_norms` vs
  `potential_feats_step_{2k+ch}`：≤3.8e-15（证实块输入=归一后势特征，无外场）；
- 链内自洽 13 项（pe 三分量和、cat(dp,se)=nl_in、gate1=ro_l1_out、gate2=ro_lmid_out、
  out=ro_l2_out、dp/tp 输入=块输入 node_feats 等）**全部逐位 0.0**。

## 未决项 / 注意事项

1. G1-G3 补丁**未施行**（引擎源码树禁止修改）；需引擎侧负责人按上述指令更新
   `extract_polar_blocks.py` 并重跑 `bench/polar_json_regen.py` 重生成 JSON。
2. `readout_gate.act` 语义（scalars 用 silu、gates 用 sigmoid）是 `GeneralNonLinearBiasReadoutBlock`
   构造常量（blocks.py L331-332），JSON 现有 info 只能推出 cst 数值；建议补丁同时写 `act_hint`。
3. cueq 若开启：PE/readout 的布局转置模块非 None、sparse_uvu 权重为 ir_mul 布局——本 ckpt
   cueq_config=None、layout='mul_ir'，无此风险；换 ckpt 时需复核（JSON 已存 layout 字段）。
4. `field_norm_factor` 为 dead buffer（前向 0 引用，源码 L563-594 + 实测）；保留序列化仅为 schema 兼容。
5. edge_* 三个入参当前前向未用（若未来版本 field_blocks 改为消费边特征，需重审计）。
