# P3 模块⑦ `local_electron_energy` —— 精确补丁提案（最终版，2026-10-15）

**定位**：把 `docs/module7_patch_draft.md`（草案）细化为**可直接应用**的补丁提案 = 文本锚点 +
插入代码全文。**不写行号 diff**——引擎树正被模块⑤并行落码（本次编写期间 tpp 已两次变更，
行号累计漂移 >500 行）。合并前先跑 §5 自检清单第 1 组（锚点存在性）。

**引擎快照（提案编写读数，md5）**：

| 文件 | md5 | 行数/时刻 |
|---|---|---|
| `libsymmetrix/source/mace_nonlinear_kokkos.hpp` | `b463c96587b3e04e3f171cf8e6aaa8d2` | 814 行 / 09:09 |
| `libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp` | `7d6136b871e38d372b142bbb00bb88b4` | 8284 行 / 09:08 |
| `symmetrix/source/cpp/mace_nonlinear_kokkos.cpp`（binding） | `ba096f9a333db0df3b8e2fdd719f4d8f` | 587 行 / 09:11 |
| `symmetrix/source/symmetrix/extract_polar_blocks.py` | `d1defd28df78bc3ff525d92c18993d43` | — / 08:41 |
| `symmetrix/source/symmetrix/extract_mace_nonlinear.py` | `e0818e87823f7f46ed89d687fbfdc5ed` | — / 03:13 |
| `models/mace-polar/polar_symmetrix_probe.json` | `e26c7d932987edbdaa94f6c630bb4408` | — / 08:43（v3 重生成） |

依据锚点：mace 0.3.16 安装副本
`/opt/matos-tools/conda/envs/matos-mlip-py312/lib/python3.12/site-packages/mace/modules/`
（`field_blocks.py` L294-466 / L652-752、`extensions.py` L700-935、`radial.py` L361-384；
e3nn `o3/_linear.py` L150-200/L400-440）+ 引擎同树只读核对。

---

## 1. 语义复核（mace 侧实参顺序、能量口径：逐项复核通过，与草案 §1 一致）

`extensions.py::PolarMACE.forward` 调用点（实测 L867-880）：

```python
local_q_e = self.local_electron_energy(          # L867
    node_attrs=data["node_attrs"],               # (N,83)   ← 模块前向体 0 处使用
    node_feats=node_feats,                       # (N,2048) ← 末层 product 输出
    edge_attrs=edge_attrs[:, :self.from_ell_max_field_update],  # 模块前向体 0 处使用
    edge_feats=edge_feats, edge_index=data["edge_index"],       # 同上
    field_feats=potential_features,              # (N,16)   ← 末次 SCF 步势特征 α|β
    charges_0=field_independent_spin_charge_density.view(N,-1),   # (N,8) 初 Fukui 校正后
    charges_induced=spin_charge_density.view(N,-1))               # (N,8) SCF 末态
le_total = scatter_sum(src=local_q_e, index=data["batch"], dim=-1, dim_size=num_graphs)  # L879-881
if getattr(self, "add_local_electron_energy", False):
    total_energy = total_energy + le_total       # L883 ← 在 e0+inter_e（含 scale/shift）之后
```

**引擎取数映射（⑦ 运行块的输入来源，全部为引擎自身 polar 链的输出）**：

| mace 实参 | 引擎来源 | 说明 |
|---|---|---|
| `node_feats` (N,2048) | `polar_node_feats_e3nn` | ① 在层循环内逐层覆写，末层值保留；已含 −P 基变换（§14 机制） |
| `field_feats` (N,16) | `polar_step_potential_features`（⑤） | **末次迭代**势特征（α｜β，含 `/field_feature_norms`）；**勿**用末态密度重算，也**勿**用 ④ 的 `polar_potential_features`（那是初态密度特征） |
| `charges_0` (N,8) | `polar_field_independent_density`（②） | 初 Fukui 校正后密度 |
| `charges_induced` (N,8) | `polar_spin_charge_density`（⑤ 原地更新后的末态） | 与 `polar_scf_final_density` 同值 |
| `node_attrs/edge_*` | 不消费 | 引擎签名不读（契约保留，同草案 §1.3） |

能量口径：`le_total` = 图内逐原子 `local_q_e` 求和，**无原子数/分子数缩放**；能量装配 =
`e0 + inter_e(scale/shift 之后) + le_total`（再 + electro + 外场·偶极，后者生产=0）。
`calc.results["electron_energy"] = le_total × 1.0`（`energy_units_to_eV` 实测 =1.0）。

## 2. 当前 JSON 序列化状态（对 08:43 v3 版实测；草案 4 类缺口已分化）

| 草案缺口 | 当前状态（实测） | 本提案处置 |
|---|---|---|
| G2 `dot_products_*.instructions[*].path_weight` | **已修复**：`0.03125`×2 在案（`extract_polar_blocks._sparse_uvu` 08:41 已加） | 引擎自写 uvu kernel 直接消费 `path_weight` 字段（与⑤的 `parse_uvu_paths` 同源） |
| G1 `mlp.layers[1/4/7]` LayerNorm | **已结构化**：`{"type":"layernorm","normalized_shape":[128],"eps":1e-05,"weight","bias"}` | 但引擎 `AffineMLP` schema 拼写为 `"layer_norm"`（`affine_mlp.cpp` L57 / `affine_mlp_kokkos.cpp` L52）→ 见前置 E2 + 引擎侧容错 |
| G3 `wigner_3j` | 仍缺 | **路线变更后废弃**：本提案自写 uvu kernel，不再用 `E3TensorProductKokkosT`（其官方 layout 解析在 `e3nn_kokkos.cpp` L1748 对 `wigner_3j` 硬抛） |
| G4 `linear_up_q/v.bias`（512 ≠ 输出 2048） | **仍缺**（bias.shape=[512]，bias 指令 `(-1,0,[512],1.0)`） | **必须**前置 E1；否则引擎装载即抛（`e3nn.cpp` L129-130 `"linear bias has an invalid size"`）。**同缺口还阻塞⑤**：`readout.linear_mid.bias=[64]`（输出 96）、`readout.linear_2.bias=[4]`（输出 10）——E1 一并修复 |

## 3. 前置提取器补丁（必须先于引擎门；⑦ 与⑤ 共同依赖）

### E1（必须）`_linear`：e3nn 部分 bias 展开为满维

锚点（`symmetrix/source/symmetrix/extract_mace_nonlinear.py`，替换整个 `_linear` 函数）：

```python
def _linear(module) -> dict[str, Any]:
    return {
        "irreps_in": str(module.irreps_in),
        "irreps_out": str(module.irreps_out),
        "instructions": _instructions(module),
        "weight": _tensor(module.weight),
        "bias": _tensor(module.bias),
        "output_mask": _tensor(module.output_mask),
    }
```

替换为：

```python
def _linear(module) -> dict[str, Any]:
    return {
        "irreps_in": str(module.irreps_in),
        "irreps_out": str(module.irreps_out),
        "instructions": _instructions(module),
        "weight": _tensor(module.weight),
        "bias": _tensor(_full_width_linear_bias(module)),
        "output_mask": _tensor(module.output_mask),
    }


def _full_width_linear_bias(module) -> torch.Tensor:
    """e3nn o3.Linear(biases=True) 的 bias 仅覆盖标量(0e)输出块（_linear.py L177-186），
    常量经 i_in==-1 的 bias 指令消费（_linear.py L412-424）；引擎侧契约要求
    bias.numel()==irreps_out.dimension()（e3nn.cpp L129-130）。部分覆盖时按 bias 指令的
    i_out 块位置零填充展开为满维；bias 为空或已满维时原样（对现役 MH1 模型零改动）。
    """
    bias = module.bias
    dimension = module.irreps_out.dimension()
    if bias is None or bias.numel() == dimension:
        return bias
    expanded = bias.new_zeros(dimension)
    cursor = 0
    out_slices = list(module.irreps_out.slices())
    for instruction in module.instructions:
        if getattr(instruction, "i_in", None) != -1:
            continue
        width = int(instruction.path_shape[0])
        start = out_slices[int(instruction.i_out)].start
        expanded[start:start + width] = bias[cursor:cursor + width]
        cursor += width
    if cursor != bias.numel():
        raise RuntimeError(
            f"linear bias expansion mismatch: consumed {cursor} of {bias.numel()} "
            f"values (irreps_out={module.irreps_out})"
        )
    return expanded
```

零影响论证：现役 MH1 全链路 linear 的 bias 均为空或 `numel()==dimension()`
（fukui 读出头 2x0e → bias 2；backbone 读出 1x0e → bias 1），走 `return bias` 原路。
修复对象（实测）：`local_electron_energy.linear_up_q/v`（512→2048）、
`field_dependent_charges_maps[*].readout.linear_mid`（64→96）、`.linear_2`（4→10）。

### E2（推荐）`_torch_mlp_net`：LayerNorm 拼写对齐引擎 schema

锚点（`extract_polar_blocks.py::_torch_mlp_net`，`cname == "LayerNorm"` 分支）：

```python
                "type": "layernorm",
```

改为：

```python
                "type": "layer_norm",
```

说明：引擎侧 4.2 已内置双向容错（`"layernorm"` 归一为 `"layer_norm"`），故 E2 不阻塞
本模块；但为消除跨模块 schema 漂移（⑥ 及以后任何走 `AffineMLPKokkosT` 的 polar MLP），
建议随 E1 一并落地并**重生成 JSON**：`cd mlip && <polar python> bench/polar_json_regen.py`。
重生成后 §5 自检第 3 组核对 `bias.shape==[2048]` 与 `mlp.layers[*].type`。

> 注：草案 G2/G3 的 extractor 补丁**不再是前置**——G2 已在 JSON 中（uVU kernel 消费同名字段），
> G3 随路线变更废弃。

---

## 4. 引擎补丁（五段插入；全复用既有原语 + 自写 uvu kernel）

**路线裁定（与草案的关键差异）**：`dot_products_q/v` **不复用** `E3TensorProductKokkosT`。
理由：(a) 其官方 layout 解析要求 `wigner_3j`（G3，JSON 无且需要 extractor 补 2×2 个 3j 张量）；
(b) `SparseUvuTensorProduct` 是 uvu 特化（element 归一 1/√d1 + 逐路径 path_weight），语义比
通用张量积窄，(c) ⑤ 已用同一路线落码（PART1 `parse_uvu_paths` + 前向内联 `launch_uvu`，
tpp L51-87 / L6031-6099），⑦ 镜像之可保证两模块数值口径一致。
**跨 TU 注意**：tpp 分 3 个 TU 编译（`PART==1/2/3`，各自 .cpp），PART1 的 `parse_uvu_paths`
在 PART2（ctor 所在）不可见——因此 4.2 内联同逻辑解析、4.3 内联同款 kernel，两个新块完全
自包含，不触动⑤已落码。

### 4.1a hpp：公开中间量视图（锚点：⑤暂存块之后）

锚点（`mace_nonlinear_kokkos.hpp`，在此**之后**插入）：

```cpp
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_mixer_temp,
        polar_mixer_temp_storage,polar_scratch_a,polar_scratch_a_storage,
        polar_scratch_b,polar_scratch_b_storage,polar_nl_input,
        polar_nl_input_storage,polar_nl_a,polar_nl_a_storage,polar_nl_b,
        polar_nl_b_storage;                                             // ⑤暂存
```

插入：

```cpp
    // P3 模块⑦: local_electron_energy（OneBodyMLPFieldReadout; 仅 polar_le_enabled 填充）
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_le_q_in,
        polar_le_q_in_storage;                                          // (N,8)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_le_q_up,
        polar_le_q_up_storage,polar_le_v_up,polar_le_v_up_storage;      // (N,512)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_le_dot_q,
        polar_le_dot_q_storage,polar_le_dot_v,polar_le_dot_v_storage;   // (N,512)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_le_invar,
        polar_le_invar_storage;                                         // (N,1024)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_le_mlp_out,
        polar_le_mlp_out_storage;                                       // (N,1)
    Kokkos::View<Precision*> polar_le_atom,polar_le_atom_storage;       // (N,)
    Precision polar_le_total=Precision(0);                              // 末次求值总量
```

### 4.1b hpp：私有表/成员（锚点：`polar_updates` 之后）

锚点（`mace_nonlinear_kokkos.hpp`，在此**之后**、`std::vector<LayerState> states;` 之前插入）：

```cpp
    std::vector<E3LinearKokkosT<Precision>> polar_mixer_linears;
    std::vector<PolarUpdate> polar_updates;
```

插入：

```cpp
    // --- P3 模块⑦: local_electron_energy 表与配置（ctor 解析） ---
    // 依据: field_blocks.py L652-752（前向链）+ L294-466（SparseUvu mode0/mode1）。
    bool polar_le_enabled=false,polar_le_add=false;
    E3LinearKokkosT<Precision> polar_le_linear_up_q,polar_le_linear_up_v;
    Kokkos::View<int*> polar_le_q_meta,polar_le_v_meta;                 // (np,8)
    Kokkos::View<Precision*> polar_le_q_path_weight,polar_le_q_weight,
        polar_le_q_mask,polar_le_v_path_weight,polar_le_v_weight,
        polar_le_v_mask;
    AffineMLPKokkosT<Precision> polar_le_mlp;   // RadialMLP 1024→128×3→1
```

### 4.2a tpp ctor：解析（锚点：`charges_maps` 计数校验之后）

锚点（`mace_nonlinear_kokkos_impl.tpp`，PART2，在此**之后**插入）：

```cpp
        if(static_cast<int>(polar_updates.size())!=polar_recursion_steps)
            throw std::invalid_argument(
                "polar_blocks: charges_maps count must match num_recursion_steps.");
    } else polar_enabled=false;
```

插入（在 `throw ...;` 与 `} else polar_enabled=false;` 之间）：

```cpp
        // ---- P3 模块⑦: local_electron_energy（OneBodyMLPFieldReadout L652-752） ----
        // uvu 路径表列 = [in1_off,in2_off,out_off,w_off,mul1,mul2,mode,d1]
        // （与 PART1 parse_uvu_paths L51-87 同一语义；PART 为独立 TU，此处内联）。
        if(pb.contains("local_electron_energy")) {
            const auto& le=pb.at("local_electron_energy");
            polar_le_enabled=true;
            polar_le_add=pb.at("configs").value("add_local_electron_energy",false);
            polar_le_linear_up_q=E3LinearKokkosT<Precision>(le.at("linear_up_q"));
            polar_le_linear_up_v=E3LinearKokkosT<Precision>(le.at("linear_up_v"));
            const auto parse_le_uvu=[&](const nlohmann::json& data,
                    Kokkos::View<int*>& meta_view,
                    Kokkos::View<Precision*>& paths_view,
                    Kokkos::View<Precision*>& weight_view,
                    Kokkos::View<Precision*>& mask_view,
                    const char* meta_label,const char* paths_label,
                    const char* weight_label,const char* mask_label) {
                const Irreps in1(data.at("irreps_in1").get<std::string>());
                const Irreps in2(data.at("irreps_in2").get<std::string>());
                const Irreps out(data.at("irreps_out").get<std::string>());
                const auto weights=tensor_values<Precision>(data.at("weight"));
                const auto mask=tensor_values<Precision>(data.at("output_mask"));
                std::vector<int> meta;
                std::vector<Precision> paths;
                int weight_offset=0;
                for(const auto& value:data.at("instructions")) {
                    const auto& a=in1.blocks.at(value.at("i_in1").get<int>());
                    const auto& b=in2.blocks.at(value.at("i_in2").get<int>());
                    const auto& c=out.blocks.at(value.at("i_out").get<int>());
                    const auto shape=value.at("path_shape")
                        .get<std::vector<int>>();
                    const int d1=2*a.l+1;
                    int mode=0;
                    if(c.l!=0) {
                        mode=1;
                        if(b.l!=0||a.l!=c.l)
                            throw std::invalid_argument(
                                "polar le uvu: mode1 requires l2==0 and l1==l_out.");
                    } else if(a.l!=b.l)
                        throw std::invalid_argument(
                            "polar le uvu: mode0 requires l1==l2.");
                    meta.insert(meta.end(),{a.offset,b.offset,c.offset,
                        weight_offset,a.multiplicity,b.multiplicity,mode,d1});
                    paths.push_back(value.at("path_weight").get<Precision>());
                    weight_offset+=static_cast<int>(shape[0])
                        *static_cast<int>(shape[1]);
                }
                if(weight_offset!=static_cast<int>(weights.size()))
                    throw std::invalid_argument(
                        "polar le uvu: weight size mismatch.");
                meta_view=toKokkosView(meta_label,meta,
                    static_cast<int>(paths.size()),8);
                paths_view=toKokkosView(paths_label,paths);
                weight_view=toKokkosView(weight_label,weights);
                mask_view=toKokkosView(mask_label,mask);
            };
            parse_le_uvu(le.at("dot_products_q"),polar_le_q_meta,
                polar_le_q_path_weight,polar_le_q_weight,polar_le_q_mask,
                "polar le q meta","polar le q path weight",
                "polar le q weight","polar le q mask");
            parse_le_uvu(le.at("dot_products_v"),polar_le_v_meta,
                polar_le_v_path_weight,polar_le_v_weight,polar_le_v_mask,
                "polar le v meta","polar le v path weight",
                "polar le v weight","polar le v mask");
            // RadialMLP = [Linear,LayerNorm,SiLU]×3 + Linear（radial.py L361-384）。
            // 提取器当前以 "layernorm" 落盘；引擎 AffineMLP schema 为 "layer_norm"
            // （affine_mlp.cpp L57）→ 归一（两种拼写均接受；"layer_norm" 原样）。
            nlohmann::json mlp=le.at("mlp");
            for(auto& layer:mlp.at("layers"))
                if(layer.value("type",std::string())=="layernorm")
                    layer["type"]="layer_norm";
            polar_le_mlp=AffineMLPKokkosT<Precision>(mlp);
            // 尺寸契约（装载即校验，失败信息见模块名）。
            if(polar_le_linear_up_q.input_dimension()!=8
                ||polar_le_linear_up_v.input_dimension()!=16)
                throw std::invalid_argument(
                    "polar_blocks: local_electron_energy linear_up inputs must be 8/16.");
            if(polar_le_linear_up_q.output_dimension()!=512
                ||polar_le_linear_up_v.output_dimension()!=512)
                throw std::invalid_argument(
                    "polar_blocks: local_electron_energy linear_up outputs must be 512.");
            if(static_cast<int>(polar_le_q_mask.extent(0))!=512
                ||static_cast<int>(polar_le_v_mask.extent(0))!=512)
                throw std::invalid_argument(
                    "polar_blocks: local_electron_energy dot_products outputs must be 512.");
            if(polar_le_mlp.input_size()!=1024||polar_le_mlp.output_size()!=1)
                throw std::invalid_argument(
                    "polar_blocks: local_electron_energy mlp must map 1024 -> 1.");
        } else polar_le_enabled=false;
```

### 4.2b tpp setter：backend 接线（锚点：⑤ backend 循环之后）

锚点（`set_e3_linear_backend` 内，在此**之后**、`if(polar_enabled){...}` 的闭括号之前插入）：

```cpp
        for(auto& linear:polar_mixer_linears)linear.set_backend(selected);
        for(auto& update:polar_updates)update.set_e3_linear_backend(selected);
    }
```

插入：

```cpp
        if(polar_le_enabled) {
            polar_le_linear_up_q.set_backend(selected);
            polar_le_linear_up_v.set_backend(selected);
        }
```

### 4.2c tpp setter：workspace 接线（锚点：⑤ workspace 循环之后）

锚点（ctor 尾部 workspace 接线块，在此**之后**、`if(polar_enabled){...}` 的闭括号之前插入）：

```cpp
        for(auto& update:polar_updates)
            update.set_e3_linear_workspace(e3_linear_workspace);
    }
```

插入：

```cpp
        if(polar_le_enabled) {
            polar_le_linear_up_q.set_workspace(e3_linear_workspace);
            polar_le_linear_up_v.set_workspace(e3_linear_workspace);
        }
```

（`polar_le_mlp` 使用 `AffineMLPKokkosT` 自带默认 workspace，按需自动增长——
`affine_mlp_kokkos.cpp` L289-309；单实例，无复用/别名问题。）

### 4.3 tpp 前向运行块（锚点：readout 累加循环之前 = polar 段（含⑤ SCF）之后）

锚点（`compute_node_energies_forces` 内，在此**之前**插入）：

```cpp
    for(int layer=0;!generated_node_program
        &&layer<static_cast<int>(readouts.size());++layer) {
```

插入：

```cpp
    if(polar_enabled&&polar_le_enabled) {
        // ---- P3 模块⑦: local_electron_energy（OneBodyMLPFieldReadout L652-752） ----
        // 输入口径 = extensions.py L867-880：q_in = charges_induced(⑤末态)
        // + charges_0(②校正后初态)；v_in = ⑤末次迭代势特征 α|β；node_feats =
        // ① 末层 e3nn 基张量。前向 = linear_up → 两条 uvu 内积 → concat → RadialMLP
        // → 逐原子 local_q_e；总量 = 图内和（单图=整盒，无缩放）。
        if(static_cast<int>(polar_step_potential_features.extent(0))!=num_nodes
            ||polar_step_potential_features.extent(1)!=16)
            throw std::runtime_error(
                "polar_blocks: local_electron_energy requires module ⑤ SCF step "
                "potential features before it runs.");
        const int le_dot_width=static_cast<int>(polar_le_q_mask.extent(0));
        const int le_up_width=polar_le_linear_up_q.output_dimension();
        ensure_view(polar_le_q_in,polar_le_q_in_storage,num_nodes,8);
        ensure_view(polar_le_q_up,polar_le_q_up_storage,num_nodes,le_up_width);
        ensure_view(polar_le_v_up,polar_le_v_up_storage,num_nodes,le_up_width);
        ensure_view(polar_le_dot_q,polar_le_dot_q_storage,num_nodes,le_dot_width);
        ensure_view(polar_le_dot_v,polar_le_dot_v_storage,num_nodes,le_dot_width);
        ensure_view(polar_le_invar,polar_le_invar_storage,num_nodes,
            2*le_dot_width);
        ensure_view(polar_le_mlp_out,polar_le_mlp_out_storage,num_nodes,1);
        ensure_view(polar_le_atom,polar_le_atom_storage,num_nodes);
        {
            auto q_in=polar_le_q_in;
            auto charges_zero=polar_field_independent_density;
            auto charges_final=polar_spin_charge_density;
            Kokkos::parallel_for("polar le q_in",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,8}),
                KOKKOS_LAMBDA(int node,int column) {
                    q_in(node,column)=charges_final(node,column)
                        +charges_zero(node,column);
                });
        }
        polar_le_linear_up_q.evaluate(polar_le_q_in,polar_le_q_up);
        polar_le_linear_up_v.evaluate(
            polar_step_potential_features,polar_le_v_up);
        // 自写 uvu 内核：镜像 field_blocks.py L405-466 的 mode0(l×l→0)/mode1(l×0→l),
        // 逐路径乘序列化 path_weight（JSON 0.03125）；与⑤ launch_uvu 同构。
        {
            const auto launch_le_uvu=[&](const int nodes,const auto& meta_host,
                    const auto& path_weight,Kokkos::View<Precision*> weight_view,
                    Kokkos::View<Precision*> mask_view,
                    Kokkos::View<Precision**> x1,
                    Kokkos::View<Precision**> x2,
                    Kokkos::View<Precision**> output) {
                // 注: meta_host/path_weight 为 HostMirror, 形参必须 auto&（同⑤ launch_uvu）;
                // 显式 Kokkos::View<Precision*> 无法绑定 HostSpace 镜像。
                const int paths=static_cast<int>(meta_host.extent(0));
                Kokkos::parallel_for("polar le uvu zero",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {nodes,static_cast<int>(output.extent(1))}),
                    KOKKOS_LAMBDA(int node,int column) {
                        output(node,column)=Precision(0);
                    });
                for(int p=0;p<paths;++p) {
                    const int in1_off=meta_host(p,0),in2_off=meta_host(p,1),
                        out_off=meta_host(p,2),w_off=meta_host(p,3),
                        mul1=meta_host(p,4),mul2=meta_host(p,5),
                        mode=meta_host(p,6),d1=meta_host(p,7);
                    const Precision pw=path_weight(p);
                    const Precision inv_sqrt_d1=Precision(1)
                        /Kokkos::sqrt(Precision(d1));
                    if(mode==0) {
                        Kokkos::parallel_for("polar le uvu mode0",nodes,
                            KOKKOS_LAMBDA(int node) {
                                for(int u=0;u<mul1;++u) {
                                    Precision mixed=Precision(0);
                                    for(int v=0;v<mul2;++v) {
                                        Precision pair=Precision(0);
                                        for(int c=0;c<d1;++c)
                                            pair+=x1(node,in1_off+u*d1+c)
                                                *x2(node,in2_off+v*d1+c);
                                        mixed+=pair*inv_sqrt_d1
                                            *weight_view(w_off+u*mul2+v);
                                    }
                                    output(node,out_off+u)+=pw*mixed;
                                }
                            });
                    } else {
                        Kokkos::parallel_for("polar le uvu mode1",nodes,
                            KOKKOS_LAMBDA(int node) {
                                for(int u=0;u<mul1;++u) {
                                    Precision mixed=Precision(0);
                                    for(int v=0;v<mul2;++v)
                                        mixed+=x2(node,in2_off+v)
                                            *weight_view(w_off+u*mul2+v);
                                    const Precision contribution=
                                        pw*mixed*inv_sqrt_d1;
                                    for(int c=0;c<d1;++c)
                                        output(node,out_off+u*d1+c)+=
                                            contribution*x1(node,
                                                in1_off+u*d1+c);
                                }
                            });
                    }
                }
                Kokkos::parallel_for("polar le uvu mask",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {nodes,static_cast<int>(output.extent(1))}),
                    KOKKOS_LAMBDA(int node,int column) {
                        output(node,column)*=mask_view(column);
                    });
            };
            const auto le_q_meta=Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),polar_le_q_meta);
            const auto le_q_path_weight=Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),polar_le_q_path_weight);
            const auto le_v_meta=Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),polar_le_v_meta);
            const auto le_v_path_weight=Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),polar_le_v_path_weight);
            auto node_feats=polar_node_feats_e3nn;
            launch_le_uvu(num_nodes,le_q_meta,le_q_path_weight,
                polar_le_q_weight,polar_le_q_mask,node_feats,
                polar_le_q_up,polar_le_dot_q);
            launch_le_uvu(num_nodes,le_v_meta,le_v_path_weight,
                polar_le_v_weight,polar_le_v_mask,node_feats,
                polar_le_v_up,polar_le_dot_v);
        }
        {
            auto invar=polar_le_invar,dot_q=polar_le_dot_q,dot_v=polar_le_dot_v;
            Kokkos::parallel_for("polar le invar concat",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,le_dot_width}),
                KOKKOS_LAMBDA(int node,int column) {
                    invar(node,column)=dot_q(node,column);
                    invar(node,le_dot_width+column)=dot_v(node,column);
                });
        }
        polar_le_mlp.evaluate(polar_le_invar,polar_le_mlp_out);
        {
            auto atom=polar_le_atom,mlp_out=polar_le_mlp_out;
            Kokkos::parallel_for("polar le atom readout",num_nodes,
                KOKKOS_LAMBDA(int node) {
                    atom(node)=mlp_out(node,0);
                });
        }
        // 总量：每图 scatter_sum（单图=整盒；无原子数缩放，mace L879-881）。
        const auto le_atom_host=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),polar_le_atom);
        Precision le_sum=Precision(0);
        for(int node=0;node<num_nodes;++node)
            le_sum+=le_atom_host(node);
        polar_le_total=le_sum;
    }
```

### 4.4 tpp 能量接线（锚点：`scale nonlinear energies` 之后）

锚点（在此**之后**插入）：

```cpp
    Kokkos::parallel_for("scale nonlinear energies",num_nodes,KOKKOS_LAMBDA(int node) {
        energies(node)=e0(node_types(node))+energy_scale*energies(node)+energy_shift;
    });
```

插入：

```cpp
    if(polar_enabled&&polar_le_enabled&&polar_le_add) {
        // 模块⑦ 能量装配（mace extensions.py L866-883）:
        //   total = e0 + inter_e（已含 scale/shift） + le_total
        // le 是 scale/shift 之后的加项（不乘 energy_scale），与 ⑥ electro 并列。
        auto energies=node_energies;
        auto le=polar_le_atom;
        Kokkos::parallel_for("polar le energy add",num_nodes,
            KOKKOS_LAMBDA(int node) {
                energies(node)+=le(node);
            });
    }
```

### 4.5 binding 访问器（锚点：`polar_scf_final_density` 访问器之后）

锚点（`symmetrix/source/cpp/mace_nonlinear_kokkos.cpp`，把该 `.def` 块整体替换为
"原块 + 新增块"，保持 `.def` 链式语法；新增块置于链尾 `;` 之前）：

```cpp
        .def("polar_scf_final_density",[](const Evaluator& self) {
            return view2vector(self.polar_scf_final_density);
        });
```

替换为：

```cpp
        .def("polar_scf_final_density",[](const Evaluator& self) {
            return view2vector(self.polar_scf_final_density);
        })
        .def("polar_le_q_in",[](const Evaluator& self) {
            // 模块⑦: q_in = charges_induced + charges_0(N×8)。
            return view2vector(self.polar_le_q_in);
        })
        .def("polar_le_q_up",[](const Evaluator& self) {
            // 模块⑦: linear_up_q 输出(N×2048)。
            return view2vector(self.polar_le_q_up);
        })
        .def("polar_le_v_up",[](const Evaluator& self) {
            // 模块⑦: linear_up_v 输出(N×2048)。
            return view2vector(self.polar_le_v_up);
        })
        .def("polar_le_dot_q",[](const Evaluator& self) {
            // 模块⑦: uvu 内积 dot_products_q(node_feats, q_up)(N×512)。
            return view2vector(self.polar_le_dot_q);
        })
        .def("polar_le_dot_v",[](const Evaluator& self) {
            // 模块⑦: uvu 内积 dot_products_v(node_feats, v_up)(N×512)。
            return view2vector(self.polar_le_dot_v);
        })
        .def("polar_le_invar",[](const Evaluator& self) {
            // 模块⑦: cat(dot_q, dot_v)(N×1024; MLP 输入)。
            return view2vector(self.polar_le_invar);
        })
        .def("polar_le_mlp_out",[](const Evaluator& self) {
            // 模块⑦: RadialMLP 输出(N×1)。
            return view2vector(self.polar_le_mlp_out);
        })
        .def("polar_local_electron_energy",[](const Evaluator& self) {
            // 模块⑦: 逐原子 local_q_e(N,)（= mace 模块直出 squeeze(-1)）。
            return view2vector(self.polar_le_atom);
        })
        .def("polar_le_energy",[](const Evaluator& self) {
            // 模块⑦: le_total = 图内 Σ local_q_e（单图=整盒; 无缩放）。
            return static_cast<double>(self.polar_le_total);
        });
```

---

## 5. 自检清单（合并前后逐条执行；grep 均为只读动作）

**A. 锚点存在性（应用前；每条应命中唯一）**
1. `grep -n "polar_nl_b_storage;" libsymmetrix/source/mace_nonlinear_kokkos.hpp` → 1 处（4.1a 插入点）。
2. `grep -n "std::vector<LayerState> states;" .../mace_nonlinear_kokkos.hpp` → 1 处（4.1b 插入点在其前）。
3. `grep -n "charges_maps count must match num_recursion_steps" .../mace_nonlinear_kokkos_impl.tpp` → 1 处（4.2a）。
4. `grep -c "update.set_e3_linear_backend(selected);" ...impl.tpp` → 1（4.2b）。
5. `grep -c "update.set_e3_linear_workspace(e3_linear_workspace);" ...impl.tpp` → 1（4.2c）。
6. `grep -c "for(int layer=0;!generated_node_program" ...impl.tpp` → 1（4.3 插入点）。
7. `grep -c "scale nonlinear energies" ...impl.tpp` → 1（4.4 插入点）。
8. `grep -n "polar_scf_final_density" symmetrix/source/cpp/mace_nonlinear_kokkos.cpp` → 访问器链尾；以完整 `.def` 文本替换（4.5）。
9. `grep -n "def _linear" symmetrix/source/symmetrix/extract_mace_nonlinear.py` → 1（E1）；`grep -n '"type": "layernorm"' symmetrix/source/symmetrix/extract_polar_blocks.py` → 1（E2）。

**B. 应用后语法/结构核对**
10. `grep -c "polar_le_enabled" ...impl.tpp` → 6（4.2a 置位/清位 ×2、4.2b/4.2c setter ×2、4.3/4.4 各 1）。
11. `grep -c "P3 模块⑦" hpp` → 2（4.1a/4.1b）、`...tpp` → 2（4.2a/4.3）；binding 用 `模块⑦` 字样注释，不依赖计数。
12. `grep -n "launch_le_uvu" ...impl.tpp` → 3（定义 + 2 次调用）。
13. 交叉核对 ⑤ 的 meta 列约定仍在案：`grep -n "P3 模块⑤: sparse_uvu 路径表解析" -A 6 ...impl.tpp`，确认列序
    `[in1_off,in2_off,out_off,w_off,mul1,mul2,mode,d1]` 与本提案 4.2a 一致；若⑤后续把 `parse_uvu_paths`
    提升为共享头声明，可将 4.2a 的 `parse_le_uvu` 替换为对它的调用（语义等价，二选一）。

**C. JSON/提取器核对（E1/E2 + 重生成后）**
14. `python - <<'P'`（只读 JSON）断言：`linear_up_q/v.bias.shape==[2048]`；
    `readout.linear_mid.bias.shape==[96]`、`readout.linear_2.bias.shape==[10]`；
    `dot_products_*.instructions[*].path_weight` 全在且为 0.03125；
    `mlp.layers[*].type ∈ {"linear","layer_norm","silu"}`；`mlp.layers[1].eps==1e-5`；
    `configs.add_local_electron_energy is True`、`num_recursion_steps==2`。`P`
15. 负例（装载期契约，照 ⑤ 的 NEG smoke 方式，只改 JSON 副本、不重编译）：
    把 `dot_products_q.weight.values` 截 1 个 → 引擎应抛 `"polar le uvu: weight size mismatch."`；
    把 `linear_up_q.irreps_in` 改成与 v 相同的 16 维串 → 应抛 `linear_up inputs must be 8/16`。

**D. 构建与装载冒烟（隔离环境）**
16. 重建（CUDA config-settings，同 ⑤ 既有流程）→ `REBUILD_EXIT=0`；随后
    `bench/polar_engine_load_smoke.py` 应 POS-OK（`polar_blocks parsed`），退出阶段
    `cudaStreamSynchronize/driver shutting down` 为已留档的无害项。
17. `grep -n "polar_le_energy\|polar_local_electron_energy\|polar_le_invar" <env>/site-packages/symmetrix*.so` 不可行——
    改为 Python：`python -c "from symmetrix.calculator import Symmetrix; ...; print(hasattr(ev,'polar_le_energy'))"`
    → `True`（门脚本首段已内置同等检查）。

**E. 数值门（§6）**
18. `CUDA_VISIBLE_DEVICES=<gpu> <polar python> bench/polar_module7_gate.py --live 2` → `GATE PASS`。

## 6. 验收门 `bench/polar_module7_gate.py`（本提案随附；只写不跑）

- **源 A（主）**：`bench/polar_fixtures/le_hooks/{mol}-f{0,1}.npz` × 8 构型（4 分子 ×2 帧）。
  构型从 npz 的 `symbols/positions/cell` 原样重建（与预检同几何、同原子序），引擎单次前向
  后逐级读访问器比对：`q_in / q_up / v_up / dot_q / dot_v / invar / mlp_out(↔mlp_h9) /
  local_q_e / le_total`。
- **源 B（可选 `--live K`）**：mace_polar + forward hooks 现取 q_in/q_up/v_up/dot_q/dot_v/
  invar/mlp 输出/le_total，与引擎在同一构型上对账。
- **判据（参照①②门，标度感知）**：中间级 `abs ≤ 5e-6 或 rel ≤ 1e-6`；
  出口（`local_q_e`、`le_total vs npz.le_total`、`le_total vs npz.fixture_electron_energy`）
  仅绝对判据 `abs ≤ 1e-6`（与草案 §4 表⑥⑦一致；不做 rel 兜底，防大值放行）。
  理论预期：预检已证模块本体与活体差 ≤9.4e-16，引擎全链与夹具差 = 上游①-⑤骨干 f64
  漂移级（~1e-8 rel；le_total O(10) → ~1e-7 abs），裕度 ≥10×。
- **产物**：`bench/polar_module7_gate.{json,png}`；访问器缺失时打印 `MODULE7-ACCESSORS-MISSING`
  并返回 2（区分"未重建"与"数值 FAIL"）。

## 7. 与草案 `module7_patch_draft.md` 的偏差（差异清单）

| # | 草案 | 本版（最终） | 理由 |
|---|---|---|---|
| 1 | `dot_products_q/v` 复用 `E3TensorProductKokkosT` + extractor 补 `wigner_3j`（G3） | **自写 uvu kernel**（镜像 `SparseUvuTensorProduct.forward` mode0/mode1 + 序列化 `path_weight`）；G3 废弃 | `E3TensorProductKokkosT` 官方 layout 需 `wigner_3j`（`e3nn_kokkos.cpp` L1748 硬抛）；uvu 是更窄语义，自写更短更准；与⑤已落码同一路线，避免两套数值口径 |
| 2 | 置⑦块"⑤之后、⑥之前" | 锚点=**readout 累加循环之前**（即 polar 段末尾） | ⑥ 未落码，锚点须对⑤任意重构鲁棒；readout 循环文本唯一且稳定 |
| 3 | 用 ④ 的 `polar_potential_features` 作为 `field_feats`（草案 §3.4 未区分） | 明确用⑤的 **`polar_step_potential_features`**（末次迭代值） | mace 语义：potential_features 是末次 SCF 步输入密度的特征（L818-820），④ 产出的是初态密度特征 |
| 4 | 表解析直接调 `parse_uvu_paths`（未察 TU 边界） | 4.2a **内联**同款解析（PART2 不可见 PART1 符号） | tpp 三个 PART 各自独立 TU（`#if PART==1/2/3`） |
| 5 | G4 仅列为"extractor 补丁指令" | 升级为 **E1（必须）**，并指出其同时阻塞⑤的 `readout.linear_mid/linear_2` 装载 | 现状 JSON 实测 bias=[512]/[64]/[4]，均在引擎装载期硬抛 |
| 6 | G1 视为未修（"raw_state_dict、缺 eps"） | 实测**已结构化**；新增拼写差异（`layernorm` vs `layer_norm`）→ E2 + 引擎双向容错 | 08:41 extractor/08:43 JSON 已重生成 |
| 7 | 独立 `polar_le_mlp_workspace` 成员与接线 | 不新增（`AffineMLPKokkosT` 默认 workspace 自增长） | 单实例、安全、减面 |
| 8 | 访问器含每层 MLP 调试视图（可选） | 不设；逐级对账止于 `invar` + `mlp_out`（=mlp_h9） | h0 幅值 O(1e4) 的逐层门可由夹具既有数据约束，引擎无必要增表 |
| 9 | 能量接线伪代码 `polar_le_atom` 累加（未区分路径） | 与 4.4 同，但**明确在 `scale nonlinear energies` 之后**、带 `polar_le_add` 门控 | 防"le 被 scale 乘"的经典错误；生产模型 scale/shift 实测 1.0/0.0 但语义必须按 mace |

## 8. 依赖、风险与闭合条件

- **依赖顺序**：E1 + JSON 重生成（解⑤/⑦装载）→合并引擎五段→重建→装载冒烟→门。
  若门的中间级在 `q_in` 即 FAIL：先查⑤末态密度/②校正密度（源 A 前置已由 scf/scf_steps 夹具锁定）；
  若 `invar` 之后才漂：查 ①② 的基变换与 uvu 口径（预检已锁定模块本体 ≤9.4e-16）。
- **⑤ 契约点（不回写⑤代码，仅依赖）**：⑦ 读 `polar_step_potential_features` 必须停在
  "末次迭代的势特征"；若⑤以后加了末态重算，本模块语义会静默错——已在 4.3 加 extent 守卫，
  建议⑤门（charges_hooks step1）复跑时同步核对该视图。
- **⑧（力）不属本提案**：⑦ 能量项对位置的直接依赖为零（位置经上游 node_feats/charges/field
  进入）；反向只需把 `dL/d local_q_e` 注入⑤/②/① 的伴随链（草案 §3.7 轮廓有效）。
- 退出阶段 `cudaStreamSynchronize/driver shutting down` 噪声为已留档无害项（v11 起）。
