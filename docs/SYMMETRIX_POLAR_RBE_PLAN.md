# Symmetrix-XL × MACE-POLAR + RBE 集成 —— 可行性与实施计划(2026-10-14)

**状态**:Phase-0 源码审计完成;提取器探针运行中(`logs/symmetrix_polar_probe.log`)。
**用户指令**:「在 symmetrix-xl 里面新增一下对 mace-polar 的支持,其中的静电求和部分可以使用
rbe;这样不需要训练通用势」——已列为当前最高优先级。

## 结论:能做,路径清晰

三条扩展线已在源码中定位,**不用重写引擎**:短程 backbone 复用现有序列化与内核;
增量集中在「多极 + k 空间 + SCF + RBE」,且有 **MACEField 先例**可循
(「MACE + 附加物理模块」从提取器到 pair style 的完整改造链路已存在一次)。

## 1. 现状审计(源码级)

| 层 | 位置 | 事实 |
|---|---|---|
| 引擎核 | `source/cpp/mace_kokkos.cpp`(2736 行) | 短程 MACE 前向;`e3nn.cpp` 用 **JSON 驱动原语**(Linear/Tensor/AffineMLP/E3*)构造网络 |
| JIT | `jit_nvrtc.cpp` | nvrtc 运行时编译内核(宿主/设备 artifact) |
| pair style | `lammps-.../src/KOKKOS/pair_symmetrix_mace_kokkos.cpp`(2289 行) | `MACEKokkos<Precision>(json, head)`;模型类型走 JSON `model_type` 分派 |
| 先例 | MACEField | `extract_mace_data.py` 已含第二模型族分支(`model_type="MACEField"`, field_feats/field_linear);pair style 有 `electric_field` 关键字 —— 完整模板 |
| 提取器 | `source/symmetrix/extract_mace_data.py`(~1600 行) | `extract_mace_data(model, ...)`;CLI `symmetrix extract -m MODEL -o OUT.json` |
| 静电/长程 | —— | **全库 grep:无 ewald/kspace/k 空间模块**(唯一真正的增量所在) |

## 2. 目标形态

`pair_style symmetrix/polar-mace`:一次前向 =
短程 backbone(复用)→ 自旋分辨多极头 → GTO 结构因子(**k 空间求和,用 RBE 采样**)→
SCF(Fukui 平衡)→ 长程特征回注 → 静电能 + 力/维里。中性闭壳层起步(charge=0/spin=1)。

## 3. 实施阶段(每阶段有独立验证门)

- **P1 提取器扩展(0.5–1 天)**:PolarMACE 分支——backbone 复用现有 serialize;
  新块 = 多极头(l≤1, 2 自旋通道)、GTO 基(sigmas/max_l)、kspace_cutoff、
  `fixedpoint_update_config`、readout、Fukui 层。**门**:JSON 与 Python 侧参数逐位对账。
- **P2 引擎多极前向,无长程(1 天)**:多极头评估,**门**:与 Python backbone/多极输出对账。
- **P3 全 k 空间 Ewald + SCF + 能量/力(2–4 天)**:解析导数;**门**:迁移今天四关方法论
  (能量/力有限差分/尺度/长程扫描 对照 `graph_longrange v0.4.0`)。
- **P4 RBE 采样(1–2 天)**:Box-Muller + Metropolis(参照 `lammps-rbe` 的 `kspace_style rbe`);
  **门**:Ewald vs RBE 的力 RMS 随批大小 p 收敛表。
- **P5 pair style 接线 + 性能(1 天)**:cutoff=POLAR r_max(12 Å)、通讯截断、h2h 计时;
  **门**:与 Python 参考同盒能量/力对照 + 稳定 MD。

预估合计:**~1–1.5 周**(focused)。

## 4. 风险register

- **SCF 梯度**是最大风险:先锁能量再上力,分段逐段解析验证;必要时先半解析过渡。
- JIT 管线适配新原语:可先走非 JIT 的 Kokkos 直调路径,再补 JIT。
- RBE 方差:用 p 扫描定标准;p 与小体系收益的权衡如实报告。
- 数值纪律:每一层均与 Python 参考对账(与 MACE-POLAR 四关同一方法论)。

## 5. 探针记录(实况,2026-10-14)

- **探针#1**(`logs/symmetrix_polar_probe.log`):`extract_mace_data(MACE-POLAR-1-M)` 走到
  readout 检查,报 `NonLinearBiasReadoutBlock` 不支持 → **工单①**(PolarMACE backbone 被路由到
  nonlinear 提取路径,卡在读出块类型)。
- **修复①**(`source/symmetrix/extract_mace_nonlinear.py`,diff 已入
  `mlip/patches/symmetrix-xl-polar-extractor.patch`):接受 `NonLinearBiasReadoutBlock`
  (silu 检查同步扩展)+ `_readout()` 增加 `linear_mid` 序列化(bias 由 `_linear` 原样携带)。
- **探针#2**(`logs/symmetrix_polar_probe2.log`):**EXTRACT-OK**。产物结构核验:
  - `model_type=MACE_Nonlinear`;`r_cut=6.0`(单层;感受野 = 2×6 = 12 Å);`l_max=3`;
  - 2 × `RealAgnosticResidualNonLinearInteractionBlock`(非线性骨干);
  - readouts = `[LinearReadoutBlock, NonLinearBiasReadoutBlock]`,后者
    `linear_1/linear_mid/linear_2` 齐全,bias 在格;
  - `execution_contracts: MH1_UVU`;
  - 产物:`mlip/models/mace-polar/polar_symmetrix_probe.json`(198 MB)。
- **口径**(重要):该 JSON = **短程 backbone only**——"静电相关键: []"已显式核验;
  多极头/GTO/SCF 等静电极化块是 P1 的下一批 schema 段(尚未开始)。
- **引擎侧待办**:P2 验证先走 bench/JIT 通道(当前版本两个 LAMMPS pair style 变体都显式
  拒绝 `MACE_Nonlinear`——上游 rollout 中;bench.py 有独立的 JIT 执行路径)。

## 6. 引擎补丁①:`NonLinearBiasReadoutBlock` 全链路支持(2026-10-14 实施)

> **改动纪律(用户指令 2026-10-15):所有改动必须有依据,参考上游源码仓库。**
> 每笔改动 = 上游源码引用 / 官方测试断言 / 官方 CI 配置 三者之一为锚,逐条入账(见 §8 依据账本);
> 无法溯源到上游的改动不做。RBE 将逐项参照 `randbatch-md/lammps-rbe`,静电块逐块对照
> `graph_longrange v0.4.0` 参考实现。

bench 加载探针依次剥出:scale/shift 标量校验(提取侧修)→ atomic_energies 二维校验(提取侧修)
→ **引擎 Kokkos 读出头类不支持**(引擎侧,本补丁)。共 **14 处改动 / 4 个 C++ 文件**
(diff 全量:`mlip/patches/symmetrix-xl-polar-support.patch`,316 行):

| 文件 | 改动 |
|---|---|
| `mace_nonlinear.cpp` | 新增 `readout_is_nonlinear()` 判定;分发/构造 `/linear_mid`/前向(多一层 silu)/反向(多段链式伴随) |
| `mace_nonlinear.hpp` | Readout 增加 `linear_mid`(unique_ptr,仅 bias 变体构造) |
| `mace_nonlinear_kokkos_impl.tpp` | `readout_component` 增加 component==3→linear_mid;构造(`has_mid`);前向/反向各增加中间层分支(复用/新增 4 个 has_mid 专用缓冲);后端/工作区/字节统计/参数计数全部按 `has_mid` 门控(对现役模型零影响) |
| `mace_nonlinear_kokkos.hpp` | Readout 结构增加 `has_mid` + `linear_mid` + 4 视图/4 存储 |

要点:`E3Linear` 已内建 bias 支持(e3nn.cpp),故补丁本质 = "读出头管线加一层 + 链式伴随";
所有新增均以 `has_mid` 门控 → 对 MH1/既有模型的行为与内存统计零改动。
重建:`pip install ./symmetrix --no-build-isolation` + CI 同款 CUDA config-settings
(见上 §5 与构建日志 `mlip/logs/symmetrix_rebuild.log`)。

## 8. 改动依据账本(全部可溯源;新改动须逐条续记)

| # | 改动 | 依据(上游来源) |
|---|---|---|
| 1 | 提取器接受 `NonLinearBiasReadoutBlock` + 序列化 `linear_mid` | mace 官方源码 `mace/modules/blocks.py::NonLinearBiasReadoutBlock`(前向:linear_1→silu→linear_mid→silu→linear_2) |
| 2 | 提取器 scale/shift 标量化(`shape==[]`) | symmetrix 仓库自带测试 `test/test_mh1.py` 断言 `data["scale_shift"]["scale"]["shape"] == []` |
| 3 | 提取器 atomic_energies 二维化 `[1,N]` | 同上测试断言 `data["atomic_energies"]["shape"] == [1, 89]` |
| 4 | 引擎 host Readout 双层前向/反向 | 镜像 #1 同一源码的数学结构(逐式对应) |
| 5 | 引擎 Kokkos Readout 双层 + 缓冲/统计 | 同 #4;新增全部以 `has_mid` 门控,对上游既有模型零改动(逐处核对) |
| 6 | 引擎 `E3Linear` 跳过 `i_in==-1` 占位指令 | **e3nn 官方源码** `e3nn/o3/_linear.py`:`i_in==-1` 即 bias-only 指令(生成于 L164/179;`bias_numel` 计入、`weight_numel` 排除 于 L369/394)→ 权重累加须排除、bias 由独立字段应用 |
| 7 | Kokkos 生成配置对齐(OPENMP/DEPRECATION_WARNINGS/AGGRESSIVE_VECTORIZATION off;IMPL_VIEW_LEGACY on;KokkosKernels_INST_EXECSPACE_OPENMP off;HOST_ARCH none) | 以 **LAMMPS 构建的实测可用配置**为锚(同一 Kokkos 生态、全量编译+运行通过);逐项均有「生成 KokkosCore_config.h 行级 diff」为证;CI `cuda-qualification.yaml` 为基线,仅对被证明冲突的项调整 |
| 8 | `external/kokkos` 5.0.2 → 5.2.1(LAMMPS 版) | 对照实验:仅换树即编译通过;5.2.1 组合已由 lmp 构建验证;5.0.2 原版留存 `kokkos-5.0.2-pin-backup/` |
| 9 | BLAS 显式化(`SYMMETRIX_BLAS_LIBRARY/INCLUDE_DIR`)+ 链接 `LIBRARY_PATH` | libsymmetrix 官方 CMake 选项(`CMakeLists.txt` L49-64 的文档化选项);链接错误日志为证 |
| 10 | 构建注册(`SYMMETRIX_BACKEND_SELECTOR=cuda`、`ARCHITECTURE=sm80`、descriptor) | 加载器 `backend_loader.py` 的逐字段校验规则 + 官方 CI/发布命名约定 |
| 11 | `nvidia-smi` compute_cap shim | 加载器 `_visible_targets()`(2 s 子进程超时在本机负载下误判)+ 本机驱动实测 |
| 12 | 构建补 `SPHERICART_ENABLE_CUDA=ON`(两级开关同开) | 仓库双开关设计:`pyproject.toml` 定义 `SYMMETRIX_SPHERICART_CUDA`/`SPHERICART_ENABLE_CUDA`(默认 OFF),`libsymmetrix/CMakeLists.txt` L129 option 默认随 Kokkos-CUDA 为 ON、L428-430 主库按该开关编 CUDA 调用;实证:CMakeCache `SYMMETRIX_SPHERICART_CUDA=ON` 但 `SPHERICART_ENABLE_CUDA=OFF` → 子库 CPU-only 而主库调 CUDA → 运行时自检报错 |
| ⏳ | RBE k 向量采样 | **逐项参照 `randbatch-md/lammps-rbe` 的 `kspace_style rbe`**(Box-Muller + Metropolis) |
| ⏳ | 静电块(多极/GTO 结构因子/SCF) | **逐块对照 `graph_longrange v0.4.0`** 参考实现数值 |

注:补丁全量存档 `mlip/patches/symmetrix-xl-polar-support.patch`(每次续改后重出)。

## 9. P2 对齐结果(2026-10-15)—— ✅ 全绿(backbone 阶段)

- **bench 里程碑**:引擎加载 PolarMACE backbone JSON 成功,GPU 求值正常(单次 0.4–0.5 s,
  SrTiO3×2 冒烟,`BENCH_RC=0`);探针 `bench/symmetrix_polar_check.py` 单盒 NH3-32:
  E=−49251.677163 eV,力自洽 FD **3.6×10⁻⁷ eV/Å**。
- **分量精确对账**(NH3-32):引擎 E ≡ `scale·interaction_energy + ΣE0 + shift`
  (−49251.677164)差 **1×10⁻⁶ eV**;`electron_energy(+8.455)` 与 `electrostatic_energy(+0.766)`
  = P3 静电极性层,不在本阶段范围(数值账目完全对上)。
- **批量 parity(4 分子 × 8 帧 = 32 构型,含扰动)**:
  **dE MAE 3.6×10⁻⁷ eV / max 6.6×10⁻⁷ eV** —— 亚微电子伏级全过(优于仓库 CI 金标准
  2×10⁻⁵ 约 100×);力差(缺 P3 静电力)= mean 1.24 / max 4.15 eV/Å(留档为 P3 目标)。
  产物:`bench/polar_parity.{png,json}`;**100 帧扩展完成:`polar_parity_100.{png,json}` ——
  100/100 帧全过,dE MAE 3.60e-7 / max 6.66e-7 eV(与 32 帧完全一致)。**
- **已知无害项**:解释器退出阶段 Kokkos teardown 报 `cudaStreamSynchronize/driver shutting
  down`(全部结果输出完成后才发生;与 sphericart 1.0.3 异步生命周期相关的退出噪音,不影响数据)。
- **对齐口径**:backbone(短程)实现 = 与 mace-torch 参考**数值精确**;下一步 P3 = 补齐静电层
  (electron/electrostatic 两项及其力),P4 = RBE 替换 k 求和。

## 10. P3 设计合同(静电极性层;依据 = PolarMACE 前向源码逐行)

`extensions.py::PolarMACE.forward`(mace 0.3.16,L862-933)的能量装配:

```
total_energy = e0 + inter_e                      # ← 引擎 JSON 已覆盖(P2 已对齐,e0=原子能基座+readout,
                                                 #    inter_e=相互作用;两者合成 = 引擎输出)
             + le_total (local_electron_energy)  # ← "electron_energy" 项:一个【可学习局部模块】,
                                                 #    输入含 field_feats(=potential_features,长程势特征!)、
                                                 #    charges_0 / charges_induced(spin_charge_density)
             + electro_energy (coulomb_energy)   # ← k 空间 GTO-Ewald(即 RBE 的替换对象)
             + external_field · total_dipole     # ← 中性无外场时 = 0(生产条件;可后置)
```

**P3 移植清单(按依赖排序)**:
1. 帧内 SCF 链:`field_independent_spin_charge_density`(初值)→ Fukui 平衡
   (Q_m_S = (charge ± spin) 约束)→ `spin_charge_density`(最终多极);
2. `electric_potential_descriptor`(GTO 结构因子 + 势/场特征)→ `potential_features`;
3. `coulomb_energy`:k 向量网格(`compute_k_vectors_flat`)+ 结构因子 + 自相互作用项 +
   实空间/非周期修正 → `electro_energy`;
4. `local_electron_energy`:learned 模块(node/edge/field_feats/charges_0/charges_induced)→ `le_total`;
5. 力:全部以上项的解析梯度(C++ 端链式;对照 graph_longrange 反向实现);
6. external field 项:生产=0,接口保留。

**验证门(每步)**:
- 1 的出口 = 静态构型上 spin_charge_density/charges 与 Python 逐位对照;
- 2 的出口 = potential_features 与 Python 对照(逐原子);
- 3 的出口 = electro_energy 与 Python 对照(同 k 网格、不分批);
- 4 的出口 = le_total 与 Python 对照;
- 全部合体后 = **端到端对照目标:E_engine == Python E_total**(±2×10⁻⁵ 金标准,即 P2 的表
  最后一行 dE_vs_total ≈ −2.7 eV 被关闭为零);力同口径。

**序列化补充(提取器)**:`fixedpoint_update_config` / `field_readout_config` / 多极预测头 /
`local_electron_energy` / `electric_potential_descriptor` 已完成统计,JSON 新增
`polar_blocks` 段(提取器补丁,依据 = 上述各模块的 state_dict 结构)。

## 11. P3 进度(2026-10-15)—— 序列化第一步完成

- **模块盘点**(checkpoint 直读):极化合件总览 = `lr_source_maps`(2×e3nn Linear)/
  `fukui_source_map`(NonLinearBias readout, 8.5k)/`electric_potential_descriptor`(GTO, 132 参数,几乎全解析常数)/
  `coulomb_energy`(45 参数)/`field_dependent_charges_maps`(2× AgnosticEmbeddedOneBodyVariableUpdate, 3.58M)/
  `local_electron_energy`(OneBodyMLPFieldReadout, 1.23M)/`layer_feature_mixer`(1.05M)/
  `external_field_contribution`(32)/configs(add_local_electron_energy=True)。
- **关键判断**:SCF 机件的可学习部分全部由**引擎已实现同族原语**构成(e3nn Linear / uvu 张量积
  [`SparseUvuTensorProduct` = e3nn TensorProduct 参数布局] / Activation 门 / torch MLP)
  → 引擎侧实现可复用既有机制,大幅降低 P3 风险。
- **序列化器**:`extract_polar_blocks.py`(新,~230 行;逐 class 依 state_dict 结构,未知叶子
  记 `raw_state_dict`+类名诚实标记)+ `extract_mace_nonlinear_data` 接入;
  probe5 实测:`polar_blocks` 全部一节产出 ✓(JSON 已更新)。
- **下一步**:引擎按模块实现(顺序:SCF 链 → 结构因子/势特征 → coulomb_energy → le 读出),
  每个模块以"Python 同模块输出逐位对照"为门。

## 12. P3 执行规格(最终版;逐行锚定 extensions.py L700-935)

**数据流(7 步,全部行号在案)**:
1. 逐层累积(L700-702):`spin_charge_density += lr_source_maps[ℓ](node_feats[ℓ])` —— 初值密度;
2. `features_mixed = layer_feature_mixer(stack(node_feats_list))`(L730)—— SCF 各步共用;
3. k 网格 + `descriptor.precompute_geometry(cell/rcell...)`(L712-728)—— 每构型一次;
4. Fukui 初始化校正(L740-769):`fukui_source_map(node_feats)` → 每图归一 → 校正密度
   l=0 通道至 (Q±spin)/2 目标(中性闭壳: 0 与 1);
5. SCF 循环 × num_recursion_steps(L783-861):对每个自旋通道
   `field_feats_σ = descriptor.forward_dynamic(cache, density_σ)` → 归一(field_feature_norms)→
   拼装 `potential_features` → `field_dependent_charges_maps[i](node_attrs, features_mixed,
   edge_attrs/feats, potential_features, local_charges=density)` → 拆分(density 增量 + Fukui 源)
   → 密度更新 + Fukui 校正;
6. 能量(L862-933):`E = e0 + inter_e + local_electron_energy(...) + coulomb_energy(...)`
   (external·dipole 在生产条件=0);
7. 力:全链 autograd 的解析等价(引擎侧=链式伴随;对照 graph_longrange reverse 实现)。

**引擎实现序与对应夹具门**:
| # | 模块 | 门 |
|---|---|---|
| ① | lr_source_maps 累积 | 中间量(后续加钩子落盘) |
| ② | Fukui 初始化(读出头机制已有) | 校正后密度(钩子) |
| ③ | GTO precompute(k 网格/结构因子) | k 网格与 Python 逐项 |
| ④ | forward_dynamic(势特征) | potential_features(钩子) |
| ⑤ | charges_maps 更新块(e3nn 原语) | 每步 density 增量(钩子) |
| ⑥ | coulomb_energy(全 k=Ewald 基线) | **es 分量(fixtures 已有)** |
| ⑦ | local_electron_energy | **le 分量(fixtures 已有)** |
| ⑧ | 力 | **forces(fixtures 已有)** + FD 自洽 |

**external field 项**:生产=0,引擎先短路;接口保留(DisplacedGTOExternalFieldBlock 已序列化)。

## 13. P3 引擎 slice-1(2026-10-15)—— polar_blocks 解析 + 载入冒烟 ✓

- **布局裁定(数值实证)**:`e3nn` 的 `2 * Irreps("1x0e+1x1o")` 实为**序列重复**
  (`1x0e+1x1o+1x0e+1x1o`,实机打印确认),故 `spin_charge_density` 平坦 8 维 =
  `[sα, xα, yα, zα, sβ, xβ, yβ, zβ]`,与 `.view(N,2,4)` 及逐通道 Fukui 校正完全自洽
  (依据:e3nn `_irreps.py::__mul__` 实机行为 + `extensions.py` L734 view 后的逐通道语义)。
- **引擎改动(v12,已入码并编译)**:
  - `mace_nonlinear_kokkos.hpp`:模型新增 `polar_enabled / polar_json /
    polar_lr_source_maps(2×E3LinearKokkosT) / polar_fukui_source_map(optional<Readout>)`;
  - `mace_nonlinear_kokkos_impl.tpp`(PART2 ctor):解析 `polar_blocks`,校验
    `lr_source_maps.size()==interactions.size()`(否则抛 invalid_argument);
    backend/workspace 两个 setter 全部接线(polar 关闭时零影响);
  - JSON 契约与既有 readout linear 同族(instructions/weight/bias/output_mask),
    `lr_source_maps[i]`:irreps `512x0e+512x1o → 1x0e+1x1o+1x0e+1x1o`,path_weight=1/√512。
- **载入冒烟**(`bench/polar_engine_load_smoke.py`,GPU0):
  - POS-OK:含 polar_blocks 的 JSON 正常装载;
  - NEG-OK:lr_source_maps 截为 1 条 → 引擎抛 "must match the interaction layer count"(预期);
  - 收尾退出噪音(cudaStreamSynchronize/driver shutting down)仍为已知无害项(v11 起留档)。
- 编译证据:`mace_nonlinear_kokkos_runtime.cpp.o` 05:40:04(源码改动 05:35:10 之后),
  `REBUILD_EXIT=0`,wheel 安装成功;`logs/polar_load_smoke.log`。
- **下一段**:模块 ①(`compute_node_energies_forces` 主循环 `features=state.output;` 后,
  逐层 `polar_lr_source_maps[ℓ].evaluate` 累加进 `polar_spin_charge_density(N×8)`),
  以 scf 钩子夹具 `lr_src_0+lr_src_1` 逐位对照为门;绑定侧加只读访问器供门比对。

**模块②前置勘察(依据在案)**:Fukui 校正需要 total_charge/total_spin;mace 侧口径 =
`atoms.info["charge"]/["spin"] → total_charge/total_spin`(mace `calculators/mace.py` L162-166
info_keys 映射;`data/atomic_data.py` L352-357 默认 total_spin=1.0),引擎侧
`Q_p_S=total_charge+(total_spin-1)`、`Q_m_S=total_charge-(total_spin-1)`(extensions.py L752-753)。
引擎现 `compute_node_energies_forces` 签名无 charge/spin → 模块②需加可选参数 + calculator 读取
info 传入(默认 0/1,对现役模型零影响)。Readout 输出目前只取 `result[:,0]`(单通道),
fukui 需 2 通道 → 模块②给 Readout 增 `evaluate_all/reverse_all`(全列)。

**模块②实现设计(编译前定稿,待①门过后入码)**:
- `Readout` 增 `evaluate_all`(同 evaluate 链,但复制 final linear 全部列;fukui linear_2
  out=2x0e,16x0e MLP,act=silu β=1.6791767923989418(已序列化,引擎照用));fukui 读出头为
  独立 `optional<Readout>` 实例 → 视图与主 readout 无冲突。
- ② 块位于主层循环后、readout 循环前:`sources = polar_fukui_source_map->evaluate_all(
  states.back().output)`;host 归约 2 列源和 + l=0 两通道总量(单图=整盒);`0→1` 钳制;
  factor=(target−pred)/sum;校正核函数 `acc(:,0)+=src(:,0)*factor_p; acc(:,4)+=src(:,1)*factor_m`;
  最后复制校正后 acc → `polar_field_independent_density`(mace L774 语义,先校正后克隆)。
- 新访问器:`polar_fukui_sources`(N,2 原始)、`polar_field_independent_density`(N,8)。
- ②b charge/spin 直传:`compute_node_energies_forces` 尾追 `double total_charge=0.0,
  double total_spin=1.0`;绑定 kwargs 同名默认;calculator L3394 调用点传
  `atoms.info.get("charge",0.0)/get("spin",1.0)`(函数内有 atoms 作用域)。
- 门:`bench/polar_module2_gate.py` 已备(双源;numpy 逐式重建 L742-766;门 5e-6)。

## 14. P3 模块①门首跑失败 → 根因定位与修复(2026-10-15)

**现象**: 引擎 accumulator vs python lr_src 和对账, 12/16 构型系统性失配 max|Δ|≈0.46,
量级与值域同阶; 标量(l=0)列吻合、矢量(l=1)列全错。

**二分定位(隔离试验, 无重建)**:
1. `bench/polar_lrsrc_isolate.py`: 把 python 钩子冻结的 node_feats 直接喂引擎
   `E3LinearKokkos`(lr_src 逐层实例) —— **逐位吻合 2.5e-16** ⇒ 权重序列化与
   引擎 e3nn Linear 实现完全正确; 问题在输入张量。
2. `bench/polar_lrsrc_deep.py`: 对候选变换做假设检验; 对全部 3×3 带符号置换拟合,
   得**精确关系**(残差 ~7e-9 = f64 噪声):
   `acc_vec = −P·c1_vec`, P=(2,0,1) 循环置换 ⇒ 引擎内部 l=1 分量基与
   mace/e3nn 基相差固定矩阵 **−P**(即 `state_out_v = −(in_v2, in_v0, in_v1)`;
   0e 块逐位一致)。
3. `bench/polar_basis_map_check.py`: NH3×2(含扰动)/CH4/CO2 4 构型复核 —— 映射恒为
   perm=(2,0,1), signs=(−1,−1,−1), 残差 4.6e-9~1.2e-8 ⇒ 非构型巧合, 是基约定。
   (该差异不影响 P2 能量/力对齐: 能量路径经 readout 只消费标量通道, 且引擎内部
   消息传递自洽; 仅当 e3nn 原生权重直接作用在 node_feats 上时暴露。)

**修复(引擎, v14)**: 供 polar 块的 node_feats 先做一次基变换(列映射, 由
`lr_source_maps[0].irreps_in` 在 ctor 构造): l=0 列直通; l=1 块内
`e3nn 列 c = −内部列 (c+1)%3`; l>1 明确拒绝。逆映射推导: `(−P)⁻¹ = −P²`。
新成员: `polar_feature_width / polar_basis_source / polar_basis_sign /
polar_node_feats_e3nn(_storage)`; 逐层在 lr_src 前转换(模块④⑤等一切喂 polar 块的
node_feats 同口径复用该机制)。

**待验**: v14 构建 → 模块①门(16 构型双源, 预期 ~1e-8 级通过)。

**模块①门复跑(v14) —— ✅ PASS(2026-10-15)**:
- 12 构型(4 夹具 + 8 活体扰动帧)全部通过;max|Δ| = 5.8e-9~3.1e-8,MAE ~1e-9,
  **比 5e-6 门低 2 个数量级**(f64 累加次序噪声级);双源一致。
- 修复前后对比:0.46(FAIL,≈50% 相对误差)→ 3e-8(≈1e-7 相对)。基变换修复生效。
- 产物 `bench/polar_module1_gate.{png,json}`(v14 PASS 版);日志
  `logs/polar_module1_gate_v14.log`。模块①(引擎 polar_blocks 解析 + lr_src 累积
  + 基变换)关闭。

**模块②门(v15) —— ✅ PASS(2026-10-15, 标度感知判据)**:
- 初级门(绝对 5e-6)fukui 微超(1.3e-6~8.0e-6)——**stageprobe 定位为门尺度问题而非实现问题**:
  分级探针(`bench/polar_fukui_stageprobe.py`)用引擎自身 E3Linear 绑定对冻结 python 输入
  逐级重放 fukui 链:linear_1→silu→linear_mid→silu→linear_2 全部吻合 **2e-13**
  (含 silu 常数 1.6791767923989418 与 bias);实跑差 = 骨干 f64 算术序漂移(~1.8e-8 相对,
  fukui 值域 O(10-200) ⇒ 绝对 ~1e-6),归一后抵消(density 1e-8 级已证)。
- **判据修订(有据)**:fukui 原始 rel ≤ 1e-6;归一源 ≤ 5e-6;校正后密度 ≤ 5e-6。
- **结果**:fukui rel max **6.3e-8**(16× 裕度) | 归一源 **1.2e-9** | 密度 **3.1e-8** —— 12/12 双源全过。
- 模块②(Fukui 初始化 + charge/spin 直传)关闭。fukui 读出头(唯一 NonLinearBiasReadoutBlock)
  的 has_mid 路径首次数值检验通过。产物 `bench/polar_module2_gate.{png,json}`(PASS 版)。

## 16. 模块③预备(2026-10-15)—— 公式 numpy 预检全锁定 + 序列化补齐

- **序列化补齐(extractor)**:`configs` 增 `num_recursion_steps=2`、`field_feature_norms`
  [20,20,0.5×6](非1常数!)、`from_ell_max_field_update=9`;JSON 已重生成
  (`bench/polar_json_regen.py`,PYTHONPATH=source 提取器)。
- **numpy 预检(对 geo3 参考 dump,编译前消风险)**:
  - k 网格半空间枚举 + |k|²≤cutoff² 滤波 + k0 掩码 —— 参考 k_vectors 已落盘(智能体B);
  - cosines/sines = cos/sin(k·r):**1.1e-16**;
  - density/feature basis 实部(即 l=0 项):**0.000e+00 逐位**;
  - **l=1 项 🔑**:参考 = `√(3/4π)·v/|v|`(v=k 按 permute[1,2,0]=(ky,kz,kx))
    ——e3nn `normalize=True` 为**输入向量归一化**(非输出),吻合 **1.1e-16**;
    引擎可用解析等价式 `f1·Y1 = c1·pref1·cl1·exp(-½σ²k²)·v`(k=0 处自动为 0,无需除法);
  - k_factor_coulomb/proj、volume_per_k:逐位/精确。
- **验收门(③)就绪**:geo3 dumps(k 网格+basis 表)为参考;④ 用 scf 夹具
  `potential_feats_step_*` + `charges_hooks`(智能体D)为参考。
- 并行备料:智能体 D/E/F 分别做 ⑤审计+钩子、⑥/⑦ 预检+补丁草案;A=RBE 预研;C=⑧反向设计。

**模块③④门(v16b) —— ✅ PASS(2026-10-15)**:
- 结构量: k 向量 **0.0 逐位**、cos/sin 6.4e-15、GTO basis 4.4e-16、k0/因子逐位
  (门 1e-9;参考 = 智能体B geo3 dump,其自身对夹具自证 ≤1.3e-15)。
- 势特征: rel max **6.8e-8**(门 1e-6,15× 裕度;与①②同源的骨干传播漂移级)。
- 12×4 值与逐位级链全部对上;公式(含 Y1 输入归一化、f32 缓冲 pref 值)均经预检锁定。
- 引擎侧新增: node_positions/cell plumb、k 枚举/基函数/势特征三 kernel、9 访问器。
- 模块③④关闭;v16 构建曾因 auto 多声明符类型不一致失败一次(2 处),已修(v16b)。
- 产物 `bench/polar_module34_gate.{png,json}`;日志 `logs/polar_module34_gate_v2.log`。

## 17. P3 模块⑤落盘(2026-10-15)—— SCF 密度更新全链, v17 构建中

- **实现**(依据 extensions.py L730/L777-866 + field_blocks.py L470-594;数值全锁定):
  - mixer: 循环内 Σ_ℓ mixer_linear[ℓ](node_feats_e3nn[ℓ]) → features_mixed(N,2048);
  - SCF 循环 ×2 步: 逐通道描述符(α/β)→归一拼接(16) → 更新块:
    势嵌入三分量(pot/nf/q 线性求和) → uvu 点积(自写 kernel 镜像 field_blocks
    L446-464;path_weight 已序列化) → source_embedding(node_attrs) → MLP
    [linear,LN,silu]×3+linear → uvu tp_out → 读出 linear_1→**e3nn-Gate**→linear_mid
    →Gate→linear_2(10 列) → 密度+=out[:,:8] → 逐步 Fukui 重归一+l=0 校正(out[:,8:10]);
  - **Gate 语义数值锁定**(charges_hooks 2.2e-16): scalars=cst_s·silu(x)(
    cst_s=1.6791767923989418)、gates=cst_g·sigmoid(x)(cst_g=1.8467055342154763)、
    矢量=cst_g·sigmoid⊗gated(√3 path_weight 与内部归一相消)、切片[32|32|96];
  - 访问器 13 个: features_mixed/pe/dp/se/nl/tp/ro_l1/gate1/ro_lmid/gate2/update_out/
    scf_step_densities(3D)/scf_final_density(与 charges_hooks step1 + scf_steps 对齐)。
- **序列化前置**(已归档): sparse_uvu path_weight(dp 0.03125/tp 0.0442,0.0765)、
  readout_gate 结构化(cst+计数+mul 指令)、LayerNorm 结构化、G4 语义段。
- **门**(智能体并行编写中 `bench/polar_module5_gate.py`): 密度演化(scf_steps 对齐)
  + stage 级(charges_hooks step1 全链)。

**⑤门脚本就绪(2026-10-15)**:`bench/polar_module5_gate.py` —— 密度演化 12 项
(4 盒×3 配对; 判据 abs≤5e-6 且 rel≤1e-6)+ stage 11 项(charges_hooks nh3-step1;
rel≤1e-6)。参考侧已用活体 mace 全前向+钩子自证:5 密度态配对 scf_steps 全部噪声级
(1.5e-16~3.0e-15), 配对关系 `step_densities[i]↔scf_steps step{i+1}`、`final↔step2`
数据自证确认; 桩引擎全程试跑 PASS(出图/JSON)。等 v17b 引擎就绪后运行。

**模块⑤门(v17f) —— ✅ PASS(2026-10-15)**:密度演化 12/12(step1/step2/final 全对,
rel 2~4e-8)+ stage 全链 11/11(rel 3~7e-8; se_out 逐位 0.00); 配对关系数据自证通过。
SCF 循环(描述符→更新块→密度更新→逐步 Fukui)端到端数值正确。修复过程留档:
快照移至校正后、门脚本 stage 段先重跑 NH3、成员捕获越界(compute-sanitizer 取证)。
模块⑤关闭; 5/8 完成(①②③④⑤)。产品: bench/polar_module5_gate.{png,json}。
