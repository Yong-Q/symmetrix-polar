# P3 模块⑧（力 / 全链反向）—— 精确补丁提案（最终版，2026-10-15）

**定位**：把 `docs/POLAR_REVERSE_DESIGN.md`（429 行设计）+ `docs/SYMMETRIX_POLAR_RBE_PLAN.md` §12/§14-17
细化为**可直接应用**的补丁提案 = 文本锚点 + 插入代码全文。**不写行号 diff**（引擎树被⑤等并行落码，
行号漂移 >500 行；本提案全部使用唯一文本锚点，见 §8 自检第 1 组）。**只写 `mlip/docs/` 与 `mlip/bench/`，
未改引擎树、未构建、未 git。**

**引擎快照（提案编写读数，md5 / 行数 / 时刻 UTC）**：

| 文件 | md5 | 行数 |
|---|---|---|
| `libsymmetrix/source/mace_nonlinear_kokkos.hpp` | `fa20b4fd45302d3b5384dc3980307526` | 815 |
| `libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp` | `a4c3b5a5f4202819b74db2ba670aabc6` | 8292 |
| `symmetrix/source/cpp/mace_nonlinear_kokkos.cpp`（binding） | `ba096f9a333db0df3b8e2fdd719f4d8f` | 587 |
| `pair_symmetrix/pair_symmetrix_mace_kokkos.cpp` | `482ddc73466e2cb182e4388c1650082b` | 2289 |

（快照时刻 2026-10-15 09:54 UTC。**合并前先跑 §8 自检第 1 组**核对锚点存在性；锚点若失配，
以自检给出的近似锚为准重定位。）

依据锚点：mace 0.3.16 安装副本 `/opt/matos-tools/conda/envs/matos-mlip-py312/.../mace/modules/`
（`extensions.py` L700-935 实测读数、`field_blocks.py`、`radial.py`）；graph_longrange v0.4.0
`/workspace/caches/conda/envs/polar/lib/python3.12/site-packages/graph_longrange/`
（`features.py` L28-135 实测读数、`energy.py` L17-30 实测读数）；引擎同树只读核对。

---

## 0. 闭合程度总表（先说结论）

| 交付面 | 状态 | 说明 |
|---|---|---|
| a. hpp 前向缓存清单 + 新增视图 | **完整** | §2.1 清单（逐模块 × 逐量 × 保存/重算决策）+ §2.2-2.4 插入文本 |
| b. `Readout::reverse_all` | **完整** | §3 全文（与 `evaluate_all` 逐行镜像；`impl.tpp` PART1） |
| c. 极性链反向 kernel（lr_src / 初始 Fukui / SCF 逐步 / 描述符+相位→节点力 / ⑥ ρ̂） | **完整**（文本级） | §4 全文；⑥ 分支门控 `polar_coulomb_enabled`（⑥ 落码时置位）；⑦ le 反向**不在本提案**（module7 §8 口径），本提案只定义种子视图与一次性消费语义 |
| d. cell 伴随（NPT 维里） | **下一 slice**（结构+累加位置已定，阻断理由 4 条） | §6；本 slice 落 `polar_cell_adjoint` 占位 + 静默错误 guard |
| e. binding 新访问器 | **仅调试必需**（`polar_node_forces` / `polar_reverse_seed_terms`；建议项单列） | §7 |
| 节点级力接线（两处修改） | **完整** | §5（tpp `reduce_node_forces` + pair style `force_reduction`） |

**关键裁定（与设计文档的差异，共 3 条）**：

1. **重放（replay）取代逐步激活缓存**（设计 §4 "保存 vs 重算"旋钮的最终选型）：SCF 各步的更新块/描述符
   前向**不逐步保存激活**（预算 ~53 kB/原子 → 10⁵ 原子 5.3 GB，超设计 §4 上限），改为反向时按步
   **重放前向**（同输入、同算子、同次序 ⇒ 位级一致）+ 只保存 4 类小量（校正后密度链、Fukui 标量、
   P_last 由种子进、几何缓存）。反向净成本 ≈ +1 次 SCF 前向。新增缓存体量：~24 B/原子/步 + 种子，
   记入 §2.1。
2. **`step_densities_post` 新增**：逐步反向的链输入 d_{i+1} 是**校后**密度；现⑤只快照了**校前**密度
   （`polar_scf_step_densities`，快照位于 Fukui 校正之前）。新增 `(S,N,8)` 校后快照视图，
   **不触碰**⑤已导出视图的语义与⑤门配对（见 §4.0-C）。
3. **⑥ coulomb 种子内联自包含**（不引用⑥的中间视图，反向时按同一 GTO 表重算 ρ 前向值；
   f64 差异在 eps 级）；⑥ 落码时只需把 `polar_coulomb_enabled` 置 true（若⑥同时暴露其
   `rho_re/im` 视图，可将重算段替换为直接消费，位级一致）。

---

## 1. 反向链序与语义裁定（实现前必读）

### 1.1 链序（与 `POLAR_REVERSE_DESIGN.md` §3.2 一致，按本文件视图名）

```
[种子] polar_reverse_seed_*（⑥/⑦ 投递）+ ⑥ coulomb 内联种子（q̄→d_final、相位→r̄）
   → for step = M-1 … 0（M=num_recursion_steps=2）:
        replay(step, d_in)                     # 前向重放:描述符(③④)+更新块(⑤)→ 重放视图
        Fukui 校正反向（缓存标量）              # d̄ += pred̄(l=0)、s̄ → out̄[8:10]
        out̄ = [ d̄_pre(0:8) | s̄(0:2) ] → 更新块反向（读出 Gate→uvu tp→MLP/LN→uvu dot→势嵌入）
        → fmix̄ += ; P̄_i ; d̄_ce +=
        P̄_i += 种子(仅末步) → 描述符反向（逐通道:顺 permute/SI→project→assemble→q̄+相位）
        d̄_i = d̄_pre + q̄_α + q̄_β + d̄_ce      # 与下一（更早）步接力
   → 初始化 Fukui 反向（异构源=fukui 读出头）→ s̄0 → Readout::reverse_all → ⑧-A → 末层 layer_adjoint +=
   → ⑧-1 lr_src 逐层反向 → ⑧-A → 各层 layer_adjoint +=
   → ⑧-5 mixer 逐层反向 → ⑧-A → 各层 layer_adjoint +=
   → polar_node_forces(3n+c) = −r̄(n,c)
[骨架] 已在播种后、主反向循环前完成注入 ⇒ 骨架反向（products→gate→conv→edge）正常消费 layer_adjoint
```

### 1.2 语义裁定（逐条与设计 §3.3/§3.5 对账）

1. **累加注入**：所有进入 `states[l].layer_adjoint` 的写作必须 `init=false`（读出头播种是**覆盖**：
   `impl.tpp` "readout" 播种段 `readouts[layer].reverse(state.output,scale,state.layer_adjoint)` 第三参
   缺省 → `Readout::reverse` 内层 `linear.reverse(seed, input_adjoint)` 缺省 `initialize_input_adjoint=true`
   → `e3nn_kokkos.cpp` "reverse" 实现先 `ordered_kokkos_deep_copy(input_adjoint,0)`）。本提案的
   ⑧-A 散射核一律 `+=`（不带清零）★。
2. **scale 无关**：极化各项不乘 `scale`；种子一律 ∂E/∂· = 1（设计 §3.5）。
3. **展开式反向**：重放必须逐算子复刻前向，禁止"再跑一个更收敛的 SCF"（设计 R7）。
4. **⑧-A 不自伴**：前向 `e3nn 列 c = s_c·内部列 src(c)`，`src(c)=off+3m+((comp+1)%3)`，
   `s_c=−1`（l=1，ctor 已证 l>1 拒绝）。伴随 = `Tᵀ`，而 `Tᵀ = −P² ≠ T = −P`（P 为 3-循环，`P³=I`）
   ⇒ **必须用同一映射做散射（c→src(c)），不能"正向再用一遍"**。散射的单射性：`c ↦ src(c)` 在每个
   l 块内是双射（l=0 恒等、l=1 三循环）⇒ 无写冲突；实现仍用 `atomic_add` 自卫（§4.2 核）。
5. **Fukui 校正反向闭式**（初始与逐步同构，派生见设计 §2 ⑧-2；本文件给出最终式）：
   前向 `d_out(l=0) = d_pre(l=0) + s̃·F`，`s̃ = s/S`，`pred = Σ_n d_pre(n,0/4)`，`F = (T−pred)/S`；S=0 时
   前向用 1 替换（钳制）。记 `F̄_σ = Σ_n s̃_nσ·d̄_out(n,c_σ)`，`Sinv = clamped ? 1 : S_raw`：
   ```
   d̄_pre(n,c_σ) += pred̄_σ = −F̄_σ/Sinv
   s̄(n,σ) = F_σ·d̄_out(n,c_σ)/Sinv − [clamped ? 0 : 2·F̄_σ·F_σ/S_raw]     (c_σ = 0 / 4)
   ```
   （第二项含 `∂s̃/∂S` 与 `∂F/∂S` 两条链之和，见 §4.4 反向核内注释。）
6. **节点级力的符号**：骨架约定 `node_forces(3e+c) = −∂E/∂xyz(edge)`（"nonlinear edge forces" 段
   `forces(coordinate)=-vector_adjoint`），`reduce_node_forces`/pair style 归约后即 `F = −∂E/∂r`。
   极化位置伴随写 `polar_node_forces(3n+c) = −r̄(n,c)`，语义与 `node_forces` 完全一致（可直接相加）。
7. **种子一次性消费**：`polar_reverse_seed_*` 视图 + `polar_reverse_seed_terms` 位掩码在
   `compute_polar_reverse_forces` 末尾清零（consumed-once）。⑥/⑦ 的反向段在调用本函数**之前**
   投递种子；⑦ 未落码时掩码 bit0=0（该能量项在引擎中不存在 ⇒ 力恒零，与引擎能量自洽；门的
   `--stage post` 才检查 python 总力 parity，见 §9）。
8. **无种子早退**：`polar_reverse_seed_terms==0` ⇒ 只清零 `polar_node_forces` 后返回
   （对本轮①-⑤前向 = 零开销；保证 ①-⑤ 门的数值与访问器完全不受⑧落码影响）。

---

## 2. hpp 补丁（`mace_nonlinear_kokkos.hpp`）

### 2.1 前向缓存清单（反向需要什么；保存 / 重算 / 无需）

| 模块 | 反向需要的量 | 现状 | 本提案决策 |
|---|---|---|---|
| ① lr_src | 无（线性反向不需前向输入；权重常量） | — | 无需 |
| ② 初始化 Fukui | 原始源 `polar_fukui_sources`（N×2）；`S/pred/F` 标量（2 通道）；校前 l=0 | 源已有；标量**被丢弃** | **新增** `polar_fukui_init_scalars`（6 个宿主标量，~48 B/次） |
| ③ 几何 | k/|k|²/k0/两套 basis/cos/sin/**volume**/rcell | 视图已有；`volume` 为宿主局部量、rcell 未留 | **新增** `polar_cell_volume`（double）+ `polar_rcell_host[9]`（维里预留，见 §6） |
| ④ 初始态势特征 | **无能量消费方**（mace 侧 initial-density 描述符不产出能量；仅诊断访问器） | — | 反向 **no-op**（文档化；若未来⑥/⑦改用初态特征，需补 project/assemble 反向） |
| ⑤ SCF 逐步 | 每步：`d_in(i)`（= 初态/校后链）、`out_i`(N×10)、Fukui 标量、更新块全部激活（pe/dp/nl/gate/ro）与 `P_i`(N×16) | 只有旧步被覆写的最后一步视图；校后密度缺失 | **重放**（d_in/`out` 由重放重建）；**新增** `polar_scf_step_densities_post`(S,N,8) + `polar_scf_fukui_scalars`(S×6) |
| ⑥ coulomb | 前向 ρ（实/虚, Nk）；`volume`；q（末态密度和） | 未落码 | 反向时用③的 basis 表**重算 ρ**（同一算子/次序；⑥ 落码后可换成其 ρ 视图） |
| ⑦ le | `q_up/v_up/nf_last`（若保存则 ~12 kB/原子） | 未落码 | 重放可全员覆盖（输入全在）；**本提案不实现**，只定义种子视图（见 §4.1） |
| ⑧-A | 无（用 ctor 的 `polar_basis_source/sign`，双向共用） | 已有 | 无需 |

**重放视图体量（本提案新增，单帧）**：`polar_rev_pe`(N,2048) + `tp`(N,2048) + `fmix_adj`(N,2048) +
`nfeat_adj`(N,2048) + `dp/se/pe_adj/dp_adj/se_adj`(各 N,512) + `nl_in`/`nl_in_adj`(N,1024) +
MLP h/a(各 3×64) + ro/gate/out 及伴随（各 ≤160, ~12 张）≈ **41 kB/原子**（f32；f64 ×2），
外加 `polar_node_forces` 24 B/原子。对比设计 §4 的 2.0-3.2 GB 预算（10⁵ 原子）仍在上限内
（若超预算，`polar_rev_tp/fmix_adj/nfeat_adj` 三张最大视图可降级为"先 d̄、后混入"的顺序复用，
见 §4 实现注）。

### 2.2 hpp 插入 A：公开视图/成员 + 反向入口声明

**锚点**（`mace_nonlinear_kokkos.hpp`，唯一，count=1）：

```cpp
        polar_nl_b_storage;                                             // ⑤暂存
```

在其**之后**插入：

```cpp
    // --- P3 模块⑧: 力/全链反向 (2026-10-15; 前向缓存增量 + 反向工作区 + 节点力累加器) ---
    // 语义: node 级力累加器,写作 −∂E/∂r(n,·),与 node_forces(边级,−∂E/∂xyz_edge) 同符号约定;
    // 归约/消费点见 §5。仅标准 materialized 路径填充(polar 前向已拒绝 direct/prepared)。
    Kokkos::View<double*> polar_node_forces,polar_node_forces_storage;   // (3N,)
    // 种子(⑥/⑦ 反向段在 compute_polar_reverse_forces 之前投递;consumed-once,调用后清零):
    int polar_reverse_seed_terms=0;                                      // bit0=⑦le, bit1=⑥coulomb
    // ⑥ coulomb 反向开关:本提案默认 false(⑥ 未落码);⑥ 落码时在其 ctor 段置 true。
    bool polar_coulomb_enabled=false;
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_seed_density_adjoint,
        polar_seed_density_adjoint_storage;                              // (N,8) ∂E/∂d_final(末态密度)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_seed_density0_adjoint,
        polar_seed_density0_adjoint_storage;                             // (N,8) ∂E/∂d_init(初校正后密度)
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_seed_potential_adjoint,
        polar_seed_potential_adjoint_storage;                            // (N,16) ∂E/∂P_last(末步势特征)
    // 前向缓存增量(§2.1):
    Kokkos::View<Precision***,Kokkos::LayoutRight> polar_scf_step_densities_post;
                                                                        // (S,N,8) 每步 Fukui 校正【后】密度
    std::array<Precision,6> polar_fukui_init_scalars{};                  // [Sα,Sβ,predα,predβ,Fα,Fβ]
    std::vector<std::array<Precision,6>> polar_scf_fukui_scalars;        // (S,6) 同上(逐步)
    double polar_cell_volume=1.0;                                        // 模块③ volume(宿主缓存)
    Precision polar_rcell_host[9]={};                                    // 模块③ rcell(宿主;维里预留)
    // cell 伴随占位(硬缺口 3;本 slice 仅清零与累加接口,消费者见 §6):
    Kokkos::View<Precision*> polar_cell_adjoint,polar_cell_adjoint_storage;  // (9,)
    // 反向工作区(首次调用 ensure_view 按对象维度分配;命名沿 §4):
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_rev_density_adjoint,
        polar_rev_density_adjoint_storage,                               // (N,8) d̄ 链
        polar_rev_channel_adjoint,polar_rev_channel_adjoint_storage,     // (N,8) 逐步 d̄ 汇入
        polar_rev_source_adjoint,polar_rev_source_adjoint_storage,       // (N,8) ⑥ q̄(用 0..3 列)
        polar_rev_feats_adjoint,polar_rev_feats_adjoint_storage,         // (N,8) 单通道 P̄ 拆分
        polar_rev_potential_adjoint,polar_rev_potential_adjoint_storage, // (N,16) P̄
        polar_rev_position_adjoint,polar_rev_position_adjoint_storage,   // (N,3) r̄
        polar_rev_nfeat_adjoint,polar_rev_nfeat_adjoint_storage,         // (N,2048) e3nn node-feat 伴
        polar_rev_fmix_adjoint,polar_rev_fmix_adjoint_storage,           // (N,2048)
        polar_rev_fukui_source_adjoint,polar_rev_fukui_source_adjoint_storage, // (N,2)
        polar_rev_out_adjoint,polar_rev_out_adjoint_storage,             // (N,10)
        polar_rev_gate1_adjoint,polar_rev_gate1_adjoint_storage,         // (N,gate_w)
        polar_rev_gate2_adjoint,polar_rev_gate2_adjoint_storage,         // (N,gate_w)
        polar_rev_ro1_adjoint,polar_rev_ro1_adjoint_storage,             // (N,ro_w)
        polar_rev_ro2_adjoint,polar_rev_ro2_adjoint_storage,             // (N,ro_w)
        polar_rev_tp_adjoint,polar_rev_tp_adjoint_storage,               // (N,tp_w)
        polar_rev_nl_adjoint,polar_rev_nl_adjoint_storage,               // (N,hidden) 滚动激活伴
        polar_rev_h_adjoint,polar_rev_h_adjoint_storage,                 // (N,hidden) LN 输入伴
        polar_rev_nlin_adjoint,polar_rev_nlin_adjoint_storage,           // (N,nl_in_w)
        polar_rev_dp_adjoint,polar_rev_dp_adjoint_storage,               // (N,512)
        polar_rev_se_adjoint,polar_rev_se_adjoint_storage,               // (N,512)
        polar_rev_pe_adjoint,polar_rev_pe_adjoint_storage;               // (N,pe_w)
    // 重放前向值(不触碰⑤导出视图):
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_rev_step_potential,
        polar_rev_step_potential_storage,                                // (N,16)
        polar_rev_pe,polar_rev_pe_storage,                               // (N,pe_w)
        polar_rev_dp,polar_rev_dp_storage,                               // (N,512)
        polar_rev_se,polar_rev_se_storage,                               // (N,512)
        polar_rev_nl_in,polar_rev_nl_in_storage,                         // (N,nl_in_w)
        polar_rev_nl_h,polar_rev_nl_h_storage,                           // (N,3*hidden)
        polar_rev_nl_a,polar_rev_nl_a_storage,                           // (N,3*hidden)
        polar_rev_nl_out,polar_rev_nl_out_storage,                       // (N,nl_out_w)
        polar_rev_tp,polar_rev_tp_storage,                               // (N,tp_w)
        polar_rev_ro1,polar_rev_ro1_storage,                             // (N,ro_w)
        polar_rev_ro2,polar_rev_ro2_storage,                             // (N,ro_w)
        polar_rev_gate1,polar_rev_gate1_storage,                         // (N,gate_w)
        polar_rev_gate2,polar_rev_gate2_storage,                         // (N,gate_w)
        polar_rev_out,polar_rev_out_storage,                             // (N,10)
        polar_rev_nlout_adjoint,polar_rev_nlout_adjoint_storage;         // (N,nl_out_w)
    // 诊断:最近一次反向实际消费的种子掩码(0=无能量项⇒零力;3=⑥+⑦)。非 consume-once。
    int polar_reverse_terms_last=0;
    // k 空间反向暂存:
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_rev_ab,
        polar_rev_ab_storage;                                            // (Nk,16)=[Ā|Ŷ] 投影反向
    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_rev_acc_cos,
        polar_rev_acc_cos_storage,polar_rev_acc_sin,polar_rev_acc_sin_storage;
                                                                         // (Nk,4) assemble 反向
    Kokkos::View<Precision*> polar_rev_rho_adj_re,polar_rev_rho_adj_re_storage,
        polar_rev_rho_adj_im,polar_rev_rho_adj_im_storage,                // (Nk)
        polar_rev_rho_fwd_re,polar_rev_rho_fwd_re_storage,
        polar_rev_rho_fwd_im,polar_rev_rho_fwd_im_storage,                // (Nk) ⑥ 重算 ρ
        polar_rev_transform,polar_rev_transform_storage;                  // (8,) Fukui 反向系数
    // 反向入口:必须在 readout 播种之后、骨架反向主循环之前调用(§4.0-A)。
    void compute_polar_reverse_forces(int num_nodes);
    // ⑧-A 公开投递口(⑦ le 反向的 n̄f_last 路径;内部 init=false 累加;定义见 §4.2 尾)。
    void polar_inject_node_features_adjoint(int layer,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> adjoint);
    // pair style 消费口(§5.2;binding §7.1 同源)。
    bool polar_blocks_enabled() const { return polar_enabled; }
```

（注：`std::array` 已由 `<array>` 引入——`mace_nonlinear_kokkos.hpp` 现含 `<limits>/<memory>/...`，
若编译报缺头则在 include 区补 `#include <array>`；tpp 已用 `std::array`（`mh1_*` 访问器），
实际已可用。）

### 2.3 hpp 插入 B：`Readout::reverse_all` 声明

**锚点**（唯一，count=1）：

```cpp
        void evaluate_all(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
                          Kokkos::View<Precision**,Kokkos::LayoutRight> output);
```

在其**之后**插入：

```cpp
        // 模块⑧(R1): evaluate_all 的伴随。seed_all(N×k) = ∂E/∂output(k=linear_2 输出列数,
        // fukui 头 k=2);中间伴覆盖写,最终伴按 initialize_input_adjoint 选择清零/累加
        // (进 states[ℓ].layer_adjoint 必须 false —— 播种为覆盖写,见 mace_nonlinear_kokkos.hpp
        //  polar 段注释)。依赖:调用前必须先对同一实例执行过 evaluate_all(缓存 hidden/
        // hidden_mid),与现有 reverse 的隐藏依赖完全同构。
        void reverse_all(Kokkos::View<const Precision**,Kokkos::LayoutRight> seed_all,
                         Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
                         bool initialize_input_adjoint);
```

### 2.4 hpp 插入 C：私有表（无；全部并入 §2.2 公开块）

无额外私有成员——本提案新增成员全部置于公开区（与⑤已落码风格一致：polar 视图均公开，
binding 与门脚本可读；`compute_polar_reverse_forces` 声明为公开也是 nvcc 扩展 lambda 的硬要求，
与同文件既有注释 "nvcc requires functions enclosing Kokkos extended lambdas to be public" 一致）。

---

## 3. tpp PART1：`Readout::reverse_all` 全文

**锚点**（唯一，count=1；`evaluate_all` 定义的结尾三行 + 闭括号）：

```cpp
    ensure_view(result,result_storage,input.extent(0),linear_2.output_dimension());
    linear_2.evaluate(activated,result);
    ordered_kokkos_deep_copy(output,result);
}
```

在其**之后**插入（`SYMMETRIX_MACE_NONLINEAR_KOKKOS_PART == 1` 区内、`Readout::reverse` 定义之前/之后均可，
该文本唯一即可）：

```cpp
template<typename Precision>
void MaceNonlinearKokkosT<Precision>::Readout::reverse_all(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> seed_all,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
    const bool initialize_input_adjoint)
{
    // 模块⑧ (R1): evaluate_all 的伴随(供 fukui_source_map 的 2x0e 读出)。
    // 与 evaluate_all 同链、同缓冲: hidden/hidden_mid 由前置 evaluate_all 缓存,
    // 本函数只读它们并写伴随; 中间层伴(activated_adj/mid_adj/act_mid_adj/hidden_adj)
    // 一律覆盖写(init=true),仅最终 linear_1 的输入伴按 initialize_input_adjoint 处理。
    if(!nonlinear)
    {
        linear.reverse(seed_all,input_adjoint,initialize_input_adjoint);
        return;
    }
    if(seed_all.extent(1)!=static_cast<std::size_t>(
            linear_2.output_dimension())
        ||seed_all.extent(0)!=input_adjoint.extent(0))
        throw std::invalid_argument(
            "MACE_Nonlinear readout reverse_all seed dimensions are inconsistent.");
    ensure_view(hidden,hidden_storage,seed_all.extent(0),linear_1.output_dimension());
    ensure_view(activated_adj,activated_adj_storage,seed_all.extent(0),
        linear_1.output_dimension());
    ensure_view(hidden_adj,hidden_adj_storage,seed_all.extent(0),
        linear_1.output_dimension());
    auto local_hidden=hidden;auto local_activated_adj=activated_adj;
    auto local_hidden_adj=hidden_adj;
    const Precision constant=activation_constant;
    if(has_mid)
    {
        ensure_view(hidden_mid,hidden_mid_storage,seed_all.extent(0),
            linear_1.output_dimension());
        ensure_view(mid_adj,mid_adj_storage,seed_all.extent(0),
            linear_1.output_dimension());
        ensure_view(act_mid_adj,act_mid_adj_storage,seed_all.extent(0),
            linear_1.output_dimension());
        auto local_hidden_mid=hidden_mid;
        auto local_mid_adj=mid_adj;
        auto local_act_mid_adj=act_mid_adj;
        linear_2.reverse(seed_all,activated_adj);
        Kokkos::parallel_for("reverse_all readout mid silu",hidden_mid.size(),
            KOKKOS_LAMBDA(std::size_t flat){const std::size_t i=flat/local_hidden_mid.extent(1);const int j=flat%local_hidden_mid.extent(1);const Precision x=local_hidden_mid(i,j),probability=Precision(1)/(Precision(1)+Kokkos::exp(-x));local_mid_adj(i,j)=local_activated_adj(i,j)*constant*(probability+x*probability*(Precision(1)-probability));});
        linear_mid.reverse(mid_adj,act_mid_adj);
        Kokkos::parallel_for("reverse_all readout head silu",hidden.size(),
            KOKKOS_LAMBDA(std::size_t flat){const std::size_t i=flat/local_hidden.extent(1);const int j=flat%local_hidden.extent(1);const Precision x=local_hidden(i,j),probability=Precision(1)/(Precision(1)+Kokkos::exp(-x));local_hidden_adj(i,j)=local_act_mid_adj(i,j)*constant*(probability+x*probability*(Precision(1)-probability));});
        linear_1.reverse(hidden_adj,input_adjoint,initialize_input_adjoint);
        return;
    }
    linear_2.reverse(seed_all,activated_adj);
    Kokkos::parallel_for("reverse_all readout silu",hidden.size(),
        KOKKOS_LAMBDA(std::size_t flat){const std::size_t i=flat/local_hidden.extent(1);const int j=flat%local_hidden.extent(1);const Precision x=local_hidden(i,j),probability=Precision(1)/(Precision(1)+Kokkos::exp(-x));local_hidden_adj(i,j)=local_activated_adj(i,j)*constant*(probability+x*probability*(Precision(1)-probability));});
    linear_1.reverse(hidden_adj,input_adjoint,initialize_input_adjoint);
}
```

**前置条件（写入门脚本，R1 待核实项的闭式回答）**：`reverse_all` 使用的 `hidden`/`hidden_mid` 与
`evaluate_all` **同名同缓冲**；标准路径中 `polar_fukui_source_map` 在模块②前向只调用一次
`evaluate_all` 且其后无任何调用，故缓存必有效。**门脚本断言顺序**：`evaluate_all → reverse_all`
必须成对且中间不得触碰该实例（§8 第 14 条）。

---

## 4. tpp PART2/PART3：极性链反向 kernel（插入文本）

### 4.0 三处小插入（前向缓存增量 + 调用点）

#### A. 调用点（PART3，`compute_node_energies_forces` 内）

**锚点**（唯一，count=1）：

```cpp
    ensure_view(harmonic_adjoints,harmonic_adjoints_storage,edges,num_lm);
```

在其**之前**插入：

```cpp
    // P3 模块⑧: 极性链全链反向(全节点特征伴 → §4.1 主函数;必须在 readout 播种【之后】、
    // 骨架反向主循环【之前】,否则层伴随被播种覆盖/被主循环错过)。
    if(polar_enabled)
        compute_polar_reverse_forces(num_nodes);
```

理由（逐行核对）：本位置在所有 `readouts[layer].reverse(state.output,scale,state.layer_adjoint)`
（覆盖写）之后、骨架反向主循环（`for(int layer=...-1;layer>=0;--layer)`）与
`harmonic_adjoints/radial_adjoints/cutoff_adjoints` 清零之前——极化反向不写这三个边级伴随
（本 ckpt 极化链不消费 edge 输入，见设计 §3.4），但放在清零前/后均安全；选此锚点对⑥/⑦落码
（可能插入 readout 相邻段）鲁棒。

#### B. 模块③ volume/rcell 缓存（PART3，③ 前向块）

**锚点**（唯一，count=1）：

```cpp
        const double volume=std::abs(
            lattice[0][0]*(lattice[1][1]*lattice[2][2]
                -lattice[1][2]*lattice[2][1])
            -lattice[0][1]*(lattice[1][0]*lattice[2][2]
                -lattice[1][2]*lattice[2][0])
            +lattice[0][2]*(lattice[1][0]*lattice[2][1]
                -lattice[1][1]*lattice[2][0]));
```

在其**之后**插入：

```cpp
        // P3 模块⑧: 缓存 volume/rcell 供反向(⑥ ρ̄ 的 V 因子 + 未来 cell 伴随,见 §6)。
        polar_cell_volume=volume;
        for(int row=0;row<3;++row)
            for(int column=0;column<3;++column)
                polar_rcell_host[3*row+column]=Precision(reciprocal[row][column]);
```

#### C. ⑤ 前向的两处缓存（校后密度 + Fukui 标量）

**C-1 逐步校正**：锚点（唯一，count=1）：

```cpp
                for(int channel=0;channel<2;++channel)
                    if(source_sums[channel]==Precision(0))
                        source_sums[channel]=Precision(1);
```

替换为（先留原始 S 再钳制）：

```cpp
                Precision raw_source_sums[2]={source_sums[0],source_sums[1]};
                for(int channel=0;channel<2;++channel)
                    if(source_sums[channel]==Precision(0))
                        source_sums[channel]=Precision(1);
```

并在紧邻其后的 Fukui 校正 kernel（锚点 `Kokkos::parallel_for("polar scf fukui correction"`）**之后**插入：

```cpp
                // P3 模块⑧: 逐步缓存(校后密度 + Fukui 标量),供反向一次性消费。
                Kokkos::realloc(Kokkos::WithoutInitializing,
                    polar_scf_step_densities_post,polar_recursion_steps,
                    num_nodes,8);
                auto post_density=polar_scf_step_densities_post;
                Kokkos::parallel_for("polar scf post-correction snapshot",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,8}),
                    KOKKOS_LAMBDA(int node,int column) {
                        post_density(step,node,column)=density_all(node,column);
                    });
                if(static_cast<int>(polar_scf_fukui_scalars.size())
                    !=polar_recursion_steps)
                    polar_scf_fukui_scalars.assign(polar_recursion_steps,
                        std::array<Precision,6>{});
                polar_scf_fukui_scalars[step]={
                    raw_source_sums[0],raw_source_sums[1],
                    predicted_zero[0],predicted_zero[1],
                    factor_alpha,factor_beta};
```

**C-2 初始化校正**：锚点（唯一，count=1）：

```cpp
        for(int channel=0;channel<2;++channel)
            if(source_sums[channel]==Precision(0))
                source_sums[channel]=Precision(1);
```

替换为：

```cpp
        const Precision raw_source_sums[2]={source_sums[0],source_sums[1]};
        for(int channel=0;channel<2;++channel)
            if(source_sums[channel]==Precision(0))
                source_sums[channel]=Precision(1);
```

并在初始化校正 kernel（锚点 `Kokkos::parallel_for("polar fukui correction"`）**之后**插入：

```cpp
        // P3 模块⑧: 初始化 Fukui 标量缓存([Sα,Sβ,predα,predβ,Fα,Fβ];S 为钳制前原始值)。
        polar_fukui_init_scalars={
            raw_source_sums[0],raw_source_sums[1],
            predicted_zero[0],predicted_zero[1],
            factor_alpha,factor_beta};
```

> 注：C-1/C-2 只**新增**落地量，不改动任何已有表达式与导出视图语义；`predicted_zero`/`factor_*`
> 均为已有变量，作用域内直接可读。

### 4.1 种子接口（供 ⑥/⑦ 反向段投递；写入门脚本不测）

⑦ le 反向（module7 §8 落点，**不在本提案**）与⑥ coulomb 反向的投递契约：

1. 在调用 `compute_polar_reverse_forces(num_nodes)` **之前**：
   - ⑦：`polar_reverse_seed_terms|=1`；把 ∂E_le/∂d_final 累加进 `polar_seed_density_adjoint(N,8)`；
     ∂E_le/∂d_init 累加进 `polar_seed_density0_adjoint(N,8)`；∂E_le/∂P_last 累加进
     `polar_seed_potential_adjoint(N,16)`；`n̄f_last` 路径直接调
     `polar_inject_node_features_adjoint(last_layer, e3nn_adjoint)`（§4.2 尾部的公开小工具，
     内部走 ⑧-A 散射 + `init=false`；声明已在 §2.2 给出）。
   - ⑥：置 `polar_coulomb_enabled=true`（本提案内联种子，见 §4.2 (S2) 段）。
2. 语义：`polar_reverse_seed_terms` 本提案内核消费后**清零**（consume-once）；种子视图不做自动清零
   （由投递方覆盖写或累加，约定：投递方先 `ensure_view` + 清零再写）。
3. 若 ⑥/⑦ 均未落码（掩码=0）：本函数清零 `polar_node_forces` 后立即返回——**对现有①②③④⑤门零影响**。

### 4.2 主函数 `compute_polar_reverse_forces` 全文

**插入位置**：PART3 末尾（`reduce_prepared_batched_stress` 定义之后），锚点（唯一，count=1）：

```
#else
#error "SYMMETRIX_MACE_NONLINEAR_KOKKOS_PART must select one implementation partition"
```

在其**之前**插入以下两块（主函数 + ⑧-A 公开小工具）。**本代码为定稿文本，逐段可直接落地**
（唯一需要读者注意的实现顺序约束：全部 λ 在小工具区定义、其后才被种子段/主循环使用）。

```cpp
// =====================================================================================
// P3 模块⑧: 极性链全链反向(力)。
// 链序: 种子(⑥/⑦) → SCF 逐步(replay→Fukui 校正逆→更新块逆→描述符逆) → 初始化 Fukui 逆
//       → lr_src 逆 / mixer 逆 → ⑧-A 注入 layer_adjoint(init=false) → polar_node_forces。
// 数学逐项对照: docs/POLAR_REVERSE_DESIGN.md §2; 前向锚: mace extensions.py L700-935,
// graph_longrange features.py L28-135 / energy.py L17-30; 引擎前向: 本文件 P3 模块①②③④⑤ 段。
// =====================================================================================
template<typename Precision>
void MaceNonlinearKokkosT<Precision>::compute_polar_reverse_forces(
    const int num_nodes)
{
    if(!polar_enabled) return;
    constexpr double two_pi=6.283185307179586476925286766559;
    constexpr double two_pi_cubed=two_pi*two_pi*two_pi;
    constexpr double polar_field_constant=1.0/(5.526349406e-3);
    const int layers=static_cast<int>(interactions.size());
    const int num_k=static_cast<int>(polar_k_vectors.extent(0));
    const int feature_width=polar_feature_width;
    if(static_cast<int>(polar_cosines.extent(0))!=num_k
        ||static_cast<int>(polar_cosines.extent(1))!=num_nodes
        ||static_cast<int>(polar_k_vectors.extent(0))!=num_k||num_k==0)
        throw std::runtime_error(
            "polar reverse: geometry cache missing or stale "
            "(run the polar forward first).");

    // ---------- 0) 累加器清零 ----------
    ensure_view(polar_node_forces,polar_node_forces_storage,
        3*static_cast<std::size_t>(num_nodes));
    ordered_kokkos_deep_copy(polar_node_forces,0.0);
    ensure_view(polar_cell_adjoint,polar_cell_adjoint_storage,9);
    ordered_kokkos_deep_copy(polar_cell_adjoint,Precision(0));
    ensure_view(polar_rev_density_adjoint,polar_rev_density_adjoint_storage,
        num_nodes,8);
    ensure_view(polar_rev_position_adjoint,polar_rev_position_adjoint_storage,
        num_nodes,3);
    ordered_kokkos_deep_copy(polar_rev_density_adjoint,Precision(0));
    ordered_kokkos_deep_copy(polar_rev_position_adjoint,Precision(0));

    // ---------- 1) 公共小工具(纯 λ,无副作用;先定义后使用) ----------
    // (2a) ⑧-A: e3nn 基伴 → 内部基伴,散射累加(单射,atomic 自卫;见 §1.2-4)。
    const auto basis_adjoint_add=[&](Kokkos::View<const Precision**> e3nn_adj,
            Kokkos::View<Precision**> internal_adj) {
        if(static_cast<int>(e3nn_adj.extent(1))!=feature_width
            ||static_cast<int>(internal_adj.extent(1))!=feature_width)
            throw std::runtime_error(
                "polar reverse: node-feature adjoint width mismatch.");
        auto source=polar_basis_source;
        auto sign=polar_basis_sign;
        Kokkos::parallel_for("polar basis adjoint scatter",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                {num_nodes,feature_width}),
            KOKKOS_LAMBDA(int node,int column) {
                // e3nn 列 c = sign_c·内部列 source(c); c↦source(c) 为分块双射
                // ⇒ 逐 (node,j) 唯一写者;atomic 仅为鲁棒。
                Kokkos::atomic_add(&internal_adj(node,source(column)),
                    sign(column)*e3nn_adj(node,column));
            });
    };
    // (2b) 逐通道 Fukui 校正反向(初始与逐步同构; §1.2-5)。
    //      scalars_host=[S_rawα,S_rawβ,predα,predβ,Fα,Fβ];sources(N,2) 从 source_offset 列起;
    //      s_bar(N,2) 输出;dbar 就地 += pred̄(l=0 列)。
    //      transform 槽位=[1/Sinvα,pred̄α,1/Sinvβ,pred̄β,term2α,term2β,Fα,Fβ](索引 0..7)。
    const auto fukui_reverse=[&](Kokkos::View<Precision**> dbar,
            Kokkos::View<const Precision**> sources,const int source_offset,
            const std::array<Precision,6>& scalars_host,
            Kokkos::View<Precision**> s_bar) {
        const auto dbar_host=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),dbar);
        const auto sources_host=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),sources);
        ensure_view(polar_rev_transform,polar_rev_transform_storage,8);
        auto transform=Kokkos::create_mirror_view(
            Kokkos::HostSpace(),polar_rev_transform);
        for(int channel=0;channel<2;++channel) {
            const Precision s_raw=scalars_host[channel];
            const bool clamped=(s_raw==Precision(0));
            const Precision s_used=clamped?Precision(1):s_raw;
            const Precision f=scalars_host[4+channel];
            double fbar=0.0;                                  // F̄ = Σ_n s̃·d̄(l=0)
            for(int node=0;node<num_nodes;++node)
                fbar+=static_cast<double>(dbar_host(node,4*channel))
                    *(static_cast<double>(sources_host(node,source_offset+channel))
                        /static_cast<double>(s_used));
            transform(2*channel)=Precision(1)/s_used;
            transform(2*channel+1)=Precision(-fbar/static_cast<double>(s_used));
            transform(4+channel)=clamped?Precision(0)
                :Precision(-2.0*fbar*static_cast<double>(f)
                    /static_cast<double>(s_raw));
            transform(6+channel)=f;
        }
        Kokkos::deep_copy(polar_rev_transform,transform);
        auto dbar_l=dbar;auto s_bar_l=s_bar;auto tr=polar_rev_transform;
        Kokkos::parallel_for("polar reverse fukui correction",num_nodes,
            KOKKOS_LAMBDA(int node) {
                for(int channel=0;channel<2;++channel) {
                    const int column=4*channel;
                    s_bar_l(node,channel)=tr(6+channel)*tr(2*channel)
                        *dbar_l(node,column)+tr(4+channel);
                    dbar_l(node,column)+=tr(2*channel+1);
                }
            });
    };
    // (2c) uvu 反向(mode0/mode1; 语义=⑤前向 launch_uvu 的逐路径伴随,设计 §2 ⑧-4b)。
    const auto uvu_reverse=[&](const std::vector<int>& meta,
            const std::vector<Precision>& path_weight,
            Kokkos::View<Precision*> weight_view,Kokkos::View<Precision*> mask_view,
            Kokkos::View<const Precision**> x1,Kokkos::View<const Precision**> x2,
            Kokkos::View<const Precision**> out_adj,
            Kokkos::View<Precision**> x1_adj,Kokkos::View<Precision**> x2_adj) {
        const int paths=static_cast<int>(meta.size())/8;
        for(int p=0;p<paths;++p) {
            const int in1_off=meta[p*8+0],in2_off=meta[p*8+1],
                out_off=meta[p*8+2],w_off=meta[p*8+3],
                mul1=meta[p*8+4],mul2=meta[p*8+5],mode=meta[p*8+6],
                d1=meta[p*8+7];
            const Precision pw=path_weight[p];
            const Precision inv_sqrt_d1=Precision(1)
                /Kokkos::sqrt(Precision(d1));
            if(mode==0) {
                Kokkos::parallel_for("polar reverse uvu mode0",num_nodes,
                    KOKKOS_LAMBDA(int node) {
                        for(int u=0;u<mul1;++u) {
                            const Precision mixed_bar=pw*out_adj(node,out_off+u)
                                *mask_view(out_off+u);
                            for(int v=0;v<mul2;++v) {
                                const Precision pair_bar=mixed_bar*inv_sqrt_d1
                                    *weight_view(w_off+u*mul2+v);
                                for(int c=0;c<d1;++c)
                                    Kokkos::atomic_add(&x1_adj(node,
                                            in1_off+u*d1+c),
                                        pair_bar*x2(node,in2_off+v*d1+c));
                                for(int c=0;c<d1;++c)
                                    Kokkos::atomic_add(&x2_adj(node,
                                            in2_off+v*d1+c),
                                        pair_bar*x1(node,in1_off+u*d1+c));
                            }
                        }
                    });
            } else {
                Kokkos::parallel_for("polar reverse uvu mode1",num_nodes,
                    KOKKOS_LAMBDA(int node) {
                        for(int u=0;u<mul1;++u) {
                            Precision mixed=Precision(0);
                            for(int v=0;v<mul2;++v)
                                mixed+=x2(node,in2_off+v)
                                    *weight_view(w_off+u*mul2+v);
                            const Precision contribution=pw*mixed*inv_sqrt_d1;
                            Precision mixed_bar=Precision(0);
                            for(int c=0;c<d1;++c) {
                                const int i1=in1_off+u*d1+c;
                                const Precision o=out_adj(node,out_off+u*d1+c)
                                    *mask_view(out_off+u*d1+c);
                                Kokkos::atomic_add(&x1_adj(node,i1),
                                    contribution*o);
                                mixed_bar+=x1(node,i1)*o;
                            }
                            for(int v=0;v<mul2;++v)
                                Kokkos::atomic_add(&x2_adj(node,in2_off+v),
                                    pw*inv_sqrt_d1*mixed_bar
                                        *weight_view(w_off+u*mul2+v));
                        }
                    });
            }
        }
    };
    // (2d) 模块⑤内联 Gate 的反向(scalars=cst_s·silu;gates=cst_g·sigmoid⊗gated)。
    const auto readout_gate_reverse=[&](Kokkos::View<const Precision**> src,
            Kokkos::View<const Precision**> dst_adj,
            Kokkos::View<Precision**> src_adj,
            const Precision scalar_cst,const Precision gate_cst,
            const int scalars,const int gates) {
        Kokkos::parallel_for("polar reverse readout gate",num_nodes,
            KOKKOS_LAMBDA(int node) {
                for(int i=0;i<scalars;++i) {
                    const Precision s=src(node,i);
                    const Precision sig=Precision(1)
                        /(Precision(1)+Kokkos::exp(-s));
                    src_adj(node,i)+=dst_adj(node,i)*scalar_cst
                        *(sig+s*sig*(Precision(1)-sig));
                }
                for(int i=0;i<gates;++i) {
                    const Precision gs=src(node,scalars+i);
                    const Precision sig=Precision(1)
                        /(Precision(1)+Kokkos::exp(-gs));
                    const Precision g=gate_cst*sig;
                    Precision gate_bar=Precision(0);
                    for(int c=0;c<3;++c) {
                        const int i3=scalars+3*i+c;
                        const int iv=scalars+gates+3*i+c;
                        src_adj(node,iv)+=g*dst_adj(node,i3);
                        gate_bar+=src(node,iv)*dst_adj(node,i3);
                    }
                    src_adj(node,scalars+i)+=gate_bar*gate_cst*sig
                        *(Precision(1)-sig);
                }
            });
    };
    // (2e) LayerNorm+SiLU 反向: dx_c = inv·γ_c·dz_c − inv/W·Σγdz
    //      − inv³·y_c/W·Σ(γdz·y);dz = ā·σ(z)(1+z(1−σ(z))),z=LN(h),a=silu(z)。
    const auto ln_silu_reverse=[&](Kokkos::View<const Precision**> h,
            Kokkos::View<const Precision**> a,
            Kokkos::View<const Precision**> a_adj,
            Kokkos::View<Precision**> h_adj,
            Kokkos::View<const Precision*> gamma,
            Kokkos::View<const Precision*> beta,
            const Precision eps) {
        const int width=static_cast<int>(h.extent(1));
        auto h_l=h;auto a_adj_l=a_adj;auto h_adj_l=h_adj;auto g=gamma;auto b=beta;
        Kokkos::parallel_for("polar reverse layernorm silu",num_nodes,
            KOKKOS_LAMBDA(int node) {
                Precision mean=Precision(0);
                for(int c=0;c<width;++c) mean+=h_l(node,c);
                mean/=Precision(width);
                Precision var=Precision(0);
                for(int c=0;c<width;++c) {
                    const Precision delta=h_l(node,c)-mean;
                    var+=delta*delta;
                }
                var/=Precision(width);
                const Precision inv=Precision(1)/Kokkos::sqrt(var+eps);
                Precision sum_gz=Precision(0),sum_gzy=Precision(0);
                for(int c=0;c<width;++c) {                    // 两趟:先归约后写
                    const Precision y=h_l(node,c)-mean;
                    const Precision z=y*inv*g(c)+b(c);
                    const Precision sig=Precision(1)
                        /(Precision(1)+Kokkos::exp(-z));
                    const Precision dz=a_adj_l(node,c)*sig
                        *(Precision(1)+z*(Precision(1)-sig));
                    sum_gz+=g(c)*dz;
                    sum_gzy+=g(c)*dz*y;
                }
                for(int c=0;c<width;++c) {
                    const Precision y=h_l(node,c)-mean;
                    const Precision z=y*inv*g(c)+b(c);
                    const Precision sig=Precision(1)
                        /(Precision(1)+Kokkos::exp(-z));
                    const Precision dz=a_adj_l(node,c)*sig
                        *(Precision(1)+z*(Precision(1)-sig));
                    Kokkos::atomic_add(&h_adj_l(node,c),
                        inv*g(c)*dz-inv/Precision(width)*sum_gz
                        -inv*inv*inv*y/Precision(width)*sum_gzy);
                }
                (void)a;
            });
    };
    // (2f) 稠密层反向(输入伴 = Σ_o W(o,i)·out̄(o);权重/偏置常量)。
    const auto dense_reverse=[&](Kokkos::View<const Precision**> weight,
            Kokkos::View<const Precision**> out_adj,
            Kokkos::View<Precision**> in_adj,const bool initialize) {
        const int out_w=static_cast<int>(weight.extent(0));
        const int in_w=static_cast<int>(weight.extent(1));
        if(initialize) ordered_kokkos_deep_copy(in_adj,Precision(0));
        auto w=weight;auto o_adj=out_adj;auto i_adj=in_adj;
        Kokkos::parallel_for("polar reverse dense",num_nodes,
            KOKKOS_LAMBDA(int node) {
                for(int i=0;i<in_w;++i) {
                    Precision acc=Precision(0);
                    for(int o=0;o<out_w;++o)
                        acc+=w(o,i)*o_adj(node,o);
                    Kokkos::atomic_add(&i_adj(node,i),acc);
                }
            });
    };
    // (2g) 共用: ρ̄ → assemble 反向(源伴 q̄ 写 dest 的 4 列;相位伴并入 r̄)。
    //      source(N,8) 提供 q = source(node,section_offset..+3)。
    const auto kspace_assemble_reverse=[&](Kokkos::View<const Precision*> rho_bar_re,
            Kokkos::View<const Precision*> rho_bar_im,
            Kokkos::View<const Precision**> source,const int section_offset,
            Kokkos::View<Precision**> dest,const int dest_offset) {
        auto dbre=polar_density_basis_re;auto dbim=polar_density_basis_im;
        ensure_view(polar_rev_acc_cos,polar_rev_acc_cos_storage,num_k,4);
        ensure_view(polar_rev_acc_sin,polar_rev_acc_sin_storage,num_k,4);
        auto acc_cos=polar_rev_acc_cos,acc_sin=polar_rev_acc_sin;
        auto kcos=polar_cosines,ksin=polar_sines;
        auto kv=polar_k_vectors,pos_adj=polar_rev_position_adjoint;
        const Precision rho_scale=Precision(two_pi_cubed/polar_cell_volume);
        Kokkos::parallel_for("polar rev assemble",num_k,
            KOKKOS_LAMBDA(int k) {
                for(int m=0;m<4;++m) {
                    acc_cos(k,m)=rho_scale*(rho_bar_re(k)*dbre(k,m)
                        +rho_bar_im(k)*dbim(k,m));
                    acc_sin(k,m)=rho_scale*(rho_bar_re(k)*dbim(k,m)
                        -rho_bar_im(k)*dbre(k,m));
                }
            });
        Kokkos::parallel_for("polar rev phase and source",num_nodes,
            KOKKOS_LAMBDA(int node) {
                for(int m=0;m<4;++m) {
                    Precision q_bar=Precision(0);
                    for(int k=0;k<num_k;++k)
                        q_bar+=kcos(k,node)*acc_cos(k,m)
                            +ksin(k,node)*acc_sin(k,m);
                    Kokkos::atomic_add(&dest(node,dest_offset+m),q_bar);
                }
                Precision rx=Precision(0),ry=Precision(0),rz=Precision(0);
                for(int k=0;k<num_k;++k) {
                    Precision cos_bar=Precision(0),sin_bar=Precision(0);
                    for(int m=0;m<4;++m) {
                        const Precision q=source(node,section_offset+m);
                        cos_bar+=acc_cos(k,m)*q;
                        sin_bar+=acc_sin(k,m)*q;
                    }
                    // (d5): ∂cos/∂r[c]=−sin·k[c];∂sin/∂r[c]=cos·k[c]
                    const Precision inner=-ksin(k,node)*cos_bar
                        +kcos(k,node)*sin_bar;
                    rx+=kv(k,0)*inner;ry+=kv(k,1)*inner;rz+=kv(k,2)*inner;
                }
                pos_adj(node,0)+=rx;pos_adj(node,1)+=ry;pos_adj(node,2)+=rz;
            });
    };
    // (2h) 描述符逐通道反向(feats̄(N,8) → q̄ + 相位伴;密度源=d_in)。
    const auto descriptor_reverse_channel=[&](const int channel,
            Kokkos::View<const Precision**> density_in) {
        auto feats_bar=polar_rev_feats_adjoint;
        auto channel_adj=polar_rev_channel_adjoint;
        auto si_sel=polar_si_select,si_ov=polar_si_overlap;
        auto permutation=polar_output_permutation;
        auto fbre=polar_feature_basis_re,fbim=polar_feature_basis_im;
        auto kn2=polar_k_norm2,k0m=polar_k0_mask;
        auto kcos=polar_cosines,ksin=polar_sines;
        ensure_view(polar_rev_ab,polar_rev_ab_storage,num_k,16);
        ensure_view(polar_rev_rho_adj_re,polar_rev_rho_adj_re_storage,num_k);
        ensure_view(polar_rev_rho_adj_im,polar_rev_rho_adj_im_storage,num_k);
        auto ab=polar_rev_ab;
        auto rho_bar_re=polar_rev_rho_adj_re,rho_bar_im=polar_rev_rho_adj_im;
        const Precision proj_scale=Precision(2.0/two_pi_cubed);
        // (i) 自作用子项反(sourcē += selectᵀ(C ⊙ (−feats̄)))。
        Kokkos::parallel_for("polar desc si reverse",num_nodes,
            KOKKOS_LAMBDA(int node) {
                for(int j=0;j<8;++j)
                    Kokkos::atomic_add(&channel_adj(node,
                            4*channel+si_sel(j)),
                        -si_ov(j)*feats_bar(node,j));
            });
        // (ii) Ā/Ŷ = Σ_i cos/sin·proj_scale·feat̄(逆置换并入:j ↦ m=perm(j) 为双射)。
        Kokkos::parallel_for("polar desc ab reverse",num_k,
            KOKKOS_LAMBDA(int k) {
                for(int j=0;j<8;++j) {
                    const int m=permutation(j);
                    Precision a=Precision(0),b=Precision(0);
                    for(int node=0;node<num_nodes;++node) {
                        const Precision z=proj_scale*feats_bar(node,j);
                        a+=kcos(k,node)*z;
                        b+=ksin(k,node)*z;
                    }
                    ab(k,m)=a;ab(k,8+m)=b;
                }
            });
        // (iii) project+kernel 反向一步到位:
        //       pr̄=Σ_m(X̄·FBR+Ŷ·FBI); pī=Σ_m(X̄·FBI−Ŷ·FBR); ρ̄=pr̄·inv_k2。
        Kokkos::parallel_for("polar desc rho adjoint",num_k,
            KOKKOS_LAMBDA(int k) {
                const Precision kfp=(k0m(k)>Precision(0))
                    ?Precision(0.5):Precision(1);
                const Precision inv_k2=(k0m(k)>Precision(0))
                    ?Precision(0):Precision(polar_field_constant)/kn2(k);
                Precision rr=Precision(0),ri=Precision(0);
                for(int m=0;m<8;++m) {
                    const Precision x_bar=ab(k,m)*kfp;
                    const Precision y_bar=ab(k,8+m)*kfp;
                    const Precision pr_bar=x_bar*fbre(k,m)+y_bar*fbim(k,m);
                    const Precision pi_bar=x_bar*fbim(k,m)-y_bar*fbre(k,m);
                    rr+=pr_bar*inv_k2;ri+=pi_bar*inv_k2;
                }
                rho_bar_re(k)=rr;rho_bar_im(k)=ri;
            });
        // (iv) assemble 反向 → q̄(4 列)+ 相位。
        kspace_assemble_reverse(rho_bar_re,rho_bar_im,density_in,
            4*channel,channel_adj,4*channel);
    };
    // (2i) 逐步前向重放(③④ + ⑤; 输入 d_in,输出写 rev 视图;与⑤已落码逐算子镜像)。
    const auto replay_step_forward=[&](const int step,
            Kokkos::View<const Precision**,Kokkos::LayoutRight> density_in) {
        auto& update=polar_updates[step];
        ensure_view(polar_rev_step_potential,polar_rev_step_potential_storage,
            num_nodes,16);
        ensure_view(polar_rev_rho_fwd_re,polar_rev_rho_fwd_re_storage,num_k);
        ensure_view(polar_rev_rho_fwd_im,polar_rev_rho_fwd_im_storage,num_k);
        auto step_potential=polar_rev_step_potential;
        for(int channel=0;channel<2;++channel) {
            { // assemble(与⑤"polar scf assemble fourier"同式)
                auto rho_re=polar_rev_rho_fwd_re,rho_im=polar_rev_rho_fwd_im;
                auto dbre=polar_density_basis_re,dbim=polar_density_basis_im;
                auto kcos=polar_cosines,ksin=polar_sines;
                const Precision rho_scale=Precision(two_pi_cubed
                    /polar_cell_volume);
                Kokkos::parallel_for("polar rev replay assemble",num_k,
                    KOKKOS_LAMBDA(int k) {
                        Precision acc_cos[4],acc_sin[4];
                        for(int m=0;m<4;++m) {
                            acc_cos[m]=Precision(0);acc_sin[m]=Precision(0);
                        }
                        for(int node=0;node<num_nodes;++node) {
                            const Precision c=kcos(k,node);
                            const Precision s=ksin(k,node);
                            for(int m=0;m<4;++m) {
                                const Precision q=density_in(node,4*channel+m);
                                acc_cos[m]+=c*q;acc_sin[m]+=s*q;
                            }
                        }
                        Precision re=Precision(0),im=Precision(0);
                        for(int m=0;m<4;++m) {
                            re+=dbre(k,m)*acc_cos[m]+dbim(k,m)*acc_sin[m];
                            im+=dbim(k,m)*acc_cos[m]-dbre(k,m)*acc_sin[m];
                        }
                        rho_re(k)=rho_scale*re;rho_im(k)=rho_scale*im;
                    });
            }
            { // project + SI + permute + /knorm(与⑤"polar scf project features"
              //  及随后的归一 kernel 逐元素等价;仅省去中间 feats 存储)
                auto rho_re=polar_rev_rho_fwd_re,rho_im=polar_rev_rho_fwd_im;
                auto fbre=polar_feature_basis_re,fbim=polar_feature_basis_im;
                auto kn2=polar_k_norm2,k0m=polar_k0_mask;
                auto kcos=polar_cosines,ksin=polar_sines;
                auto si_ov=polar_si_overlap,si_sel=polar_si_select;
                auto permutation=polar_output_permutation;
                auto knorm=polar_field_feature_norms;
                const Precision proj_scale=Precision(2.0/two_pi_cubed);
                Kokkos::parallel_for("polar rev replay project",num_nodes,
                    KOKKOS_LAMBDA(int node) {
                        Precision feat[8];
                        for(int m=0;m<8;++m) {
                            Precision proj_cos=Precision(0);
                            Precision proj_sin=Precision(0);
                            for(int k=0;k<num_k;++k) {
                                const Precision inv_k2=(k0m(k)>Precision(0))
                                    ?Precision(0)
                                    :Precision(polar_field_constant)/kn2(k);
                                const Precision pr=rho_re(k)*inv_k2;
                                const Precision pi=rho_im(k)*inv_k2;
                                const Precision kfp=(k0m(k)>Precision(0))
                                    ?Precision(0.5):Precision(1);
                                const Precision fa=(pr*fbre(k,m)
                                    +pi*fbim(k,m))*kfp;
                                const Precision fb=(pr*fbim(k,m)
                                    -pi*fbre(k,m))*kfp;
                                proj_cos+=fa*kcos(k,node);
                                proj_sin+=fb*ksin(k,node);
                            }
                            feat[m]=proj_scale*(proj_cos+proj_sin);
                        }
                        for(int j=0;j<8;++j) {
                            const int source_index=permutation(j);
                            const Precision value=feat[source_index]
                                -si_ov(j)*density_in(node,
                                    4*channel+si_sel(j));
                            step_potential(node,8*channel+j)=value/knorm(j);
                        }
                    });
            }
        }
        const int dp_width=update.source_embedding.output_dimension();
        const int hidden=update.nl_hidden_width;
        const int gate_w=update.gate_scalar_count+3*update.gate_gated_count;
        const int ro_w=update.readout_linear_1.output_dimension();
        ensure_view(polar_rev_pe,polar_rev_pe_storage,num_nodes,
            update.potential_linear.output_dimension());
        ensure_view(polar_rev_dp,polar_rev_dp_storage,num_nodes,dp_width);
        ensure_view(polar_rev_se,polar_rev_se_storage,num_nodes,dp_width);
        ensure_view(polar_rev_nl_in,polar_rev_nl_in_storage,num_nodes,
            2*dp_width);
        ensure_view(polar_rev_nl_h,polar_rev_nl_h_storage,num_nodes,3*hidden);
        ensure_view(polar_rev_nl_a,polar_rev_nl_a_storage,num_nodes,3*hidden);
        ensure_view(polar_rev_nl_out,polar_rev_nl_out_storage,num_nodes,
            static_cast<int>(update.nl_w3.extent(0)));
        ensure_view(polar_rev_tp,polar_rev_tp_storage,num_nodes,
            static_cast<int>(update.readout_linear_1.input_dimension()));
        ensure_view(polar_rev_ro1,polar_rev_ro1_storage,num_nodes,ro_w);
        ensure_view(polar_rev_gate1,polar_rev_gate1_storage,num_nodes,gate_w);
        ensure_view(polar_rev_ro2,polar_rev_ro2_storage,num_nodes,ro_w);
        ensure_view(polar_rev_gate2,polar_rev_gate2_storage,num_nodes,gate_w);
        ensure_view(polar_rev_out,polar_rev_out_storage,num_nodes,
            static_cast<int>(update.readout_linear_2.output_dimension()));
        { // 势嵌入三分量(与⑤同序:potential_linear + node_feats_linear + charge_embedding)
            update.potential_linear.evaluate(polar_rev_step_potential,
                polar_rev_pe);
            update.node_feats_linear.evaluate(polar_features_mixed,
                polar_scratch_a);
            update.charge_embedding.evaluate(density_in,polar_scratch_b);
            auto pe=polar_rev_pe,ra=polar_scratch_a,rb=polar_scratch_b;
            Kokkos::parallel_for("polar rev replay pe sum",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,static_cast<int>(pe.extent(1))}),
                KOKKOS_LAMBDA(int node,int column) {
                    pe(node,column)+=ra(node,column)+rb(node,column);
                });
        }
        const auto to_host_vector=[&](const auto& view) {
            const auto mirror=Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),view);
            std::vector<typename decltype(mirror)::value_type>
                values(mirror.size());
            for(std::size_t i2=0;i2<values.size();++i2)
                values[i2]=mirror.data()[i2];
            return values;
        };
        const std::vector<int> dot_meta=to_host_vector(update.uvu_dot_meta);
        const std::vector<Precision> dot_pw=
            to_host_vector(update.uvu_dot_path_weight);
        const std::vector<int> tp_meta=to_host_vector(update.uvu_tp_meta);
        const std::vector<Precision> tp_pw=
            to_host_vector(update.uvu_tp_path_weight);
        const auto launch_uvu_forward=[&](const std::vector<int>& meta,
                const std::vector<Precision>& path_weight,
                Kokkos::View<Precision*> weight_view,
                Kokkos::View<Precision*> mask_view,
                Kokkos::View<const Precision**> x1,
                Kokkos::View<const Precision**> x2,
                Kokkos::View<Precision**> output) {
            const int paths=static_cast<int>(meta.size())/8;
            const int width=static_cast<int>(output.extent(1));
            Kokkos::parallel_for("polar rev replay uvu zero",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,width}),
                KOKKOS_LAMBDA(int node,int column) {
                    output(node,column)=Precision(0);
                });
            for(int p=0;p<paths;++p) {
                const int in1_off=meta[p*8+0],in2_off=meta[p*8+1],
                    out_off=meta[p*8+2],w_off=meta[p*8+3],
                    mul1=meta[p*8+4],mul2=meta[p*8+5],mode=meta[p*8+6],
                    d1=meta[p*8+7];
                const Precision pw=path_weight[p];
                const Precision inv=Precision(1)/Kokkos::sqrt(Precision(d1));
                if(mode==0) {
                    Kokkos::parallel_for("polar rev replay uvu mode0",num_nodes,
                        KOKKOS_LAMBDA(int node) {
                            for(int u=0;u<mul1;++u) {
                                Precision mixed=Precision(0);
                                for(int v=0;v<mul2;++v) {
                                    Precision pair=Precision(0);
                                    for(int c=0;c<d1;++c)
                                        pair+=x1(node,in1_off+u*d1+c)
                                            *x2(node,in2_off+v*d1+c);
                                    mixed+=pair*inv*weight_view(w_off+u*mul2+v);
                                }
                                output(node,out_off+u)+=pw*mixed;
                            }
                        });
                } else {
                    Kokkos::parallel_for("polar rev replay uvu mode1",num_nodes,
                        KOKKOS_LAMBDA(int node) {
                            for(int u=0;u<mul1;++u) {
                                Precision mixed=Precision(0);
                                for(int v=0;v<mul2;++v)
                                    mixed+=x2(node,in2_off+v)
                                        *weight_view(w_off+u*mul2+v);
                                const Precision contribution=pw*mixed*inv;
                                for(int c=0;c<d1;++c)
                                    output(node,out_off+u*d1+c)+=contribution
                                        *x1(node,in1_off+u*d1+c);
                            }
                        });
                }
            }
            Kokkos::parallel_for("polar rev replay uvu mask",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,width}),
                KOKKOS_LAMBDA(int node,int column) {
                    output(node,column)*=mask_view(column);
                });
        };
        launch_uvu_forward(dot_meta,dot_pw,update.uvu_dot_weight,
            update.uvu_dot_mask,polar_features_mixed,polar_rev_pe,
            polar_rev_dp);
        update.source_embedding.evaluate(attrs,polar_rev_se);
        { // nl_in = cat(dp, se)
            auto ni=polar_rev_nl_in,dp=polar_rev_dp,se=polar_rev_se;
            Kokkos::parallel_for("polar rev replay nl concat",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,dp_width}),
                KOKKOS_LAMBDA(int node,int column) {
                    ni(node,column)=dp(node,column);
                    ni(node,dp_width+column)=se(node,column);
                });
        }
        { // MLP 重放: h_k → nl_h 块 k;a_k → nl_a 块 k(k=0..2);dense3 → nl_out
            auto h=polar_rev_nl_h;auto a=polar_rev_nl_a;
            auto ni=polar_rev_nl_in;auto out_nl=polar_rev_nl_out;
            const Precision eps=update.ln_eps;
            const auto ln_block=[&](Kokkos::View<Precision**> hb,
                    Kokkos::View<Precision**> ab,Kokkos::View<const Precision*> g,
                    Kokkos::View<const Precision*> b) {
                Kokkos::parallel_for("polar rev replay ln",num_nodes,
                    KOKKOS_LAMBDA(int node) {
                        Precision mean=Precision(0);
                        for(int c2=0;c2<hidden;++c2) mean+=hb(node,c2);
                        mean/=Precision(hidden);
                        Precision var=Precision(0);
                        for(int c2=0;c2<hidden;++c2) {
                            const Precision delta=hb(node,c2)-mean;
                            var+=delta*delta;
                        }
                        var/=Precision(hidden);
                        const Precision inv=Precision(1)
                            /Kokkos::sqrt(var+eps);
                        for(int c2=0;c2<hidden;++c2) {
                            const Precision normalized=(hb(node,c2)-mean)
                                *inv*g(c2)+b(c2);
                            ab(node,c2)=normalized
                                /(Precision(1)+Kokkos::exp(-normalized));
                        }
                    });
            };
            const auto dense_forward=[&](Kokkos::View<const Precision**> w,
                    Kokkos::View<const Precision*> b,
                    Kokkos::View<const Precision**> src,
                    Kokkos::View<Precision**> dst,int dst_offset,
                    const int src_width) {
                Kokkos::parallel_for("polar rev replay dense",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,static_cast<int>(w.extent(0))}),
                    KOKKOS_LAMBDA(int node,int o) {
                        Precision acc=b(o);
                        for(int c2=0;c2<src_width;++c2)
                            acc+=w(o,c2)*src(node,c2);
                        dst(node,dst_offset+o)=acc;
                    });
            };
            {
                auto w0=update.nl_w0;auto b0=update.nl_b0;
                const int src_width=static_cast<int>(w0.extent(1));
                Kokkos::parallel_for("polar rev replay dense0",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,static_cast<int>(w0.extent(0))}),
                    KOKKOS_LAMBDA(int node,int o) {
                        Precision acc=b0(o);
                        for(int c2=0;c2<src_width;++c2)
                            acc+=w0(o,c2)*ni(node,c2);
                        h(node,o)=acc;
                    });
            }
            ln_block(Kokkos::subview(h,Kokkos::ALL,
                    std::make_pair(0,hidden)),
                Kokkos::subview(a,Kokkos::ALL,std::make_pair(0,hidden)),
                update.ln0_weight,update.ln0_bias);
            dense_forward(update.nl_w1,update.nl_b1,
                Kokkos::subview(a,Kokkos::ALL,std::make_pair(0,hidden)),
                h,hidden,hidden);
            ln_block(Kokkos::subview(h,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden)),
                Kokkos::subview(a,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden)),
                update.ln1_weight,update.ln1_bias);
            dense_forward(update.nl_w2,update.nl_b2,
                Kokkos::subview(a,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden)),
                h,2*hidden,hidden);
            ln_block(Kokkos::subview(h,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden)),
                Kokkos::subview(a,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden)),
                update.ln2_weight,update.ln2_bias);
            { // dense3 → nl_out(末层无 LN)
                auto w3=update.nl_w3;auto b3=update.nl_b3;
                const int src_width=static_cast<int>(w3.extent(1));
                Kokkos::parallel_for("polar rev replay dense3",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                        {num_nodes,static_cast<int>(w3.extent(0))}),
                    KOKKOS_LAMBDA(int node,int o) {
                        Precision acc=b3(o);
                        for(int c2=0;c2<src_width;++c2)
                            acc+=w3(o,c2)*a(node,2*hidden+c2);
                        out_nl(node,o)=acc;
                    });
            }
        }
        launch_uvu_forward(tp_meta,tp_pw,update.uvu_tp_weight,
            update.uvu_tp_mask,polar_features_mixed,polar_rev_nl_out,
            polar_rev_tp);
        update.readout_linear_1.evaluate(polar_rev_tp,polar_rev_ro1);
        {
            const Precision scalar_cst=update.gate_scalar_cst;
            const Precision gate_cst=update.gate_gate_cst;
            const int scalars=update.gate_scalar_count;
            const int gates=update.gate_gate_count;
            auto src=polar_rev_ro1;auto dst=polar_rev_gate1;
            Kokkos::parallel_for("polar rev replay gate1",num_nodes,
                KOKKOS_LAMBDA(int node) {
                    for(int i2=0;i2<scalars;++i2) {
                        const Precision s=src(node,i2);
                        dst(node,i2)=scalar_cst*s
                            /(Precision(1)+Kokkos::exp(-s));
                    }
                    for(int i2=0;i2<gates;++i2) {
                        const Precision g=gate_cst
                            /(Precision(1)+Kokkos::exp(-src(node,scalars+i2)));
                        for(int c2=0;c2<3;++c2)
                            dst(node,scalars+3*i2+c2)=g
                                *src(node,scalars+gates+3*i2+c2);
                    }
                });
        }
        update.readout_linear_mid.evaluate(polar_rev_gate1,polar_rev_ro2);
        {
            const Precision scalar_cst=update.gate_scalar_cst;
            const Precision gate_cst=update.gate_gate_cst;
            const int scalars=update.gate_scalar_count;
            const int gates=update.gate_gate_count;
            auto src=polar_rev_ro2;auto dst=polar_rev_gate2;
            Kokkos::parallel_for("polar rev replay gate2",num_nodes,
                KOKKOS_LAMBDA(int node) {
                    for(int i2=0;i2<scalars;++i2) {
                        const Precision s=src(node,i2);
                        dst(node,i2)=scalar_cst*s
                            /(Precision(1)+Kokkos::exp(-s));
                    }
                    for(int i2=0;i2<gates;++i2) {
                        const Precision g=gate_cst
                            /(Precision(1)+Kokkos::exp(-src(node,scalars+i2)));
                        for(int c2=0;c2<3;++c2)
                            dst(node,scalars+3*i2+c2)=g
                                *src(node,scalars+gates+3*i2+c2);
                    }
                });
        }
        update.readout_linear_2.evaluate(polar_rev_gate2,polar_rev_out);
    };

    // ---------- 2) 种子早退判定(掩码=0 ⇒ 零力返回,对①-⑤门零影响) ----------
    const int seed_terms=polar_reverse_seed_terms
        |(polar_coulomb_enabled?2:0);
    polar_reverse_terms_last=seed_terms;   // 诊断(非 consume-once;门脚本分档用)
    if(seed_terms==0) {
        polar_reverse_seed_terms=0;
        return;
    }

    // ---------- 3) ⑦ le 种子并入(若落码): d_final / P_last 链头;d_init 留到 SCF 之后 ----------
    if(seed_terms&1) {
        if(static_cast<int>(polar_seed_density_adjoint.extent(0))!=num_nodes
            ||static_cast<int>(polar_seed_density_adjoint.extent(1))!=8
            ||static_cast<int>(polar_seed_density0_adjoint.extent(0))!=num_nodes
            ||static_cast<int>(polar_seed_density0_adjoint.extent(1))!=8
            ||static_cast<int>(polar_seed_potential_adjoint.extent(0))!=num_nodes
            ||static_cast<int>(polar_seed_potential_adjoint.extent(1))!=16)
            throw std::runtime_error(
                "polar reverse: le seed extents are inconsistent "
                "(see module8_patch_final.md 4.1).");
        auto dbar=polar_rev_density_adjoint;
        auto seed_final=polar_seed_density_adjoint;
        Kokkos::parallel_for("polar reverse seed density final",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,8}),
            KOKKOS_LAMBDA(int node,int column) {
                dbar(node,column)+=seed_final(node,column);
            });
    }

    // ---------- 4) ⑥ coulomb 种子(polar_coulomb_enabled 由 ⑥ 落码置位) ----------
    // E = 0.5·V/(2π)⁶·Σ_k 2(ρ_r·pot_r+ρ_i·pot_i),pot=FIELD_CONSTANT·kfac_c·ρ ⇒
    // ρ̄ = 2·V/(2π)⁶·FIELD_CONSTANT·kfac_c·ρ;q(n,m)=d_final(n,m)+d_final(n,4+m)。
    if(seed_terms&2) {
        ensure_view(polar_rev_source_adjoint,polar_rev_source_adjoint_storage,
            num_nodes,8);
        ensure_view(polar_rev_channel_adjoint,polar_rev_channel_adjoint_storage,
            num_nodes,8);
        ordered_kokkos_deep_copy(polar_rev_channel_adjoint,Precision(0));
        {
            auto q_view=polar_rev_source_adjoint;
            auto final_density=polar_spin_charge_density;
            Kokkos::parallel_for("polar coulomb q",num_nodes,
                KOKKOS_LAMBDA(int node) {
                    for(int m=0;m<4;++m)
                        q_view(node,m)=final_density(node,m)
                            +final_density(node,4+m);
                });
        }
        ensure_view(polar_rev_rho_fwd_re,polar_rev_rho_fwd_re_storage,num_k);
        ensure_view(polar_rev_rho_fwd_im,polar_rev_rho_fwd_im_storage,num_k);
        ensure_view(polar_rev_rho_adj_re,polar_rev_rho_adj_re_storage,num_k);
        ensure_view(polar_rev_rho_adj_im,polar_rev_rho_adj_im_storage,num_k);
        { // assemble 前向重算 ρ(q)(与 ⑥ 前向同表同式;⑥ 落码后可换其视图,位级一致)
            auto q_view=polar_rev_source_adjoint;
            auto rho_re=polar_rev_rho_fwd_re,rho_im=polar_rev_rho_fwd_im;
            auto dbre=polar_density_basis_re,dbim=polar_density_basis_im;
            auto kcos=polar_cosines,ksin=polar_sines;
            const Precision rho_scale=Precision(two_pi_cubed
                /polar_cell_volume);
            Kokkos::parallel_for("polar coulomb assemble",num_k,
                KOKKOS_LAMBDA(int k) {
                    Precision acc_cos[4],acc_sin[4];
                    for(int m=0;m<4;++m) {
                        acc_cos[m]=Precision(0);acc_sin[m]=Precision(0);
                    }
                    for(int node=0;node<num_nodes;++node) {
                        const Precision c=kcos(k,node);
                        const Precision s=ksin(k,node);
                        for(int m=0;m<4;++m) {
                            const Precision q=q_view(node,m);
                            acc_cos[m]+=c*q;acc_sin[m]+=s*q;
                        }
                    }
                    Precision re=Precision(0),im=Precision(0);
                    for(int m=0;m<4;++m) {
                        re+=dbre(k,m)*acc_cos[m]+dbim(k,m)*acc_sin[m];
                        im+=dbim(k,m)*acc_cos[m]-dbre(k,m)*acc_sin[m];
                    }
                    rho_re(k)=rho_scale*re;rho_im(k)=rho_scale*im;
                });
        }
        { // ρ̄(含 V 因子与 kfac_c;k=0 处 kfac=0)
            auto rho_bar_re=polar_rev_rho_adj_re,rho_bar_im=polar_rev_rho_adj_im;
            auto rho_re=polar_rev_rho_fwd_re,rho_im=polar_rev_rho_fwd_im;
            auto kn2=polar_k_norm2,k0m=polar_k0_mask;
            const Precision rho_bar_scale=Precision(
                2.0*polar_cell_volume/(two_pi_cubed*two_pi_cubed)
                    *polar_field_constant);
            Kokkos::parallel_for("polar coulomb rho adjoint",num_k,
                KOKKOS_LAMBDA(int k) {
                    const Precision kfac=(k0m(k)>Precision(0))
                        ?Precision(0):Precision(1)/kn2(k);
                    rho_bar_re(k)=rho_re(k)*kfac*rho_bar_scale;
                    rho_bar_im(k)=rho_im(k)*kfac*rho_bar_scale;
                });
        }
        kspace_assemble_reverse(polar_rev_rho_adj_re,
            polar_rev_rho_adj_im,polar_rev_source_adjoint,0,
            polar_rev_channel_adjoint,0);
        { // q̄ 拆分到两自旋通道(sum dim=1 的伴随)
            auto dbar=polar_rev_density_adjoint;
            auto q_bar=polar_rev_channel_adjoint;
            Kokkos::parallel_for("polar coulomb q adjoint split",num_nodes,
                KOKKOS_LAMBDA(int node) {
                    for(int m=0;m<4;++m) {
                        const Precision value=q_bar(node,m);
                        Kokkos::atomic_add(&dbar(node,m),value);
                        Kokkos::atomic_add(&dbar(node,4+m),value);
                    }
                });
        }
    }

    // ---------- 5) SCF 逐步反向(step = M-1 … 0) ----------
    ensure_view(polar_rev_fmix_adjoint,polar_rev_fmix_adjoint_storage,
        num_nodes,feature_width);
    ordered_kokkos_deep_copy(polar_rev_fmix_adjoint,Precision(0));
    for(int step=polar_recursion_steps-1;step>=0;--step) {
        auto& update=polar_updates[step];
        const int hidden=update.nl_hidden_width;
        const int dp_width=update.source_embedding.output_dimension();
        const int gate_w=update.gate_scalar_count+3*update.gate_gated_count;
        const int ro_w=update.readout_linear_1.output_dimension();
        const int pe_w=update.potential_linear.output_dimension();
        // d_in(i): i=0 → 初始校正后密度; i>0 → i-1 步校后密度快照。
        Kokkos::View<const Precision**,Kokkos::LayoutRight> density_in=
            step==0?Kokkos::View<const Precision**,Kokkos::LayoutRight>(
                        polar_field_independent_density)
                   :Kokkos::View<const Precision**,Kokkos::LayoutRight>(
                        Kokkos::subview(polar_scf_step_densities_post,
                            step-1,Kokkos::ALL,Kokkos::ALL));
        replay_step_forward(step,density_in);
        ensure_view(polar_rev_channel_adjoint,polar_rev_channel_adjoint_storage,
            num_nodes,8);
        ordered_kokkos_deep_copy(polar_rev_channel_adjoint,Precision(0));
        // (a) 逐步 Fukui 校正反向:d̄(校正后) → d̄_pre + s̄。
        ensure_view(polar_rev_out_adjoint,polar_rev_out_adjoint_storage,
            num_nodes,static_cast<int>(update.readout_linear_2.output_dimension()));
        {
            auto dbar=polar_rev_density_adjoint;
            auto out_adj=polar_rev_out_adjoint;
            Kokkos::parallel_for("polar rev out adjoint head",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,8}),
                KOKKOS_LAMBDA(int node,int column) {
                    out_adj(node,column)=dbar(node,column);
                });
        }
        ensure_view(polar_rev_fukui_source_adjoint,
            polar_rev_fukui_source_adjoint_storage,num_nodes,2);
        fukui_reverse(polar_rev_density_adjoint,polar_rev_out,/*source_offset=*/8,
            polar_scf_fukui_scalars.at(step),polar_rev_fukui_source_adjoint);
        {
            auto out_adj=polar_rev_out_adjoint;
            auto s_bar=polar_rev_fukui_source_adjoint;
            Kokkos::parallel_for("polar rev out adjoint tail",num_nodes,
                KOKKOS_LAMBDA(int node) {
                    out_adj(node,8)=s_bar(node,0);
                    out_adj(node,9)=s_bar(node,1);
                });
        }
        // (b) 更新块反向: out̄ → gate2 → ro2 → gate1 → ro1 → tp → MLP → nl_in → dp/se
        //     → pe → P̄ / d̄_ce / fmix̄。
        ensure_view(polar_rev_gate2_adjoint,polar_rev_gate2_adjoint_storage,
            num_nodes,gate_w);
        ensure_view(polar_rev_ro2_adjoint,polar_rev_ro2_adjoint_storage,
            num_nodes,ro_w);
        ensure_view(polar_rev_gate1_adjoint,polar_rev_gate1_adjoint_storage,
            num_nodes,gate_w);
        ensure_view(polar_rev_ro1_adjoint,polar_rev_ro1_adjoint_storage,
            num_nodes,ro_w);
        ensure_view(polar_rev_tp_adjoint,polar_rev_tp_adjoint_storage,
            num_nodes,static_cast<int>(polar_rev_tp.extent(1)));
        ensure_view(polar_rev_nlout_adjoint,polar_rev_nlout_adjoint_storage,
            num_nodes,static_cast<int>(polar_rev_nl_out.extent(1)));
        ensure_view(polar_rev_nl_adjoint,polar_rev_nl_adjoint_storage,
            num_nodes,3*hidden);
        ensure_view(polar_rev_h_adjoint,polar_rev_h_adjoint_storage,
            num_nodes,3*hidden);
        ensure_view(polar_rev_nlin_adjoint,polar_rev_nlin_adjoint_storage,
            num_nodes,static_cast<int>(polar_rev_nl_in.extent(1)));
        ensure_view(polar_rev_dp_adjoint,polar_rev_dp_adjoint_storage,
            num_nodes,dp_width);
        ensure_view(polar_rev_se_adjoint,polar_rev_se_adjoint_storage,
            num_nodes,dp_width);
        ensure_view(polar_rev_pe_adjoint,polar_rev_pe_adjoint_storage,
            num_nodes,pe_w);
        ensure_view(polar_rev_potential_adjoint,
            polar_rev_potential_adjoint_storage,num_nodes,16);
        update.readout_linear_2.reverse(polar_rev_out_adjoint,
            polar_rev_gate2_adjoint);
        ordered_kokkos_deep_copy(polar_rev_ro2_adjoint,Precision(0));
        readout_gate_reverse(polar_rev_ro2,polar_rev_gate2_adjoint,
            polar_rev_ro2_adjoint,update.gate_scalar_cst,
            update.gate_gate_cst,update.gate_scalar_count,
            update.gate_gate_count);
        update.readout_linear_mid.reverse(polar_rev_ro2_adjoint,
            polar_rev_gate1_adjoint);
        ordered_kokkos_deep_copy(polar_rev_ro1_adjoint,Precision(0));
        readout_gate_reverse(polar_rev_ro1,polar_rev_gate1_adjoint,
            polar_rev_ro1_adjoint,update.gate_scalar_cst,
            update.gate_gate_cst,update.gate_scalar_count,
            update.gate_gate_count);
        update.readout_linear_1.reverse(polar_rev_ro1_adjoint,
            polar_rev_tp_adjoint);
        { // (i) uvu_tp 反向: x1=fmix(→fmix̄), x2=nl_out(→nl̄_out)
            const auto to_host_vector=[&](const auto& view) {
                const auto mirror=Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(),view);
                std::vector<typename decltype(mirror)::value_type>
                    values(mirror.size());
                for(std::size_t i2=0;i2<values.size();++i2)
                    values[i2]=mirror.data()[i2];
                return values;
            };
            const std::vector<int> tp_meta=to_host_vector(update.uvu_tp_meta);
            const std::vector<Precision> tp_pw=
                to_host_vector(update.uvu_tp_path_weight);
            ordered_kokkos_deep_copy(polar_rev_nlout_adjoint,Precision(0));
            uvu_reverse(tp_meta,tp_pw,update.uvu_tp_weight,
                update.uvu_tp_mask,polar_features_mixed,polar_rev_nl_out,
                polar_rev_tp_adjoint,polar_rev_fmix_adjoint,
                polar_rev_nlout_adjoint);
        }
        { // (ii) MLP 反向滚动: a 伴缓冲=nl_adjoint,h 伴缓冲=h_adjoint
            auto h_adj=polar_rev_h_adjoint;
            auto a_adj=polar_rev_nl_adjoint;
            auto h=polar_rev_nl_h;auto a=polar_rev_nl_a;
            // dense3 的输入伴写 a2 块([2h,3h)):以子视图承接(initialize 只清零该块)。
            dense_reverse(update.nl_w3,polar_rev_nlout_adjoint,
                Kokkos::subview(a_adj,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden)),true);
            {
                auto h2=Kokkos::subview(h,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden));
                auto a2=Kokkos::subview(a,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden));
                auto a2_adj=Kokkos::subview(a_adj,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden));
                auto h2_adj=Kokkos::subview(h_adj,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden));
                ordered_kokkos_deep_copy(h2_adj,Precision(0));
                ln_silu_reverse(h2,a2,a2_adj,h2_adj,update.ln2_weight,
                    update.ln2_bias,update.ln_eps);
            }
            {
                auto h2_adj=Kokkos::subview(h_adj,Kokkos::ALL,
                    std::make_pair(2*hidden,3*hidden));
                auto a1_adj=Kokkos::subview(a_adj,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden));
                dense_reverse(update.nl_w2,h2_adj,a1_adj,true);
            }
            {
                auto h1=Kokkos::subview(h,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden));
                auto a1=Kokkos::subview(a,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden));
                auto a1_adj=Kokkos::subview(a_adj,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden));
                auto h1_adj=Kokkos::subview(h_adj,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden));
                ordered_kokkos_deep_copy(h1_adj,Precision(0));
                ln_silu_reverse(h1,a1,a1_adj,h1_adj,update.ln1_weight,
                    update.ln1_bias,update.ln_eps);
            }
            {
                auto h1_adj=Kokkos::subview(h_adj,Kokkos::ALL,
                    std::make_pair(hidden,2*hidden));
                auto a0_adj=Kokkos::subview(a_adj,Kokkos::ALL,
                    std::make_pair(0,hidden));
                dense_reverse(update.nl_w1,h1_adj,a0_adj,true);
            }
            {
                auto h0=Kokkos::subview(h,Kokkos::ALL,
                    std::make_pair(0,hidden));
                auto a0=Kokkos::subview(a,Kokkos::ALL,
                    std::make_pair(0,hidden));
                auto a0_adj=Kokkos::subview(a_adj,Kokkos::ALL,
                    std::make_pair(0,hidden));
                auto h0_adj=Kokkos::subview(h_adj,Kokkos::ALL,
                    std::make_pair(0,hidden));
                ordered_kokkos_deep_copy(h0_adj,Precision(0));
                ln_silu_reverse(h0,a0,a0_adj,h0_adj,update.ln0_weight,
                    update.ln0_bias,update.ln_eps);
            }
            {
                auto h0_adj=Kokkos::subview(h_adj,Kokkos::ALL,
                    std::make_pair(0,hidden));
                dense_reverse(update.nl_w0,h0_adj,
                    polar_rev_nlin_adjoint,true);
            }
        }
        { // (iii) nl_in 伴拆分: [dp | se](se=attrs 常量路径,丢弃)
            auto nlin_adj=polar_rev_nlin_adjoint;
            auto dp_adj=polar_rev_dp_adjoint;
            auto se_adj=polar_rev_se_adjoint;
            Kokkos::parallel_for("polar rev nl_in split",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,dp_width}),
                KOKKOS_LAMBDA(int node,int column) {
                    dp_adj(node,column)=nlin_adj(node,column);
                    se_adj(node,column)=nlin_adj(node,dp_width+column);
                });
        }
        { // (iv) uvu_dot 反向: x1=fmix(→fmix̄), x2=pe(→pe伴)
            const auto to_host_vector=[&](const auto& view) {
                const auto mirror=Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(),view);
                std::vector<typename decltype(mirror)::value_type>
                    values(mirror.size());
                for(std::size_t i2=0;i2<values.size();++i2)
                    values[i2]=mirror.data()[i2];
                return values;
            };
            const std::vector<int> dot_meta=
                to_host_vector(update.uvu_dot_meta);
            const std::vector<Precision> dot_pw=
                to_host_vector(update.uvu_dot_path_weight);
            ordered_kokkos_deep_copy(polar_rev_pe_adjoint,Precision(0));
            uvu_reverse(dot_meta,dot_pw,update.uvu_dot_weight,
                update.uvu_dot_mask,polar_features_mixed,polar_rev_pe,
                polar_rev_dp_adjoint,polar_rev_fmix_adjoint,
                polar_rev_pe_adjoint);
        }
        { // (v) 势嵌入反: pe伴 → P̄ / fmix̄(累加) / d̄_ce(累加)
            update.potential_linear.reverse(polar_rev_pe_adjoint,
                polar_rev_potential_adjoint);
            update.node_feats_linear.reverse(polar_rev_pe_adjoint,
                polar_rev_fmix_adjoint,/*initialize_input_adjoint=*/false);
            update.charge_embedding.reverse(polar_rev_pe_adjoint,
                polar_rev_channel_adjoint,/*initialize_input_adjoint=*/false);
        }
        // (vi) 末步并入 ⑦ 的 ∂E/∂P_last;描述符反向(逐通道)。
        if(step==polar_recursion_steps-1&&(seed_terms&1)) {
            auto pbar=polar_rev_potential_adjoint;
            auto seed=polar_seed_potential_adjoint;
            Kokkos::parallel_for("polar rev last step P seed",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,16}),
                KOKKOS_LAMBDA(int node,int column) {
                    pbar(node,column)+=seed(node,column);
                });
        }
        ensure_view(polar_rev_feats_adjoint,polar_rev_feats_adjoint_storage,
            num_nodes,8);
        {
            auto pbar=polar_rev_potential_adjoint;
            auto feats_bar=polar_rev_feats_adjoint;
            auto knorm=polar_field_feature_norms;
            for(int channel=0;channel<2;++channel) {
                const int base=8*channel;
                Kokkos::parallel_for("polar rev feats split",num_nodes,
                    KOKKOS_LAMBDA(int node) {
                        for(int j=0;j<8;++j)
                            feats_bar(node,j)=pbar(node,base+j)/knorm(j);
                    });
                descriptor_reverse_channel(channel,density_in);
            }
        }
        // (vii) d̄_i = d̄_pre + q̄(两通道) + d̄_ce
        {
            auto dbar=polar_rev_density_adjoint;
            auto extra=polar_rev_channel_adjoint;
            Kokkos::parallel_for("polar rev density chain merge",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,8}),
                KOKKOS_LAMBDA(int node,int column) {
                    dbar(node,column)+=extra(node,column);
                });
        }
    }
    // ⑦ 的 ∂E/∂d_init(初校正后密度)在 SCF 链尾并入。
    if(seed_terms&1) {
        auto dbar=polar_rev_density_adjoint;
        auto seed_init=polar_seed_density0_adjoint;
        Kokkos::parallel_for("polar reverse seed density init",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,8}),
            KOKKOS_LAMBDA(int node,int column) {
                dbar(node,column)+=seed_init(node,column);
            });
    }

    // ---------- 6) 初始化 Fukui 反向 → s̄0 → Readout::reverse_all → ⑧-A → 末层 ----------
    {
        ensure_view(polar_rev_fukui_source_adjoint,
            polar_rev_fukui_source_adjoint_storage,num_nodes,2);
        fukui_reverse(polar_rev_density_adjoint,polar_fukui_sources,
            /*source_offset=*/0,polar_fukui_init_scalars,
            polar_rev_fukui_source_adjoint);
        ensure_view(polar_rev_nfeat_adjoint,polar_rev_nfeat_adjoint_storage,
            num_nodes,feature_width);
        polar_fukui_source_map->reverse_all(polar_rev_fukui_source_adjoint,
            polar_rev_nfeat_adjoint,/*initialize_input_adjoint=*/true);
        ensure_view(states.back().layer_adjoint,
            states.back().layer_adjoint_storage,num_nodes,feature_width);
        basis_adjoint_add(polar_rev_nfeat_adjoint,
            states.back().layer_adjoint);
    }

    // ---------- 7) ⑧-1 lr_src 逐层反向(全量 d̄ 平权,sum 线性) ----------
    for(int layer=0;layer<layers;++layer) {
        ensure_view(states[layer].layer_adjoint,
            states[layer].layer_adjoint_storage,num_nodes,feature_width);
        polar_lr_source_maps[layer].reverse(polar_rev_density_adjoint,
            polar_rev_nfeat_adjoint,/*initialize_input_adjoint=*/true);
        basis_adjoint_add(polar_rev_nfeat_adjoint,
            states[layer].layer_adjoint);
    }

    // ---------- 8) ⑧-5 mixer 逐层反向 ----------
    for(int layer=0;layer<layers;++layer) {
        polar_mixer_linears[layer].reverse(polar_rev_fmix_adjoint,
            polar_rev_nfeat_adjoint,/*initialize_input_adjoint=*/true);
        basis_adjoint_add(polar_rev_nfeat_adjoint,
            states[layer].layer_adjoint);
    }

    // ---------- 9) 位置伴 → 节点力(语义同 node_forces: −∂E/∂r) ----------
    {
        auto pos=polar_rev_position_adjoint;
        auto node_force=polar_node_forces;
        Kokkos::parallel_for("polar node force write",num_nodes,
            KOKKOS_LAMBDA(int node) {
                node_force(3*node)=-static_cast<double>(pos(node,0));
                node_force(3*node+1)=-static_cast<double>(pos(node,1));
                node_force(3*node+2)=-static_cast<double>(pos(node,2));
            });
    }

    // ---------- 10) 收尾: 种子 consume-once ----------
    polar_reverse_seed_terms=0;
    if(polar_seed_density_adjoint.extent(0)!=0)
        ordered_kokkos_deep_copy(polar_seed_density_adjoint,Precision(0));
    if(polar_seed_density0_adjoint.extent(0)!=0)
        ordered_kokkos_deep_copy(polar_seed_density0_adjoint,Precision(0));
    if(polar_seed_potential_adjoint.extent(0)!=0)
        ordered_kokkos_deep_copy(polar_seed_potential_adjoint,Precision(0));
}

// ⑧-A 公开小工具(⑦ le 反向的 n̄f_last 路径投递口;hpp 声明在 §2.2)。
template<typename Precision>
void MaceNonlinearKokkosT<Precision>::polar_inject_node_features_adjoint(
    const int layer,Kokkos::View<const Precision**,Kokkos::LayoutRight> adjoint)
{
    if(!polar_enabled||layer<0||layer>=static_cast<int>(states.size()))
        throw std::invalid_argument(
            "polar node-feature adjoint injection layer is invalid.");
    if(static_cast<int>(adjoint.extent(1))!=polar_feature_width)
        throw std::invalid_argument(
            "polar node-feature adjoint width mismatch.");
    ensure_view(states[layer].layer_adjoint,states[layer].layer_adjoint_storage,
        adjoint.extent(0),polar_feature_width);
    auto source=polar_basis_source;
    auto sign=polar_basis_sign;
    auto internal=states[layer].layer_adjoint;
    const int num_nodes=static_cast<int>(adjoint.extent(0));
    Kokkos::parallel_for("polar inject node-feature adjoint",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
            {num_nodes,polar_feature_width}),
        KOKKOS_LAMBDA(int node,int column) {
            Kokkos::atomic_add(&internal(node,source(column)),
                sign(column)*adjoint(node,column));
        });
}
```

**配套（核对项；对应文本均已在 §2.2 一并给出，此处仅复核**）：

1. hpp 声明 `polar_inject_node_features_adjoint`（§2.2 已含；tpp 定义见上）。
2. hpp 视图 `polar_rev_nlout_adjoint,polar_rev_nlout_adjoint_storage`（§2.2 已含）。
3. hpp 成员 `int polar_reverse_terms_last=0;`（§2.2 已含；主函数第 2 段赋值）。
4. **几何就绪检查**已**内联**在主函数开头（不新增成员/辅助函数），文本即主函数第 0 段之前的三行
   判断（`polar_cosines/polar_k_vectors` 的 extent 校验）；其余几何尺寸在各自 λ 内由
   `ensure_view` 按对象维度校验；该检查只保证"前向确已跑过"。

### 4.3 实现注（数值顺序与位级一致性）

1. **重放位级一致**：`replay_step_forward` 与⑤已落码的算子次序、循环次序、归约次序逐一相同
   （逐算子镜像，仅缓冲名不同）；唯一"形状融合"是 project 与 `/knorm` 两个 kernel 的合并——
   逐元素算式与次序不变（先 `feat[perm]−si·q` 再 `/knorm`），bitwise 等价。若未来⑤前向改动，
   重放必须同步修改（§8 第 20 条给出机器可查的对照 grep）。
2. **单射散射**：⑧-A 散射与 `polar_inject_node_features_adjoint` 使用 `atomic_add`；`basis_source`
   的分块双射性由 ctor 构造保证（l=0 恒等、l=1 三循环）。
3. **d̄ 就地更新次序**：`fukui_reverse` 先做宿主归约（F̄）再写 `dbar`；`dbar` 的 l=0 列在同一
   kernel 内"先读后写"（`s_bar` 用旧值、`dbar` 用 `+=pred̄`），无跨线程冲突（逐 node 独立）。
4. **device 内核次序**：跨 kernel 的 `pos_adj` 累加依赖同一执行流的顺序语义（同 execution space
   连续 launch），无 atomics 需求；kernel 内同一 (node,·) 的多写者一律 atomic。

---

## 5. 节点级力接线（两处修改文本 + 消费点清单）

### 5.1 归约点 1：tpp `reduce_node_forces`（PART3）

**锚点**（唯一，count=1）：

```cpp
void MaceNonlinearKokkosT<Precision>::reduce_node_forces(
```

在该函数**体末尾**（边级归约 `Kokkos::parallel_for("MaceNonlinearKokkos::reduce_node_forces", ...)`
之后、函数闭括号之前）插入：

```cpp
    // P3 模块⑧: 节点级极性力并入(= −∂E/∂r;与 node_forces 同符号约定)。
    // 覆盖所有 atom_forces 消费方: calculator `_reduce_atom_forces` 绑定、
    // reduce_prepared_node_forces(极化在 prepared 路径不可达,此处为一致性防御)。
    if(polar_enabled) {
        if(polar_node_forces.extent(0)<3u*static_cast<std::size_t>(num_nodes))
            throw std::invalid_argument(
                "MH-1 atom-force reduction exceeds the polar node-force extent.");
        const auto polar_forces=polar_node_forces;
        Kokkos::parallel_for(
            "MaceNonlinearKokkos::reduce_polar_node_forces",
            Kokkos::RangePolicy<decltype(factorized_execution_space),
                Kokkos::IndexType<int>>(
                factorized_execution_space,0,num_nodes),
            KOKKOS_LAMBDA(const int node) {
                for(int component=0;component<3;++component)
                    Kokkos::atomic_add(&reduced_forces(3*node+component),
                        polar_forces(3*node+component));
            });
    }
```

（`reduced_forces` 是本函数内既有局部变量 `const auto reduced_forces=atom_forces;`，
作用域覆盖插入点——插入点必须在边级 `parallel_for` **之后**、函数尾之前。）

### 5.2 归约点 2：pair style `force_reduction`（唯一可达路径）

消费点盘点（grep 证据）：

- `pair_symmetrix/pair_symmetrix_mace_kokkos.cpp` **L1179** `PairSymmetrixMACEKokkos::force_reduction`
  （`compute_no_domain_decomposition`）——**唯一**会在极化模型上执行的力归约
  （该路径调用标准 `compute_node_energies_forces`，L1144/L1153）。
- 同文件 L1809-1836 "Extract Fixed-Workspace Reduced Forces" 消费 `mace->atom_forces`
  （= §5.1 修改后自动含极性项；但其所在路径 `compute_mpi_message_passing` 走 legacy
  `reverse_A0` 管线、不执行极化前向 ⇒ 极性项恒零，无需额外修改）。
- 同文件 L2205 "force reduction"（`compute_no_mpi_message_passing`）同样走 legacy 管线
  （`mace->compute_Y/compute_R0/reverse_A0`），极化前向不可达 ⇒ 无需修改。
- **判据**：只有包含 `compute_node_energies_forces(` 调用的路径才可能产出极性力；
  `grep -n "compute_node_energies_forces(" pair_symmetrix_mace_kokkos.cpp` 只命中
  L1144/L1153（均在 `compute_no_domain_decomposition` 内），与上述一致（§8 第 21 条）。

**修改**：在 L1179 `force_reduction` 的 `Kokkos::parallel_for` **之后**插入：

```cpp
  // P3 模块⑧: 节点级极性力(每 owned 节点一段;此处 f 为 LAMMPS 力数组,语义=累加)。
  if (mace->polar_blocks_enabled() && mace->polar_node_forces.extent(0) >=
      static_cast<std::size_t>(3*num_nodes)) {
    auto polar_forces = mace->polar_node_forces;
    Kokkos::parallel_for("PairSymmetrixMACEKokkos::polar_node_force_addition",
      num_nodes,
      KOKKOS_LAMBDA (const int ii) {
        const int i = node_indices(ii);
        for (int c = 0; c < 3; ++c)
          Kokkos::atomic_add(&f(i,c), polar_forces(3*ii+c));
      });
  }
```

**配套 C++ 访问器**（hpp 公开区；binding 不需要但 pair style 需要）：

```cpp
    bool polar_blocks_enabled() const { return polar_enabled; }
```

（`node_indices(ii)` 为 pair style 既有视图，`ii` 为节点槽位、`i` 为全局原子索引——
与 `force_reduction` 中 `const int i = node_indices(ii);` 的用法完全一致；
§5.2 的循环必须放在 `force_reduction` 之后以避免与它的 per-node 写竞争同一 f 元素——
atomic_add 已可保证正确性，次序仅为可读性。）

### 5.3 与 python 计算器的接缝

`symmetrix/source/symmetrix/calculator.py` 的 `_compute_*`（L3872-3899）优先走
`evaluator._reduce_atom_forces(...)` → §5.1 ⇒ bench/ase 路径自动获得极性力；
其 numpy 回退支（无 `_reduce_atom_forces` 的旧引擎）读 `evaluator.node_forces` 手工 bincount，
**不含**极性节点力——已在 §8 第 22 条给出 grep 断言（该回退支对新引擎不可达）。

---

## 6. cell 伴随（硬缺口 3）—— 结构、累加位置与"下一 slice"阻断理由

### 6.1 已完成的结构（本 slice 只落占位与静默错误防护）

- `polar_cell_adjoint(9)` 已在校正实现中清零（§4.2 第 0 段）；本 slice **不产生**消费者。
- **∂k/∂cell 结构（写给下一 slice，实现时按下式，再以 3×3 cell 的 FD-strain 复核转置记账）**：
  前向锚（本文件 ③ 块，已对 numpy 逐位锁定）：
  ```
  rec[i][j] = 2π·inv_cell[j][i]            (i,j 循环方向见 ③ 块 reciprocal[·][·] 赋值)
  k_i(k点)  = Σ_j n_j·rec[j][i]            (n = 半空间枚举整数向量)
  V         = |det(cell)|;  ρ_scale = (2π)³/V
  ```
  相位反向已给出 `k̄_i(k) = Σ_n r_i(n)·inner̄(k,n)`（§4.2 (2g) 的 `inner` 即 k̄ 的被积量，
  加一行 `atomic_add(&kbar(k,i), positions(i)·inner)` 即得）。随后：
  ```
  r̄ec[j][i] = Σ_k n_j(k)·k̄_i(k)                      (∂k/∂rec)
  inv̄_cell[i][j] = 2π·Σ_n k̄_i·n_j                     (rec[j][i]=2π·inv_cell[i][j])
  cell̄ += −inv_cellᵀ·inv̄_cell·inv_cellᵀ              (d(A⁻¹)=−A⁻¹ dA A⁻¹)
  V̄ = −(1/V)·Σ_terms(ρ̄·ρ)  [descriptor 两通道×步 + ⑥]  (ρ=(2π)³/V·ρ̃ 的 V 链)
  cell̄ += V̄·V·inv_cellᵀ                              (∂det/∂A = det·A⁻ᵀ)
  ```
  累加位置：`polar_cell_adjoint`（9, 行主序 = cell[3×3]）；写入点在 k̄ 归约 kernel 与
  两个 ρ̄ kernel 各加一个原子累加即可；k 网格边界（|k|≤cutoff 的枚举集合）随 cell 分段常量，
  导数忽略（不可微点集合测度零，FD 复核时避开即可）。
- **位置部分（可与本 slice 同时闭合的半个维里）**：`polar_node_forces` 已含全部位置伴；
  周期性体系维里中"位置部分"= `Σ_i r_i ⊗ F_i^polar`（须用每原子力而不是边级力，
  故不能借道 `reduce_stress` 的 `node_forces⊗xyz(edge)`）。

### 6.2 阻断理由（"下一 slice"裁定，4 条）

1. **⑥ 未落码**：cell 伴随的最大项（V̄ 与 ρ̄→k̄ from 电荷）的消费者不存在；单测③④的 k̄
   只能覆盖 descriptor 半条链，数值门无法闭环。
2. **RBE/R8 未定**：P4 采样 k 集合由采样器生成；若采样器协方差/权重依赖 cell（Box-Muller
   协方差 = cell⁻¹ 派生），cell 伴随需加权重项，且 Metropolis 接受域在 cell 上分段常量
   ⇒ 导数不连续。必须等 P4 的"反向冻结样本"语义锁定。
3. **应力 API 缺消费者**：`reduce_stress` 只消费 `node_forces⊗xyz(edge)`；极化维里需要
   (i) `Σ_i r_i⊗F_i` 新项、(ii) `cell̄` 显式项、(iii) binding `_reduce_stress` 与 pair style
   `vflag_global` 两处接线。三处都属应力面，超出模块⑧（力）范围。
4. **转置记账需专门门**：`cell̄` 公式含两处矩阵转置（inv_cell 的伴随、V 的 cofactor），
   必须以应变 FD（strain-FD，本仓库尚无协议）逐元素复核后才能启用。

### 6.3 防静默错误（本 slice 必做）

`reduce_stress`（tpp，锚点 `void MaceNonlinearKokkosT<Precision>::reduce_stress(`）
函数体首部插入：

```cpp
    // P3 模块⑧: 极化 cell 伴随未实现(见 docs/module8_patch_final.md §6)。
    // 禁止静默输出"缺极化项"的解析维里。
    if(polar_enabled)
        throw std::runtime_error(
            "MH-1 stress reduction: polar cell adjoint (virial) is not "
            "implemented yet; disable virial/stress for polar models.");
```

（pair style 侧若要防 `vflag_global` 静默错误，可在 `compute_no_domain_decomposition`
的 `if (vflag_global)`（锚点 `PairSymmetrixMACEKokkos::virial_reduction`）前加
`if (mace->polar_blocks_enabled()) error->all(FLERR, "polar virial (cell adjoint) not implemented yet; set virial off");`——
建议随本 slice 一并落地。）

---

## 7. binding（`symmetrix/source/cpp/mace_nonlinear_kokkos.cpp`）

**裁定：无需新访问器即可完成力接线**（pair style 直读 C++ 成员；calculator 走 tpp 归约）。
仅建议（§7.1 两条为门脚本必需；7.2 为调试可选）。

### 7.1 建议新增（门脚本依赖；锚点=`.def("polar_scf_final_density", ...)` 链尾整体替换为"原块 + 新增块"）

```cpp
        .def("polar_scf_final_density",[](const Evaluator& self) {
            return view2vector(self.polar_scf_final_density);
        })
        .def("polar_node_forces",[](const Evaluator& self) {
            // 模块⑧: 节点级极性力(3N;−∂E/∂r;由 reduce 消费前的原始累加器)。
            return view2vector(self.polar_node_forces);
        })
        .def("polar_reverse_terms_last",[](const Evaluator& self) {
            // 模块⑧: 最近一次反向实际消费的种子掩码(bit0=⑦le,bit1=⑥coulomb;
            // 诊断用,非 consume-once;0 ⇒ 引擎无极化能量项 ⇒ 力恒零)。
            return self.polar_reverse_terms_last;
        })
        .def("polar_blocks_enabled",[](const Evaluator& self) {
            // 模块⑧: polar_enabled 只读镜像(pair style 用同名 C++ 访问器)。
            return self.polar_blocks_enabled();
        })
        .def("polar_scf_step_densities_post",[](const Evaluator& self) {
            // 模块⑧: 每步 Fukui 校正【后】密度(S×N×8;= 下一步步输入/末态)。
            return view2vector(self.polar_scf_step_densities_post);
        });
```

（`polar_reverse_terms_last` 为 hpp 公开 int 成员（§2.2）；`polar_reverse_seed_terms` 保持
consume-once 语义、**不**绑定到脚本，防误用。）

### 7.2 调试可选（默认不加；FD 定位用）

- `.def("polar_cell_adjoint", ...)`——cell 伴随下一 slice 落地后启用；
- `.def("polar_node_forces")` 已有；如需 per-step 的 P̄/fmix̄ 诊断，可在反向末尾暴露
  `polar_rev_potential_adjoint`/`polar_rev_fmix_adjoint` 的 **norm**（不建议暴露全量）。

### 7.3 明确不加

- 不加 “polar_reverse_all” 之类的 Readout 级绑定——反向前向必须成对且由引擎内部驱动，
  暴露给脚本会打破 `evaluate_all → reverse_all` 的隐藏缓存前提（§3）。
---

## 8. 自检清单（合并前后逐条执行；全部为只读 grep/读数动作）

**A. 锚点存在性（应用前；每条应命中唯一）**

1. `grep -c "polar_nl_b_storage;" libsymmetrix/source/mace_nonlinear_kokkos.hpp` → 1（§2.2 插入点）。
2. `grep -c "void evaluate_all(Kokkos::View<const Precision\*\*,Kokkos::LayoutRight> input," mace_nonlinear_kokkos.hpp` → 1（§2.3 插入点）。
3. `py: 断言 tpp 中三行组合 "ensure_view(result,result_storage,input.extent(0),linear_2.output_dimension());\n    linear_2.evaluate(activated,result);\n    ordered_kokkos_deep_copy(output,result);\n}" 出现 1 次`（§3 插入点；注意与 `evaluate` 的 `...subview(result,Kokkos::ALL,0))` 分支区分）。
4. `grep -c "ensure_view(harmonic_adjoints,harmonic_adjoints_storage,edges,num_lm);" mace_nonlinear_kokkos_impl.tpp` → 1（§4.0-A 调用点）。
5. `grep -c "const double volume=std::abs(" mace_nonlinear_kokkos_impl.tpp` → 1（§4.0-B）。
6. `grep -n "for(int channel=0;channel<2;++channel)" mace_nonlinear_kokkos_impl.tpp` → 4 处；其中 §4.0-C 的两处钳制锚必须同时含**前一行的**
   `predicted_zero[1]+=polar_accumulator_host(node,4);`（初始化）/`predicted_zero[1]+=density_host(node,4);`（逐步）——用这两行做上下文锚，
   各自 count=1；插入后 `grep -c "raw_source_sums" ` → 4（两处各 2 次引用）。
7. `grep -c "polar fukui correction" mace_nonlinear_kokkos_impl.tpp` → 1；`grep -c "polar scf fukui correction" ` → 1（§4.0-C 快照插入点）。
8. `grep -c "#error \"SYMMETRIX_MACE_NONLINEAR_KOKKOS_PART must select one implementation partition\"" mace_nonlinear_kokkos_impl.tpp` → 1（§4.2 插入点）。
9. `grep -c "void MaceNonlinearKokkosT<Precision>::reduce_node_forces(" mace_nonlinear_kokkos_impl.tpp` → 1（§5.1 插入点）。
10. `grep -c "PairSymmetrixMACEKokkos::force_reduction" pair_symmetrix/pair_symmetrix_mace_kokkos.cpp` → 1（§5.2 插入点）。
11. `grep -c '.def("polar_scf_final_density"' symmetrix/source/cpp/mace_nonlinear_kokkos.cpp` → 1（§7 替换锚）。
12. `grep -c "void MaceNonlinearKokkosT<Precision>::reduce_stress(" mace_nonlinear_kokkos_impl.tpp` → 1（§6.3 guard 插入点）。

**B. 应用后结构与计数核对（grep 预期值）**

13. `grep -c "compute_polar_reverse_forces"`：hpp = 2（声明 + 注释或如实现集中）**以最终文本为准**；tpp = 2（定义 + 调用）；
    `grep -c "polar_inject_node_features_adjoint"`：hpp = 1（声明）+ tpp = 2（定义 + 注释引用）=
    以实际计数核对即可（本清单的作用是"变化有据"，不追求跨文件固定数）。
14. `grep -c "reverse_all" mace_nonlinear_kokkos_impl.tpp` → 3（定义 1 + 调用 2：初始化 Fukui 处 1；`polar_fukui_source_map->evaluate_all` 已有 1，勿混）。
15. `grep -c "kspace_assemble_reverse" mace_nonlinear_kokkos_impl.tpp` → 3（1 定义 + 2 调用：⑥ 与描述符）。
16. `grep -c "编辑注\|占位\|TODO\|KOKKOS_KERNEL_LAMBDA" mace_nonlinear_kokkos_impl.tpp` → 0（应用后代码不得残留提案注释字串）。
17. `grep -n "polar_rev_nlout_adjoint" mace_nonlinear_kokkos_impl.tpp` → ≥4（ensure + uvu 目标 + dense3 输入 + 清零）；
    hpp 同名字段 1 行。
18. `grep -c "polar_coulomb_enabled" mace_nonlinear_kokkos_impl.tpp` → 2（掩码位 1 + ⑥ 段头 1）；hpp 1（定义）。
    ⑥ 落码后此处应增至 ≥4（parse 置位 + 注释）。
19. `grep -c "polar_reverse_terms_last"`：hpp = 2（定义 + 注释/绑定用）、tpp = 2（赋值 + 注释）、binding = 1。
20. **重放镜像对照**（关键正确性自检，逐对 grep）：对⑤前向 kernel 标签逐条确认重放存在同构体——
    `polar scf assemble fourier` ↔ `polar rev replay assemble`；`polar scf project features` ↔ `polar rev replay project`；
    `polar uvu mode0/1`（新增 `launch_uvu_forward` 内两处）↔ `polar rev replay uvu mode0/1`；
    `polar pe sum` ↔ `polar rev replay pe sum`；`polar nl concat` ↔ `polar rev replay nl concat`；
    `polar layernorm silu` ↔ `polar rev replay ln`；`polar readout gate` ↔ 重放 gate1/gate2 与反向 `polar reverse readout gate`；
    `polar density update` 必须**不在**重放出现（重放不写密度；若出现即错）。

**C. 数据/JSON 与消费点核对**

21. `grep -c "compute_node_energies_forces(" pair_symmetrix/pair_symmetrix_mace_kokkos.cpp` → 4（含 `_field` 变体；
    普通版恰为 §5.2 列出的 L1144/L1153 两处）——用 `grep -n` 逐行确认其余调用点（L1138/1150）为 `_field` 变体且不在极化路径。
22. `grep -n "reduce_atom_forces = getattr" symmetrix/source/symmetrix/calculator.py` → 1（§5.3 接缝；
    回退支 `results["forces"]` 的 numpy bincount 分支不含极性项——新引擎必有 `_reduce_atom_forces`）。
23. JSON 只读断言（pro 环境 python，只读）：
    `polar_blocks.configs.num_recursion_steps==2`、`field_feature_norms` 8 值、`lr_source_maps` 长度==interactions 长度、
    `field_dependent_charges_maps` 长度==2（§4 重放/种子按步索引的前提）。
24. 反向语义对账（读数，非 grep）：对 `charges_hooks/nh3-step1.npz` 的 `gate_in`,`gate_in2`,`ro_l1_in`,`nl_in`
    与引擎反向前重放视图逐位对比（可用 §9 门的 `--stage replay` 子模式，或临时访问器）；
    预期 ≤5e-6 绝对（与⑤门同源漂移）。

**D. 回归与数值门（合并后）**

25. 重跑 ①②③④⑤ 既有门（`bench/polar_module{1,2,34,5}_gate.py`）：⑧ 落码后应**逐位不变**（无种子 ⇒ 早退；
    且新增仅缓存/清零，不动前向数值）。任一门漂移 >0 即回查 §4.0-C 快照插入是否越界改了变量。
26. `bench/polar_module8_gate.py`（§9）→ `pre` 阶段（seed mask=0）应 PASS 于 FD_PRE=1e-5；
    ⑥/⑦ 落码后 `post` 阶段应 PASS 于 FORCE_GATE=1e-3 eV/Å 与 FD_POST=1e-3。
27. 负例（装载期/运行期契约，只改内存不重编译）：把 `polar_scf_step_densities_post` 以错 S 切片触发
    `step-1` 越界 → 引擎应抛 Kokkos 越界断言或 extent 异常（打印即可，不强制）；把
    `polar_reverse_seed_terms` 手工置 1 但不投种子视图 → 应抛
    `"polar reverse: le seed extents are inconsistent"`（§4.2 第 3 段守卫）。

---

## 9. 验收门 `bench/polar_module8_gate.py`（完整脚本；已落盘，只写不跑）

> 脚本已写入 `mlip/bench/polar_module8_gate.py`（11.7 kB；`python3 -m py_compile` 与
> polar 环境 `py_compile` 双双通过；未执行）。下方代码与落盘文件为同一文本。

设计要点：

- **分阶段**：`--stage auto` 用 `ev.polar_reverse_terms_last()`（0=pre：引擎尚无极化能量项消费方 ⇒ 力应恒零，
  做无极性回归；3=both ⇒ post：全量力/FD 校验）。判据两套。
- **FD 自洽（引擎自身有限差分 vs 解析力，抽样原子）**：中央差分 `h=1e-3 Å`，f64 引擎；
  `F_x ≈ −(E(+h)−E(−h))/(2h)`。pre 阶段判据 `FD_PRE=1e-5 eV/Å`（P2 自洽 3.6e-7 的回归上界），
  post 阶段 `FD_POST=1e-3 eV/Å`。
- **P2 遗留力差目标（python 总力 parity）**：夹具/活体帧上 `|ΔF| = |F_engine − F_python|` 的
  mean/p50/p90/p99/max 分布；打印 P2 遗留基线（mean 1.24 / max 4.15 eV/Å 为"缺 P3 静电层"时的差），
  post 阶段判据 `FORCE_GATE=1e-3 eV/Å`（先看分布再判 max）。
- **活体帧**：`mm.build_gas(mol,32,0.01,seed,2.0)` + rattle（与 module7 门同构）。
- 访问器缺失 → `MODULE8-ACCESSORS-MISSING` + rc=2（区分"未重建"与"数值 FAIL"）。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 模块⑧验收门: 力/全链反向 (2026-10-15; 只写不跑, 由集成方执行)。

判据(两阶段; 由 ev.polar_reverse_terms_last() 自动分档):
  pre  (mask==0: 引擎尚无极化能量项消费方):
    - 引擎自身 FD 自洽(FD_PRE=1e-5 eV/Å; P2 自洽 3.6e-7 的回归上界);
    - polar_node_forces ≡ 0(全节点);且不含极性项的 python 力差与 P2 遗留分布一致(打印)。
  post (mask==3: ⑥/⑦ 已落码):
    - FD 自洽(FD_POST=1e-3 eV/Å, 抽样原子, h=1e-3 Å, f64);
    - python 总力 parity(FORCE_GATE=1e-3 eV/Å; 先看分布 mean/p50/p90/p99/max);
    - Σ_n F_n ≈ 0(平移不变, ≤1e-8·max|F|)。
用法(参照 module5/7 门; descriptor 环境 + unset PYTHONPATH + nvidia-smi shim):
  cd mlip && env -u PYTHONPATH \
    SYMMETRIX_BACKEND_DESCRIPTOR_PATH=$PWD/models/mace-polar/backend_cuda12_sm80.json \
    SYMMETRIX_BACKEND=cuda12-sm80 \
    PATH=$PWD/bench/nvidia_smi_shim:$PATH \
    LD_LIBRARY_PATH=/opt/matos-tools/cuda/12.9/lib64:/opt/matos-tools/conda/envs/matos-compute/lib \
    CUDA_VISIBLE_DEVICES=<gpu> /workspace/caches/conda/envs/polar/bin/python \
    bench/polar_module8_gate.py [--fd-atoms 4] [--h 1e-3] [--live 1] [--stage auto]
退出码: 0=PASS, 1=FAIL, 2=访问器缺失/装载失败。
注: 解释器退出阶段 cudaStreamSynchronize/driver shutting down 为已知无害噪音。
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/bench")
import mace_md as mm  # noqa: E402

J = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
POLAR = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"
FIX = f"{ROOT}/bench/polar_fixtures"
MOLS = [("NH3", 770600), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]
FD_PRE = 1e-5        # eV/Å(pre 阶段: 无极性项回归)
FD_POST = 1e-3       # eV/Å(post 阶段: 全量解析力自洽)
FORCE_GATE = 1e-3    # eV/Å(post 阶段: 与 python 总力 parity)
P2_LEGACY = {"mean": 1.24, "max": 4.15}   # 缺 P3 静电层时的历史力差(eV/Å),留档对照
ACCESSORS = ("polar_node_forces", "polar_reverse_terms_last",
             "polar_blocks_enabled")


def dist(a, b):
    d = np.abs(np.asarray(a, dtype=np.float64).reshape(-1)
               - np.asarray(b, dtype=np.float64).reshape(-1))
    return {"mean": float(d.mean()), "p50": float(np.percentile(d, 50)),
            "p90": float(np.percentile(d, 90)),
            "p99": float(np.percentile(d, 99)), "max": float(d.max()),
            "n": int(d.size)}


def build_atoms_from_npz(d):
    from ase import Atoms
    return Atoms(symbols=[str(s) for s in d["symbols"]],
                 positions=np.asarray(d["positions"], dtype=np.float64),
                 cell=np.asarray(d["cell"], dtype=np.float64), pbc=True)


def engine_energy_forces(engine, ev, atoms):
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = engine
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64).reshape(-1, 3)
    n = len(atoms)
    polar = np.asarray(ev.polar_node_forces(), dtype=np.float64).reshape(-1, 3)
    if polar.shape[0] != n:
        raise RuntimeError(f"polar_node_forces shape {polar.shape} != ({n},3)")
    terms = int(ev.polar_reverse_terms_last())
    return energy, forces, polar, terms


def fd_check(engine, atoms, atom_ids, h):
    """中央差分: F_x ≈ −(E(+h)−E(−h))/(2h); 返回 (fd_forces, energies)。"""
    fd = {}
    for ai in atom_ids:
        for comp in range(3):
            vals = []
            for sign in (+1.0, -1.0):
                a = atoms.copy()
                p = a.get_positions()
                p[ai, comp] += sign * h
                a.set_positions(p)
                a.calc = engine
                vals.append(float(a.get_potential_energy()))
            fd[(ai, comp)] = -(vals[0] - vals[1]) / (2.0 * h)
    return fd


def run_frame(engine, ev, atoms, atom_ids, h, py_calc, want_python):
    energy, forces, polar, terms = engine_energy_forces(engine, ev, atoms)
    fd = fd_check(engine, atoms, atom_ids, h)
    fd_vals = np.array([fd[(ai, c)] for (ai, c) in fd], dtype=np.float64)
    an_vals = np.array([-forces[ai, c] for (ai, c) in fd], dtype=np.float64)
    fd_d = dist(fd_vals, an_vals)
    out = {"energy": energy, "terms": terms, "fd": fd_d,
           "polar_max": float(np.abs(polar).max()),
           "force_balance": float(np.abs(forces.sum(axis=0)).max())}
    if want_python and py_calc is not None:
        a = atoms.copy()
        a.info["charge"] = 0
        a.info["spin"] = 1
        a.calc = py_calc
        py_forces = np.asarray(a.get_forces(), dtype=np.float64).reshape(-1, 3)
        out["parity"] = dist(forces, py_forces)
        out["python_force_max"] = float(np.abs(py_forces).max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd-atoms", type=int, default=4, help="每帧 FD 抽样原子数")
    ap.add_argument("--h", type=float, default=1e-3, help="FD 步长 Å")
    ap.add_argument("--live", type=int, default=1, help="每分子活体帧数(0=只用夹具)")
    ap.add_argument("--stage", choices=("auto", "pre", "post"), default="auto")
    ap.add_argument("--out", default=f"{ROOT}/bench/polar_module8_gate")
    args = ap.parse_args()

    from symmetrix.calculator import Symmetrix

    engine = Symmetrix(J, dtype="float64")
    ev = engine.evaluator
    missing = [n for n in ACCESSORS if not hasattr(ev, n)]
    if missing:
        print(f"MODULE8-ACCESSORS-MISSING: {missing} 引擎未含模块⑧访问器"
              f"(先应用 docs/module8_patch_final.md 并重建)", flush=True)
        return 2
    if not bool(ev.polar_blocks_enabled()):
        print("MODULE8: polar_blocks 未启用(JSON/装载问题)", flush=True)
        return 2

    py_calc = None
    if args.stage != "pre":
        from mace.calculators import mace_polar
        py_calc = mace_polar(model=POLAR, device="cuda", default_dtype="float64")

    rows = []
    stage_seen = None

    # ---------------- 源 A: 夹具(4 分子 × 2 帧) ----------------
    print("== 源 A: 夹具 polar_fixtures/*.npz(8 构型) ==", flush=True)
    for mol, seed in MOLS:
        d0 = np.load(f"{FIX}/{mol.lower()}-f0.npz", allow_pickle=False)
        n = len(d0["symbols"])
        atom_ids = list(range(0, n, max(1, n // args.fd_atoms)))[:args.fd_atoms]
        for k in (0, 1):
            d = np.load(f"{FIX}/{mol.lower()}-f{k}.npz", allow_pickle=False)
            atoms = build_atoms_from_npz(d)
            row = {"case": f"{mol.lower()}-f{k}"}
            row.update(run_frame(engine, ev, atoms, atom_ids, args.h,
                                 py_calc, args.stage != "pre"))
            stage_seen = row["terms"] if stage_seen is None else stage_seen
            rows.append(row)
            print(f"  {row['case']:10s} terms={row['terms']} "
                  f"FD max={row['fd']['max']:.3e}  "
                  f"parity={row.get('parity', {}).get('max', float('nan')):.3e}  "
                  f"|F|max={row['python_force_max'] if 'python_force_max' in row else float('nan'):.3f}  "
                  f"ΣF={row['force_balance']:.3e}", flush=True)

    # ---------------- 源 B: 活体帧 ----------------
    if args.live > 0:
        print(f"== 源 B: 活体重建(--live {args.live}/分子) ==", flush=True)
        for mol, seed in MOLS:
            base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
            for k in range(args.live):
                atoms = base.copy()
                atoms.rattle(stdev=0.05, seed=seed + 7 + 4 * k)
                n = len(atoms)
                atom_ids = list(range(0, n, max(1, n // args.fd_atoms)))[:args.fd_atoms]
                row = {"case": f"{mol.lower()}-live{k}"}
                row.update(run_frame(engine, ev, atoms, atom_ids, args.h,
                                     py_calc, args.stage != "pre"))
                rows.append(row)
                print(f"  {row['case']:12s} terms={row['terms']} "
                      f"FD max={row['fd']['max']:.3e}  "
                      f"parity={row.get('parity', {}).get('max', float('nan')):.3e}",
                      flush=True)

    stage = args.stage
    if stage == "auto":
        stage = "pre" if stage_seen == 0 else "post"
    fd_gate = FD_PRE if stage == "pre" else FD_POST
    ok_fd = all(r["fd"]["max"] <= fd_gate for r in rows)
    ok_zero = (all(r["polar_max"] == 0.0 for r in rows)
               if stage == "pre" else True)
    ok_parity = True
    if stage == "post":
        ok_parity = all(("parity" in r) and (r["parity"]["max"] <= FORCE_GATE)
                        for r in rows)
    ok_balance = all(r["force_balance"] <= 1e-8 * max(1.0, r["python_force_max"]
                     if "python_force_max" in r else 1.0) for r in rows)
    ok = bool(ok_fd and ok_zero and ok_parity and ok_balance)
    verdict = "PASS" if ok else "FAIL"

    fd_max = max(r["fd"]["max"] for r in rows)
    par_max = max((r.get("parity", {}).get("max", 0.0)) for r in rows)
    print(f"\nGATE {verdict}: stage={stage} configs={len(rows)}  "
          f"FD max={fd_max:.3e} (gate {fd_gate:.0e})  "
          f"parity max={par_max:.3e} (gate {FORCE_GATE:.0e}; "
          f"P2 legacy mean/max={P2_LEGACY['mean']}/{P2_LEGACY['max']})", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(rows))
    axes[0].bar(x, np.maximum([r["fd"]["max"] for r in rows], 1e-18))
    axes[0].axhline(fd_gate, color="r", ls="--", lw=1,
                    label=f"FD gate={fd_gate:.0e}")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Engine FD self-consistency ({stage})")
    axes[1].bar(x, np.maximum([r.get("parity", {}).get("max", 0.0)
                               for r in rows], 1e-18), label="|ΔF| max")
    axes[1].axhline(FORCE_GATE, color="r", ls="--", lw=1,
                    label=f"parity gate={FORCE_GATE:.0e}")
    axes[1].axhline(P2_LEGACY["max"], color="gray", ls=":", lw=1,
                    label="P2 legacy max=4.15")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].set_title("python total-force parity (max per frame)")
    axes[2].text(0.5, 0.5,
                 f"Module 8 (forces / full reverse)\nstage: {stage}\n"
                 f"terms mask: {stage_seen}\nverdict: {verdict}",
                 ha="center", va="center", fontsize=12)
    axes[2].axis("off")
    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels([r["case"] for r in rows], rotation=60, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    with open(f"{args.out}.json", "w") as fh:
        json.dump({"rows": rows, "stage": stage,
                   "gates": {"fd": fd_gate, "parity": FORCE_GATE,
                             "p2_legacy": P2_LEGACY},
                   "verdict": verdict}, fh, indent=1)
    print(f"saved: {args.out}.png , {args.out}.json", flush=True)
    print("MODULE8-GATE-DONE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

**脚本注**：

1. `pre` 阶段（无⑥/⑦）FD 判据刻意取 1e-5（P2 已实测 3.6e-7；留 30× 裕度），
   同时断言 `polar_node_forces ≡ 0`（回归不变性的机器证据）。
2. `post` 阶段"先看分布"：`rows[*]["parity"]` 输出 mean/p50/p90/p99/max，判据只卡 max ≤ 1e-3；
   若 max 超而 p99 合格，先在 JSON 里定位离群原子（建议后续把 FD 抽样点覆盖到 β 通道 l=0 校正异常节点）。
3. FD 的 `atom_ids` 若取到 `n//fd_atoms` 步长，注意分子盒中 128 原子时默认抽 4 个（0,32,64,96）。
4. 依赖：`--stage post` 需要 ⑥/⑦ 已落码且 `polar_reverse_terms_last()==3`；否则该档自动失败并在
   摘要里给出 terms 值。

---

## 10. 依赖、风险与合并条件

1. **合并顺序**：⑧ 只依赖已落码的①②③④⑤ + 本提案（无需 ⑥/⑦ 即可编译/装载；
   `polar_coulomb_enabled=false` 默认 + 种子掩码早退 ⇒ 对现役模型零影响）。
   §4.0-C 的 ⑤ 段小插入（快照/标量）是唯一"触碰⑤代码"的改动——只增不改，合并后立即用
   §8 第 25 条回归验证 ①-⑤ 门逐位不变。
2. **⑥ 落码接口**：置 `polar_coulomb_enabled=true`（ctor 内）；若同时暴露 `rho_re/im` 视图，
   把 §4.2 第 4 段的两处"重算"替换为直接消费（位级一致；否则差异在 f64 eps 级）。
3. **⑦ 落码接口**：见 §4.1（三视图 + 掩码 + `polar_inject_node_features_adjoint` 投递）；
   ⑦ 反向若在 ⑧ 之后落码，仅需在其反向函数内新增 ~40 行（模块 ⑦ 的 MLP/uvu 反向可直接复用
   §4.2 的 `dense_reverse`/`ln_silu_reverse`/`uvu_reverse` 语义——建议⑦实现时把这三个 λ 提升为
   类内静态/成员函数以避免重复；本提案先保持⑧自包含）。
4. **R8/RBE**：反向对 k 样本无感知（样本生成不依赖 positions）；P4 落码时必须在反向**冻结样本**
   （把 k 集合当作常量），否则 FD 会被采样噪声淹没。门脚本在 post 阶段的 FD 若用小体系全网格
   （非 RBE）跑，不受此影响。
5. **R11 拒绝纪律**：本 ckpt 的开关矩阵（`zero_charges=False`、`field_si=False`、
   `include_electrostatic_self_interaction=True`、`quadrupole_feature_corrections=False`）已在
   前向对账锁定；若未来 ckpt 打开未实现伴随的开关，应在 ctor 直接拒绝（不静默近似）。
6. **R12 精度**：门在 f64 下判定；f32 生产实例化仅在 post 阶段复跑 FD 时允许放宽到
   ~5e-2 eV/Å（能量量化噪声主导），需单独判据并在门 JSON 标注 dtype。
7. **cell 伴随**：见 §6；本 slice 的 guard（`reduce_stress` 抛错）必须在 P5（NPT）前替换为
   完整实现，否则 NPT 不可用（这是有意的"响亮失败"而非静默错误）。

---

*本文档所有插入文本均以 2026-10-15 09:54 UTC 引擎快照核对；行号一律不写，锚点唯一性由 §8-A 机器校验。
未执行任何构建/写引擎树操作。*
