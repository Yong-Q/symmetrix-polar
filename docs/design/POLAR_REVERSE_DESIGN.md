# PolarMACE 静电极性链反向(模块⑧ = 力)设计预研

**性质**:只读调研 + 设计文档(2026-10-15)。未改任何源码、未跑 GPU/构建、未提交。
**依据**:graph_longrange v0.4.0(`/workspace/caches/conda/envs/polar/lib/python3.12/site-packages/graph_longrange/`)、
mace 0.3.x(`/opt/matos-tools/conda/envs/matos-mlip-py312/lib/python3.12/site-packages/mace/modules/{extensions,field_blocks,blocks,radial,utils}.py`)、
引擎 symmetrix-xl(`/workspace/group/shared/bza/external-repos/symmetrix-xl/libsymmetrix/source/`)、
计划文档 `mlip/docs/SYMMETRIX_POLAR_RBE_PLAN.md` §12/§14。
**配置实证**:MACE-POLAR-1-M 已在本机以 CPU 载入核对(`torch.load(...)` 读取属性),关键数字见 §1.4;
偏振块 JSON 结构见 `mlip/models/mace-polar/polar_symmetrix_probe.json`。

**下文简称**:
`impl.tpp` = `external-repos/symmetrix-xl/libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp`;
`nl.hpp` = 同目录 `mace_nonlinear_kokkos.hpp`;
`e3nn_kokkos.cpp` / `e3nn_kokkos.hpp` = 同目录;
`extensions.py` / `field_blocks.py` / `blocks.py` / `utils.py`(mace) = mace 侧;
`features.py` / `gto_utils.py` / `energy.py` / `kspace.py` = graph_longrange 侧。

---

## 0. 速览(TL;DR)

1. **参考实现没有自定义 autograd**。graph_longrange 全包 grep 无 `autograd.Function` / `ctx.save_for_backward` / `backward`;
   反向 = 原生 torch 算子自动微分。因此引擎侧必须**从前向数学逐式推导伴随**,本文给出全部算子级伴随与推导要点(§2)。
2. **引擎挂接点唯一且明确**:在 `compute_node_energies_forces` 内、readout 播种循环(`impl.tpp` L5448-5460)**之后**、
   骨架反向主循环(`impl.tpp` L5549)**之前**,新增 `polar_reverse(...)`。
   理由:readout 播种是**覆盖**(`E3LinearKokkosT::reverse` 默认 `initialize_input_adjoint=true`,先清零再写,`e3nn_kokkos.cpp` L1209-1210;
   `Readout::reverse` 调用处 `impl.tpp` L1245-1246 未传该参数);而骨架跨层累加用 `init=false`(`impl.tpp` L7034)。
   极化链进入 `state.layer_adjoint` 必须**累加(init=false)**,否则会抹掉 readout 种子。
3. **本检查点的重要简化(配置实证)**:
   - `charges_maps = AgnosticEmbeddedOneBodyVariableUpdate` 的 forward **不消费** `edge_attrs/edge_feats/edge_index`(纯节点级,`field_blocks.py` L563-594),
     因此极化链**不产生** harmonic/radial/cutoff 伴随——极化对力的贡献只有两条路:节点特征伴随(经骨架)与 k 空间相位的直接位置伴随。
   - SCF 迭代数 `num_recursion_steps=2`,两自旋通道;`local_electron_energy` 开启;`coulomb` 自作用项**关闭**(`include_self_interaction=True`,`energy.py` L157 跳过);
     descriptor 自作用项**开启**(`field_si=False`,`features.py` L652-655 生效);非周期/单层修正对 TTT 恒零(`energy.py` L168-192),可跳过。
4. **必须新增的引擎件**(现有代码均缺,见 §3.4/§5):
   ① `Readout::reverse_all`(全列种子 + init 标志;现只有 `evaluate_all`,`nl.hpp` L461,`reverse_all` 全库 grep 为空,
      且 `Readout::reverse` 的 seed 固定为 N×1,`impl.tpp` L1246,对 2 列输出的 fukui 头无法使用);
   ② 极化**节点级**力累加器 `polar_atom_forces(N×3)`(极化位置伴随无法表达为"每条有向边的力",见 §3.4);
   ③ `cell/volume/rcell` 传入标准评估路径(现签名 `impl.tpp` L3733 无 cell;且 polar 明确拒绝 prepared/direct 路径,`impl.tpp` L4205-4207);
   ④ 极化伴随缓存视图(§4)。
5. **未决问题 12 条**(§5),其中高风险:`reverse_all` 与 fukui 头的接缝(R1)、RBE 采样对伴随的影响(R8)、
   virial/cell 链缺位会让 NPT 下的解析维里静默缺失极化项(R9)、SH 约定(内部 sphericart vs e3nn)仅对 node_feats l=1 实证过(R4)。

---

## 1. 参考实现反向现状核验

### 1.1 无自定义反向(证据)
```
grep -n -E "backward|autograd\.Function|ctx\.save_for_backward|def forward|torch\.autograd" graph_longrange/*.py
→ 命中全部为 def forward / @staticmethod;无 backward、无 autograd.Function、无 ctx.save_for_backward。
```
- `features.py` L28-67(`assemble_fourier_series_batch`)、L70-83(`apply_coulomb_kernel_batch`)、L86-117(`project_to_features_batch`):
  全部 torch matmul/sum 组合 —— 反向由 autograd 逐个算子给出。
- `energy.py` L17-30(`energy_product_batch`):`per_k = 2*sum(density*potential,dim=-1)`,`index_add_` 到图;`0.5*V*.../(2π)^6`。
- `kspace.py` L11-127(`compute_k_vectors_flat`):纯张量代数;`gto_utils.py` L33-48(`RadialIntegralDirect.forward`):exp/mul。
- 结论:**"参考反向"= 前向算子雅可比链**。下文 §2 按算子给出伴随;凡与 mace/graph_longrange 前向公式一一对应的,均给 file:line。

### 1.2 mace 侧力的生成
- `get_outputs`(mace `utils.py` L167-222)→ `compute_forces`(L22-38):`forces = -autograd.grad(sum(energy), positions)`;
  polar 的所有能量项都并入同一个 `total_energy`(`extensions.py` L866-929),没有独立的反传代码。
- 因此引擎侧模块⑧ = "把 `total_energy` 里 polar 各项对 positions 的链式贡献解析化",并同时把对 node_feats 的贡献接回骨架反向链。

### 1.3 引擎现有反向机制(接口与语义,只读核对)
| 机制 | 位置 | 关键语义 |
|---|---|---|
| readout 播种 | `impl.tpp` L5448-5460 | `readouts[layer].reverse(state.output, scale, state.layer_adjoint)`;**覆盖**(init=true 默认) |
| `E3LinearKokkosT::reverse(out_adj, in_adj, bool init=true)` | `nl.hpp` L95-97;impl `e3nn_kokkos.cpp` L1198-1233 | init=true 先清零目标再写;false 纯累加;**不需要前向输入** |
| `Readout::reverse` | `impl.tpp` L1245-1269 | seed 恒为 N×1(`ensure_view(seed,...,1)`),`scale` 广播;has_mid 分支=线性_1→act→linear_mid→act→linear_2 的链式伴随 |
| 跨层累加 | `impl.tpp` L7032-7036 | `interaction.linear_up.reverse(state.up_adj, states[layer-1].layer_adjoint, /*init=*/false)` —— **累加**示范 |
| 反向主循环 | `impl.tpp` L5549-7037 | `layer = N-1 … 0`;product reverse(L5727)→linear_2/gate/linear_1→卷积反向(积累 `harmonic_adjoints`/`radial_adjoints`/`cutoff_adjoints`,L6883-6941) |
| 边级→力 | `impl.tpp` L7039-7092 | `F_edge = -(dE/dd·r̂ + Σ_lm harmonic_adjoint·∂Y/∂xyz)`,写 `node_forces(3*edge+c)` |
| 归约 | `impl.tpp` L7128-7160(`reduce_node_forces`);pair style `pair_symmetrix/pair_symmetrix_mace_kokkos.cpp` L1185-1220 | node_forces(source)+=F_edge, (receiver)-=F_edge;pair style 直接读 `mace->node_forces` |
| 应力 | `impl.tpp` L7177-7205 | 只由 `node_forces ⊗ xyz(edge)` 生成 → **无独立 cell 伴随通路**(见 R9) |
| scale 语义 | `impl.tpp` L5471-5472;mace L707 | `scale` 只乘骨架 readout 能量;**polar 各项不乘 scale**(mace L879-883/L925-929) |
| 层数一致性 | `impl.tpp` L3437 | `interactions==products==readouts` 强制 → 每层 `layer_adjoint` 都被播种(不存在未初始化层) |

### 1.4 MACE-POLAR-1-M 关键配置(载入实测,供伴随复现)
- `num_recursion_steps=2`;两自旋通道;`add_local_electron_energy=True`。
- `atomic_multipoles_max_l=1`(charges 8 维 = `1x0e+1x1o+1x0e+1x1o`,mace `extensions.py` L431);
  `field_feature_max_l=1`、`field_feature_widths=[1.5,3.0]` → `field_irreps=2x0e+2x1o`(8 维)、`potential_irreps=16 维`(L450-455)。
- `kspace_cutoff=1.8467166`;density basis:(max_l=1, σ=[1.5], multipoles),feature basis:(max_l=1, σ=[1.5,3.0], receiver);
  两组相位常数 = real `[1,0,0,0]` / imag `[0,-1,-1,-1]`;`permute_indices=[1,2,0]`(JSON 实证)。
- `field_feature_norms = [20,20,0.5,0.5,0.5,0.5,0.5,0.5]`(buffer,常量)。
- `field_si=False`(descriptor 减自作用)、`include_electrostatic_self_interaction=True`(coulomb 不减)、`quadrupole_feature_corrections=False`。
- `lr_source_maps[0].linear`: `512x0e+512x1o → 8 维`,`zero_charges=False`;`fukui_source_map`: 512x0e+512x1o → 隐藏 16x0e → 2x0e。
- `charges_maps[0].readout.irreps_out = 1x0e+1x1o+1x0e+1x1o+2x0e`(8 电荷 + 2 Fukui 源,与夹具 `charges_step_*(:,:10)` 一致)。
- `layer_feature_mixer.linears` = 2(与层数一致);`external_field_contribution` 序列化为 `raw_state_dict`(未展开;生产=0 → 短路)。
- 夹具佐证:`bench/polar_fixtures/scf/*.npz` 键 `lr_src_0/1`、`fukui_src`、`charges_step_0/1`、`potential_feats_step_0..3`(2 步 × 2 通道);
  `bench/polar_fixtures/*.npz` 含 `forces(128,3)` 参考力(模块⑧门的现成基线)。

---

## 2. 逐模块伴随清单

记号:单图(引擎每次调用=整盒单图);对任意中间量 z,`z̄ ≡ ∂E/∂z`(E=体系总能量);N=原子数;f32/f64 以引擎 `Precision` 为准。
通用原则:**伴随按前向逆序逐算子作用;一切"线性算子"的伴随 = 转置算子;索引类(select/index_add/scatter)伴随 = 反向索引累加。**

### ⑧-A 基变换列映射(内部基 ⇄ e3nn 基)
**a) 前向**。引擎内部 node_feats(`state.output`)与 e3nn 基在 l=1 上相差固定矩阵 **−P**(计划 §14 实证)。
引擎 forward(`impl.tpp` L5373-5379,映射由 ctor 依 `irreps_in` 构造,L3362-3384):
```
x_e3nn[c] = s_c · x_int[ src(c) ],   src(c)=blk.offset+3m+((comp+1)%3),  s_c = −1 (l=1) / +1 (l=0)
```
可微输入:内部 node_feats(仅作输入,不直接进 E;经 polar 块进入)。
**b) 伴随**(推导):前向即线性映射 `e3nn = T·int,T = D·P`(D=diag(±1),P=循环置换 `(c+1)%3`)。
伴随 = `Tᵀ`,而 `Tᵀ = P⁻¹D = −P² ≠ T = −P`(P³=I)⇒ **T 不是自伴的,方向必须反转**(这正是计划 §14"逆映射 `(−P)⁻¹=−P²`"的出处;
调用方要求论证自伴性,结论:不自伴)。实现:
```
x̄_int[src(c)] += s_c·x̄_e3nn[c]      ⇔      x̄_int[j] = −x̄_e3nn[(j+2)%3]   (l=1,逐 m 三元组)
x̄_int[j] = x̄_e3nn[j]                                                (l=0)
```
l>1 前向已明确拒绝(`impl.tpp` L3373-3375)。
**c) 引擎挂接**:新增一个与 L5373-3379 同构的逆映射核(源映射 `(comp+2)%3`,符号不变);输出累加进
`states[ℓ].layer_adjoint`(**init=false**)。前向缓存:无(仅需 ctor 的 `polar_basis_source/sign`,可复用于双向)。
**d) 去向**:密度/节点特征伴随(→骨架→力);无位置/细胞依赖。

### ⑧-1 lr_source_maps(EnvironmentDependentSpinSourceBlock)
**a) 前向**(mace `field_blocks.py` L54-73;调用 `extensions.py` L672-703;引擎 L5359-5390):
`y_ℓ = W_ℓ·(T·x_ℓ)`(e3nn Linear;`zero_charges` 分支本 ckpt=False,配置实证);`density0 = Σ_ℓ y_ℓ`(N×8)。
可微输入:`x_ℓ`(每层 node_feats);实参 `W` 常量。
**b) 伴随**:`d̄ensity0` 由上游(⑧-2 链尾)给出;因 sum 为线性,`ȳ_ℓ = d̄ensity0`(每层同一 N×8 全量);
`x̄_e3nn = E3LinearKokkosT::reverse(ȳ_ℓ,·)`(`nl.hpp` L95-97;反向不需要前向输入);
再经 ⑧-A 得 `x̄_int → states[ℓ].layer_adjoint +=`(init=false)。
**c) 引擎挂接**:`E3LinearKokkosT::reverse` 现成;输出视图=各层 `layer_adjoint` 累加。
缓存:无。
**d) 去向**:仅密度路径(→节点特征→骨架反向→力);无直接位置/细胞项。

### ⑧-2 Fukui 源归一 + l=0 校正(初始化与每步同构)
**a) 前向**(mace L737-751 初始化,L742-766 校正;循环内 L842-864;引擎初始化版 L5393-5446):
```
s_iσ = fukui_map(x_last)_iσ          (N×2;首步输入=末层 node_feats e3nn)
S_σ  = Σ_i s_iσ   (整图求和;S=0→1 钳制)
s̃    = s/S
pred_σ = Σ_i d_iσ (d = 当前密度的 l=0 列,即列 0/4)
F_σ  = (T_σ − pred_σ)/S ,  T_α=Q_p_S/2 , T_β=Q_m_S/2
d_iσ(l=0) += s̃_iσ·F_σ            (其余 7 列不变)
```
可微输入:`x_last`(末层 node_feats),`d`(密度 l=0 两列),`Q_p_S/Q_m_S`(常量,由 total_charge/total_spin 传入,L5421-5428)。
**b) 伴随**(逐通道独立;推导要点)= 反向链:
```
s̃̄_i  += F·d̄_i ;            F̄ = Σ_i s̃_i·d̄_i
pred̄ = −F̄/S ;              S̄ = −F̄·F/S          (S≠0 主分支;与引擎 L5418-5420 钳制同分支)
d̄⁰_j += pred̄               (校正前 l=0 列;其余列穿透)
s̄_i  = s̃̄_i/S ;             S̄ += −Σ_i s̃̄_i·s_i/S²
s̄_i += S̄                   (S=Σs)
(S==0 分支:pred̄=−F̄,s̄ 经恒等映射;必须镜像引擎分支)
```
`s̄`(N×2)→ `Readout::reverse_all`(**新接口**,R1)→ `x̄_e3nn` → ⑧-A → 末层 `layer_adjoint +=`。
**c) 引擎挂接**:校正核的伴随是一个小 `parallel_for`(图内 rank-1 项,宿主归约即可,与 L5406-5417 同型);
`polar_fukui_sources`(N×2 原始源,**已存**,L5400-5402)可直接复用;**需新增保存**:`S_σ, pred_σ, F_σ`(各 2 标量/次;
目前 host 计算后丢弃,L5410-5432)以及校正前的 l=0 列(现被原地覆盖,L5435-5439)。
**d) 去向**:密度伴随(图内全对全,无位置依赖);`x_last` 路径→节点特征伴随。无位置/细胞项。

### ⑧-3 forward_dynamic(GTO 势特征;per 迭代 × per 通道)
**a) 前向**(graph_longrange `features.py` L617-658;几何缓存 L568-615;算子 L28-117;`gto_utils.py` L152-178):
```
ρ(k)   = assemble(source; cos, sin, B_d, V_k)          [L28-67]   (乘 (2π)³/V_k)
pot(k) = FIELD_CONSTANT·kfac_c·ρ                        [L70-83]   kfac_c=1/k²(k≠0),0(k=0)
feat   = project(pot; B_f, cos, sin, kfac_p) → permute  [L86-117, L408-409]
feat  -= overlap_constants ⊙ select(source)             [L652-655;field_si=False ⇒ 该支生效]
(+ non-periodic corrections:TTT 恒 0,跳过 [L640-657])
```
其中 `cosines[k,i]=cos(k_k·r_i)·mask`,`sines=sin(·)·mask`(L579-583);`kfac_p=0.5@k=0`(L592-593)。
可微输入:**source(density, N×4)**、**node_positions**(仅经 cos/sin)、**cell**(经 k 向量与 volume)、volume(单图标量)。
**b) 伴随**(逐算子;矩阵形状按单图):
```
(d1) assemble 反向:  cc̄ = (2π)³/V_k·(B_d,r·ρ̄_r + B_d,i·ρ̄_i);  cs̄ = (2π)³/V_k·(B_d,i·ρ̄_r − B_d,r·ρ̄_i)
                     sourcē = cosᵀ·cc̄ + sinᵀ·cs̄ ;  cos̄ += cc̄·sourceᵀ ;  sin̄ += cs̄·sourceᵀ
                     V̄_k   += −(2π)³/V_k²·Re(ρ̃·ρ̄)      (ρ̃=未乘常数前的密度)
(d2) kernel 反向:    ρ̄ += FIELD_CONSTANT·kfac_c·p̄ot
(d3) project 反向:   feat̄ 先 un-permute(index_add 反向)并 reshape 8×N;
                     proj_cos¯ = proj_sin¯ = 2/(2π)³·feat̄ᵀ
                     Ā[k,j] = Σ_i proj_cos¯[j,i]·cos[k,i] ;  B̄[k,j] = Σ_i proj_sin¯[j,i]·sin[k,i]
                     Ā,B̄ 再乘 kfac_p ;  cos̄[k,i] += Σ_j A[k,j]·proj_cos¯[j,i] ;  sin̄ 同式
                     (A,B = 前向 ρ 的线性组合中间量,需保留或由 ρ 重算)
                     pot̄_r = B_f,r·Ā + B_f,i·B̄ ;  pot̄_i = B_f,i·Ā − B_f,r·B̄
(d4) 自作用反向:      sourcē += selectᵀ( C ⊙ (−feat̄) )      (C=overlap_constants 常量对角)
(d5) 位置(直接力):  内积̄ = (−sin⊙cos̄ + cos⊙sin̄)⊙mask ;  r̄_i += Σ_k k_k·内积̄(k,i) ;  k̄_k += Σ_i r_i·内积̄(k,i)
(d6) k 依赖(cell):  k̄ ← 经 RadialIntegralDirect(L33-48:f0=p0·e^{−k²σ²/2},f1=p1·k·e^{…})与 Y_lm(k)、
                     kfac_c=1/k² 汇总,k̄_norm2 → k̄ += 2k·k̄_norm2;
(d7) cell:          k = n·rcell(kspace.py L86-88),rcell = 2π·(cell⁻¹)ᵀ(data/atomic_data.py L371):
                     r̄cell[i,j] = Σ_n kvecs[n,i]·k̄[n,j];  d(cell⁻¹) = −cell⁻¹(dcell)cell⁻¹ ⇒
                     cell̄ = −2π·cell⁻¹·r̄cell·cell⁻¹(实现时以 3×3 cell 的 FD 复核转置记账);
                     V̄ = Σ_k V̄_k + (density 缩放 (2π)³/V 的贡献)
```
（d1)(d2)(d3) 与 (d5)(d6) 的"势场↔相位↔位置"三件套同时适用于 ⑧-6。）
**c) 引擎挂接**:目标模块④的前向必须保留几何缓存(§4 表);反向写入:(i)密度伴随缓冲(回 SCF),
(ii)**polar 位置伴随累加器**(→节点力,§3.4),(iii)cell 累加器(延后,R9)。缓存:几何缓存(`module③`)+ 每迭代 `source`。
**d) 去向**:密度→上游 SCF;位置→**直接力**(唯一直连 xyz 的位点,另有 off 的 external 场);cell→维里(R9)。

### ⑧-4 charges_maps 更新块(AgnosticEmbeddedOneBodyVariableUpdate)
**a) 前向**(field_blocks.py L563-594;子块 L118-203 嵌入块、L294-466 SparseUvu TP、L229-280 FieldUpdateBlock;
readout = `GeneralNonLinearBiasReadoutBlock`,blocks.py L309-355;调用 mace L821-840):
```
mixed  = PE(pot_feats, nf, attrs, q_local)             # 3 条 e3nn Linear 相加(L195-203)
invar  = dot_products(nf, mixed)                       # TP mode0: l×l→0(L446-453)
nonlin = RadialMLP( cat(invar, source_embedding(attrs)) )   # [2·invar]+[64,64,64]+[invar](L513-515)
newf   = tp_out(nf, nonlin)                            # TP mode1: l×0→l(L454-464)
out    = readout(newf)                                 # GatedEquivariantBlock ∘ l1,GatedEqBlock ∘ lmid, l2(L352-355)
```
可微输入:`nf`(node_feats=features_mixed, e3nn 基)、`pot_feats`(potential_features)、`q_local`(density)。
**本 ckpt 关键事实**:`edge_attrs/edge_feats/edge_index` 形参存在但**未被使用**(L563-594 全文),`node_attrs` 是 one-hot 常量经 `source_embedding` 进 `cat` 的**第二路**(注意它进 MLP,不是被忽略)。`nonlinearity_cls="MLPNonLinearity"` 被 `_setup` 忽略(L480),实际为硬编码 RadialMLP(L513-515)⇒ 引擎镜像**具体类**,不要按配置名实现。
**b) 伴随**(算子级):
- SparseUvu TP(`field_blocks.py` L405-466):
  - mode0(l×l→0):`pair[b,u,v]=Σ_d x1[b,u,d]x2[b,v,d]/√d1; mixed_u=Σ_v pair·w[u,v]`
    ⇒ `m̄ixed_u= pw·out̄_u; p̄air[b,u,v]=m̄ixed_u·w[u,v]; x̄1[b,u,d]+=Σ_v p̄air·x2/√d1; x̄2[b,v,d]+=Σ_u p̄air·x1/√d1`(需缓存 x1,x2 或重算);
  - mode1(l×0→l):`m_u=Σ_v x2[b,v]w[u,v]; out += pw·x1·m_u/√d1`
    ⇒ `x̄1 += pw·m_u·out̄/√d1; m̄_u = pw·Σ_d x1·out̄/√d1; x̄2[b,v] += Σ_u m̄_u·w[u,v]`;
  - `out *= output_mask` ⇒ `out̄ *= mask`(常量)。
- 嵌入块:3 条 e3nn Linear(`L124-142`)⇒ 各自 `E3LinearKokkosT::reverse`;`pot_feats̄`、`q̄_local`、`n̄f`。
- RadialMLP(`radial.py` L361-384,Linear→LayerNorm→SiLU):标准链;需缓存每层 pre-activation(或重算),
  LayerNorm 反向按 weight/bias=常量、对输入求导(**当前 X 的均值/方差**参与)。
- `GeneralNonLinearBiasReadoutBlock`(blocks.py L309-355):两次 `GatedEquivariantBlock` + linear_1/lmid/l2。
  引擎已有等价机件:`Gate::reverse`(标量 silu + 向量 sigmoid 门,L510-518)与 `Readout::reverse` has_mid(L1248-1268);
  本块的激活是 **GatedEquivariantBlock(scalars=act, gates=sigmoid, gated=vectors×gates)**,与引擎 `Gate` 的
  标量/张量双路径一致 ⇒ 用 `Gate::reverse` + `E3LinearKokkosT::reverse` 组合实现,不能直接套 `Readout::reverse`(该分支是纯 silu 链)。
- 输出拆分(mace L833-840):`out[:, :-2]`=8 列密度增量(reshape(N,2,4) 加到密度),`out[:, -2:]`=Fukui 源(进 ⑧-2)。
**c) 引擎挂接**:全部用现有原语反向;需按迭代保存的激活见 §4(或"保存输入+重算内部")。
**d) 去向**:`n̄f→features_mixed 伴随→(mixer 反向)逐层 node_feats→⑧-A→layer_adjoint`;
`q̄_local→密度伴随(上一迭代/初值密度)`;`p̄ot_feats→⑧-3(通道拆分后)`;**无位置/细胞项**(本 ckpt)。

### ⑧-5 SCF 循环整体(2 步;把 ⑧-2/3/4 串成逆序)
**a) 前向**(mace L777-864;引擎待实现):对 i=0,1:
`d_i → (α,β 拆分 → forward_dynamic×2 → ÷field_feature_norms → cat) → P_i → charges_map[i] → (Δd_i|fk_i) → d+=Δ → Fukui 校正 → d_{i+1}`;
`P_i = potential_features` 变量在循环后被写入最终值(le 消费,L873),`d_{M}` 为最终密度。
**b) 伴随(逆序驱动)**:
```
给定 d̄_final(最终密度)与 P̄_final(仅末迭代的 potential_features;后文 ⑧-6/⑧-7 的种子),外部场=0 短路:
for i = M-1 … 0:
   ⑧-2 反向(第 i 步校正):入口 d̄(=∂E/∂d_{i+1});
        出口 d̄_pre(=∂E/∂(d_i+Δ_i)) 与 s̄_i(N×2,原始 Fukui 源伴随)
   out̄ := [ d̄_pre(8 列) | s̄_i(2 列) ]        # out 的 10 列各自就位
   ⑧-4 反向(out̄) → q̄_local(加回 d̄_pre ⇒ 得 d̄_i)、P̄_i、fmix̄
   P̄ 拆分 → F̄_α,F̄_β;÷field_feature_norms ⇒ 逐元素 ×1/norms
   ⑧-3 反向(α,β 各一次) → d̄_σ 加回本迭代输入密度 d_i、polar 位置/细胞累加器
   d̄ ← d̄_i(交给 i-1;末迭代的 P̄_final 已并入末迭代的 P̄)
循环后:初始化 Fukui(⑧-2 一次)→ 初值密度 d̄0;d̄0 同时含 i=0 的 d̄ 与 charges_0 通道(⑧-7)
```
**c) 引擎挂接**:**两个阶段**的投递(关键设计决定):
  (i)"节点特征伴随"阶段:`fmix̄→mixer 反向→(⑧-A)→各层 layer_adjoint += ;x_last 路径(⑧-2/⑧-7)→末层 +=`。
       必须在骨架反向主循环(L5549)**之前**完成,否则层伴随被错过;
  (ii)"几何伴随"阶段:polar 位置伴随写 `polar_atom_forces`,`cell̄` 写 `polar_cell_adjoint`(若实现维里)。
       可在主循环之后、边级→力(L7039)之前统一投递(或直接在阶段(i)里写累加器,效果相同)。
**d) 去向**:密度→节点特征→骨架→力;位置→直接力;细胞→维里。

### ⑧-6 coulomb_energy(GTO-Ewald;RBE 的替换对象)
**a) 前向**(energy.py L113-194;调用 mace L887-924):
`q = d[:,0]+d[:,1]`(mace L887;`charges_to_mul_ir` / `field_from_mul_ir` / `fukui_to_mul_ir` 在无 cuet 环境下
为 `None` ⇒ **恒等**(wrapper_ops.py L413-434),伴随亦为恒等;实现时按"存在则旁路"处理以兼容 cuet ckpt);
`ρ=assemble(q)`(coulomb 自带 density_basis,参数与 descriptor 的**完全相同**→ 引擎可共享缓存);
`pot=FIELD_CONSTANT·kfac_c·ρ`;`E = 0.5·V/(2π)⁶·Σ_k 2(ρ_r pot_r+ρ_i pot_i)`(L17-30);
自作用项:**关闭**(配置实证 `include_self_interaction=True` → L157 分支不进;若日后开启,见 ⑧-3(d4) 同款线性伴随);
pbc 修正:TTT 下 `is_molecule/is_slab` 全 False ⇒ 恒 0(L168-192),跳过。
可微输入:`q(density)`、positions(cos/sin)、cell、volume。
**b) 伴随**:`Ē=1`,`pot=f·ρ(f=FIELD_CONSTANT·kfac_c)` ⇒ `E=V/(2π)⁶·Σ_k f·(ρ_r²+ρ_i²)` ⇒
`ρ̄ = 2·V/(2π)⁶·pot`(经 ρ 与 pot 两条链求和后恰好为 2×;实现时按"对 (density,potential) 两输入各取一条伴随再相加"逐步核对);
`V̄ = E/V` 再加 ⑧-3(d1) 的 `V̄_k` 项;
`q̄` 经 `sum dim=1` 反向:**两自旋通道各得全量** `q̄`(线性 sum);
其余经 ⑧-3(d1)(d2)(d3) 同款算子;(d5)(d6)(d7)=位置/细胞同 ⑧-3。
**c) 引擎挂接**:目标模块⑥前向需暴露与 ⑧-3 共用的几何缓存与 `ρ(k)` 重算入口;反向写同一组累加器。
**d) 去向**:`q̄→`两通道密度伴随(→SCF 链);位置→直接力;cell→维里。

### ⑧-7 local_electron_energy(OneBodyMLPFieldReadout)
**a) 前向**(field_blocks.py L718-752;调用 mace L867-885):
`q = q_ind + q_0`(两者均 N×8,相加仍 N×8)→ `q_up=Lin_q(q)`;`v_up=Lin_v(P_final)`;
`invar = cat(dot_q(nf_last,q_up), dot_v(nf_last,v_up))`(两个 mode0 TP);`e_node = MLP(invar)`(RadialMLP [2·invar]+[128,128,128]+[1]);
`le = Σ_nodes`。
可微输入:`q_ind`(=最终密度 `spin_charge_density.view(N,-1)`)、`q_0`(=初值校正密度 `field_independent`)、
`P_final`(末迭代 potential_features)、`nf_last`(末层 node_feats)。
**b) 伴随**:种子 `ē_node=1`(每节点)→ MLP 反向(LayerNorm/SiLU 需缓存)→ `īnvar` → 拆两路:
`dot_q`/`dot_v` 的 mode0 伴随(同 ⑧-4 公式)→ `n̄f_last`(两路相加,经 ⑧-A → 末层 layer_adjoint)、`q̄_up`、`v̄_up`;
`Lin_q/Lin_v` 反向 → `q̄`(→ `q̄_ind`、`q̄_0` 两分支)、`P̄_final`(→ 进 ⑧-5 最后一次迭代的 P 分支)。
**c) 引擎挂接**:全部现有原语;缓存:q_up/v_up/nf_last 与 MLP 激活(或重算)。**注意 `q̄_ind` 与 ⑧-6 的 `q̄` 同落
最终密度 → 二者相加,不可覆盖。**
**d) 去向**:`q̄_0→`初值密度(→初始化 Fukui→lr_source_maps→各层节点特征);`P̄_final→`SCF;`n̄f_last→`节点特征;无直接位置/细胞项。

### ⑧-8 external field / dipole / 非周期修正(生产关闭,接口保留)
- `external·total_dipole` 项(mace L925-929;dipole 构造 `utils.py` L507-529):若启用,伴随为
  `q̄_i += E·r_i`,`r̄_i += E·q_i`(位置直连力),`dipole` 的 l=1 排列 `[2,0,1]` 反向索引,`total_charge` 不参与能量。
- `DisplacedGTOExternalFieldBlock`(gto_utils.py L305-363)+ barycenter gauge(mace L800-816):barycenter 求导把
  `−1/N_g·Σ` 摊回全图原子;矩阵为常量。
- 本 ckpt:`external_field_contribution` 未序列化(JSON 仅 `raw_state_dict`),生产 external=0 ⇒ **建议引擎在前向显式短路置零,伴随恒零**;
  接口保留位在 `compute_node_energies_forces_field`(nl.hpp L178-184 同族)。
**d) 去向**:仅当启用时新增位置/电荷项;默认零。

---

## 3. 链序与挂接点

### 3.1 前向顺序(参照)
mace:`lr_src 累积(L666-703)→ k 网格+几何 precompute(L710-730)→ mixer(L733)→ Fukui 初始化(L737-766)→ SCF ×2(L777-864)→ le(L867-885)→ charge/dipole(L887-912)→ coulomb(L913-924)→ 装配(L925-929)→ get_outputs(L931-943)`。
引擎现状:模块①(L5359-5390)、②(L5393-5446)已实现且位置与 mace 一致;③-⑦为待实现前向(JSON 键已就位:`electric_potential_descriptor`/`coulomb_energy`/`field_dependent_charges_maps`/`layer_feature_mixer`/`local_electron_energy`/`configs`,见 probe JSON)。

### 3.2 反向链序(逆序)与接缝视图
```
[seed 1] E_total 的极化分项(无 scale;mace L879-883/L925-929 与引擎 L5471-5472 对比)
   ├─ ⑧-7 le 反向        → q̄_ind(最终密度) + q̄_0(初值密度) + P̄_final + n̄f_last
   ├─ ⑧-6 coulomb 反向   → q̄(最终密度,加在 q̄_ind 上;两通道各得全量) + 位置累加 + cell 累加
   └─ ⑧-5 SCF 逆序(i=1→0):每步 ⑧-2 → ⑧-4(out̄ 组装)→ ⑧-3(×2 通道)→ 接力 d̄
   └─ ⑧-2 初始化反向 + ⑧-1 lr_src 反向(逐层) + mixer 反向(逐层)
[投递 A] 所有 e3nn→内部 的 ⑧-A 逆映射,累加进 states[ℓ].layer_adjoint(init=false)
[投递 B] polar_atom_forces(节点级)与 polar_cell_adjoint(可选)
[骨架] L5549 主循环(消费已被加权的 layer_adjoint)→ L7039 边级→力 → reduce/pair 归约
```
逆序对照表:

| 前向步 | 反向步 | 伴随落点 |
|---|---|---|
| ⑧-1 逐层 lr_src | 最后 | layer_adjoint(各层,累加) |
| ⑧-2 初始化 Fukui | 倒二 | 末层 layer_adjoint + 初值密度 |
| ⑧-3 forward_dynamic(i,σ) | SCF 内逆序 | 密度 + 位置 + cell |
| ⑧-4 charges_map[i] | 同上 | fmix/末层特征 + 密度 + P |
| ⑧-2 每步 Fukui | 同上 | 密度 + 末层特征 |
| ⑧-6 coulomb | 先(S 层) | 密度(两通道)+ 位置 + cell |
| ⑧-7 le | 最先 | q_ind/q_0/P_final/nf_last |

### 3.3 进入 `state.layer_adjoint`:覆盖 vs 累加(逐行核对结论)
- readout 播种(`impl.tpp` L5457-5459):`ensure_view(...); readouts[layer].reverse(state.output, scale, state.layer_adjoint)` →
  `Readout::reverse`(L1245-1246)`linear.reverse(seed, input_adjoint)` **未传第三参** ⇒ 默认 `initialize_input_adjoint=true`
  (`nl.hpp` L95-97;`e3nn_kokkos.cpp` L1209-1210 先 `deep_copy(0)` 再写)⇒ **覆盖**。
- 骨架跨层(`impl.tpp` L7032-7036):`linear_up.reverse(state.up_adj, states[layer-1].layer_adjoint, false)` ⇒ **累加**。
- 结论:极化链的节点特征伴随**必须累加**(`init=false`),且必须在 L5460 播种**之后**注入,
  否则被播种清零;注入点选在 L5466(生成路径清零块)之后、L5549 主循环之前。
- 层完整性:ctor L3437 强制 `interactions==products==readouts`,播种覆盖每层,无未初始化层。

### 3.4 力的汇合(极化位置伴随是节点级,必须新增累加器)
- 现有唯一出力的地方是 L7039-7092(**边级**):`node_forces(3·e+c) = −(d̄_dist·r̂ + Σ_lm h̄_lm·∂Y/∂xyz)`,
  之后 `reduce_node_forces`(L7128-7160)按 `edge_receivers/edge_sources` 把有向边力摊到原子;pair style 则直接读
  `mace->node_forces` 再聚合(`pair_symmetrix_mace_kokkos.cpp` L1185-1220)。
- 极化的位置伴随来自 `cos(k·r_i)`(及 off 的 external 场),是**逐原子**的 `r̄_i`,**无法**写成逐有向边力的和
  (相位依赖绝对位置,不依赖边矢量)。
- 设计:`nl.hpp` 新增 `polar_atom_forces(MaceNonlinearKokkosT 成员, N×3, 每次调用清零)`;
  ⑧-5 阶段(ii)写入 `F_i = −r̄_i`(与现有力约定一致:骨架 `node_forces = −∂E/∂xyz(edge)`(`impl.tpp` L7090),
  `reduce_node_forces`/pair style 归约后即 `F = −∂E/∂r`);
  `reduce_node_forces` 末尾把 `polar_atom_forces` 加到 `atom_forces`;pair style 的 force_reduction 同样补加。
  （两处消费点都要改;计划 §12 的"力"门=fixtures `forces` + FD,可直接验证该符号约定。）
- 本 ckpt 极化链**不产生** `harmonic_adjoints/radial_adjoints/cutoff_adjoints` 贡献(⑧-4 不用 edge 输入);
  若未来 ckpt 的 field_update 类消费 edge_attrs/feats,需补"内部 SH→e3nn SH"的逐 l 伴随(R4)。

### 3.5 种子与 scale 的边界
- 骨架 readout 种子带 `scale`(L5459;`energy_scale` 亦在 L5471 能量行);polar 各项**不带 scale**(mace L866-883/L925-929)。
  ⑧ 的种子一律 `∂E_total/∂·`(=1),不要复用 `scale`。

---

## 4. 缓存清单(前向为反向必须留下什么)

| 模块 | 必须留存 | 体量/原子 | 备注 |
|---|---|---|---|
| ① lr_src | 无(W 常量;线性反向不需输入) | 0 | — |
| ② Fukui(初始化+每步) | 原始源 `s`(已存 `polar_fukui_sources`)、`S_σ/pred_σ/F_σ` 标量、校正前 l=0 | ~0 | `pred` 现被覆盖,需新增保存(L5410-5439) |
| ③ 几何(module③) | k 向量/k_norm2/k0_mask/cosines/sines/两套 basis_fs/kfac_c/kfac_p/volume_per_k/output_permutation | cos/sin 各 n_k·N | **全量最大项**;SCF 两通道两迭代共用,必须存活到反向 |
| ④ forward_dynamic | 每迭代×通道的 `source`(N×4) 或 `ρ(k)` | N·4·(M·2) 或 n_k·2·(M·2) | 建议**存 source、反向重算 ρ(k)**(2 个 matmul/k) |
| ⑤ charges_maps | 每迭代:`d_in`(N×8)、`P_i`(N×16)、`out`(N×10);另有**一次**的 `fmix`(N×1024,e3nn;mace L733) | 每迭代 ≈ 136 B/原子;fmix ≈ 4 kB/原子(一次) | 块内激活(mixed N×1024、nonlin N×512、readout 内部)建议**重算**(需存 d_in/P/fmix 与权重)或以预算旋钮权衡 |
| ⑥ coulomb | 复用 ③ 的 density_basis 缓存(参数全同)| 0 | `q` 与最终密度同源 |
| ⑦ le | `q_up/v_up`(N×1024×2)、`nf_last`(N×1024)、MLP 激活 | ~ (2048+1024)·4B ≈ 12 kB/原子 | 可重算(输入全在) |
| ⑧-A | 无(用 ctor 的 `polar_basis_source/sign`) | 0 | 逆映射核共用映射表 |

**内存算式(单帧)**:`C ≈ 2·n_k·N·sizeof(P) (cos/sin) + N·(8+16+10)·M·sizeof(P) (charges 输入) + N·1024·sizeof(P) (fmix,一次) + N·(2048+1024)·sizeof(P) (le,若保存) + O(n_k)`;
其中 descriptor 的 ρ(k)、charges/le 内部激活按"重算"计。
例:`N=10^5,n_k≈2×10³,sizeof=4` ⇒ cos/sin ≈ **1.6 GB**;其余:charges 输入 ≈ 27 MB(M=2)、fmix ≈ 0.4 GB、
le(若保存)≈ 1.2 GB ⇒ 合计约 **2.0-3.2 GB**(视 le 保存/重算)。
⇒ 建议:(i)cos/sin 必须留(重算=再跑一遍三角+掩码,约等于一次前向的 10-20%);(ii)charges/le 内部激活按
`execution_mh1_scratch_budget`(`impl.tpp` L3951-3989)同型预算做"保留 vs 重算"旋钮并记录计数。

---

## 5. 风险与未决

- **R1(接口,必做)**:fukui 头输出 2 列,而 `Readout::reverse` seed 固定 N×1(`impl.tpp` L1246)且 `linear.reverse` 校验列宽
  (`e3nn_kokkos.cpp` L1203-1206)⇒ 必须新增 `Readout::reverse_all(seed(N×k), input_adjoint, init)`;同时 ⑧-1/⑧ 的
  `linear.reverse` 调用要与 `initialize_input_adjoint=false`(累加)配对。**待核实**:`fukui_source_map` 的 `linear`
  (非 nonlinear 分支)与 `has_mid` 分支在 `reverse_all` 下的激活缓存复用是否与 `evaluate_all` 一致(同一实例、同一前向序)。
- **R2(自伴性,已论证)**:⑧-A 不是自伴映射;实现时必须用反向源映射 `(comp+2)%3`(前向 `(comp+1)%3`)。
  逆映射核与前向核同型(同一套符号、索引方向相反),但**不能**直接复用前向核。**待核实**:−P 关系仅在 l=1 实证;
  若未来 ckpt 的 node_feats 出现 l>1,引擎当前直接拒绝(`impl.tpp` L3373-3375),需要新的逐 l 映射(注意 polar 内部
  descriptor/charges 的数据流本身是 e3nn 基,**不经过** ⑧-A)。
- **R3(数值,高中风险)**:Fukui 归一的 1/S 与其伴随 1/S²;S 是整图源和(量级 O(1)·N),钳制只处理 S==0 精确零,
  **小 S 时伴随放大**。mace 用 `.double()` 归约(L742-747、L842-847),引擎模块②目前在 `Precision` 下做宿主归约
  (L5406-5417)⇒ 建议 ⑧ 前先把 S/pred 归约升级为 double(顺带也是前向对账的精度修正)。**待核实**:夹具中是否出现过
  小 S 构型(可用 `fukui_src` 夹具直接查 S 分布)。
- **R4(约定,待核实)**:内部 SH(sphericart,`compute_Y` L3588-3730)与 e3nn SH 的逐 l 换算。当前仅对 node_feats
  l=1 实证 `−P`(§14);edge SH/P 的换算以及 l=2 符号无实证。本 ckpt 不消费 edge SH,但 k 空间 `Y_lm(k)`
  必须按 e3nn 约定计算(`permute_indices=[1,2,0]`,JSON 实证)——其伴随只影响 cell(R9)。验证配方:取几组 k 向量,
  比较引擎 Y 与 `e3nn.o3.SphericalHarmonics(...,normalization='integral')` 的逐 l 符号/排列。
- **R5(缓存/内存)**:cos/sin 是最大项(§4 算式);SCF M=2、两通道使同一缓存被 4 次取用 ⇒ 不可提前释放。
  **与 ⑤/⑦ 的"重算 vs 保存"**需要预算旋钮;重算必须严格复刻前向算子次序(含 f32 舍入),否则 FD 门上会看到漂移。
- **R6(路径限制)**:polar 前向目前显式拒绝 direct/prepared 路径(L4205-4207);模块⑧ 的挂接点也只在标准物化路径存在。
  JIT/直调路径的极化伴随**未设计**——若 P5 想让 polar 走 RTC,需要单独的方案(或明确不支持)。**待核实**:P5 计划口径。
- **R7(多步 SCF 的反向语义)**:参考实现是"展开式 autograd"(L777-864 无隐式函数定理/不动点梯度)。引擎必须同构展开;
  任何收敛加速/提前退出都会改变梯度 ⇒ 禁止在反向重算时"再跑一个更收敛的 SCF"。
- **R8(P4/RBE 关联)**:RBE 用随机 k 样本替换网格和。对伴随的三点影响:
  (i)生产实现建议**反向冻结样本**(把 k 样本当常量),位置/力伴随结构不变(样本生成不依赖 positions);
  (ii)力方差 ∝ 1/√p,需要 p 扫描给出力的 RMS(计划 P4 门);
  (iii)若采样器带重要性权重或 proposal 协方差由 cell 推得(`cell⁻¹` 进 Box-Muller 协方差),则 cell 伴随要补
      权重/协方差项,且 Metropolis 接受域在 cell 上分段常量 ⇒ 导数不连续。**待核实**:`lammps-rbe` 的 `kspace_style rbe`
      是否为无权重采样(决定 (iii) 是否必须)。FD 自洽检查必须**固定随机种子并对 E±h 复用同一组 k**(否则 FD 被采样噪声淹没)。
- **R9(cell/virial 缺口,范围外但须挂账)**:引擎现应力只由 `node_forces⊗xyz` 得到(L7177-7205),没有 cell 伴随。
  极化引入真实 cell 依赖(k∝rcell、体积因子、k 网格选择),NPT 下若不补会导致**解析维里静默缺极化项**(能量对但压强错)。
  模块⑧(力)可不做,但设计上必须保留 `polar_cell_adjoint` 累加器接口并在 P5 前补齐(R9-公式见 §2 ⑧-3(d6)(d7))。
  **待核实**:pair style 的 virial 开关与 `reduce_stress` 是否在极化开启时被强制禁用(避免静默错误)。
- **R10(符号/常数)**:`field_feature_norms` 的除法是**逐元素常量**(JSON 实证),不是归一化统计量 ⇒ 伴随为乘 const⁻¹;
  不要误当 LayerNorm 处理。同理 `output_mask`/`select_indices`/`overlap_constants` 全为常量(JSON 已序列化)。
- **R11(配置开关矩阵)**:`zero_charges`(lr_src)、`field_si`、`include_electrostatic_self_interaction`、
  `quadrupole_feature_corrections`、`include_pbc_corrections` 都会增删伴随项。本 ckpt 取值已实证(§1.4);
  提取器/引擎应以 JSON 逐项门控,并**拒绝**未能实现伴随的开关组合(而不是静默近似)。
- **R12(精度)**:生产 `Precision=float`(`impl.tpp` L7327-7328 同时实例化 float/double);参考夹具与 FD 门是 f64。
  引擎侧建议先以 double 实例化通过力门(相对误差目标 ~1e-5-1e-6,FD 步长 h≈1e-4 Å),再评估 f32 下的降级口径。

---

## 6. 锚点索引(实现时跳转)

**引擎**
- 挂接点:`impl.tpp` L5448-5460(readout 播种/覆盖)、L5461-5467(生成路径清零)、L5475-5497(几何伴随视图清零)、L5549(反向主循环起)、L7032-7036(跨层累加 init=false)、L7039-7092(边级→力)、L7128-7160(reduce_node_forces)。
- 现有极化前向:模块① L5359-5390;② L5393-5446;ctor 基映射 L3362-3384;polar 成员 `nl.hpp` L343-352、L712-729。
- 原语反向:`E3LinearKokkosT::reverse` `nl.hpp` L95-97 / `e3nn_kokkos.cpp` L1198-1233;`E3TensorProductKokkosT::reverse` `nl.hpp` L199;`Gate::reverse` L510-518;`Readout::{evaluate,evaluate_all,reverse}` L1184-1207/L1208-1244/L1245-1269。
- 签名/限制:`compute_node_energies_forces` L3733(无 cell);prepared 路径 cell 入参 `nl.hpp` L177-184;polar 拒绝 direct/prepared L4205-4207;readouts==interactions 强制 L3437;应力 L7177-7205;pair style 消费 `pair_symmetrix/pair_symmetrix_mace_kokkos.cpp` L1185-1220。

**mace**
- 前向链:`extensions.py` L666-708(lr_src)、L710-730(k 网格/precompute)、L733(mixer)、L737-766(初始化 Fukui)、L777-864(SCF)、L867-885(le)、L887-912(charge/dipole)、L913-929(coulomb+装配)、L931-943(get_outputs)。
- 子块:`field_blocks.py` L45-50(mixer)、L54-73(lr_src 块)、L118-203(嵌入块)、L294-466(TP)、L470-562/L563-594(charges_maps)、L652-752(le);`blocks.py` L120-156(NonLinearBiasReadout)、L309-355(GeneralNonLinearBiasReadout);`radial.py` L361-384(RadialMLP);`utils.py` L22-38(compute_forces)、L167-222(get_outputs)、L507-529(dipole);`data/atomic_data.py` L371(rcell)。

**graph_longrange v0.4.0**
- `features.py` L28-67(assemble)、L70-83(kernel)、L86-117(project)、L120-134(reconstruct)、L137-179(通道版)、L463-489/L491-502/L568-658(几何缓存与 PBC 动态前向);`gto_utils.py` L33-48(radial)、L152-178(basis)、L221-302(自作用)、L305-363(外场块);`energy.py` L17-30(能量乘积)、L113-194(PBC 能量);`kspace.py` L11-127(k 网格);`utils.py` L12(FIELD_CONSTANT)、L18-20(permute);`slabs.py` L59-69/L185-...(修正,仅非 PBC)。
- 夹具:`bench/polar_fixtures/*.npz`(含 `forces`)、`bench/polar_fixtures/scf/*.npz`(SCF 钩子)、`bench/polar_fd_f64.py`(f64 FD 协议)。

---
*本文档所有结论均带源码锚点;标注"待核实"的 7 处(§5 R1/R2/R3/R4/R6/R8/R9,共 8 个"待核实"标记)建议在模块③-⑦前向实现时同步闭合,避免反向设计回改前向。*
