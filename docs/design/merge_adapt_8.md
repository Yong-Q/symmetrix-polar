# ⑧ 引擎补丁 v3 适配 —— polar 独立 TU + ⑥⑦已落码新结构(2026-10-15)

**定位**: 把 `docs/module8_patch_final.md`(⑧ 力/全链反向; 2269 行提案, 锚点基于
"polar 未拆 TU" 旧布局)适配到**当前新结构**——polar 运行期代码已拆入独立编译单元,
且 ⑥ coulomb / ⑦ local_electron 前向**已落码**(v19c; 前向全门 PASS, 端到端能量 rel≤1.1e-11):

- `libsymmetrix/source/mace_polar_impl.tpp`(polar TU, 1429 行): ②③④⑤ 全前向 +
  `parse_polar_extension_tables()` + ⑥/⑦ 前向运行段(⑥-C1/C2、⑦-4.3/4.4 v2 合成段);
  主 tpp 仅 ①/mixer 循环块、ctor 解析块、setter 接线、**单调用点** `run_polar_pipeline(...)`;
- 反向主链在主 tpp(`mace_nonlinear_kokkos_impl.tpp`): readout 播种(L5483-5485, 覆盖写)→
  "scale nonlinear energies"(L5497-5499)→ 前向 polar 调用(L5501-5504)→ 骨架反向主循环;
  `reduce_node_forces`/`reduce_stress` 在 PART3; binding 链尾已是 ⑥ 的
  `polar_coulomb_rho_im` 块; pair style 自提案快照(md5 `482ddc73…`)**未变**。

**交付**:
1. 本文(适配裁定 + 偏离全表 + 结构核对 + 全绿证据, 见 §5);
2. `bench/apply_polar_patches_v3.py`(应用器: 块清单 = 6 文件 / 16 个插入点;
   dry-run 已全绿(exit 0), 日志 `logs/apply_patches_v3_dryrun.log`(235 行);
   模拟产物 `logs/apply_v3_sim/` 经 `bench/polar_lint.py` 三文件全 CLEAN)。

**纪律(已遵守)**: 本任务只写 `mlip/docs/` 与 `mlip/bench/`(+ 日志/模拟产物于 `mlip/logs/`);
引擎树(`/workspace/group/shared/bza/external-repos/symmetrix-xl`)**零写入**——dry-run 仅内存
模拟(dry-run 前后五文件 md5 不变, 无 `.polar_tmp` 残留); 未构建; 未调用 git; 未跑 GPU。
`--apply` 已实现但**未执行**。

---

## 0. 新结构基线与只读快照

### 0.1 只读快照(v3 适配完成时刻)

| 文件 | 行数 | md5 | 备注 |
|---|---|---|---|
| `libsymmetrix/source/mace_polar_impl.tpp` | 1429 | `1a8f4913dd692f8d37e3256f18ba13ff` | ⑥⑦ 前向(含 v2 合成段)已落码 |
| `libsymmetrix/source/mace_nonlinear_kokkos.hpp` | 861 | `087d5f010825d195f2d22111f784785b` | 含 v2 ⑥⑦ hpp 增补 |
| `libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp` | 7359 | `4bd42dcc190ff92235d997aa574be744` | v2 单行(parse 调用)已落 |
| `symmetrix/source/cpp/mace_nonlinear_kokkos.cpp`(binding) | 636 | `ed512afe0152727cd80649f9d59df677` | v2 链已含 ⑦9+⑥3 条 |
| `pair_symmetrix/pair_symmetrix_mace_kokkos.cpp` | 2289 | `482ddc73466e2cb182e4388c1650082b` | **与提案 §0 快照逐字节一致**(09:54 后未变) |

应用器 dry-run 结束后复验五文件 md5 与上表**逐字节相同**(引擎树零写入)。

### 0.2 树活跃状态关键事实(逐项实测; 直接决定 D-02/D-03/D-04/D-05/D-10)

1. **⑧ 未落码**: 16 项幂等标记应用前全为 0(hpp/tpp/tu/cpp/pair; 见应用器 `[3]` 段)。
2. **⑤ 快照口径已变(相对提案假设)**: `polar_scf_step_densities` 的
   `"polar scf density snapshot"` 内核(polar TU L978-984)位于
   `"polar scf fukui correction"`(L970-976)**之后**, 注释明示
   "快照 = 逐步 Fukui 校正后的完整状态(参考口径 scf_steps step{i+1})" ⇒ **校后口径**。
   提案 §0-裁定2/§4.0-C 假设的"现⑤只快照校前密度"已过时。
3. **密度源命名(现状)**: ⑥ 前向实读 `polar_scf_final_density`(polar TU L1024);
   ⑦ 前向实读 `polar_scf_final_density`(L1136, v2/D05); `polar_spin_charge_density`
   自 12:13 起为只读的 ① 原始累积; `polar_working_density` 为 `run_polar_pipeline`
   内局部工作密度(L209-212)。
4. **`attrs` 为类成员**(hpp L725 `Kokkos::View<Precision**,LayoutRight> attrs,…`)⇒
   提案 §4.2 的 `compute_polar_reverse_forces(int num_nodes)` 签名与 replay 中
   `update.source_embedding.evaluate(attrs,…)` **逐字保留, 无需扩参**。
5. **pair 文件未漂移**: §5.2 的 `force_reduction` 块(25 行锚点)与
   `compute_no_domain_decomposition` 消费点均按提案文本命中(count==1)。
6. **前置注意**: 引擎树现存 v2 时代的 `*.bak`(hpp 10-15 11:27 / tpp 11:29 /
   tu 12:13 / cpp 09:11)。`--apply` 的备份纪律为"`.bak` 已存在即拒绝覆盖"
   ⇒ 真实应用前需人工移走旧 `.bak`(见 §4-4)。

### 0.3 关键结构核对(任务点名; 已在应用器 `[2]` 段机器化, 13/13 全 OK)

1. **readout 播种 = 覆盖写, 极性伴随一律 `init=false` 累加注入**:
   播种段 `readouts[layer].reverse(state.output,scale,state.layer_adjoint)`(tpp L5485)
   第三参缺省 ⇒ `Readout::reverse` 内层 `linear.reverse(seed,input_adjoint)` 缺省
   `initialize_input_adjoint=true`, 即覆盖; 故 ⑧ 的注入必须**在播种之后**且用累加:
   ⑧-A 散射核(`basis_adjoint_add`)与 §7.2 小工具(`polar_inject_node_features_adjoint`)
   全部 `atomic_add(+)=`; `Readout::reverse_all` 的最终输入伴按参数注入
   `polar_rev_nfeat_adjoint`(覆盖, `true`), 随后 `basis_adjoint_add` 散射(累加)进
   `states.back().layer_adjoint` —— 与提案 §1.2-1/§3 语义一致。
2. **基变换伴随 = 散射式逆映射(Tᵀ=−P²)**: 前向 `e3nn[c]=sign_c·int[src(c)]`
   (主 tpp L5441-5447); 伴随实现 `internal[src(c)] += sign_c·e3nn̄[c]`
   (count==1 的 `polar basis adjoint scatter` 核), 单射+atomic 自卫, 非"正向再用一遍"。
3. **⑤ 快照口径核对 → 重放+逆序**(见 §0.2-2/3): 反向的逐步输入 d_in(i) 直接消费
   **既有** `polar_scf_step_densities`(校后口径, 与提案 `_post` 同值同源);
   `polar_scf_final_density` 为末态; 逐步重放 `replay_step_forward` 只读既有成员视图
   (几何/k 表/工作密度链), 不改 ⑤ 前向任何表达式(D-03/D-04)。
4. **`Readout::reverse_all` 落位裁定 = 主 tpp PART1**(提案原落点): 其依赖 =
   Readout 私有缓冲(hidden/activated_adj/…) + 文件级匿名空间 `ensure_view` 助手 +
   PART1 TU 的类模板显式实例化(文件尾 `template class MaceNonlinearKokkosT<…>`);
   若迁 polar TU, 需依赖"外层类显式实例化连带实例化嵌套成员类成员函数"这一
   非保证语义(链接风险), 且 `evaluate_all → reverse_all` 成对使用的缓存前提属
   Readout 基础设施(非 polar 专属) ⇒ 选择与提案一致的主 tpp PART1(块 2a)。
5. **节点级力接线两处(不可避免在主 tpp/其他文件, 列独立插入点)**:
   ②c tpp `reduce_node_forces` 边级归约之后(`reduced_forces` 局部作用域内, 逐字);
   ⑤ pair `force_reduction` 平行写之后(逐字; 若放其前会与 per-node 写竞争同一 f 元素,
   atomic 虽保正确、按提案置于其后)。
6. **cell 伴随 = 下一 slice(保留 loud guard)**: ②d `reduce_stress` 函数体首
   `if(polar_enabled) throw …`(逐字); `polar_cell_adjoint` 占位清零在 §4.2 第 0 段。
   提案 §6.3 "建议"的 pair style `vflag_global` guard **未纳入**(列 §4-1)。

---

## 1. 适配规则落实对照

| 规则 | 落实 | 对应块 |
|---|---|---|
| 1. ⑧ 运行期代码(伴随 kernel/缓存/反向块)全落 polar TU; 主 tpp 仅单一调用点少量行 | §4.2 主函数 + ⑧-A 小工具 + §4.0-B/C 缓存全落 TU(显式实例化之前); 主 tpp 仅 1 个调用点块(4 行+注释); **例外(按提案强制保留)**: `Readout::reverse_all`(Readout 基础设施, 依赖裁定见 §0.3-4)、`reduce_node_forces`、`reduce_stress` guard、pair style 接线 | 2a–2d、3a–3g |
| 2. 照抄提案已确认的关键裁定 | ① 播种覆盖→累加注入; ② 基变换散射(Tᵀ=−P²); ③ 重放+逆序(快照口径按现状重定位); ④ 节点力累加器+两处归约接线逐字; ⑤ cell 伴随下一 slice+loud guard; ⑥ 种子一并 ⑥ coulomb 内联重算保留。唯一系数级修正是 D-08(FD 证据) | §2.2 |
| 3. 与 ⑥⑦ 现状对齐 | 反向入口/缓存落点按当前代码重定位(前向调用点后紧邻); `pb→polar_json` 无需再改(v2 已完成); **密度源命名按现状**: ⑥ 种子 `polar_spin_charge_density`→`polar_scf_final_density`(D-05); d_in 步链 `_post`→既有 `step_densities`(D-04); ⑥ 开关置位落 parse 函数(D-10) | 3f、3g |
| 4. 应用器同 v2 纪律 | 锚点 `count==1` 断言(+前后哨兵行)、dry-run 默认、`--apply` 未执行(备份/写前重assert/写后复验/回滚)、模拟产物+`polar_lint` 强制 0、`[8]` 数值自检 | 脚本 |
| 5. 自检 | dry-run exit 0 全绿(13 前提 + 16 块 + 35 探针 + 4 顺序探针); 三文件 lint CLEAN(6 类全 0); 数值自检通过; 证据见 §5.1 | — |

### 2. 偏离清单(块号、原锚点、新落点、原因)

以下每条 = 相对 `module8_patch_final.md` 原文的偏差; 代码正文除表列内容外逐字保留
(提取位置见 §5.2 表)。

| # | 块号 | 原锚点(提案) | 新落点(v3) | 原因 |
|---|---|---|---|---|
| D-01 | 3f、3b–3e | §4.2 主函数+⑧-A 插主 tpp PART3 尾(`#else/#error` 之前); §4.0-C-1/C-2 插旧主 tpp 内联 ⑤ 段 | 3f: polar TU 显式实例化之前(全文 1198 行); 3b–3e: polar TU `run_polar_pipeline` ⑤ 段内(锚点文本 count==1) | 规则 1(polar 已独立 TU; ⑥⑦ 同线先例); 提案插入点语义("函数尾/⑤ 段内")在新树逐位对应 |
| D-02 | 1b | §2.2 含新视图 `polar_scf_step_densities_post`(S,N,8) | 删该成员(2 行)+改为 2 行 [v3] 说明 | 提案增设 `_post` 的动机(现⑤只快照校前)已不成立: 既有 `polar_scf_step_densities` 即校后口径(§0.2-2); 不新增同值重复视图 |
| D-03 | 3c | §4.0-C-1 插入含"校后快照"(realloc+alias+快照 kernel, 10 行) | 删该 10 行, 仅保留 Fukui 标量缓存(12 行含 [v3] 注释) | 同 D-02; 标量缓存与 `raw_source_sums` 捕获逐字保留 |
| D-04 | 3f | §4.2 d_in 步链 `Kokkos::subview(polar_scf_step_densities_post, step-1,…)` | `Kokkos::subview(polar_scf_step_densities, step-1,…)`(+3 行 [v3] 注释) | 同值同源(校后口径); 直接消费既有视图, 与 §8-20 的门配对(view 语义)完全不触碰 |
| D-05 | 3f | §4.2 ⑥ 种子 `auto final_density=polar_spin_charge_density;` | `auto final_density=polar_scf_final_density;`(+2 行 [v3] 注释) | 与 ⑥ 前向(Polar TU L1024 实读)同源; `polar_spin_charge_density` 已退回只读①累积(v2/D05 同类) |
| D-06 | 3f | §4.2 lambda 形参 `Kokkos::View<Precision**>` / `<const Precision**>`(无 LayoutRight) | 全部 13+18=31 处补 `,Kokkos::LayoutRight`(计数断言) | `bench/polar_lint.py` R3(v17c 教训)对非 const `View<Precision**>` 参数拦截; 树内既有风格一律显式 LayoutRight; 语义不变 |
| D-07 | 4 | §7.1 锚点=链尾 `polar_scf_final_density` 块(提案时刻) | 锚点=**现链尾** `polar_coulomb_rho_im` 块(v2 D11 先例); 一次尾展开为 +3 条; 删 `polar_scf_step_densities_post` 访问器 | 新树链尾已变; `_post` 视图删除(D-02) ⇒ 其访问器一并删(既有 `polar_scf_step_densities` 访问器仍在, 数据可达) |
| D-08 | 3f | §4.2 `fukui_reverse` 三处系数: `tr(2)=1/s_used; tr(3)=−F̄/s_used; tr(4)=−[未钳制]2·f·F̄/s_raw` | `tr(2)=1; tr(3)=−F̄; tr(4)=−[未钳制]f·F̄`(+5 行核内 [v3] 注释) | **语义级修正(唯一)**: 提案按设计 §2 ⑧-2 的 `s̃·F(F=(T−pred)/S)` 参数化推导, 而引擎/mace 前向为 `d_out=d_pre+s_n·f, f=(T−pred)/S_used`(单一 1/S; mace extensions.py L742-766 实测), 提案式多含 1/S 且第 2 项系数应为 `−f·F̄`。FD 证据见 §2.1(修正前 err≈6.1e2, 修正后 3.0e-9); 不修则 post FD 门必失败 |
| D-09 | 1a | 提案 §2.2 注"若编译报缺头则在 include 区补 `#include <array>`" | hpp include 区补 `#include <array>`(+1 行注) | hpp 现未含 `<array>` 且 `std::array<Precision,6>` 成员为 ⑧ 新增; 提案注预先授权 |
| D-10 | 3g | §0-3/§10-2:"⑥ 落码时在其 **ctor 段**置 `polar_coulomb_enabled=true`" | 置位落 `parse_polar_extension_tables()` ⑥ 表解析块尾(+2 行注) | ⑥ 已落码 ⇒ 按提案结论置位; 新树 ctor 解析已迁入该函数(v2 D06/D07), 此即"ctor 段"的当代替身; hpp 默认 false 保留 |
| D-11 | 全部 | — | 新增 `[v3 适配]/[v3 修正]` 标注与原因注释共 **10 行**(tu 7、hpp 3; tpp/cpp/pair 为逐字块无标注) | 使适配痕迹与提案原文可区分、可审计(同 v2 D12 口径) |

**偏离计数: 11 条**(D-08 为唯一语义级; D-01/D-07 为落点迁移; 其余为名字/成员级
被迫适配、lint 适配或标注)。

### 2.1 D-08 证据(数值级, 可复现)

**引擎/mace 前向**(单通道, 初始与逐步同构):
`S̃=clamp(S)= (S==0 ? 1 : S); f=(T−pred)/S̃; d_out(n,0)=d_pre(n,0)+s_n·f`
(polar TU L238-258/L957-976 与 mace `extensions.py` L737-766 实测一致)。

**闭式伴随**(对输入 `d̄=∂E/∂d_out(n,0)`、`F̄:=Σ_n (s_n/S̃)·d̄(n,0)`):
```
d̄_pre(n,0) += −F̄                       (提案: −F̄/S̃  ⇒ 多一个 1/S̃)
s̄(n)        = f·d̄(n) − [未钳制] f·F̄     (提案: f·d̄/S̃ − [未钳制] 2·f·F̄/S_raw)
```
**中央差分对拍**(应用器 `[8]` 段内嵌, 纯 stdlib; 随机化 N=3 用例, h=1e-6):

| 对象 | max abs err |
|---|---|
| 提案原文(初版) | **6.110e+02** |
| v3 修正(落地文本) | **3.013e-09** |
| 钳制分支(S≡0)闭式 | 0.0(精确) |

旁证: 设计文档 §2 ⑧-2 原文即写作 `F=(T−pred)/S; d+=s̃·F`(增量 = s·(T−pred)/S²),
与 mace 参考实现(增量 = (s/S)·(T−pred))相差一个 1/S —— 属设计→提案文本的代数笔误,
本次按其"闭式伴随"结论修正常数, 不改变代码其余任何结构。

### 2.2 其余数值链核查(无偏离, 已对拍)

- ⑧-A 散射: 前向 `e3nn[c]=sign_c·int[src(c)]` ⇒ 伴随 `int̄[src(c)]+=sign_c·e3nn̄[c]`
  (数据流单射 + atomic) — 与设计 §2 ⑧-A 一致, 实现逐字。
- ⑥ coulomb 种子链(内联重算 ρ、ρ̄=2V/(2π)⁶·FC·kfac·ρ、assemble 反向):
  应用器 `[8]` 段以 `E=FC/(2V)·Σ_k 2(ρ̂_r²+ρ̂_i²)/kn2`(当前 ⑥ 前向口径)对拍
  `q̄ ≡ ∂E/∂q`, **rel=9.6e-13**(h=1e-4)⇒ 与 ⑥ 前向同系数, 无需改动(⑥ 已暴露
  `polar_coulomb_rho_re/im`, 提案允许的"直接消费"替换留待后续 slice, 见 §4-2)。
- 描述符/相位/位置链常数(proj_scale=2/(2π)³、rho_scale=(2π)³/V、d5 相位式)按
  设计 §2 ⑧-3 推导复算一致; 重放核与 ⑤ 前向逐算子镜像(§8-20 grep 对照时须同步维护)。

---

## 3. 保真保留(非偏离; 说明)

1. **⑦ le 反向不在本提案**(module7 §8 口径): 只落种子视图/位掩码契约
   (`polar_reverse_seed_terms` bit0 由未来 ⑦ slice 驱动); ⑧ 掩码消费 bit1(⑥),
   `polar_coulomb_enabled=true`(D-10)。
2. **`Readout::reverse_all` 全文落主 tpp PART1**(依赖选择及理由见 §0.3-4)——
   与提案 §3 落点一致, 非偏离。
3. **§5.2 配套访问器 `polar_blocks_enabled()` 已在 §2.2 块内**(提案原文即在 §2.2
   尾), 不重复插入; pair style 直读 C++ 成员(§5.2 语义)。
4. 提案 §4.3 实现注(重放位级镜像、单射散射、d̄ 就地更新次序、跨 kernel 顺序语义)
   全部随文保留于代码注释, 未改动。
5. §5.1 注释"覆盖所有 atom_forces 消费方…"按原文保留(calculator `_reduce_atom_forces`
   接缝见 §5.3; 本次无需改动 calculator)。
6. 数值自检自身也作为 `[8]` 段落盘在应用器内(可在无 GPU/无引擎环境下复跑)。

---

## 4. 未纳入项与存疑

1. **pair style `vflag_global` guard 未纳入**: 提案 §6.3 为"建议"(
   `if (mace->polar_blocks_enabled()) error->all(...)`); 本 slice 的 loud guard =
   `reduce_stress` throw(逐字落地)。pair 侧 guard 涉及 `vflag_global` 分支与
   `error->all` 语义, 建议随 cell 伴随(下一 slice)一并落地 —— 当前 pair 侧
   `virial_reduction` 仍会静默给出缺极化项维里, **对极化模型使用 LAMMPS virial 的
   用户需注意**(或等下一 slice)。
2. **⑥ 内联重算 vs 直接消费**: ⑥ 现已暴露 `polar_coulomb_rho_re/im` 视图, 提案
   §0-3 允许把 §4.2 第 4 段两处"重算"替换为直接消费(位级一致); 本次按提案默认
   保留自包含重算(f64 eps 级差异, 已有 FD 证据), 替换留作可选优化。
3. **module8 门分档提示(结构事实)**: ⑥⑦ 已落码后, 极化模型上
   `polar_reverse_terms_last ≥ 2`(⑥ 位), `--stage auto` 将走 post 档;
   而 ⑦ le 反向尚未落地 ⇒ engine 力缺 le 项, post 档 python 总力 parity
   (FORCE_GATE=1e-3)在 ⑦ 反向落地前**预期超差**(pre 档的"零力回归"亦不再可达)。
   这是 ⑥⑦ 先于 ⑧ 落码的顺序性结果, 非本补丁缺陷; 建议 ⑦ 反向 slice 紧随其后。
4. **真实 `--apply` 前需处理 v2 旧备份**: 树中现存 `mace_nonlinear_kokkos.hpp.bak`
   /`mace_nonlinear_kokkos_impl.tpp.bak`/`mace_polar_impl.tpp.bak`/`mace_nonlinear_kokkos.cpp.bak`
   (v2 应用留存); 应用器按纪律"`.bak` 已存在即拒绝覆盖" ⇒ 先人工移走(如改名
   `.bak.v2`)。`pair_symmetrix_mace_kokkos.cpp` 无旧 `.bak`, 不受影响。
5. **树持续活跃**: 应用器锚点为纯文本 + 前后哨兵; **真实 `--apply` 前必须重跑
   `--dry-run`**(写前自动重断言, 版本不符即中止并打印诊断)。

---

## 5. 附: 全绿证据摘要 + 块清单表 + 后续步骤

### 5.1 dry-run 全绿证据摘要(2026-10-15; 日志 235 行, 退出码 0)

命令:
```bash
cd /workspace/group/shared/bza/task3_sampling_database/mlip
python3 bench/apply_polar_patches_v3.py --dry-run --log logs/apply_patches_v3_dryrun.log
# 完整输出: logs/apply_patches_v3_dryrun.log(235 行, 退出码 0)
```

- **锚点**: 16/16 插入点 `count==1` 唯一命中(块内断言 + 前后哨兵行校验);
- **结构前提核查**: 13/13 OK(含 §0.3 六项 + `attrs` 成员/`polar_json`/前向调用点/
  校后快照内核/⑥⑦ 密度源/binding 既有访问器/pair 唯一块);
- **幂等标记**: 16 项应用前全为 0(树未被本补丁改动);
- **应用后结构探针**: 35/35 OK(hpp §2.2/§2.3、tpp reverse_all/调用点/节点力/guard、
  tu 缓存/标量/主函数签名/D-04/D-05/D-08 三处修正/LayoutRight 清零、cpp 3 条访问器、
  pair 力写入等); **顺序探针 4/4 OK**(播种<反向注入<骨架反向; 校正<标量缓存<快照;
  init 校正<init 缓存; pipeline<反向函数);
- **`[8]` 数值自检**: Fukui 提案原文 err=6.110e+02 / v3 修正 err=3.013e-09;
  ⑥ 种子链 rel=9.554e-13; 钳制分支闭式 OK(见 §2.1);
- **`[9]` `bench/polar_lint.py`(对 dry-run 模拟产物 `logs/apply_v3_sim/`)**:
  main tpp / polar TU / pair style 三文件全 `CLEAN`,
  **6 类规则发现数 = 0 处(R1=0 R2=0 R3=0 R4=0 R5=0 R6=0, rc=0)**;
- **引擎树零写入**: 五文件 md5 与应用前快照逐字节一致; 无 `.polar_tmp`。

行数变化预估(difflib 内存; git 未调用):
```
 libsymmetrix/source/mace_polar_impl.tpp             | 1225 ++++  1429 -> 2654 (+1225)
 libsymmetrix/source/mace_nonlinear_kokkos.hpp       |   94 ++    861 ->  955 (+94)
 libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp  |   86 ++   7359 -> 7445 (+86)
 symmetrix/source/cpp/mace_nonlinear_kokkos.cpp      |   13 +     636 ->  649 (+13)
 pair_symmetrix/pair_symmetrix_mace_kokkos.cpp       |   13 +    2289 -> 2302 (+13)
 5 files changed, 1431 insertions(+), 0 deletions(-)
```

### 5.2 块清单表(6 文件 / 16 个插入点; dry-run 实测)

| # | 块 id | 文件 | op / 锚点(应用时刻) | 操作(行) | 内容来源(逐字提取) |
|---|---|---|---|---|---|
| 1 | 1a-hpp-array | `mace_nonlinear_kokkos.hpp` | insert_before `#include <cstdint>` | +2 | 新增(D-09; 提案 §2.2 注) |
| 2 | 1b-hpp-public | 同上 | after ⑤暂存块 5 行(`polar_nl_b_storage;`) | +84 | module8 §2.2(L139-222), 删 `_post` 2 行(D-02) |
| 3 | 1c-hpp-rev-all-decl | 同上 | after `evaluate_all` 声明 2 行块 | +8 | module8 §2.3(L240-248) |
| 4 | 2a-tpp-rev-all-def | `mace_nonlinear_kokkos_impl.tpp` | after `evaluate_all` 定义尾 4 行(`…deep_copy(output,result);}`) | +54 | module8 §3(L273-326) |
| 5 | 2b-tpp-rev-call | 同上 | before `ensure_view(harmonic_adjoints,…)` | +5 | module8 §4.0-A(L350-354) |
| 6 | 2c-tpp-node-force | 同上 | after 边级归约 kernel 尾(`…receiver),-force);});`) | +20 | module8 §5.1(L1708-1727) |
| 7 | 2d-tpp-stress-guard | 同上 | after `reduce_stress` 签名+`{` | +7 | module8 §6.3(L1832-1838) |
| 8 | 3a-tu-volume | `mace_polar_impl.tpp` | after ③ `const double volume=std::abs(…)` 7 行块 | +5 | module8 §4.0-B(L379-384) |
| 9 | 3b-tu-c1-clamp | 同上 | replace 逐步 clamp 3 行(16 空格) | 3→4 | module8 §4.0-C-1(L399-403) |
| 10 | 3c-tu-c1-scalars | 同上 | after `"polar scf fukui correction"` kernel 7 行块 | +12 | module8 §4.0-C-1(L408-427), 删快照 10 行(D-03) |
| 11 | 3d-tu-c2-clamp | 同上 | replace 初始化 clamp 3 行(8 空格) | 3→4 | module8 §4.0-C-2(L440-444) |
| 12 | 3e-tu-c2-scalars | 同上 | after `"polar fukui correction"` kernel 7 行块 | +5 | module8 §4.0-C-2(L449-454) |
| 13 | 3f-tu-reverse-main | 同上 | before `template class MaceNonlinearKokkosT<float>;` | +1198 | module8 §4.2(L487-1668) + D-04/D-05/D-06/D-08 变换 + 6 行 [v3] 头 |
| 14 | 3g-tu-coulomb-enable | 同上 | after ⑥ si_select 语句 3 行块(parse 函数内) | +3 | 新增(D-10; 提案 §0-3/§10-2) |
| 15 | 4-cpp-binding | `symmetrix/source/cpp/mace_nonlinear_kokkos.cpp` | replace 现链尾 `polar_coulomb_rho_im` 块 3 行 | 3→16 | module8 §7.1(L1855-1875), 去 `_post` 1 条(D-07) |
| 16 | 5-pair-force-wire | `pair_symmetrix/pair_symmetrix_mace_kokkos.cpp` | after `force_reduction` 全块 25 行 | +13 | module8 §5.2(L1751-1763) |

来源引用(应用器 `[0]` 段同源): 1b→L139-222; 1c→L240-248; 2a→L273-326; 2b→L350-354;
2c→L1708-1727; 2d→L1832-1838; 3a→L379-384; 3b→L399-403; 3c→L408-427; 3d→L440-444;
3e→L449-454; 3f→L487-1668; 4→L1855-1875; 5→L1751-1763。

### 5.3 应用与后续(本任务范围外; 由用户/主线程执行)

```bash
cd /workspace/group/shared/bza/task3_sampling_database/mlip
# (前置) 移走 v2 旧 .bak: hpp/tpp/tu/cpp 四文件 *.bak(见 §4-4)
python3 bench/apply_polar_patches_v3.py --dry-run     # 应用前一刻复验(树可能仍在漂移)
python3 bench/apply_polar_patches_v3.py --apply       # 写入(自动备份 .bak; 失败回滚)
```

随后: 重建(重编 polar TU + 主 tpp/hpp/binding/pair 各一次) → 装载冒烟 →
①-⑤ 回归(§8-25, 门应逐位不变) → ⑧ 门(pre 档不可达说明见 §4-3) →
⑦ le 反向 slice(补 post 档 parity) → cell 伴随 slice(替换 §6.3 loud guard)。
