# ⑥⑦ 引擎补丁 v2 适配 —— polar 独立 TU 新结构(2026-10-15)

**定位**: 把 `docs/module6_patch_final.md`(⑥ `coulomb_energy`)与
`docs/module7_patch_final.md`(⑦ `local_electron_energy`)两份"文本锚点 + 逐行代码块"
提案**适配到新结构**——polar 已拆分为独立编译单元:

- `libsymmetrix/source/mace_polar_impl.tpp`(底部匿名空间 helpers → `PolarUpdate`
  构造/接线定义 → `MaceNonlinearKokkosT<Precision>::run_polar_pipeline(...)`,②③④⑤
  全在其体内;`mace_polar_kokkos.cpp` 仅 include 该 tpp);
- 主 tpp(`mace_nonlinear_kokkos_impl.tpp`):①/mixer 循环块、ctor 解析块
  (`if(data.contains("polar_blocks")){...} else polar_enabled=false;`)、setter 接线、
  **单调用点** `if(polar_enabled) run_polar_pipeline(...)`(位于 "scale nonlinear
  energies" 之后,L5496-5503);
- hpp:polar 成员区;binding(`symmetrix/source/cpp/mace_nonlinear_kokkos.cpp`):
  polar 访问器链,链尾 = `polar_scf_final_density`(L576-578)。

**交付**:
1. 本文(适配裁定 + 偏离清单 + 结构前提核对 + 全绿证据,见 §5-附);
2. `bench/apply_polar_patches_v2.py`(应用器:块清单 = ①hpp 增补、②主 tpp 1 行、
   ③新 TU 两个插入段、④binding 链;dry-run 已全绿,日志
   `logs/apply_patches_v2_dryrun.log`)。

**纪律(已遵守)**: 本任务只写 `mlip/docs/` 与 `mlip/bench/`;引擎树
(`/workspace/group/shared/bza/external-repos/symmetrix-xl`)**零写入**——dry-run 仅内存
模拟(dry-run 前后四文件 md5 不变,无 `.bak`/`.polar_tmp` 残留);未构建;未调用 git;
未跑 GPU。

---

## 0. 新结构基线与只读快照

### 0.1 只读快照(v2 适配完成时刻)

| 文件 | 行数 | md5 | 备注 |
|---|---|---|---|
| `libsymmetrix/source/mace_polar_impl.tpp`(新 TU) | 1000 | `139bf0d40b960ea812b0ec674c66a186` | 12:13 更新后读数(见 0.2) |
| `libsymmetrix/source/mace_nonlinear_kokkos.hpp` | 821 | `a01de8b84524fdf6f50b03d4b6f78ecd` | 本任务期间无漂移 |
| `libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp` | 7358 | `c767274a54506063d46dc564f10c86ff` | 本任务期间无漂移 |
| `symmetrix/source/cpp/mace_nonlinear_kokkos.cpp`(binding) | 587 | `ba096f9a333db0df3b8e2fdd719f4d8f` | 本任务期间无漂移 |

### 0.2 树活跃修改记录(适配期间实测)

- 任务背景给定读数 983 行(旧);本任务首次读取为 **994 行**
  (`25af11b2…`,含②③④⑤与显式实例化);
- 2026-10-15 **12:13:58**,并行进程再次更新 TU(994 → **1000 行**,`139bf0d4…`):
  `run_polar_pipeline` 顶部(L207-212)新增函数内**工作密度**
  `polar_working_density`(自 `polar_spin_charge_density` 深拷贝);② 的校正改写
  (L224)与 `polar_field_independent_density` 落盘(L265)、④(L446)/⑤(L583)的
  `density_all` 别名均改指工作密度。语义变化:`polar_spin_charge_density` 恢复为
  **只读的 ① 原始累积**;⑤ 的 SCF 末态 = 工作密度 = 快照成员
  `polar_scf_final_density`(L986-996 写入)。
- v2 应用器的锚点/插入点**不随行号漂移**:12:14 重跑 dry-run 在新版本上依旧
  `7/7 count=1` 全绿(唯一随之被迫的适配 = 偏离表 D05)。
- 合成后的两段插入内容长度:③a = 128 行、③b = 287 行(随 D05 注释 +2)。

### 0.3 三项关键结构核对(任务点名;已在应用器 [2] 段机器化,读数全 `count==1`)

1. **`run_polar_pipeline` 体内 ⑤ 块之后恰无其他内容(插点在函数尾)**:
   ⑤ 块尾接 `        }`(关⑤内层作用域)→ `    }`(关 `if(polar_enabled){`)→ `}`
   (函数尾)三行紧邻;应用器以 9 行整段锚点(含 `final_density(...)=density_all(...)`
   收尾 kernel)断言唯一,插入点 = 锚点第 7/8 行之间(即 ⑤ 块闭 `        }` 与
   `if(polar_enabled){` 块闭 `    }` 之间)。
2. **`node_energies`/`polar_json` 成员存在且可达**:hpp L341
   `Kokkos::View<double*> atomic_energies,node_energies,node_forces,atom_forces,`
   (公开区;⑥-C2/⑦-4.4 以 1 元素 kernel device 端写 node 0);hpp L757
   `nlohmann::json polar_json;`(私有;成员函数可达);主 tpp L3357
   ctor `polar_enabled=true;polar_json=pb;`(新 TU 的 `parse_polar_extension_tables()`
   数据源可达)。
3. **binding 链尾锚点唯一**:`        .def("polar_scf_final_density"...` 三行块
   (cpp L576-578)`count==1`;其前后哨兵行(`        })` / `}`)逐字在案。

---

## 1. 适配规则落实对照

| 规则 | 落实 | 对应块 |
|---|---|---|
| 1. 运行期代码落新 TU(插入 `run_polar_pipeline` ⑤ 之后、函数尾 `}` 前;该函数被调用于 scale/shift 之后) | ⑥-C1 + ⑦-4.3 + ⑦-4.4 + ⑥-C2 合成**同一连续插入段**(8 空格基缩进,位于 `if(polar_enabled){` 块内、⑤ 之后),能量写入用成员 `node_energies`(1 元素 kernel 写 node 0,沿用提案 §3.3 既有修正) | ③b |
| 2. 解析代码落新 TU(`parse_polar_extension_tables()`;`polar_json` 数据源;`pb.at(...)` → `polar_json.at(...)`;主 tpp 仅 1 行) | ⑥-B + ⑦-4.2a 合并为成员函数定义(新 TU 显式实例化之前);主 tpp `} else polar_enabled=false;` 后插 `if(polar_enabled) parse_polar_extension_tables();` | ③a、② |
| 3. hpp 合并 4.1a/4.1b + 公开/私有标注 + 新声明 | 公开区:⑦-4.1a(视图)+⑥-A(视图与 bool/SI 表,与①–⑤ 同区按提案);私有区:⑦-4.1b(表与配置);新增 `void parse_polar_extension_tables();` 声明(`run_polar_pipeline` 声明之后) | ①a/①b/①c |
| 4. binding 照 ⑥-D/⑦-4.5 全文,锚定当前链尾并展开 | `polar_scf_final_density` 终端块一次展开为 ⑦ 9 条 + ⑥ 3 条(⑦ 链尾 `        });` → `        })` 后接 ⑥ 块) | ④ |
| 5. 内容保真 + 偏离列表 | 代码一律**程序化逐字提取**自两份提案文档(带来源行号,见 §5.1);偏离共 **12 条**(§2),全部为落点/名字/缩进类被迫适配 + [v2] 标注(§2-D12) | — |
| 6. 应用器与旧版同纪律 | `bench/apply_polar_patches_v2.py`:锚点 `count==1` 断言(含诊断输出)、dry-run 默认、`--apply` 已实现未执行(备份 `.bak` + 写前重断言 + 写后字节/探针复验 + 失败回滚)、写前幂等标记检查 | 脚本 |
| 7. 自检全绿 + 证据 + 三项核对 | 见 §5.2(exit 0,7/7 锚点、7 项结构前提、31 项结构探针、2 项顺序探针全 OK)与 §0.3 | — |

---

## 2. 偏离清单(块号、原锚点、新落点、原因)

以下每条 = 相对两份提案原文的偏差;代码正文除表列内容外逐字保留。

| # | 块号 | 原锚点(提案) | 新落点(v2) | 原因 |
|---|---|---|---|---|
| D01 | ⑥-C1 | 旧主 tpp ⑤ 块尾 `        }`(⑤ 内层闭合)与 `    }`(`if(polar_enabled){` 闭合)之间(旧布局 L6409/6410) | 新 TU `run_polar_pipeline` 内、⑤ 块闭(现 L989)与 `if(polar_enabled){` 块闭(L990)之间;文本与 8 空格缩进不变 | polar/⑤ 已迁独立 TU,该区间即任务指定的"⑤ SCF 块之后、函数尾前";`num_k`/`volume` 与本插入同处 `if(polar_enabled){` 块作用域,直接可见 |
| D02 | ⑥-C2 | 旧主 tpp `    if(has_zbl) {` 之前 | ③b 运行段末块(C1→4.3→4.4 之后);`auto coulomb_energies=energies;` → `=node_energies;`;整体 +4 缩进 | 规则 1:运行期代码全落新 TU。`energies` 为旧 `compute_node_energies_forces` 局部别名,新作用域不可达 → 改用成员 `node_energies`(任务规则明示);调用点在 scale/shift 之后 ⇒ "装配在 scale 后"语义保持 |
| D03 | ⑦-4.3 | 旧主 tpp readout 累加循环之前(`for(int layer=0;!generated_node_program` 前) | ③b 运行段第 2 块;整体 +4 缩进 | 规则 1;该函数体位于 ⑤ 之后、天然满足"⑤ 之后"语义 |
| D04 | ⑦-4.4 | 旧主 tpp `scale nonlinear energies` 块之后 | ③b 运行段第 3 块;整体 +4 缩进(文本原样,已用成员 `node_energies`) | `run_polar_pipeline` 由主 tpp 在 scale 块之后**单点调用**(L5496-5503)⇒ 语义等价;避免主 tpp 第二处插入 |
| D05 | ⑦-4.3(密度源) | `auto charges_final=polar_spin_charge_density;`(提案 §1 表:"⑤ 原地更新后的末态…与 polar_scf_final_density 同值") | `auto charges_final=polar_scf_final_density;`(行内加 2 行 `[v2 适配]` 注释说明) | **树 12:13 更新**:②/⑤ 改用函数内工作密度,`polar_spin_charge_density` 变为只读的①原始累积,不再等于 SCF 末态;末态已发布为快照成员 `polar_scf_final_density`(与工作密度同值,⑥-C1 同源)→ 代名替换(数值语义保持) |
| D06 | ⑥-B | 旧主 tpp ctor `charges_maps` 计数校验之后、`} else polar_enabled=false;` 之前 | ③a `parse_polar_extension_tables()` 函数体首段(新 TU 显式实例化之前定义);`pb.at("coulomb_energy")` → `polar_json.at("coulomb_energy")`(1 处);新增 `const auto& epd=polar_json.at("electric_potential_descriptor");`;整体 -4 缩进 | 规则 2:解析合并入成员函数;数据源=成员 `polar_json`;`epd` 原为 ctor ③ 段局部量,函数内按同一 JSON 路径重建(被迫作用域适配) |
| D07 | ⑦-4.2a | 同上 ctor 空档 | ③a 函数体第 2 段;`pb.contains(...)`/`pb.at(...)` → `polar_json.contains(...)`/`polar_json.at(...)`(共 3 处);整体 -4 缩进 | 规则 2(同 D06) |
| D08 | ⑥-B/⑦-4.2a(call) | ctor 解析块内直接插入(两提案各自表述) | 主 tpp 仅 1 行:`    if(polar_enabled) parse_polar_extension_tables();`(插于 `} else polar_enabled=false;` 之后,L3441) | 规则 2("主 tpp 仅在…之后插 1 行") |
| D09 | ⑦-4.1a/⑥-A/⑦-4.1b | hpp:⑤暂存块尾 `polar_nl_b_storage;` 之后(⑦-4.1a 与 ⑥-A 共享空档)、`polar_updates;` 之后(⑦-4.1b) | 同空档(锚点在新 hpp 未漂移);⑦-4.1a+⑥-A 合并为**公开区同一插入**、⑦-4.1b 置于**私有区**;附公开/私有标注 | 规则 3(合并 + 标注);两提案共用空档,合并后只插 1 次 |
| D10 | (新增)| — | hpp `run_polar_pipeline` 声明之后新增 `void parse_polar_extension_tables();`(公开区,2 行) | 规则 2/3:新成员函数的 hpp 声明(提案无,补 D06/D07 迁入所必需) |
| D11 | ⑥-D + ⑦-4.5 | cpp 链尾 `polar_scf_final_density` 终端块;两提案各按"链尾展开"先后应用两次 | 同一锚点**一次**展开:⑦ 9 条 `.def` 后接 ⑥ 3 条;⑦ 链尾 `        });` 改写为 `        })`(接 ⑥) | 规则 4:锚点=当前链尾;两提案两次顺序展开的合并等价形态(功能一致,尾 `;` 保留 1 处) |
| D12 | 全部 | — | 新增 `[v2]`/`[公开(供 binding)]`/`[私有]` 标注与说明注释共 15 行(hpp 6 + ③a 3 + ③b 6) | 规则 3 标注要求;使适配痕迹与提案原文可区分、可审计 |

**偏离计数: 12 条**(其中 D05 为树 12:13 更新引入的二次适配;全部为落点、名字、
缩进(±4 空格)或标注类,**无一处改动数值语义**)。

### 2.1 与任务适配规则 1-2 的关键语义对照

- "作用域内已有的名字直接用":③b 仅使用 `num_k`/`volume`(③ 段局部)、
  `polar_*` 成员(含 `polar_scf_final_density`/`polar_step_potential_features`/
  `polar_field_independent_density`/`polar_node_feats_e3nn`)与 `num_nodes`(函数形参);
  未新声明任何作用域别名(除提案自带的 `auto` 局部别名)。
- "能量写入用成员 `node_energies`,1 元素 kernel,不得 host 直写":⑥-C2 走
  `Kokkos::parallel_for("add polar coulomb energy",1,...)`;⑦-4.4 沿用提案
  num_nodes kernel(写 node 级、单图=整盒口径)。
- "该函数被调用于 scale/shift 之后":主 tpp L5496-5498(scale 块)→ L5501-5503
  (调用点),两处能量装配(4.4、C2)均在函数体内 ⇒ 天然满足。

---

## 3. 保真保留(非偏离; 说明)

1. ⑦-4.1a 注释 `(N,512)`(q_up/v_up 实为 N×2048)按 `merge_preflight_67.md` §8-1
   **原样保留**(数值路径用 `output_dimension()` 自适应)。
2. ⑦-4.2a 内联 `parse_le_uvu`(含 "PART 为独立 TU,此处内联" 注释)按预检 §8-2
   保留——新 TU 内匿名空间 `parse_uvu_paths` 实际可达,内联无风险且逐字保真。
3. ⑥-C1 注释内"(⑤ 未落地时换 polar_field_independent_density)"过渡句保留
   (本树 ⑤ 已落地,主方案密度源 = `polar_scf_final_density`)。
4. ⑥-C1 自含常量 `coulomb_field_constant`(不改 ③④ 代码)按提案保留。
5. ⑥-B 保持"无 contains 守卫"(polar_blocks 存在即要求 `coulomb_energy`,
   提案原语义;现役模型含该段)。
6. 提案中 hpp 的 `polar_coulomb_si_select` 取 `View<int*>`(与 descriptor 侧
   `polar_si_select` 同型)与 tpp 的 double 标量归约/1 元素落盘等既有修正,均为
   `module6/7_patch_final.md` 正文,照单执行。

---

## 4. 未纳入项与存疑

1. **⑦-4.2b/4.2c(backend/workspace setter 接线)未纳入本 v2 的 4 组块清单**:
   任务块清单①–④ 不含两块,且约束"主 tpp 仅 1 行"不允许改 `set_e3_linear_backend`
   (L3063-3081)与 ctor 尾 workspace 接线块(L3487-3495)。**影响评估**:目标平台
   CUDA 构建下 `E3LinearKokkosT` 自动后端走 scalar 路径(`use_scalar_backend` 于
   `KOKKOS_ENABLE_CUDA` 恒真),不触 `workspace_`,功能不受阻;但 host 大样本或强制
   `packed_gemm` 时应用器不会为其接线 → **建议后续 slice 补齐**:在上述两处既有
   polar 接线块各加 3 行(`if(polar_le_enabled){polar_le_linear_up_q/v.set_backend
   (selected);}` 与 `set_workspace(e3_linear_workspace);`),或把接线挂入
   `parse_polar_extension_tables()` 之后的延迟初始化(需先建 workspace,当前 ctor
   时序:polar 解析块 L3440 → 1 行调用 → workspace 创建 L3474-3477)。
2. **树持续活跃修改**:polar TU 在 2 小时内至少两次变更(994→1000;12:13 工作密度)。
   本文与脚本的锚点均为文本锚点且带哨兵行断言;**真实 `--apply` 前必须重跑
   `--dry-run`**(应用器写前会自动重断言,版本不符即中止并打印诊断)。
3. **过渡性提示(非阻塞)**:`polar_scf_final_density` 现同时服务 ⑤ 快照、⑥-C1、
   ⑦-4.3 三处(合成后 TU 内 5 次出现,含 [v2] 注释);若后续 ⑤ 重构改变该发布点,
   请同步复核本文件 D01/D05 两行与门脚本密度源选择。

---

## 5. 附: 全绿证据摘要 + 块清单表

### 5.1 dry-run 全绿证据摘要(2026-10-15,log 144 行,退出码 0)

命令:
```bash
cd /workspace/group/shared/bza/task3_sampling_database/mlip
python3 bench/apply_polar_patches_v2.py --dry-run --log logs/apply_patches_v2_dryrun.log
# 完整输出: logs/apply_patches_v2_dryrun.log(144 行,退出码 0)
```

- **锚点**:7/7 插入点 `count==1` 唯一命中(块内断言 + 前后哨兵行校验);
- **结构前提核查**:7/7 OK(§0.3 三项 + `node_energies` 公开区、`polar_json` 成员与
  ctor 赋值行、`run_polar_pipeline` 定义唯一、④/⑤ 工作密度别名 2 处、⑤ 尾插点、
  binding 链尾);
- **幂等标记**:8 项应用前全为 0(树未被本补丁改动);
- **应用后结构探针**:31/31 OK(⑥:C1 ρ̂/装配/密度源、⑦:q_in/uvu/invar/mlp/总量/
  访问器 9+3 条、解析函数、主 tpp 调用行、hpp 视图/声明等);
- **顺序探针**:2/2 OK(`⑤ 写末态密度 < ⑥-C1 读取`;`⑥-C1 < ⑦-4.4(add le) <
  ⑥-C2(add coulomb)`,与 extensions.py L866→L925-929 参考装配序一致);
- **引擎树零写入**:snapshot md5 与应用后复验一致(见 §0.1),无 `.bak`/`.polar_tmp`。

### 5.2 块清单表(4 组 / 7 个插入点;dry-run 实测)

| # | 块 | 文件 | 锚点(字面;应用时刻 count) | 操作 | 内容行数 | 内容来源(逐字提取) |
|---|---|---|---|---|---|---|
| ①a | hpp 公开增补 | `mace_nonlinear_kokkos.hpp` | ⑤暂存块 5 行(尾行 `polar_nl_b_storage;`),之后 | 后插 26 行 | 26 | module7 §4.1a(L180-193)+ module6 §1(L38-48)+ 3 [v2] 标注 |
| ①b | hpp 私有增补 | 同上 | `std::vector<PolarUpdate> polar_updates;`(2 行块),之后 | 后插 11 行 | 11 | module7 §4.1b(L207-216)+ 2 [v2] 标注 |
| ①c | hpp 声明 | 同上 | `run_polar_pipeline` 声明 6 行块,之后 | 后插 2 行 | 2 | 新增(注释 + 声明) |
| ② | 主 tpp 1 行 | `mace_nonlinear_kokkos_impl.tpp` | `    } else polar_enabled=false;` | 后插 1 行 | 1 | 新增(规则 2) |
| ③a | 新 TU 解析函数 | `mace_polar_impl.tpp` | `template class MaceNonlinearKokkosT<float>;`,之前 | 前插 128 行 | 128 | module6 §2(L73-103)+ module7 §4.2a(L232-320),经 pb→polar_json/epd 重建/-4 缩进 |
| ③b | 新 TU 运行段 | 同上 | ⑤ 尾 9 行块(`polar scf final density` 收尾 kernel…`        }`/`    }`),第 7/8 行之间 | 中插 287 行 | 287 | module6 §3.2(L142-250,原文 8 空格)+ module7 §4.3(L375-520,+4)+ §4.4(L535-546,+4)+ module6 §3.3(L271-285,+4) |
| ④ | binding 链 | `symmetrix/source/cpp/mace_nonlinear_kokkos.cpp` | 链尾 `polar_scf_final_density` 3 行块 | 整块替换 3 → 52 行 | 52 | module7 §4.5(L562-601)+ module6 §4(L320-333),一次链尾展开 |

行数变化预估(内存 difflib;git 未调用):
```
 libsymmetrix/source/mace_polar_impl.tpp             | 415 ++++  1000 -> 1415 (+415)
 libsymmetrix/source/mace_nonlinear_kokkos.hpp       |  39 +++    821 ->  860 (+39)
 libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp  |   1 +     7358 -> 7359 (+1)
 symmetrix/source/cpp/mace_nonlinear_kokkos.cpp      |  49 ++     587 ->  636 (+49)
 4 files changed, 504 insertions(+), 0 deletions(-)
```

### 5.3 应用与后续(本任务范围外;由用户/主线程执行)

```bash
cd /workspace/group/shared/bza/task3_sampling_database/mlip
python3 bench/apply_polar_patches_v2.py --dry-run     # 应用前一刻复验(树可能仍在漂移)
python3 bench/apply_polar_patches_v2.py --apply       # 写入(自动备份 .bak;失败回滚)
```
随后:重建(仅需重编 polar TU + 主 tpp/hpp/binding 一次)→ 装载冒烟 →
`--forces-expect-absent` 的 ⑥/⑦ 门与端到端(L1)。执行前先补 §4-1 的 setter 接线
决策(建议单列小补丁)。
