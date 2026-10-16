# 模块⑥+⑦ 合并应用预检 —— `module6_patch_final.md` × `module7_patch_final.md`(2026-10-15)

**范围**:对两份最终补丁提案的**全部插入点**做只读核查(锚点唯一性/漂移、⑥⑦ 冲突矩阵、
既有符号作用域),并产出可机械执行的应用步骤清单。**不含任何引擎树写操作。**

**结论(TL;DR)**:
- **锚点**:13 个插入点全部逐字命中当前树(§2),唯一性 `count=1`(单列行/块)。无一处因
  树漂移而失配;行号在两小时内已两次位移(§1),因此**以文本锚点为准、行号仅作定位参考**。
- **冲突**:**3 处真实争用**(⑥-A↔⑦-4.1a 共享 hpp 空档;⑥-B↔⑦-4.2a 共享 ctor 空档;
  ⑥-D↔⑦-4.5 共享 binding 链尾)+ 2 处相邻伪争用(⑥-C1↔⑦-4.3、⑦-4.4↔⑥-C2,文本互不重叠)。
  全部可通过固定顺序 + 锚点顺延消解,**无一需要改写任一补丁的代码内容**。
- **顺序**:推荐 **先 ⑦(8 步)后 ⑥(5 步)**;理由与反向顺序的锚点移动规则见 §5。
- **符号**:两份提案引用的既有符号全部存在且同域可达(§4)。
- **非阻塞发现**:⑦-4.1a 一句注释把 q_up/v_up 标成 `(N,512)`(实为 N×2048,仅注释);⑦ 关于
  "PART2 不可见 `parse_uvu_paths`" 的论断对当前树不成立(该 helper 在共享前导段,可调用;
  内联副本无害,保留即可)。详见 §8。

---

## 1. 只读快照与漂移(树处于活跃修改中)

本预检三次读取(2026-10-15 UTC):

| 文件 | 提案编写时快照 | 预检中间读数 09:42 | **本预检定稿读数(证据基准)** |
|---|---|---|---|
| `libsymmetrix/source/mace_nonlinear_kokkos.hpp` | `b463c965…` / 814 行 | `fa20b4fd…` / 815 行 | `fa20b4fd45302d3b5384dc3980307526` / 815 行 / 09:27:42 |
| `libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp` | `7d6136b8…` / 8284 行 | `9912b639…` / 8291 行 | `a4c3b5a5f4202819b74db2ba670aabc6` / 8292 行 / **09:52:22** |
| `symmetrix/source/cpp/mace_nonlinear_kokkos.cpp` | `ba096f9a…` / 587 行 | 同 | `ba096f9a333db0df3b8e2fdd719f4d8f` / 587 行 / 09:11:09(无漂移) |
| `symmetrix/source/symmetrix/extract_mace_nonlinear.py` | `e0818e87…` | 变更(E1 已施行) | `7af7baee…`(含 `_full_width_linear_bias`) |
| `symmetrix/source/symmetrix/extract_polar_blocks.py` | `d1defd28…` | 变更(E2 已施行) | `6f41a0a0…`(含 `"type": "layer_norm"`) |
| `models/mace-polar/polar_symmetrix_probe.json` | 08:43 v3 | **09:30 重生成** | 本预检全部字段实测通过(§4.3) |

**漂移判定**:tpp 在 09:42→09:52 之间又 +1 行(行号整体位移),但 §2 中**全部锚点文本逐字仍在**。
应用前必须重跑 §9 复验(全 `count=1`);若 md5 再变,以 §9 输出为准、本文行号作废但锚点文本有效。

**⑤ 落地判定(主方案适用性)**:`grep -n "P3 模块⑤: SCF 循环" tpp` → **命中 L5930** ⇒ 按提案
主方案应用(⑥-C1 取 `polar_scf_final_density`,§3.4 过渡方案与 `--density-mode post-fukui`
不需要)。

---

## 2. 锚点逐条核查(全部命中,证据 = 实际行摘录)

行号基准 = 上表最后列(tpp `a4c3b5a5` / hpp `fa20b4fd` / cpp `ba096f9a`)。

### 2.1 ⑦ 模块(local_electron_energy)

| # | 提案节 | 文件 | 锚点(摘录) | 实测 | 判定 |
|---|---|---|---|---|---|
| 1 | §4.1a | hpp | ⑤暂存块末行 `        polar_nl_b_storage;                                             // ⑤暂存`(其后为 `    void compute_node_energies_forces(`,**L385→L386**) | L385 单命中;其后紧邻 L386 | ✅ 唯一·无漂移 |
| 2 | §4.1b | hpp | `    std::vector<PolarUpdate> polar_updates;`(**之后、`std::vector<LayerState> states;` 之前**) | L801;`states;` L802;其后紧邻 | ✅ 唯一·无漂移 |
| 3 | §4.2a | tpp | `                "polar_blocks: charges_maps count must match num_recursion_steps.");`(其下即 `    } else polar_enabled=false;`) | L3598;`} else polar_enabled=false;` L3599 | ✅ 唯一·无漂移 |
| 4 | §4.2b | tpp | `        for(auto& linear:polar_mixer_linears)linear.set_backend(selected);` + `        for(auto& update:polar_updates)update.set_e3_linear_backend(selected);` + `    }` | L3237-3239,逐字一致 | ✅ 唯一·无漂移 |
| 5 | §4.2c | tpp | `        for(auto& update:polar_updates)` + `            update.set_e3_linear_workspace(e3_linear_workspace);` + `    }` | L3652-3654,逐字一致 | ✅ 唯一·无漂移 |
| 6 | §4.3 | tpp | `    for(int layer=0;!generated_node_program` + `        &&layer<static_cast<int>(readouts.size());++layer) {` | L6412-6413,逐字一致 | ✅ 唯一·无漂移 |
| 7 | §4.4 | tpp | `    Kokkos::parallel_for("scale nonlinear energies",num_nodes,KOKKOS_LAMBDA(int node) {` … `    });` | L6435-6437,逐字一致(其上一行 `const Precision energy_scale=scale,energy_shift=shift;` L6434) | ✅ 唯一·无漂移 |
| 8 | §4.5 | cpp | 链尾整块 `        .def("polar_scf_final_density",[](const Evaluator& self) {` / `            return view2vector(self.polar_scf_final_density);` / `        });` | L576-578(其后 L579 `}` 关闭函数) | ✅ 唯一·无漂移 |

### 2.2 ⑥ 模块(coulomb_energy)

| # | 提案节 | 文件 | 锚点(摘录) | 实测 | 判定 |
|---|---|---|---|---|---|
| 9 | §1(补丁 A) | hpp | 同 #1(`polar_nl_b_storage;` 之后、`void compute_node_energies_forces(` 之前) | 同 #1 | ✅ 唯一(**与 ⑦-4.1a 同空档**,见 §3) |
| 10 | §2(补丁 B) | tpp | 同 #3(charges_maps 校验之后、`} else polar_enabled=false;` 之前) | 同 #3 | ✅ 唯一(**与 ⑦-4.2a 同空档**) |
| 11 | §3.2(补丁 C1) | tpp | ⑤ 块尾 `            Kokkos::parallel_for("polar scf final density",` … `                });` + `        }` + `    }` | L6403-6410,逐字一致(`        }`=L6409 关 ⑤ 内层作用域、`    }`=L6410 关 `if(polar_enabled) {`) | ✅ 唯一·无漂移 |
| 12 | §3.3(补丁 C2) | tpp | 文件尾段唯一 `    if(has_zbl) {`(**之前、其上方相邻的 `    });` 空行之后**) | L8058;其上方 L8056 `    });`、L8057 空行 | ✅ 唯一·无漂移 |
| 13 | §4(补丁 D) | cpp | polar 访问器链末尾(= #8 的同一终端块) | 同 #8 | ✅ 唯一(**与 ⑦-4.5 同段落**) |

**旁证(结构与作用域,均实测)**:
- ③ 作用域:`        const double volume=std::abs(` = L5723、`        const int num_k=static_cast<int>(k_norm2_host.size());` = L5773,
  与 ⑥-C1 插入点(8 空格缩进,L6409/L6410 之间)同层——两者都在 `    if(polar_enabled) {`(L5686)
  与闭合 `    }`(L6410)之间 ⇒ `num_k`/`volume` **直接可见**。③(L5687)/④(L5843)/⑤(L5930) 相对序不变。
- ⑤ 块结构(证据):`        {` 于 L5935 开、L6409 关;`for(int step=0;step<polar_recursion_steps;++step) {`
  于 L6024 开、L6399 关;**末态密度写入 L6400-6408(循环之外)**,故 ⑥-C1 读到的必为末步密度。
- ④ 内层局部 `field_constant`(L5849,其嵌套块 `{`/`}` 为 L5848/L5928)与 ⑤ 的 `scf_field_constant`(L5936)
  均为块内局部、对 ⑥-C1 不可见 ⇒ ⑥-C1 自含 `coulomb_field_constant` 是**必要**且与既有形态一致的写法(§8-3)。

---

## 3. 冲突矩阵(⑥ × ⑦)

| 插入点(空档/终端) | ⑥ 用 | ⑦ 用 | 类型 | 判定与消解 |
|---|---|---|---|---|
| hpp `polar_nl_b_storage;` ↔ `void compute_node_energies_forces(`(L385/386) | ⑥-A | ⑦-4.1a | **真实争用**:同一插入位置 / 同一成员区(public 视图区) | 两者均为**新增同名域外成员声明**,无符号/容量争用;允许同空档先后插入。推荐序=先 ⑦-4.1a、后 ⑥-A(⑥-A 的结束边界 `void compute_node_energies_forces(` 顺延为"⑦ 块之后") |
| hpp `polar_updates` ↔ `states`(L801/802) | — | ⑦-4.1b | 无 | 独立;任何序 |
| tpp ctor `throw …charges_maps…;` ↔ `} else polar_enabled=false;`(L3598/3599) | ⑥-B | ⑦-4.2a | **真实争用**:同一 ctor 区间(`if(data.contains("polar_blocks")){` L3514 内尾段)/ 同一插入位置 | 两者逻辑互斥面为零(⑥ 读 `coulomb_energy`,⑦ 以 `pb.contains("local_electron_energy")` 守卫);先后皆可。推荐序=先 ⑦-4.2a、后 ⑥-B(⑥-B 插到 ⑦ 块与 `} else` 之间) |
| tpp setter backend(3238/3239) | — | ⑦-4.2b | 无 | 独立 |
| tpp setter workspace(3653/3654) | — | ⑦-4.2c | 无 | 独立 |
| tpp ⑤ 内层闭 `        }` ↔ 大块闭 `    }`(L6409/6410) | ⑥-C1 | — | 与 ⑦-4.3 **相邻但不同空档** | 伪争用:⑥-C1 在 `    }` 之内侧,⑦-4.3 在其外侧(见下行);互不改写对方锚点,任何序 |
| tpp readout 累加循环前(L6412 前) | — | ⑦-4.3 | 同上 | 伪争用;任何序 |
| tpp `scale nonlinear energies` 块后(6437/6439) | — | ⑦-4.4 | 与 ⑥-C2 语义同级、文本不同空档 | 伪争用;任何序(两处都只在 `energies` 上 device 相加,交换律) |
| tpp `if(has_zbl) {` 前(8057/8058) | ⑥-C2 | — | 同上 | 伪争用;任何序 |
| cpp `.def` 链尾(L576-578) | ⑥-D | ⑦-4.5 | **真实争用**:同一 binding 段落(唯一终端块 `        });`) | 串行执行;两补丁的机械操作同型("终端 `        });` → `        })` + 新链 + `;`"),后应用者锚点顺延为**当前链尾块**(名字变为先应用者的最后一个 `.def`) |

**冲突计数:3 真实争用**(均为"同空档/同终端先后插入",零语义耦合)+ **2 伪争用**(相邻独立)。
**没有任何一处需要改写 ⑥/⑦ 的插入代码本身**;所有争用只影响"锚点定位描述"。

---

## 4. 既有符号与作用域核查

### 4.1 提案引用的既有符号(全部命中)

| 符号 | 声明(hpp) | 写入/使用(tpp,当前版) | ⑥/⑦ 使用点可达性 |
|---|---|---|---|
| `polar_scf_final_density` | L379 `Kokkos::View<Precision**,Kokkos::LayoutRight> polar_scf_final_density;` | ⑤ 写:L6400-6401 realloc、L6402 alias、L6403-6408 "polar scf final density" kernel;读=⑥-C1 | ✅ 成员视图;⑥-C1 插入点在写之后同一大块,读取到末态值 |
| `polar_step_potential_features` | L376(与 `polar_field_feats_a/b` 同组,`// (N,8)/(N,16)`) | ⑤ 写:L5948-5949 realloc、L6004 alias、L6186 "polar scf normalize potential features" kernel | ✅ 成员视图;⑦-4.3 在其后(函数级)读取,值为末次迭代 |
| `polar_spin_charge_density` | L346-347 | ⑤ 原地更新(SCF 步内);② 校正读它 | ✅ 成员视图;⑦-4.3 读为末态 |
| `polar_field_independent_density` | L351-352 | ② 写:L5679-5683 | ✅ 成员视图;⑦-4.3 读为初 Fukui 校正后密度 |
| `polar_node_feats_e3nn` | L763-764 | ① 逐层写(L5593 起) | ✅ 成员视图;⑦-4.3 uvu 的 x1 |
| `polar_cosines` / `polar_sines` | L356-357 | ③ 每次前向重建(L5779 起) | ✅ 成员视图;⑥-C1 读 |
| `polar_density_basis_re/im` | L358-359 `// (Nk,4)` | ③/④/⑤ 消费(③ k 网格段) | ✅ ⑥-C1 读(与 ④ 全同表,ctor 已断言同源) |
| `polar_k_norm2` / `polar_k0_mask` | L355 | ③ 写(L5775-5776) | ✅ ⑥-C1 读 |
| `polar_updates` | L801(`std::vector<PolarUpdate>`) | ctor L3590-3591 emplace;L3596 计数校验;setter L3238/3652 | ✅ ⑦-4.2b/4.2c 与之并列接线,互不重名 |
| `num_k`(局部) | — | `const int num_k=…` L5773(8 空格) | ✅ 与 ⑥-C1 同作用域(§2 旁证) |
| `volume`(局部) | — | `const double volume=std::abs(` L5723(8 空格) | ✅ 同上 |
| `epd` / `pb`(ctor 局部) | — | `const auto& pb=data.at("polar_blocks");` L3515;`const auto& epd=pb.at("electric_potential_descriptor");` L3549 | ✅ 均在 `if(data.contains("polar_blocks")){`(L3514)同一作用域,⑥-B/⑦-4.2a 插入点(3598/3599)可直接引用 |
| 原语 `ensure_view` | — | 1 维 `(View&,View&,std::size_t)` L90;2 维 `(…,int,int)` L99 | ✅ ⑥-C1 显式传 `std::size_t(num_k)`/`std::size_t(1)` 与重载匹配 |
| 原语 `toKokkosView` + `tensor_values<int>` | — | 先例:L3579-3581(`polar_si_select`,③ 段,含 `tensor_values<int>`) | ✅ ⑥-B 的 SI select 解析同型,**先例在案** |
| `parse_uvu_paths` 元列序 `[in1_off,in2_off,out_off,w_off,mul1,mul2,mode,d1]` | — | L51-87(共享前导段) | ✅ ⑦-4.2a 内联同列序(见 §8-2 说明) |
| `E3LinearKokkosT` API | — | `set_backend`/`set_workspace`/`evaluate`/`input_dimension()`/`output_dimension()`(`e3nn_kokkos.hpp` L31-32/53/58/61) | ✅ ⑦-4.2b/4.2c/4.3 全部有 API |
| `AffineMLPKokkosT` API | — | json ctor / `input_size()` / `output_size()` / `evaluate()` / 默认 workspace 自增长(`affine_mlp_kokkos.hpp` L27-32/51;`.cpp` `prepare` L283-309) | ✅ ⑦-4.2a/4.3 成立,无需额外 workspace 接线(⑦ §4.2c 注成立) |
| `Kokkos::parallel_reduce` 标量风格 | — | 先例:L3995/3998(int/long long)、L8157(double) | ✅ ⑥-C1 的 double 归约与库内一致 |

### 4.2 ctor 解析先例(⑥-B 断言形态)

③ 段 `polar_si_overlap/polar_si_select` 的 `toKokkosView(…, tensor_values<…>(…))` 写法
(L3576-3581)即 ⑥-B 的模板;⑥-B 新增的**JSON 子树递归相等断言**无既有先例但仅用
`nlohmann::json::operator!=`(值语义),安全。

### 4.3 JSON 实值核查(09:30 重生成版,全部通过)

⑥:coulomb `include_self_interaction=True / include_pbc_corrections=True`;SI 表
coulomb `[4]/[4]` vs descriptor `[8]/[8]`(**两表分离**);`kspace_cutoff` 相等;
`density_basis` 子树 `EXACT-EQ True`(全文比对)。
⑦:`configs.add_local_electron_energy=True`、`num_recursion_steps=2`;
`linear_up_q/v.bias.shape=[2048]`(E1 已生效);`dot_products_q/v` 各 2 条指令
`path_weight=0.03125`、`weight.shape=[524288]=2×512×512`;`mlp.layers[*].type =
linear/layer_norm/silu ×3 + linear`(E2 已生效)、LN `eps=1e-5`;
`kind=OneBodyMLPFieldReadout`。顶层 `has_zbl=False`、`scale_shift=1.0/0.0`
(与 ⑥-C2 锚点说明一致)。

---

## 5. 安全合并顺序与锚点漂移说明

### 5.1 推荐顺序:**先 ⑦(8 步,§6 步骤 1-8)、后 ⑥(5 步,步骤 9-13)**

理由:
1. ⑦-4.5 是两补丁中唯一的**"整块替换"型**操作(9 个新 `.def`,文本最长);在其字面锚点
   (`polar_scf_final_density` 终端块,L576-578)**仍然逐字存在**时执行,风险最低。
2. ⑦ 的 5 组引擎段全部自包含(新成员/新守卫),不消费 ⑥ 的任何符号;反之 ⑥ 也不消费 ⑦。
   ⇒ 顺序纯属机械问题,取"大替换先做"。
3. ⑥ 的三处争用点(⑥-A/⑥-B/⑥-D)在 ⑦ 之后都退化为**单向顺延**(见 §5.2),而 ⑦ 的争用点
   在 ⑥ 之后需要"回到链中插入"的改写,更容易错。

### 5.2 反向/正向的锚点移动规则

**(a) 先 ⑦ 后 ⑥(推荐)时 ⑥ 的锚点如何移动**:

| ⑥ 节 | 原始锚点 | ⑦ 应用后 | 执行动作 |
|---|---|---|---|
| ⑥-A | `polar_nl_b_storage;` 之后、`void compute_node_energies_forces(` 之前 | 该空档现为 `…polar_nl_b_storage;` → [⑦-4.1a 块] → `void compute_node_energies_forces(` | 插入到 **⑦-4.1a 块之后、`void compute_node_energies_forces(` 之前**(紧贴声明行上方) |
| ⑥-B | `throw …charges_maps…;` 之后、`} else polar_enabled=false;` 之前 | 空档现为 `…throw …;` → [⑦-4.2a 块] → `} else polar_enabled=false;` | 插入到 **⑦-4.2a 块之后、`} else polar_enabled=false;` 之前**(或紧贴 throw 之后;二选一,推荐前者) |
| ⑥-C1 | ⑤ 尾 `        }` 与 `    }` 之间 | **不变**(⑦ 4.1b/4.2b/4.2c/4.3/4.4/4.5 均不触该空档) | 原样执行 |
| ⑥-C2 | `    if(has_zbl) {` 之前 | **不变** | 原样执行 |
| ⑥-D | 链尾 `polar_scf_final_density` 块(终端 `        });`) | ⑦-4.5 后链尾 = `polar_le_energy` 块,仍以 `        });` 结尾 | 锚点顺延为"**当前链尾终端 `        });`**";展开为 `        })` + ⑥ 的 3 个 `.def` + `;`(即 ⑥ §4 插入码原样追加在最末 `.def` 之后) |

**(b) 若强行先 ⑥ 后 ⑦(备选,不推荐)时 ⑦ 的锚点移动**:
- ⑦-4.1a 插入点改为"`polar_nl_b_storage;` 之后、[⑥-A 块]之前";
- ⑦-4.2a 插入点改为"throw … 之后、[⑥-B 块]之前";
- ⑦-4.5 的字面锚点(**以 `        });` 结尾的 `polar_scf_final_density` 整块**)将**不再存在**
  (⑥-D 已把它改成 `        })` + 3 个 `.def`),必须改写为"当前链尾终端 `        });` → `        })`
  + 9 个 `.def` + `;`"。可行但需二次重述锚点,易错。

> 结论:任选一种顺序都可得到同一最终文件;**不要混用两方向**(例如 ⑦ 先做 4.1a/4.2a 又
> 先做 ⑥-D)。

### 5.3 合并后的语义序核对(与 mace 参考装配序对照)

`extensions.py`:L707 `node_inter_es = scale_shift(...)` → L866 `total = e0 + inter_e` →
L883 `+ le_total` → L925-927 `+ electro_energy`。引擎合并后:scale 块(L6435)→ ⑦-4.4
(le,不乘 scale)→ ⑥-C2(coulomb,不乘 scale,位于 `if(has_zbl)` 前、与 ZBL 同级)。
两者都位于 `scale nonlinear energies` **之后**,语义正确;彼此交换律(device 端独立相加)。

---

## 6. 应用步骤清单(机械执行;每步已给预期 grep)

约定:`ENGINE=/workspace/group/shared/bza/external-repos/symmetrix-xl`。
行号 = §1 定稿读数;应用前先跑 §9。
"插入内容代号"指对应提案文档中的代码块(逐字复制,勿改)。

| # | 文件 | 锚点文本(定位) | 插入内容代号 | 预期 grep 验证命令(在 `$ENGINE` 下) |
|---|---|---|---|---|
| 1 | hpp | `        polar_nl_b_storage;                                             // ⑤暂存`(之后;L385 后) | `module7_patch_final.md §4.1a` 代码块 | `grep -c -F 'polar_le_q_in_storage' libsymmetrix/source/mace_nonlinear_kokkos.hpp` → `1` |
| 2 | hpp | `    std::vector<PolarUpdate> polar_updates;`(之后、`std::vector<LayerState> states;` 之前) | `module7_patch_final.md §4.1b` 代码块 | `grep -c -F 'P3 模块⑦' libsymmetrix/source/mace_nonlinear_kokkos.hpp` → `2` |
| 3 | tpp | `                "polar_blocks: charges_maps count must match num_recursion_steps.");`(之后、`    } else polar_enabled=false;` 之前) | `module7_patch_final.md §4.2a` 代码块 | `grep -c -F 'polar_le_mlp=AffineMLPKokkosT<Precision>(mlp);' libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp` → `1` |
| 4 | tpp | `        for(auto& update:polar_updates)update.set_e3_linear_backend(selected);`(之后、`    }` 之前) | `module7_patch_final.md §4.2b` 代码块 | `grep -c -F 'polar_le_linear_up_q.set_backend(selected);' libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp` → `1` |
| 5 | tpp | `            update.set_e3_linear_workspace(e3_linear_workspace);`(之后、`    }` 之前) | `module7_patch_final.md §4.2c` 代码块 | `grep -c -F 'polar_le_linear_up_q.set_workspace(e3_linear_workspace);' libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp` → `1` |
| 6 | tpp | `    for(int layer=0;!generated_node_program`(**之前**插入) | `module7_patch_final.md §4.3` 代码块 | `grep -c -F 'launch_le_uvu' libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp` → `3`;`grep -c -F 'auto q_in=polar_le_q_in;' …` → `1` |
| 7 | tpp | `    Kokkos::parallel_for("scale nonlinear energies"…` 块的 `    });`(**之后**,空行前) | `module7_patch_final.md §4.4` 代码块 | `grep -c -F '"polar le energy add"' …tpp` → `1` |
| 8 | cpp | 链尾块 `        .def("polar_scf_final_density",…);`(整体替换为"原块 + §4.5 新增",新块置于链尾 `;` 之前) | `module7_patch_final.md §4.5` 代码块 | `grep -c -F 'polar_le_energy' symmetrix/source/cpp/mace_nonlinear_kokkos.cpp` → `1` |
| 9 | hpp | ⑦-4.1a 块之后、`    void compute_node_energies_forces(int num_nodes`(**之前**) | `module6_patch_final.md §1(补丁 A)` 代码块 | `grep -c -F 'polar_coulomb_si_select' …hpp` → `1`;`grep -c -F 'polar_include_self_interaction' …hpp` → `1` |
| 10 | tpp | ⑦-4.2a 块之后、`    } else polar_enabled=false;`(**之前**) | `module6_patch_final.md §2(补丁 B)` 代码块 | `grep -c -F 'coulomb_energy density_basis/kspace_cutoff must' …tpp` → `1` |
| 11 | tpp | ⑤ 尾 `        }`(L6409)与 `    }`(L6410)**之间** | `module6_patch_final.md §3.2(补丁 C1)` 代码块 | `grep -c -F '"polar coulomb rho"' …tpp` → `1` |
| 12 | tpp | `    if(has_zbl) {`(之前、`    });` 空行之后) | `module6_patch_final.md §3.3(补丁 C2)` 代码块 | `grep -c -F '"add polar coulomb energy"' …tpp` → `1` |
| 13 | cpp | 当前链尾终端 `        });`(步骤 8 后=⑦ 的 `polar_le_energy` 块;展开为 `        })` + §4 新链 + `;`) | `module6_patch_final.md §4(补丁 D)` 代码块 | `grep -c -F '.def("polar_coulomb' …cpp` → `3`;`grep -c -F 'polar_coulomb' …cpp` → `3` |

### 6.1 应用后整体核对(与两份 §5 自检等价的一次性命令块)

```bash
ENGINE=/workspace/group/shared/bza/external-repos/symmetrix-xl
HPP=$ENGINE/libsymmetrix/source/mace_nonlinear_kokkos.hpp
TPP=$ENGINE/libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp
CPP=$ENGINE/symmetrix/source/cpp/mace_nonlinear_kokkos.cpp
# ⑦ 结构
grep -c -F 'P3 模块⑦' "$HPP"              # 2
grep -c -F 'polar_le_enabled' "$TPP"       # 6
grep -c -F 'polar_le_enabled' "$HPP"       # 2 (声明行 + 4.1a 注释;polar_le_add 同声明行)
# ⑥ 结构
grep -c -F 'P3 模块⑥' "$HPP"              # 1
grep -c -F 'P3 模块⑥' "$TPP"              # 3 (ctor/C1/C2 注释)
grep -c -F '.def("polar_coulomb' "$CPP"    # 3
# 装配序与密度源
grep -n -F 'scale nonlinear energies' "$TPP"           # < 'polar le energy add' < 'if(has_zbl) {'
grep -n -F 'polar_scf_final_density' "$TPP"            # 写(≤6408) < ⑥-C1 读取行
grep -n -F 'const double volume=std::abs' "$TPP"       # ③ 段行号 < ⑥-C1
# 只读数值自检(⑥ §5 a/b/c)
cd /workspace/group/shared/bza/task3_sampling_database/mlip && \
  /workspace/caches/conda/envs/polar/bin/python - <<'PY'
import json; d=json.load(open("models/mace-polar/polar_symmetrix_probe.json"))
ce=d["polar_blocks"]["coulomb_energy"];epd=d["polar_blocks"]["electric_potential_descriptor"]
print("si shapes",ce["self_interaction_terms"]["overlap_constants"]["shape"],
      epd["self_interaction_terms"]["overlap_constants"]["shape"])
print("flags",ce["include_self_interaction"],ce["include_pbc_corrections"])
print("EXACT-EQ",json.dumps(ce["density_basis"],sort_keys=True)
      ==json.dumps(epd["density_basis"],sort_keys=True))
PY
```

(⑥/⑦ 各自 §5 自检表其余条目——`--live` 级、负例 smoke、构建后装载冒烟、门脚本——按原文执行;
本文 §4.3 已预核 JSON 条目全部通过。)

---

## 7. 应用顺序时序图(文本)

```
[E1/E2/JSON 已就绪(09:27/09:30,已核)]
 ① ⑦-4.1a hpp  ② ⑦-4.1b hpp  ③ ⑦-4.2a tpp  ④ ⑦-4.2b tpp
 ⑤ ⑦-4.2c tpp  ⑥ ⑦-4.3 tpp   ⑦ ⑦-4.4 tpp   ⑧ ⑦-4.5 cpp
 ⑨ ⑥-A hpp     ⑩ ⑥-B tpp     ⑪ ⑥-C1 tpp    ⑫ ⑥-C2 tpp   ⑬ ⑥-D cpp
 → 重建(⑥⑦ 同一次;E1/E2 后 JSON 已于 09:30 重生成,无需再动) → 装载冒烟 → ⑥门 → ⑦门
   → L1 端到端(bench/polar_endtoend_parity.py --forces-expect-absent)
```

预检附注:重建后先跑 `bench/polar_engine_load_smoke.py`(POS-OK)与
`bench/polar_module6_gate.py --live 1`、`bench/polar_module7_gate.py --live 2`;
端到端总账见 `bench/polar_endtoend_parity.py`(与本文同日、Task B)。

---

## 8. 预检发现(非阻塞;应用时无需改代码)

1. **注释计数指误(⑦-4.1a)**:hpp 插入块把 `polar_le_q_up/polar_le_v_up` 标注为 `// (N,512)`,
   而 `linear_up_q/v` 的 `irreps_out=512x0e+512x1o` ⇒ 实际宽度 **2048**(夹具 `q_up/v_up`
   shape=(N,2048) 实测;⑦-4.5 绑定注释 `(N×2048)` 正确;4.3 用
   `polar_le_linear_up_q.output_dimension()` 自适应,数值不受影响)。应用时**原样保留**该注释
   即可(勿按注释把视图改成 512)。
2. **TU 边界论断(⑦ §4/§7-偏差 4)**:`parse_uvu_paths` 实际位于 tpp **共享前导段 L51-87**
   (`#if … PART == 1` 在 L286 才出现),即对 PART2(ctor)同样可见;'不可见'依据对当前树不成立。
   但 ⑦-4.2a 的内联副本与 L80-81 列序/错误语义逐项一致(`polar le uvu:` 前缀除外),
   **保留内联无任何风险**;若后续想减重,须注意 helper 输出 `std::vector`、内联版还做了
   view 包装,不能一行替换。
3. **⑥-C1 自含常量正确且必要**:`field_constant` 是 ④ 内层局部(L5849,其 `{` 在 L5848 开、
   L5928 闭),⑤ 亦自含 `scf_field_constant`(L5936)⇒ ⑥-C1 声明 `coulomb_field_constant`
   与既有形态一致,不得改为提升全局(会扩散改动面)。
4. **树活跃修改风险**:tpp 在 09:08→09:42→09:52 三次变更;应用者必须"先 §9 复验 → 立即应用"。
5. **E1/E2/JSON 已就绪**:无需再跑 `bench/polar_json_regen.py`(除非 extractor 又被改动;
   重生成会同时改动 ⑥ 的 `density_basis` 断言依据,重生成后须重跑 §4.3 核查)。

## 9. 应用前一刻的锚点全量复验(只读;全部应 `1`)

```bash
ENGINE=/workspace/group/shared/bza/external-repos/symmetrix-xl
HPP=$ENGINE/libsymmetrix/source/mace_nonlinear_kokkos.hpp
TPP=$ENGINE/libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp
CPP=$ENGINE/symmetrix/source/cpp/mace_nonlinear_kokkos.cpp
md5sum "$HPP" "$TPP" "$CPP"     # 对照 §1:fa20b4fd…/a4c3b5a5…/ba096f9a…
# hpp 锚点(全部应 1)
for pat in \
  "polar_nl_b_storage;                                             // ⑤暂存" \
  "std::vector<PolarUpdate> polar_updates;" \
  "    void compute_node_energies_forces(int num_nodes" \
  "std::vector<LayerState> states;" \
  ; do printf '[hpp] %s  x%d\n' "$pat" "$(grep -c -F "$pat" "$HPP")"; done
# tpp 锚点(全部应 1)
for pat in \
  "charges_maps count must match num_recursion_steps" \
  "for(auto& linear:polar_mixer_linears)linear.set_backend(selected);" \
  "update.set_e3_linear_workspace(e3_linear_workspace);" \
  "for(int layer=0;!generated_node_program" \
  'Kokkos::parallel_for("scale nonlinear energies",' \
  'Kokkos::parallel_for("polar scf final density",' \
  "    if(has_zbl) {" \
  ; do printf '[tpp] %s  x%d\n' "$pat" "$(grep -c -F "$pat" "$TPP")"; done
grep -c -F '.def("polar_scf_final_density"' "$CPP"                       # 1
```

末行提示:tpp ⑤ SCF 收尾三行(`Kokkos::realloc(…polar_scf_final_density…` → `auto final_density=…`
→ `Kokkos::parallel_for("polar scf final density",…` → `                });` → `        }` → `    }`)
必须在案——⑥-C1 的空档由此两行界定;若树再次变更,以本文件 §2 #11 的**整段锚点文本**为准。
