# P3 模块⑥ 精确补丁提案 —— `coulomb_energy`(全 k 空间 GTO-Ewald 能量)(2026-10-15)

**上游**: `docs/module6_patch_draft.md`(伪代码级草案,正确性依据全部在案);
本文 = 该草案的**可直接应用版**(文本锚点 + 插入代码全文;不写行号 diff)。
**预检**: `bench/polar_module6_preflight.py`(4 夹具双源 rel ≤1.2e-15,已锁定);
**门脚本**: `bench/polar_module6_gate.py`(本提案配套,见 §6;只写不跑)。

## 0. 锚点纪律与只读快照

引擎树仅只读访问。读取时(2026-10-15T09:15:13Z)快照:

| 文件 | 行数 | md5 | 备注 |
|---|---|---|---|
| `libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp` | 8284 | `7d6136b871e38d372b142bbb00bb88b4` | **模块⑤ 正在落码**(09:04–09:11 连续改动) |
| `libsymmetrix/source/mace_nonlinear_kokkos.hpp` | 814 | `b463c96587b3e04e3f171cf8e6aaa8d2` | ⑤ 成员已入 |
| `symmetrix/source/cpp/mace_nonlinear_kokkos.cpp` | 587 | `ba096f9a333db0df3b8e2fdd719f4d8f` | ⑤ 访问器已入 |

**⑤ 落地判定(打补丁前先跑)**:
```bash
ENGINE=/workspace/group/shared/bza/external-repos/symmetrix-xl
grep -n "P3 模块⑤: SCF 循环" $ENGINE/libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp
```
- 命中(= 本文读取时状态):补丁 C 用 **主方案**(⑥ 紧随 ⑤ 的 SCF 收尾、取 `polar_scf_final_density`);
- 未命中:按 §3.4 备选锚点(④ 块结尾)插入,密度源行换 `polar_field_independent_density`,门脚本用 `--density-mode post-fukui`。

**变更纪律**:全部公式/表/装配序均有 file:line 依据(§7 账本),与草案依据一致;仅"实现形态"按现行源码细化(偏差见 §7)。

---

## 1. 补丁 A —— `mace_nonlinear_kokkos.hpp` 成员声明

**锚点**: 公开中间量视图区,`// ⑤暂存` 视图块的最后一行
`        polar_nl_b_storage;` 之后、`    void compute_node_energies_forces(` 声明之前。
(该区所有成员必须是 **public** —— cpp 绑定在类外访问;与①–⑤ 视图同区同风格。)

**插入代码全文**:

```cpp
    // --- P3 模块⑥: coulomb_energy(全 k GTO-Ewald;前向标量 eV + ρ̂(k) 诊断) ---
    // ρ̂ = 未乘 (2π)³/V 的傅里叶密度(即 features.py L58-63 的括号项);
    // E = FIELD_CONSTANT/(2V)·Σ_k 2·(ρ̂_r²+ρ̂_i²)/|k|²(k=0 跳过)。
    bool polar_include_self_interaction=true,polar_include_pbc_corrections=true;
    Kokkos::View<Precision*> polar_coulomb_si_overlap;                   // (4)
    Kokkos::View<int*> polar_coulomb_si_select;                          // (4)
    Kokkos::View<Precision*> polar_coulomb_rho_re,polar_coulomb_rho_im,
        polar_coulomb_rho_re_storage,polar_coulomb_rho_im_storage;       // (Nk)
    Kokkos::View<double*> polar_coulomb_energy,polar_coulomb_energy_storage;
                                                                        // (1) eV
```

要点(与草案的差异见 §7-偏差 1):`polar_coulomb_si_select` 必须与 descriptor 侧
`polar_si_select` 同型 `Kokkos::View<int*>`(ctor 用 `tensor_values<int>` 解析);
草案把两者写在同一个 `Kokkos::View<Precision*>` 声明里,不能编译。

---

## 2. 补丁 B —— ctor 解析(impl.tpp,`if(data.contains("polar_blocks")){` 内)

**锚点(⑤ 已落地,主)**: ⑤ ctor 块末尾的
```cpp
        if(static_cast<int>(polar_updates.size())!=polar_recursion_steps)
            throw std::invalid_argument(
                "polar_blocks: charges_maps count must match num_recursion_steps.");
```
之后、`    } else polar_enabled=false;` 之前。此处 `pb`(`polar_blocks`)与 `epd`
(`electric_potential_descriptor`,③ 段已声明)均在同一作用域,直接可用。

**备选锚点(⑤ 未落地)**: ③④ 表解析段最后一行
`        polar_recursion_steps=pb.at("configs").at("num_recursion_steps").get<int>();` 之后。

**插入代码全文**:

```cpp
        // ---- P3 模块⑥: coulomb_energy(GTO 表与 descriptor 同源;SI 表独立) ----
        // 依据: GL energy.py L33-68(同一 density_max_l/smearing_width/kspace_cutoff
        // 构造 density_basis / self_interaction / realspace 三子块);
        // 实测(2026-10-15): JSON 中 coulomb_energy.density_basis 与
        // electric_potential_descriptor.density_basis 全子树逐值相等。
        {
            const auto& cb=pb.at("coulomb_energy");
            polar_include_self_interaction=
                cb.at("include_self_interaction").get<bool>();
            polar_include_pbc_corrections=
                cb.at("include_pbc_corrections").get<bool>();
            // 逐值一致性断言(nlohmann::json == 为递归数值比较);不等即抛错,
            // 防止未来 extractor/模型失配时静默复用错误表。
            if(cb.at("density_basis")!=epd.at("density_basis")
                ||cb.at("kspace_cutoff").get<double>()
                    !=epd.at("kspace_cutoff").get<double>())
                throw std::invalid_argument(
                    "polar_blocks: coulomb_energy density_basis/kspace_cutoff must "
                    "match electric_potential_descriptor (same-source GTO tables; "
                    "see extract_polar_blocks.py::_gto_energy).");
            // SI 表 = coulomb 自己的 [4] 表(实测 overlap=[5.4160867,0.4011916×3],
            // select=[0,1,2,3]),与 descriptor 的 [8] 表**不同源**,且 realspace
            // 分支自相互作用符号约定与 k 空间相反 → 必须单独解析,不得复用。
            polar_coulomb_si_overlap=toKokkosView("polar coulomb si overlap",
                tensor_values<Precision>(cb.at("self_interaction_terms")
                    .at("overlap_constants")));
            polar_coulomb_si_select=toKokkosView("polar coulomb si select",
                tensor_values<int>(cb.at("self_interaction_terms")
                    .at("select_indices")));
        }
```

依据:断言模式镜像 ③ 段既有写法(`tensor_values<...>` + `toKokkosView`,见 tpp
③ 段 `polar_si_overlap/polar_si_select` 解析);"逐值相等"由测得的 JSON 子树相等支撑
(§5-自检 5 可复跑)。

---

## 3. 补丁 C —— tpp 运行块(两段式:C1 计算 / C2 装配)

### 3.1 位置与作用域裁定(含 `num_k`/`volume` 处理)

当前源码结构(读取时):`if(polar_enabled){`(③④⑤ 大块)内依次为
③(k 网格/几何,`const double volume` 与 `const int num_k` 均在**该大块作用域**内)→
④(势特征)→ ⑤(SCF 循环,收尾把末态密度写入 `polar_scf_final_density`)。
**具体做法(选定)**: C1 **不**提升 `num_k`/`volume` 为函数级变量(那要改③,与⑤的
落码热点冲突),而是把 C1 插在**同一大括号作用域内、⑤ 之后、大块闭合 `}` 之前**,
`num_k`/`volume` 直接可见。函数级提升仅作为无法同域时的备选(不推荐:需把
③ 的 `const double volume=`/`const int num_k=` 改为对函数级变量的赋值,扩散改动面)。

### 3.2 C1:ρ̂(k) 计算 + 能量归约 + SI/pbc 闸门

**锚点(主方案,⑤ 已落地)**: ⑤ 块尾部
```cpp
            Kokkos::parallel_for("polar scf final density",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},
                    {num_nodes,8}),
                KOKKOS_LAMBDA(int node,int column) {
                    final_density(node,column)=density_all(node,column);
                });
        }
    }
```
中 `        }`(⑤ 内层作用域闭合)与 `    }`(③④⑤ 大块闭合)之间;新代码缩进 8 空格。
插入后,大块闭合 `}` 之前会多出 ⑥ 的 `{...}`。

**插入代码全文**:

```cpp
        // ---- P3 模块⑥: coulomb_energy(全 k GTO-Ewald 基线) ----
        // 依据: graph_longrange v0.4.0 energy.py L113-166(_pbc_energy_batch)与
        // features.py L28-83(结构因子/库仑核), gto_utils.py L283-302(自相互作用);
        // 密度源 = SCF 末态(extensions.py L887-893 自旋求和, L910-924 调用)。
        // 公式经 bench/polar_module6_preflight.py 对 4 夹具双源逐位锁定(rel ≤1.2e-15)。
        // 化简: E = FIELD_CONSTANT/(2V)·Σ_k 2·kfac·(ρ̂_r²+ρ̂_i²), kfac=1/k²(k=0 跳过;
        // 由 density=(2π)³ρ̂/V 对 energy.py L17-30 两次代入所得,与预检链等价)。
        // 作用域: num_k/volume 为③ 块局部量,本块与③④⑤ 同处一个
        // if(polar_enabled){...} 内 ⇒ 直接可见(不得移到块外,见 §3.1)。
        {
            constexpr double coulomb_field_constant=1.0/(5.526349406e-3);
            // (a) ρ̂(k) = Σ_n e^{ik·r_n}·B(k)·q_n(未乘 (2π)³/V;q_n = 自旋和)
            ensure_view(polar_coulomb_rho_re,polar_coulomb_rho_re_storage,
                std::size_t(num_k));
            ensure_view(polar_coulomb_rho_im,polar_coulomb_rho_im_storage,
                std::size_t(num_k));
            auto coulomb_rho_re=polar_coulomb_rho_re;
            auto coulomb_rho_im=polar_coulomb_rho_im;
            auto coulomb_cos=polar_cosines;
            auto coulomb_sin=polar_sines;
            auto coulomb_dbre=polar_density_basis_re;
            auto coulomb_dbim=polar_density_basis_im;
            // 密度源 = ⑤ 写入的 SCF 末态快照(N,8)。若在⑤ 未落地的分支打本补丁,
            // 把下一行换成 polar_field_independent_density(② 出口;过渡门 §3.4)。
            auto coulomb_density=polar_scf_final_density;
            Kokkos::parallel_for("polar coulomb rho",num_k,KOKKOS_LAMBDA(int k) {
                Precision acc_cos[4]={Precision(0),Precision(0),Precision(0),
                    Precision(0)};
                Precision acc_sin[4]={Precision(0),Precision(0),Precision(0),
                    Precision(0)};
                for(int node=0;node<num_nodes;++node) {
                    const Precision c=coulomb_cos(k,node);
                    const Precision s=coulomb_sin(k,node);
                    for(int m=0;m<4;++m) {
                        const Precision q=coulomb_density(node,m)
                            +coulomb_density(node,4+m);
                        acc_cos[m]+=c*q;
                        acc_sin[m]+=s*q;
                    }
                }
                Precision re=Precision(0),im=Precision(0);
                for(int m=0;m<4;++m) {
                    re+=coulomb_dbre(k,m)*acc_cos[m]
                        +coulomb_dbim(k,m)*acc_sin[m];
                    im+=coulomb_dbim(k,m)*acc_cos[m]
                        -coulomb_dbre(k,m)*acc_sin[m];
                }
                coulomb_rho_re(k)=re;
                coulomb_rho_im(k)=im;
            });
            // (b) E_k 归约(k=0 跳过;标量用 double,与库内既有 parallel_reduce
            //     标量风格一致:tpp "validate nonlinear nodes"/"reduce_stress")
            auto coulomb_kn2=polar_k_norm2;
            auto coulomb_k0=polar_k0_mask;
            double coulomb_k_energy=0.0;
            Kokkos::parallel_reduce("polar coulomb k energy",num_k,
                KOKKOS_LAMBDA(int k,double& acc) {
                    if(coulomb_k0(k)>Precision(0)) return;
                    const double re=static_cast<double>(coulomb_rho_re(k));
                    const double im=static_cast<double>(coulomb_rho_im(k));
                    acc+=2.0*(re*re+im*im)
                        /static_cast<double>(coulomb_kn2(k));
                },coulomb_k_energy);
            double coulomb_energy_value=coulomb_k_energy
                *coulomb_field_constant/(2.0*volume);
            // (c) 自相互作用(GTO self block,energy.py L156-166 减 0.5·Σ<q,self>)。
            //     本模型序列化 include_self_interaction=true ⇒ 分支不执行
            //     (反事实诊断在门脚本)。表 = coulomb 自己的 [4] 表。
            if(!polar_include_self_interaction) {
                auto coulomb_si_overlap=polar_coulomb_si_overlap;
                auto coulomb_si_select=polar_coulomb_si_select;
                double coulomb_self_energy=0.0;
                Kokkos::parallel_reduce("polar coulomb self interaction",
                    num_nodes,KOKKOS_LAMBDA(int node,double& acc) {
                        double value=0.0;
                        for(int m=0;m<4;++m) {
                            const int selected=coulomb_si_select(m);
                            const double q=static_cast<double>(
                                coulomb_density(node,m)
                                +coulomb_density(node,4+m));
                            const double q_selected=static_cast<double>(
                                coulomb_density(node,selected)
                                +coulomb_density(node,4+selected));
                            value+=q*static_cast<double>(
                                coulomb_si_overlap(m))*q_selected;
                        }
                        acc+=value;
                    },coulomb_self_energy);
                coulomb_energy_value-=0.5*coulomb_self_energy;
            }
            // (d) pbc 修正(energy.py L168-192): 本路径 = 全周期(TTT)盒,
            //     is_molecule/is_slab 选择器均为假(slabs.py L185-256/L59-69 两项
            //     不执行)⇒ 修正恒 0。include_pbc_corrections=true 的语义即
            //     "该恒 0 项已被计入"。非全周期盒暂不支持(引擎无 pbc 入参;
            //     后续 slice 显式实现 monopole/slab 解析式并拒绝未实现分支)。
            if(polar_include_pbc_corrections) {
                // TTT ⇒ 0(保留闸门语义与后续 slice 挂点)。
            }
            // (e) 标量落盘(device 端写;host 不得直接访存 CUDA 视图)
            ensure_view(polar_coulomb_energy,polar_coulomb_energy_storage,
                std::size_t(1));
            auto coulomb_energy=polar_coulomb_energy;
            const double coulomb_energy_host=coulomb_energy_value;
            Kokkos::parallel_for("polar coulomb energy store",1,
                KOKKOS_LAMBDA(int) {
                    coulomb_energy(0)=coulomb_energy_host;
                });
        }
```

对草案的两处实现修正(§7-偏差 2/3):草案 `(c)` 的 `Precision self_e=…` 用 `Precision`
做归约容器、且 `(d)` 直接 `node_energies(0)+=static_cast<double>(e_coulomb)` 是
host 写 device 视图(CUDA 后端非法)——本提案改为 double 归约 + 1 元素 kernel 落盘,
装配段(§3.3)再 device 端读回。

**公式形态复核(2026-10-15,纯 numpy 复算)**:按 C1 的引擎形态(ρ̂ 未乘 (2π)³/V、
前因子 FIELD_CONSTANT/(2V)、`if(k0>0) return` 跳过 k=0)对 4 夹具直接重算:
nh3 0.765539555956946 / ch4 0.085221931248514 / n2 0.164550958289156 /
co2 0.733896214980153,vs index.json 目标 rel ≤1.3e-15(与预检链代数等价,已实测)。

### 3.3 C2:能量装配(scale/shift 之后、与 ZBL 加法同级)

**锚点**: 文件尾段唯一的 `    if(has_zbl) {`(本模型 JSON `has_zbl=false`,块不执行,
但锚点函数级唯一)之前、其上方相邻的 `    });` 空行之后。此处 `energies`
(`auto energies=node_energies;`,"scale nonlinear energies" 行沿线)已在作用域。

**插入代码全文**:

```cpp
    if(polar_enabled) {
        // P3 模块⑥ 能量装配:参考装配序 = (e0+inter_e) → +le → +electro
        // (extensions.py L866 → L925-929);scale_shift 仅作用于 node_inter_es
        // (L706-708),electro **未缩放** ⇒ 必须位于 "scale nonlinear energies"
        // 行之后,否则会被覆写为 e0+scale·E+shift(语义错)。本模型
        // scale=1.0/shift=0.0(JSON 实值)数值无差,顺序语义必须正确。
        // 单图口径: 整盒总量加入 node 0(calculator.py L3860 汇总全部 node 能量)。
        auto coulomb_energy=polar_coulomb_energy;
        auto coulomb_energies=energies;
        Kokkos::parallel_for("add polar coulomb energy",1,
            KOKKOS_LAMBDA(int) {
                coulomb_energies(0)+=coulomb_energy(0);
            });
    }
```

### 3.4 备选锚点(⑤ 未落地时)

- C1 插在 ④ 块之后:找 `polar project features` kernel 的
  `                    });`(逐列非门 kernel 收尾)→ `            }`(for channel 闭合)→
  `        }`(④ 内层作用域闭合)→ 在**大块闭合** `    }` 之前插入(同一作用域,
  `num_k`/`volume` 仍可见);并把密度源行改为
  `auto coulomb_density=polar_field_independent_density;`
  (② 出口 = SCF 起点,过渡门判据 `abs ≤1e-6 eV`,门脚本 `--density-mode post-fukui`)。
- ⑤ 之后落地时**只改这一行**即可切回终态语义;C2/D 不受影响。

### 3.5 `field_constant` 说明

③④ 块的 `constexpr double field_constant=1.0/(5.526349406e-3)` 是 ④ 内层作用域局部量
(⑥ 处不可见)。C1 自包含声明同值常量 `coulomb_field_constant`(依据 GL utils.py L12),
**不改** ③④ 既有行(最小化与 ⑤ 落码热点的冲突);若后续统一提升为文件级
`polar_field_constant`,把 C1/④ 两处改为引用即可(纯机械替换,不影响数值)。

---

## 4. 补丁 D —— binding 访问器(`symmetrix/source/cpp/mace_nonlinear_kokkos.cpp`)

**锚点**: polar 访问器链末尾。读取时最后一条为
```cpp
        .def("polar_scf_final_density",[](const Evaluator& self) {
            return view2vector(self.polar_scf_final_density);
        });
```
(草案写作 "polar_potential_features 之后";⑤ 访问器已入,故锚点顺延到链尾;
若只在无⑤ 分支应用,则锚 `polar_potential_features` 访问器之后,语义相同。)

**插入代码全文**(即把上述链尾 `        });` 展开为 `        })` + 新链 + `;`):

```cpp
        .def("polar_coulomb_energy",[](const Evaluator& self) {
            // 模块⑥: 前向标量(1,), eV;口径 = SCF 末态密度 + 序列化
            // include_self_interaction/include_pbc_corrections(本模型 True/True,
            // TTT 修正恒 0)。polar 未启用时返回空列表。
            return view2vector(self.polar_coulomb_energy);
        })
        .def("polar_coulomb_rho_re",[](const Evaluator& self) {
            // 模块⑥ 诊断: ρ̂(k) 实部(Nk;未乘 (2π)³/V, 与预检脚本链口径见门脚本)。
            return view2vector(self.polar_coulomb_rho_re);
        })
        .def("polar_coulomb_rho_im",[](const Evaluator& self) {
            return view2vector(self.polar_coulomb_rho_im);
        });
```

---

## 5. 自检清单(全部为可执行 grep/读数动作)

全部命令设 `ENGINE=/workspace/group/shared/bza/external-repos/symmetrix-xl`,
`MLIP=/workspace/group/shared/bza/task3_sampling_database/mlip`。
注: 下表命令里的 `\|` 是 Markdown 表格内的转义写法,复制执行时替换为 `|`。

| # | 约束 | 动作 | 期望 |
|---|---|---|---|
| 1 | **表不重算**(⑥ 复用③ 的表,不重建 GTO 基) | `sed -n '/P3 模块⑥/,/^    }$/p' $ENGINE/libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp \| grep -cE "exp\(|pref[01]\|cl_scale"` | `0`(⑥ 块无任何表重建;表从 `polar_density_basis_re/im` 读) |
| 2 | **SI 表 [4] 独立** | 运行 `python`(见下 a)+ `grep -c 'cb\.at("self_interaction_terms")' <tpp>` | (a) 打印 coulomb `shape=[4]`、descriptor `[8]`(两表分离);引擎侧计数 = `2`(overlap/select 各自取自 `cb`,无一取自 `epd`) |
| 3 | **include_self_interaction=true 语义** | `python - <<'PY'`(见下 b)+ `grep -n "!polar_include_self_interaction" <tpp>` | JSON `True`;引擎只有 `if(!polar_include_self_interaction)` 一个受控分支(减法被跳过);门脚本 `si_counterfactual` 列为大值(证明未减) |
| 4 | **TTT 修正恒 0** | `sed -n '/P3 模块⑥/,/^    }$/p' <tpp> \| grep -cE "2\.837297|MADELUNG\|dipole"` | `0`(⑥ 块无任何修正项算术);门脚本打印 `is_molecule=False is_slab=False`(pbc=[T,T,T],选择器逻辑 energy.py L181-190) |
| 5 | **density_basis 与 descriptor 逐值一致(ctor 断言成立)** | `python - <<'PY'`(见下 c) | `EXACT-EQ True`(全子树,含 cl_scale/expanded_l/相位/permute/radial_spline);引擎侧 `grep -n "coulomb_energy density_basis/kspace_cutoff must" <tpp>` 命中 |
| 6 | **装配序**(scale/shift 之后、未缩放、与 ZBL 同级) | `grep -n "scale nonlinear energies\|add polar coulomb energy\|if(has_zbl) {" <tpp>` | 行号满足 `scale nonlinear energies` < `add polar coulomb energy` < `if(has_zbl) {`(同级相邻);`grep -A4 "add polar coulomb energy" <tpp> \| grep -c energy_scale` = `0`(未乘 scale) |
| 7 | **密度源 = SCF 末态** | `grep -n "polar_scf_final_density" <tpp>` | 命中 3 处:⑤ 写入(`realloc`+`final_density` 拷贝)、⑥ 读取(`auto coulomb_density=polar_scf_final_density;`);且 ⑥ 读取行号 > ⑤ 写入行号 |
| 8 | **作用域**(num_k/volume 可见) | `grep -n "const double volume=std::abs\|const int num_k=static_cast\|// ---- P3 模块⑥" <tpp>` | 三者行号递增且同层(均在同一个 `if(polar_enabled){` 与闭合 `}` 之间;可用 `sed -n` 目检 ⑥ 块尾是 `        }\n    }` 两级闭合) |
| 9 | **hpp 成员齐全** | `grep -n "polar_include_self_interaction\|polar_include_pbc_corrections\|polar_coulomb" $ENGINE/libsymmetrix/source/mace_nonlinear_kokkos.hpp` | 命中:2×bool、si_overlap、si_select(`View<int*>`)、rho_re/im(+storage)、energy(+storage) |
| 10 | **binding 访问器** | `grep -n "polar_coulomb" $ENGINE/symmetrix/source/cpp/mace_nonlinear_kokkos.cpp` | 3 条 `.def`,位于 `polar_scf_final_density` 之后、链尾 `;` 之前 |
| 11 | **门脚本先跑完 SCF** | `grep -n "atoms.get_potential_energy()\|= read_engine(atoms)" $MLIP/bench/polar_module6_gate.py` | `run_case` 内 `get_potential_energy()` 行号 < `read_engine(atoms)` 调用行号(访问器读取在完整前向之后) |
| 12 | **rho 口径**(未乘 (2π)³/V) | `grep -n "未乘 (2π)" $ENGINE/libsymmetrix/source/mace_nonlinear_kokkos.hpp $MLIP/bench/polar_module6_gate.py` | hpp 成员注释、⑥ 块注释(视实现落点)与门脚本换算行(`rho_np*volume/TWO_PI**3`)均命中 |

内嵌 python 检查片段(只读):

```bash
# a+b) SI 表独立 / include_self_interaction 语义
cd $MLIP && /workspace/caches/conda/envs/polar/bin/python - <<'PY'
import json; d=json.load(open("models/mace-polar/polar_symmetrix_probe.json"))
ce=d["polar_blocks"]["coulomb_energy"]; epd=d["polar_blocks"]["electric_potential_descriptor"]
print("coulomb si shapes:", ce["self_interaction_terms"]["overlap_constants"]["shape"],
      ce["self_interaction_terms"]["select_indices"]["shape"])
print("descriptor si shapes:", epd["self_interaction_terms"]["overlap_constants"]["shape"],
      epd["self_interaction_terms"]["select_indices"]["shape"])
print("include_self_interaction:", ce["include_self_interaction"],
      "include_pbc_corrections:", ce["include_pbc_corrections"])
PY
# c) density_basis 逐值一致
cd $MLIP && /workspace/caches/conda/envs/polar/bin/python - <<'PY'
import json; d=json.load(open("models/mace-polar/polar_symmetrix_probe.json"))
pb=d["polar_blocks"]
print("EXACT-EQ", json.dumps(pb["coulomb_energy"]["density_basis"], sort_keys=True)
      == json.dumps(pb["electric_potential_descriptor"]["density_basis"], sort_keys=True))
PY
```

(路径按 `$MLIP` 拼装;上表期望值 = 2026-10-15 实测:coulomb si `[4]/[4]`、descriptor
si `[8]/[8]`、True/True、EXACT-EQ True。)

---

## 6. 门脚本 `bench/polar_module6_gate.py`

配套文件,已落盘(只写不跑)。设计要点:

- **引擎侧**: `Symmetrix(J, dtype="float64")` + `mm.build_gas` 重建 4 夹具盒;
  **先 `atoms.get_potential_energy()` 跑完整前向(含⑤ SCF)**,再读
  `polar_coulomb_energy()/polar_coulomb_rho_re/im()` 与密度源;访问器缺失 → `return 2`。
- **参考侧**: `import polar_module6_preflight as pf`,复用
  `rebuild_density_basis/assemble_density/self_interaction_energy/pbc_correction_energy`
  与常数;能量链逐式 = 预检(rel ≤1.2e-15 已锁定)。
- **密度模式**(`--density-mode auto|scf-final|post-fukui`,auto = 探测
  `polar_scf_final_density` 访问器):`scf-final` 对 `index.json` 端到端目标;
  `post-fukui`(过渡,用② 后密度)对预检 `E_es(step0)` 过渡值,判据 `abs ≤1e-6 eV`。
- **判据**: ① ρ̂ rel ≤1e-6;② `|E_engine − E_numpy(引擎密度)|` rel ≤1e-6 且 abs ≤5e-6 eV;
  ③ 夹具端到端同②(scf-final)。活体帧(可选 `--live k`)仅 ①②。
- **诊断**: SI 反事实列(证明 `include_self_interaction=true` ⇒ 未减)、TTT 选择器
  (`is_molecule/is_slab=False`)、两种密度链的 ΔE 并列(便于发现密度源接错)。
- 产物:`bench/polar_module6_gate.{png,json}`;退出码 0/1/2。

用法(不在本任务跑):
```bash
cd /workspace/group/shared/bza/task3_sampling_database/mlip && CUDA_VISIBLE_DEVICES=<gpu> \
  /workspace/caches/conda/envs/polar/bin/python bench/polar_module6_gate.py --live 1
```

---

## 7. 与 `module6_patch_draft.md` 的偏差表(精化时发现)

| # | 草案 | 本提案 | 原因 |
|---|---|---|---|
| 1 | hpp:`polar_coulomb_si_select` 声明在 `Kokkos::View<Precision*>` 组 | 独立 `Kokkos::View<int*>` | ctor 用 `tensor_values<int>` 解析(select_indices 为整型);与既有 `polar_si_select` 同型。草案写法不能编译 |
| 2 | tpp(b):`Precision e_coulomb` + parallel_reduce 用 `Precision&` 归约;装配 `node_energies(0)+=static_cast<double>(e_coulomb)`(host 直接写视图) | 标量归约容器 `double`;能量落盘/装配均走 kernel(device 端) | CUDA 后端 host 不得直接访存 device 视图;库内既有标量 reduce 均为 int/long long/double |
| 3 | `field_constant` 提升为文件级 `constexpr`(改④ 既有行) | C1 自包含 `coulomb_field_constant`(不改③④) | 最小化与⑤ 落码热点冲突;数值等价(同值同源) |
| 4 | ctor 锚点 "L3425 `polar_recursion_steps=…` 之后";hpp 锚点 "`polar_potential_features` 声明后";binding 锚 "L536 之后" | 锚点全部顺延到 **⑤ 块之后**(ctor:charges_maps 校验后;hpp:⑤ 视图块尾;binding:`polar_scf_final_density` 访问器后) | ⑤ 在撰写期间已落码(09:04–09:11),锚点漂移;文本锚点不随行号漂 |
| 5 | ⑤ 未落地 ⇒ ⑥ 用② 后密度过渡 | 主方案直接 `polar_scf_final_density`(⑤ 已落地);过渡方案降级为 §3.4 备选 + 门脚本 `post-fukui` 模式 | 以读取到的最新引擎状态为准;两级口径均保留 |
| 6 | density_basis 一致性 "逐值比较后复用 polar_tab_density_*" | ctor 断言 = JSON 子树递归相等(`cb.at("density_basis")!=epd.at("density_basis")` 抛错)+ kspace_cutoff 相等;表仍复用③ 视图 | 实测两子树逐值相等;递归断言更完整且与④ 侧表脱耦 |
| 7 | rho 诊断口径未写明 | hpp/绑定注释与门脚本明确 "未乘 (2π)³/V"(numpy 对照需 ×V/(2π)³) | 预检 `assemble_density` 含 (2π)³/V,引擎 ρ̂ 不含;不注明易错判 |

数值语义(公式、表、SI 减号、TTT 恒 0、装配未缩放、单图 node-0 口径)与草案一致,
未改动;`bench/polar_module6_preflight.py` 仍为唯一参考实现。

**依据账本续记**(接计划 §8 第 15 条):16 = 本文 §3 实现形态细化(CUDA 写路径
[库内 zbl/readout 先例],double 标量归约 [库内 validate/reduce_stress 先例]);
17 = §2 一致性断言(extract_polar_blocks.py `_gto_energy` 与 `_gto_feature` 同源
`_gto_basis`;实测 JSON 子树相等)。
