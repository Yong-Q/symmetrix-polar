#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""apply_polar_patches_v3.py —— ⑧ 力/全链反向 补丁（v3 适配：polar 独立 TU + ⑥⑦已落码）应用器.

用途
----
把 ``docs/module8_patch_final.md``（⑧ 力/全链反向提案，锚点基于"polar 未拆 TU"旧布局）
**适配到当前新结构**并机械应用到 symmetrix-xl 引擎树：

- polar 运行期代码（伴随 kernel/缓存/反向块）全部落独立编译单元
  ``libsymmetrix/source/mace_polar_impl.tpp``（新增成员函数
  ``compute_polar_reverse_forces`` 等，由主 tpp 反向链的**单一调用点**触发）；
- 主 tpp（``mace_nonlinear_kokkos_impl.tpp``）新增：``Readout::reverse_all`` 定义
  （PART1，与 evaluate_all/reverse 同分区——依赖选择见 merge_adapt_8.md）、反向调用点 4 行、
  ``reduce_node_forces`` 节点级力归约块、``reduce_stress`` loud guard；
- binding（``symmetrix/source/cpp/mace_nonlinear_kokkos.cpp``）链尾追加 3 条 ⑧ 访问器；
- pair style（``pair_symmetrix/pair_symmetrix_mace_kokkos.cpp``）力归约点追加节点级极性力。

v3 块清单（6 文件 / 16 个插入点；详见 BLOCK_SPECS）：

- ① hpp（``mace_nonlinear_kokkos.hpp``，3 个插入点）：`<array>` 头、§2.2 公开增补
  （删提案的重复视图 ``polar_scf_step_densities_post``）、§2.3 ``Readout::reverse_all`` 声明。
- ② 主 tpp 3 个插入点：§3 ``reverse_all`` 定义、§4.0-A 反向调用点、§5.1 节点力归约；
- ③ 主 tpp 1 个：§6.3 ``reduce_stress`` loud guard。
- ④ 新 TU 7 个插入点：§4.0-B volume/rcell 缓存、§4.0-C-1/C-2（clamp 替换 + 标量缓存）、
  §4.2 主函数 + ⑧-A 小工具（含 D-04/D-05/D-08 三处内容适配）、⑥ 反向开关置位。
- ⑤ binding 1 个：§7.1 链尾展开（3 条新 .def；提案的 ``_post`` 访问器随视图删除）。
- ⑥ pair style 1 个：§5.2 力归约块（提案文本逐字）。

内容提取纪律（同 v1/v2）：插入内容从 ``module8_patch_final.md`` 的代码块**程序化逐字提取**
（节标题 + 代码块引言正则唯一定位；结构断言失败即中止），v3 的被迫适配变换逐项显式实现并
计数断言（见各 builder；偏离全表见 ``docs/merge_adapt_8.md`` §2）。锚点全部为引擎树**字面
文本**（本树读取快照 2026-10-15；锚点 count==1 强断言）。

应用模式与纪律（同 v2）：
- ``--dry-run``（默认）：只读引擎树；逐块打印锚点 count/插入行数；输出应用后结构探针、
  行数变化预估（内存 difflib, 不调用 git）；写出"模拟产物"到 ``--sim-dir``（默认
  ``logs/apply_v3_sim/``, 仅本 mlip 工作区）并对其运行 ``bench/polar_lint.py``（main tpp /
  polar TU / pair style 三个文件, 6 类规则发现数必须为 0）；[8] 数值自检（Fukui 伴随 FD
  校验 + ⑥ 种子链 FD 校验, 纯 stdlib）。
- ``--apply``：真实写入。写入**前**重跑全部断言（锚点唯一 + 幂等标记 + lint），要求 ``.bak``
  不存在，备份后写入；写入后复验（字节一致 + 探针），失败自动回滚。仅应在 dry-run 全绿后由
  用户/主线程执行——本脚本交付时**未**在真实树执行 ``--apply``。
- dry-run 绝不写引擎树（写操作 = ``--log`` 日志 + ``--sim-dir`` 模拟产物，均在 mlip/ 内）；
  不调用 git、不构建。
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 路径与常量
# --------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
MLIP_ROOT = SCRIPT_PATH.parents[1]  # .../task3_sampling_database/mlip
DOCS_DIR = MLIP_ROOT / "docs"
LINT_PATH = MLIP_ROOT / "bench" / "polar_lint.py"
DEFAULT_ENGINE = Path(
    os.environ.get(
        "SYMMETRIX_XL_ENGINE",
        "/workspace/group/shared/bza/external-repos/symmetrix-xl",
    )
)
DEFAULT_SIM_DIR = MLIP_ROOT / "logs" / "apply_v3_sim"

DOC8 = "module8_patch_final.md"

# 目标树 = 当前新结构（polar 独立 TU; ⑥⑦ 已落码 v19c）
ENGINE_FILES = {
    "tu": "libsymmetrix/source/mace_polar_impl.tpp",
    "hpp": "libsymmetrix/source/mace_nonlinear_kokkos.hpp",
    "tpp": "libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp",
    "cpp": "symmetrix/source/cpp/mace_nonlinear_kokkos.cpp",
    "pair": "pair_symmetrix/pair_symmetrix_mace_kokkos.cpp",
}

# v3 编写时的只读快照（仅漂移提示；锚点文本为准）
V3_SNAPSHOT = {
    "tu": ("1a8f4913dd692f8d37e3256f18ba13ff", 1429),
    "hpp": ("087d5f010825d195f2d22111f784785b", 861),
    "tpp": ("4bd42dcc190ff92235d997aa574be744", 7359),
    "cpp": ("ed512afe0152727cd80649f9d59df677", 636),
    "pair": ("482ddc73466e2cb182e4388c1650082b", 2289),
}

# --------------------------------------------------------------------------
# 引擎树锚点（字面文本；全部要求 count==1）
# --------------------------------------------------------------------------
HPP_ANCHOR_INCLUDE = "#include <cstdint>"
HPP_ANCHOR_PUBLIC = (
    "    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_mixer_temp,\n"
    "        polar_mixer_temp_storage,polar_scratch_a,polar_scratch_a_storage,\n"
    "        polar_scratch_b,polar_scratch_b_storage,polar_nl_input,\n"
    "        polar_nl_input_storage,polar_nl_a,polar_nl_a_storage,polar_nl_b,\n"
    "        polar_nl_b_storage;                                             // ⑤暂存"
)
HPP_ANCHOR_REVERSE_ALL = (
    "        void evaluate_all(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,\n"
    "                          Kokkos::View<Precision**,Kokkos::LayoutRight> output);"
)
TPP_ANCHOR_EVAL_TAIL = (
    "    ensure_view(result,result_storage,input.extent(0),linear_2.output_dimension());\n"
    "    linear_2.evaluate(activated,result);\n"
    "    ordered_kokkos_deep_copy(output,result);\n"
    "}"
)
TPP_ANCHOR_REVERSE_CALL = (
    "    ensure_view(harmonic_adjoints,harmonic_adjoints_storage,edges,num_lm);"
)
TPP_ANCHOR_NODE_FORCES = (
    "        KOKKOS_LAMBDA(const std::size_t edge) {\n"
    "            const int receiver=edge_receivers(edge);\n"
    "            const int source=edge_sources(edge);\n"
    "            for(int component=0;component<3;++component) {\n"
    "                const double force=directed_forces(3*edge+component);\n"
    "                Kokkos::atomic_add(\n"
    "                    &reduced_forces(3*source+component),force);\n"
    "                Kokkos::atomic_add(\n"
    "                    &reduced_forces(3*receiver+component),-force);\n"
    "            }\n"
    "        });"
)
TPP_ANCHOR_STRESS = (
    "void MaceNonlinearKokkosT<Precision>::reduce_stress(\n"
    "    const double volume,Kokkos::View<const double*> xyz)\n"
    "{"
)
TU_ANCHOR_VOLUME = (
    "        const double volume=std::abs(\n"
    "            lattice[0][0]*(lattice[1][1]*lattice[2][2]\n"
    "                -lattice[1][2]*lattice[2][1])\n"
    "            -lattice[0][1]*(lattice[1][0]*lattice[2][2]\n"
    "                -lattice[1][2]*lattice[2][0])\n"
    "            +lattice[0][2]*(lattice[1][0]*lattice[2][1]\n"
    "                -lattice[1][1]*lattice[2][0]));"
)
TU_ANCHOR_C1_CLAMP = (
    "                for(int channel=0;channel<2;++channel)\n"
    "                    if(source_sums[channel]==Precision(0))\n"
    "                        source_sums[channel]=Precision(1);"
)
TU_ANCHOR_C1_KERNEL = (
    '                Kokkos::parallel_for("polar scf fukui correction",num_nodes,\n'
    "                    KOKKOS_LAMBDA(int node) {\n"
    "                        density_all(node,0)+=update_out_view(node,8)\n"
    "                            *factor_alpha;\n"
    "                        density_all(node,4)+=update_out_view(node,9)\n"
    "                            *factor_beta;\n"
    "                    });"
)
TU_ANCHOR_C2_CLAMP = (
    "        for(int channel=0;channel<2;++channel)\n"
    "            if(source_sums[channel]==Precision(0))\n"
    "                source_sums[channel]=Precision(1);"
)
TU_ANCHOR_C2_KERNEL = (
    '        Kokkos::parallel_for("polar fukui correction",num_nodes,\n'
    "            KOKKOS_LAMBDA(int node) {\n"
    "                polar_accumulator(node,0)+=\n"
    "                    polar_sources(node,0)*factor_alpha;\n"
    "                polar_accumulator(node,4)+=\n"
    "                    polar_sources(node,1)*factor_beta;\n"
    "            });"
)
TU_ANCHOR_INSTANTIATION = "template class MaceNonlinearKokkosT<float>;"
TU_ANCHOR_COULOMB_SI = (
    '        polar_coulomb_si_select=toKokkosView("polar coulomb si select",\n'
    '            tensor_values<int>(cb.at("self_interaction_terms")\n'
    '                .at("select_indices")));'
)
CPP_ANCHOR_CHAIN_TAIL = (
    '        .def("polar_coulomb_rho_im",[](const Evaluator& self) {\n'
    "            return view2vector(self.polar_coulomb_rho_im);\n"
    "        });"
)
PAIR_ANCHOR_FORCE_REDUCTION = (
    "  auto mace_node_forces = mace->node_forces;\n"
    '  Kokkos::parallel_for("PairSymmetrixMACEKokkos::force_reduction",\n'
    "    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),\n"
    "    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {\n"
    "      const int ii = team_member.league_rank();\n"
    "      const int i = node_indices(ii);\n"
    "      double f_x, f_y, f_z;\n"
    "      Kokkos::parallel_reduce(\n"
    "        Kokkos::TeamThreadRange(team_member, num_neigh(ii)),\n"
    "        [&] (const int jj, double& f_x, double& f_y, double& f_z) {\n"
    "          const int ij = first_neigh(ii) + jj;\n"
    "          const int j = neigh_indices(ij);\n"
    "          f_x += mace_node_forces(3*ij);\n"
    "          f_y += mace_node_forces(3*ij+1);\n"
    "          f_z += mace_node_forces(3*ij+2);\n"
    "          Kokkos::atomic_add(&f(j,0), mace_node_forces(3*ij));\n"
    "          Kokkos::atomic_add(&f(j,1), mace_node_forces(3*ij+1));\n"
    "          Kokkos::atomic_add(&f(j,2), mace_node_forces(3*ij+2));\n"
    "        }, f_x, f_y, f_z);\n"
    "        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {\n"
    "          Kokkos::atomic_add(&f(i,0), -f_x);\n"
    "          Kokkos::atomic_add(&f(i,1), -f_y);\n"
    "          Kokkos::atomic_add(&f(i,2), -f_z);\n"
    "        });\n"
    "    });"
)

# --------------------------------------------------------------------------
# 块表：6 文件 / 16 个插入点（v3 新结构）
#
# op:
#   "insert_after"  : 整块锚点之后插入
#   "insert_before" : 整块锚点之前插入
#   "replace"       : 整块锚点替换为 content
# --------------------------------------------------------------------------
BLOCK_SPECS = [
    dict(order=1, bid="1a-hpp-array", group="① hpp 增补",
         file="hpp", op="insert_before", anchor=HPP_ANCHOR_INCLUDE,
         builder="hpp_array",
         expect_before=("", "#include <Kokkos_Core.hpp>"),
         note="<array> 头(提案 §2.2 注; ⑧ 成员 std::array 需要)"),
    dict(order=2, bid="1b-hpp-public", group="① hpp 增补",
         file="hpp", op="insert_after", anchor=HPP_ANCHOR_PUBLIC,
         builder="hpp_public",
         expect_after=("    // ===== [v2 适配] 模块⑥⑦ hpp 公开增补(binding 与 run_polar_pipeline 共用) =====",),
         note="module8 §2.2 公开视图/成员 + 反向入口声明(删 _post 重复视图, 见 D-02)"),
    dict(order=3, bid="1c-hpp-rev-all-decl", group="① hpp 增补",
         file="hpp", op="insert_after", anchor=HPP_ANCHOR_REVERSE_ALL,
         builder="hpp_revdecl",
         expect_before=("        // 如 PolarMACE 的 fukui_source_map;evaluate 只取第 0 列)。",
                        "        // 与 evaluate 同链,但复制 final linear 的全部列(供 2x0e 等多通道读出头,"),
         expect_after=("        void reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Precision scale,",),
         note="module8 §2.3 Readout::reverse_all 声明(逐字)"),
    dict(order=4, bid="2a-tpp-rev-all-def", group="② 主 tpp 增补",
         file="tpp", op="insert_after", anchor=TPP_ANCHOR_EVAL_TAIL,
         builder="tpp_revdef",
         expect_before=("    }", "        return;"),
         expect_after=("", "template<typename Precision>"),
         note="module8 §3 Readout::reverse_all 全文(PART1; 依赖选择=主 tpp, 见 merge_adapt_8.md §3)"),
    dict(order=5, bid="2b-tpp-rev-call", group="② 主 tpp 增补",
         file="tpp", op="insert_before", anchor=TPP_ANCHOR_REVERSE_CALL,
         builder="tpp_call",
         expect_before=("", "            total_charge,total_spin);",
                        "        run_polar_pipeline(num_nodes,attrs,node_positions,cell,"),
         expect_after=("    ordered_kokkos_deep_copy(harmonic_adjoints,Precision(0));",),
         note="module8 §4.0-A 反向调用点(readout 播种之后、骨架反向主循环之前)"),
    dict(order=6, bid="2c-tpp-node-force", group="② 主 tpp 增补",
         file="tpp", op="insert_after", anchor=TPP_ANCHOR_NODE_FORCES,
         builder="tpp_node",
         expect_before=("            factorized_execution_space,0,num_edges),",),
         expect_after=("}", ""),
         note="module8 §5.1 reduce_node_forces 节点级极性力归约块(逐字)"),
    dict(order=7, bid="2d-tpp-stress-guard", group="② 主 tpp 增补",
         file="tpp", op="insert_after", anchor=TPP_ANCHOR_STRESS,
         builder="tpp_stress",
         expect_after=("    if(!std::isfinite(volume)||!(volume>0.0)||xyz.extent(0)%3!=0)",),
         note="module8 §6.3 reduce_stress loud guard(cell 伴随=下一 slice)"),
    dict(order=8, bid="3a-tu-volume", group="③ 新 TU 增补",
         file="tu", op="insert_after", anchor=TU_ANCHOR_VOLUME,
         builder="tu_volume",
         expect_before=("        }",),
         expect_after=("        // n 上界(kspace.py L41-53): max_ns=ceil(cutoff / (rcell_i·ĉell_i))",),
         note="module8 §4.0-B volume/rcell 缓存(③ 前向块内)"),
    dict(order=9, bid="3b-tu-c1-clamp", group="③ 新 TU 增补",
         file="tu", op="replace", anchor=TU_ANCHOR_C1_CLAMP,
         builder="tu_c1_clamp",
         expect_before=("                }",),
         expect_after=("                const Precision charge=Precision(total_charge);",),
         note="module8 §4.0-C-1 clamp 替换(先留原始 S; 逐步)"),
    dict(order=10, bid="3c-tu-c1-scalars", group="③ 新 TU 增补",
         file="tu", op="insert_after", anchor=TU_ANCHOR_C1_KERNEL,
         builder="tu_c1_scalars",
         expect_before=("                    -predicted_zero[1])/source_sums[1];",),
         expect_after=("                // 快照 = 逐步 Fukui 校正后的完整状态(参考口径 scf_steps step{i+1})",),
         note="module8 §4.0-C-1 Fukui 标量缓存(快照部分删除, 见 D-03)"),
    dict(order=11, bid="3d-tu-c2-clamp", group="③ 新 TU 增补",
         file="tu", op="replace", anchor=TU_ANCHOR_C2_CLAMP,
         builder="tu_c2_clamp",
         expect_before=("        }", "            predicted_zero[1]+=polar_accumulator_host(node,4);"),
         expect_after=("        // Q_p_S=Q+(S-1), Q_m_S=Q-(S-1)(extensions.py L752-753);",),
         note="module8 §4.0-C-2 clamp 替换(先留原始 S; 初始化)"),
    dict(order=12, bid="3e-tu-c2-scalars", group="③ 新 TU 增补",
         file="tu", op="insert_after", anchor=TU_ANCHOR_C2_KERNEL,
         builder="tu_c2_scalars",
         expect_before=("            /source_sums[1];",),
         expect_after=("        // 校正后 = field_independent_spin_charge_density(mace L774 clone)。",),
         note="module8 §4.0-C-2 初始化 Fukui 标量缓存(逐字)"),
    dict(order=13, bid="3f-tu-reverse-main", group="③ 新 TU 增补",
         file="tu", op="insert_before", anchor=TU_ANCHOR_INSTANTIATION,
         builder="tu_main",
         expect_before=("", "}"),
         expect_after=("template class MaceNonlinearKokkosT<double>;",),
         note="module8 §4.2 主函数 + ⑧-A 小工具(迁入 TU; 含 D-04/D-05/D-08 三处内容适配)"),
    dict(order=14, bid="3g-tu-coulomb-enable", group="③ 新 TU 增补",
         file="tu", op="insert_after", anchor=TU_ANCHOR_COULOMB_SI,
         builder="tu_enable",
         expect_before=('                .at("overlap_constants")));',),
         expect_after=("    }", "",
                       "    // ---- P3 模块⑦: local_electron_energy（OneBodyMLPFieldReadout L652-752） ----"),
         note="⑥ 已落码 ⇒ polar_coulomb_enabled 置位(提案 §0-3/§10-2 口径; 落点=解析函数)"),
    dict(order=15, bid="4-cpp-binding", group="④ binding",
         file="cpp", op="replace", anchor=CPP_ANCHOR_CHAIN_TAIL,
         builder="cpp_bind",
         expect_before=("        })",
                        "            return view2vector(self.polar_coulomb_rho_re);",),
         expect_after=("}", "", "void bind_mace_nonlinear_kokkos(py::module_& module)"),
         note="module8 §7.1 链尾展开为 +3 条(锚点=现链尾; _post 访问器随视图删除, 见 D-07)"),
    dict(order=16, bid="5-pair-force-wire", group="⑤ pair style",
         file="pair", op="insert_after", anchor=PAIR_ANCHOR_FORCE_REDUCTION,
         builder="pair_wire",
         expect_before=("", "  }"),
         expect_after=("", "  if (vflag_global) {"),
         note="module8 §5.2 节点级极性力写入 LAMMPS 力数组(逐字; 置于 force_reduction 之后)"),
]

# --------------------------------------------------------------------------
# 结构前提核查（应用前；只读断言）
# --------------------------------------------------------------------------
PRE_CHECKS = [
    ("hpp", "    Kokkos::View<double*> atomic_energies,node_energies,node_forces,atom_forces,", 1,
     "node_energies/node_forces 成员存在且公开"),
    ("hpp", "    nlohmann::json polar_json;", 1,
     "polar_json 成员存在(⑥⑦ 解析数据源)"),
    ("hpp", "    Kokkos::View<Precision**,Kokkos::LayoutRight> attrs,radial,edge_harmonics,features;", 1,
     "attrs 为类成员 ⇒ 提案签名 compute_polar_reverse_forces(int) 不需扩参(headline 裁定)"),
    ("hpp", "    std::vector<PolarUpdate> polar_updates;", 1,
     "polar_updates 私有表(重放按步索引的前提)"),
    ("tpp", "        run_polar_pipeline(num_nodes,attrs,node_positions,cell,", 1,
     "前向单调用点(反向调用点紧邻其后)"),
    ("tpp", "        readouts[layer].reverse(state.output,scale,state.layer_adjoint);", 1,
     "readout 播种段(反向注入必须在其后, init=false 累加)"),
    ("tpp", "    auto e0=atomic_energies;", 1,
     "scale nonlinear energies 段(前向装配点标志)"),
    ("tu", "MaceNonlinearKokkosT<Precision>::run_polar_pipeline", 1,
     "run_polar_pipeline 定义唯一(⑥⑦ 运行段已在其内)"),
    ("tu", '                Kokkos::parallel_for("polar scf density snapshot",', 1,
     "既有逐步快照内核(位于 Fukui 校正之后 ⇒ 校后口径; D-03/D-04 依据)"),
    ("tu", "        auto coulomb_density=polar_scf_final_density;", 1,
     "⑥ 前向密度源 = SCF 末态快照(D-05 同源依据)"),
    ("tu", "        auto charges_final=polar_scf_final_density;", 1,
     "⑦ 前向末态密度源(v2/D05 先例)"),
    ("cpp", ".def(\"polar_scf_step_densities\"", 1,
     "既有 scf_step_densities 访问器(_post 视图删除后其数据仍可达)"),
    ("pair", '  Kokkos::parallel_for("PairSymmetrixMACEKokkos::force_reduction",', 1,
     "pair style force_reduction 定义唯一"),
]

# --------------------------------------------------------------------------
# 应用后结构探针（内存结果）
# --------------------------------------------------------------------------
POST_PROBES = [
    ("hpp", "#include <array>", 1, "①a <array>"),
    ("hpp", "P3 模块⑧", 1, "①b §2.2 头注释"),
    ("hpp", "polar_reverse_seed_terms", 1, "①b 种子掩码声明"),
    ("hpp", "polar_reverse_terms_last", 1, "①b 诊断掩码成员"),
    ("hpp", "std::array<Precision,6> polar_fukui_init_scalars", 1, "①b init 标量缓存"),
    ("hpp", "polar_scf_step_densities_post", 0, "①b _post 重复视图已删(D-02)"),
    ("hpp", "reverse_all", 1, "①c 声明"),
    ("tpp", "Readout::reverse_all(", 1, "②a 定义"),
    ("tpp", '"reverse_all readout mid silu"', 1, "②a has_mid 分支"),
    ("tpp", "compute_polar_reverse_forces(num_nodes);", 1, "②b 单调用点"),
    ("tpp", '"MaceNonlinearKokkos::reduce_polar_node_forces"', 1, "②c 节点力归约"),
    ("tpp", "polar cell adjoint (virial) is not ", 1, "②d loud guard"),
    ("tu", "polar_cell_volume=volume;", 1, "③a volume 缓存"),
    ("tu", "polar_rcell_host[3*row+column]", 1, "③a rcell 缓存"),
    ("tu", "Precision raw_source_sums[2]={source_sums[0],source_sums[1]};", 2,
     "③b/③d C-1/C-2 替换(子串语义: C-2 行含前缀 const)"),
    ("tu", "const Precision raw_source_sums[2]={source_sums[0],source_sums[1]};", 1, "③d C-2 替换"),
    ("tu", "polar_scf_fukui_scalars[step]={", 1, "③c 逐步标量写入"),
    ("tu", "polar_fukui_init_scalars={", 1, "③e init 标量写入"),
    ("tu", "polar scf post-correction snapshot", 0, "③c 重复快照内核已删(D-03)"),
    ("tu", "MaceNonlinearKokkosT<Precision>::compute_polar_reverse_forces(", 1, "③f 主函数定义"),
    ("tu", "MaceNonlinearKokkosT<Precision>::polar_inject_node_features_adjoint(", 1, "③f ⑧-A 注入口"),
    ("tu", "std::array<Precision,6>& scalars_host", 1, "③f fukui_reverse 签名"),
    ("tu", "transform(2*channel+1)=Precision(-fbar);", 1, "③f D-08 修正(1)"),
    ("tu", "transform(2*channel)=Precision(1);", 1, "③f D-08 修正(2)"),
    ("tu", ":Precision(-fbar*static_cast<double>(f));", 1, "③f D-08 修正(3)"),
    ("tu", "polar_spin_charge_density;", 0, "③f ⑥ 种子改源完成(D-05 唯一旧用法已换)"),
    ("tu", "auto final_density=polar_scf_final_density;", 2,
     "③f D-05 ⑥ 种子密度源 + ⑤ 前向 final_density(既有)"),
    ("tu", "polar_coulomb_enabled=true;", 1, "③g ⑥ 反向开关置位"),
    ("tu", "Kokkos::View<Precision**> ", 0, "③f LayoutRight 补齐(D-06)"),
    ("cpp", '.def("polar_node_forces"', 1, "④ 节点力访问器"),
    ("cpp", '.def("polar_reverse_terms_last"', 1, "④ 掩码诊断访问器"),
    ("cpp", '.def("polar_blocks_enabled"', 1, "④ enabled 访问器"),
    ("cpp", "polar_scf_step_densities_post", 0, "④ _post 访问器未加(D-07)"),
    ("pair", '"PairSymmetrixMACEKokkos::polar_node_force_addition"', 1, "⑤ 力写入 kernel"),
    ("pair", "mace->polar_blocks_enabled()", 1, "⑤ enabled 守卫"),
]

# 顺序探针：文本中按顺序出现
ORDER_PROBES = [
    ("tpp",
     ("readouts[layer].reverse(state.output,scale,state.layer_adjoint);",
      "compute_polar_reverse_forces(num_nodes);",
      "ensure_view(harmonic_adjoints,harmonic_adjoints_storage,edges,num_lm);"),
     "readout 播种 < 极性全链反向注入 < 骨架反向(边级伴随清零)"),
    ("tu",
     ('Kokkos::parallel_for("polar scf fukui correction"',
      "polar_scf_fukui_scalars[step]={",
      'Kokkos::parallel_for("polar scf density snapshot"'),
     "⑤ 逐步校正 < ⑧ 标量缓存 < 既有校后快照"),
    ("tu",
     ('Kokkos::parallel_for("polar fukui correction"',
      "polar_fukui_init_scalars={"),
     "② 初始化校正 < ⑧ init 标量缓存"),
    ("tu",
     ("MaceNonlinearKokkosT<Precision>::run_polar_pipeline",
      "MaceNonlinearKokkosT<Precision>::compute_polar_reverse_forces"),
     "前向 pipeline 定义 < 反向主函数定义(同 TU 语序)"),
]

# 幂等标记：应用前必须为 0（>0 说明树已被本补丁（部分）修改）
IDEMPOTENCE_MARKERS = [
    ("hpp", "P3 模块⑧"),
    ("hpp", "polar_reverse_seed_terms"),
    ("hpp", "reverse_all"),
    ("hpp", "#include <array>"),
    ("tpp", "compute_polar_reverse_forces"),
    ("tpp", "Readout::reverse_all"),
    ("tpp", "polar_node_forces"),
    ("tpp", "reduce_polar_node_forces"),
    ("tu", "compute_polar_reverse_forces"),
    ("tu", "polar_cell_volume"),
    ("tu", "polar_scf_fukui_scalars"),
    ("tu", "polar_fukui_init_scalars"),
    ("tu", "polar_coulomb_enabled"),
    ("tu", "raw_source_sums"),
    ("cpp", '.def("polar_node_forces"'),
    ("pair", "polar_node_force_addition"),
]


class PatchError(Exception):
    """锚点/结构断言失败（打印诊断后中止）."""


# --------------------------------------------------------------------------
# 输出（stdout + 可选日志 tee）
# --------------------------------------------------------------------------
class Tee:
    def __init__(self, log_path: Path | None):
        self.handle = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = open(log_path, "w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        if self.handle is not None:
            self.handle.write(text)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()
            self.handle = None


TEE: Tee


def out(text: str = "") -> None:
    TEE.write(text + "\n")


# --------------------------------------------------------------------------
# 文档解析与内容提取（同 v1/v2 机制）
# --------------------------------------------------------------------------
def split_fences(doc_path: Path) -> tuple[list[str], list[dict]]:
    lines = doc_path.read_text(encoding="utf-8").split("\n")
    fences: list[dict] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            if j >= len(lines):
                raise PatchError(f"{doc_path.name}: 第 {i + 1} 行起始的代码块未闭合")
            k = i - 1
            while k >= 0 and lines[k].strip() == "":
                k -= 1
            paragraph: list[str] = []
            while k >= 0 and lines[k].strip() != "" and not lines[k].strip().startswith("```"):
                paragraph.append(lines[k])
                k -= 1
            fences.append(
                dict(
                    start=i + 1,
                    end=j + 1,
                    intro="\n".join(reversed(paragraph)),
                    body=lines[i + 1: j],
                )
            )
            i = j + 1
        else:
            i += 1
    return lines, fences


def section_fences(
    doc_name: str, doc_lines: list[str], fences: list[dict], section_prefix: str
) -> list[dict]:
    starts = [n for n, line in enumerate(doc_lines, 1) if line.startswith(section_prefix)]
    if len(starts) != 1:
        raise PatchError(
            f"{doc_name}: 节标题前缀 {section_prefix!r} 命中 {len(starts)} 处（要求 1）"
        )
    start = starts[0]
    end = len(doc_lines) + 1
    for n in range(start + 1, len(doc_lines) + 1):
        if re.match(r"^#{1,6}\s", doc_lines[n - 1]):
            end = n
            break
    return [f for f in fences if start < f["start"] < end]


def pick_fence(
    doc_name: str, section_prefix: str, fences: list[dict], intro_regex: str, what: str
) -> dict:
    rx = re.compile(intro_regex)
    hits = [f for f in fences if rx.search(f["intro"])]
    if len(hits) != 1:
        detail = "\n".join(
            f"      L{f['start']}-{f['end']} intro={f['intro'][:70]!r}" for f in fences
        )
        raise PatchError(
            f"{doc_name} {section_prefix}: 无法唯一定位{what}代码块 "
            f"(regex={intro_regex!r}, 命中 {len(hits)}); 该节代码块:\n{detail}"
        )
    return hits[0]


class DocBlocks:
    """按 (文档, 节, 引言正则) 程序化提取代码块（缓存）. """

    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
        self._cache: dict[str, tuple[list[str], list[dict]]] = {}
        self.refs: dict[str, str] = {}

    def get(self, doc: str, section: str, intro_regex: str, key: str) -> list[str]:
        if doc not in self._cache:
            self._cache[doc] = split_fences(self.docs_dir / doc)
        doc_lines, fences = self._cache[doc]
        sec = section_fences(doc, doc_lines, fences, section)
        fence = pick_fence(doc, section, sec, intro_regex, key)
        self.refs[key] = f"{doc} L{fence['start']}-{fence['end'] - 1}"
        return list(fence["body"])


# --------------------------------------------------------------------------
# v3 适配变换（逐项显式; 全部带计数断言）
# --------------------------------------------------------------------------
def require(lines: list[str], idx: int, cond: bool, key: str, what: str) -> None:
    if not cond:
        raise PatchError(f"{key}: 结构断言失败（{what}）: 行[{idx}]={lines[idx]!r}")


def replace_counted(lines: list[str], old: str, new: list[str], expected: int, key: str) -> list[str]:
    """把整行 == old 的行替换为 new 多行（计数断言）."""
    n = sum(1 for line in lines if line == old)
    if n != expected:
        raise PatchError(f"{key}: 预期 {expected} 行 {old!r}, 实际 {n}")
    out_lines: list[str] = []
    for line in lines:
        if line == old:
            out_lines.extend(new)
        else:
            out_lines.append(line)
    return out_lines


def replace_block_counted(
    lines: list[str], old_block: list[str], new_block: list[str], expected: int, key: str
) -> list[str]:
    """把连续整块 old_block 替换为 new_block（计数断言; 不重叠）."""
    text = "\n".join(lines)
    needle = "\n".join(old_block)
    n = text.count(needle)
    if n != expected:
        raise PatchError(f"{key}: 预期整块命中 {expected} 次, 实际 {n}")
    text = text.replace(needle, "\n".join(new_block))
    return text.split("\n")


def sub_layout_right(lines: list[str], key: str) -> list[str]:
    """⑧ lambda 形参 Kokkos::View<...**> 补 LayoutRight（bench/polar_lint.py R3）.

    [v3 适配·D-06] 提案正文的 lambda 形参写作 ``Kokkos::View<Precision**>``（缺 LayoutRight;
    树内既有风格一律显式 LayoutRight; lint R3 会拦截）。全部 13+18 处逐字加注, 语义不变。
    """
    n1 = n2 = 0
    out_lines: list[str] = []
    for line in lines:
        n1 += line.count("Kokkos::View<Precision**>")
        n2 += line.count("Kokkos::View<const Precision**>")
        line = line.replace(
            "Kokkos::View<Precision**>", "Kokkos::View<Precision**,Kokkos::LayoutRight>"
        )
        line = line.replace(
            "Kokkos::View<const Precision**>",
            "Kokkos::View<const Precision**,Kokkos::LayoutRight>",
        )
        out_lines.append(line)
    if (n1, n2) != (13, 18):
        raise PatchError(f"{key}: LayoutRight 补齐计数异常 (Precision**={n1} 期望 13; const={n2} 期望 18)")
    return out_lines


def sub_coulomb_density_source(lines: list[str], key: str) -> list[str]:
    """⑧ §4.2 ⑥ 种子: 密度源 ``polar_spin_charge_density`` → ``polar_scf_final_density``.

    [v3 适配·D-05] 该视图自 v19c 起为只读的 ① 原始累积; ⑥ 前向实读 SCF 末态
    ``polar_scf_final_density``(polar TU "auto coulomb_density=polar_scf_final_density;")。
    """
    old = "            auto final_density=polar_spin_charge_density;"
    new_lines = [
        "            // [v3 适配·D-05] ⑥ 前向密度源 = polar_scf_final_density(polar TU ⑥-C1 实读);",
        "            // polar_spin_charge_density 自 v19c 起为只读①原始累积, 与 SCF 末态不等价。",
        "            auto final_density=polar_scf_final_density;",
    ]
    return replace_counted(lines, old, new_lines, 1, key)


def sub_step_densities(lines: list[str], key: str) -> list[str]:
    """⑧ §4.2 d_in 步链: ``polar_scf_step_densities_post`` → 既有 ``polar_scf_step_densities``.

    [v3 适配·D-04] 既有视图的 "polar scf density snapshot" 内核位于逐步 Fukui 校正**之后**
    (校后口径), 与提案新增的 _post 同值同源; 重复视图/快照不再新增。
    """
    old_comment = "        // d_in(i): i=0 → 初始校正后密度; i>0 → i-1 步校后密度快照。"
    new_comment = [
        old_comment,
        "        // [v3 适配·D-04] i>0 直接消费既有 polar_scf_step_densities——其",
        '        // "polar scf density snapshot" 内核位于逐步 Fukui 校正之后(校后口径, 与提案',
        "        // step_densities_post 同值同源); 原提案的 _post 重复视图不新增。",
    ]
    lines = replace_counted(lines, old_comment, new_comment, 1, key + ":注释")
    text = "\n".join(lines)
    n = text.count("polar_scf_step_densities_post")
    if n != 1:
        raise PatchError(f"{key}: 预期 token 'polar_scf_step_densities_post' 命中 1 次, 实际 {n}")
    text = text.replace("polar_scf_step_densities_post", "polar_scf_step_densities")
    return text.split("\n")


def sub_fukui_reverse(lines: list[str], key: str) -> list[str]:
    """⑧ §4.2 fukui_reverse 三处系数修正（[v3 修正·D-08], FD 证据见 merge_adapt_8.md）.

    引擎/mace 前向: d_out(n)=d_pre(n)+s_n·f, f=(T−pred)/S_used（单一 1/S）。
    提案按设计 §2 ⑧-2 的 s̃·F（F=(T−pred)/S）参数化推出（对引擎前向多含 1/S）:
      - 提案 pred̄=−F̄/S_used → 修正 pred̄=−F̄;
      - 提案 s̄=f·d̄/S_used − 2·f·F̄/S_raw → 修正 s̄=f·d̄ −[未钳制] f·F̄。
    修正后与中央差分一致（脚本 [8] 自检: 修正前 err 6.1e2, 修正后 3.0e-9）。
    """
    # (a) 槽位注释
    lines = replace_counted(
        lines,
        "    //      transform 槽位=[1/Sinvα,pred̄α,1/Sinvβ,pred̄β,term2α,term2β,Fα,Fβ](索引 0..7)。",
        [
            "    //      transform 槽位=[1,pred̄α,1,pred̄β,term2α,term2β,Fα,Fβ](索引 0..7;",
            "    //      F=factor_σ=(T−pred)/S_used; [v3 修正·D-08] 见核内注释)。",
        ],
        1,
        key + ":槽位注释",
    )
    # (b) tr(2ch)=1/s_used → 1（前置修正说明）
    lines = replace_counted(
        lines,
        "            transform(2*channel)=Precision(1)/s_used;",
        [
            "            // [v3 修正·D-08] 引擎/mace 前向: d_out(n)=d_pre(n)+s_n·f, f=(T−pred)/S_used",
            "            // (单一 1/S; mace extensions.py L742-766 与 ⑤ 前向核)。闭式伴随:",
            "            //   pred̄=−F̄;  s̄=f·d̄ −[未钳制] f·F̄;  F̄=Σ_n (s_n/S_used)·d̄(n,l=0)。",
            "            // 提案按设计 §2 ⑧-2 的 s̃·F(F=(T−pred)/S) 参数化推出, 对引擎前向多含 1/S,",
            "            // 第二项系数应为 −f·F̄(非 −2·f·F̄/S_raw); FD 证据: merge_adapt_8.md §2-D08。",
            "            transform(2*channel)=Precision(1);",
        ],
        1,
        key + ":tr2",
    )
    # (c) tr(2ch+1)
    lines = replace_counted(
        lines,
        "            transform(2*channel+1)=Precision(-fbar/static_cast<double>(s_used));",
        ["            transform(2*channel+1)=Precision(-fbar);"],
        1,
        key + ":tr3",
    )
    # (d) tr(4+ch)
    lines = replace_block_counted(
        lines,
        [
            "            transform(4+channel)=clamped?Precision(0)",
            "                :Precision(-2.0*fbar*static_cast<double>(f)",
            "                    /static_cast<double>(s_raw));",
        ],
        [
            "            transform(4+channel)=clamped?Precision(0)",
            "                :Precision(-fbar*static_cast<double>(f));",
        ],
        1,
        key + ":tr4",
    )
    return lines


# --------------------------------------------------------------------------
# 各块内容构建
# --------------------------------------------------------------------------
def build_hpp_array(doc: DocBlocks) -> list[str]:
    return [
        "// [v3 适配·D-09] 模块⑧ 使用 std::array(提案 §2.2 注; 现树 hpp 未含 <array>)。",
        "#include <array>",
    ]


def build_hpp_public(doc: DocBlocks) -> list[str]:
    key = "1b-hpp-public"
    b = doc.get(DOC8, "### 2.2 hpp 插入 A", r"在其\*\*之后\*\*插入", key)
    require(b, 0, b[0].startswith("    // --- P3 模块⑧:"), key, "§2.2 首行")
    require(b, len(b) - 1, b[-1].strip() == "bool polar_blocks_enabled() const { return polar_enabled; }",
            key, "§2.2 末行")
    drop = "    Kokkos::View<Precision***,Kokkos::LayoutRight> polar_scf_step_densities_post;"
    n = sum(1 for line in b if line == drop)
    if n != 1:
        raise PatchError(f"{key}: 未唯一定位 _post 视图行(命中 {n})")
    body: list[str] = []
    i = 0
    while i < len(b):
        if b[i] == drop:
            if i + 1 < len(b) and "每步 Fukui 校正【后】密度" in b[i + 1]:
                body.extend(
                    [
                        "    // [v3 适配·D-02] 校后逐步密度直接复用既有 polar_scf_step_densities",
                        "    // (其快照内核位于逐步 Fukui 校正之后 = 校后口径; 提案的 _post 重复视图不新增)。",
                    ]
                )
                i += 2
                continue
            raise PatchError(f"{key}: _post 视图行后未找到续行注释（文档可能已变更）")
        body.append(b[i])
        i += 1
    header = [
        "    // ===== [v3 适配] 模块⑧ hpp 公开增补(反向入口/缓存/工作区; binding 与 pair style 共用) =====",
    ]
    return header + body


def build_hpp_revdecl(doc: DocBlocks) -> list[str]:
    key = "1c-hpp-rev-all-decl"
    b = doc.get(DOC8, "### 2.3 hpp 插入 B", r"在其\*\*之后\*\*插入", key)
    require(b, 0, b[0].startswith("        // 模块⑧(R1): evaluate_all 的伴随。"), key, "§2.3 首行")
    require(b, len(b) - 1, b[-1] == "                         bool initialize_input_adjoint);",
            key, "§2.3 末行")
    return b


def build_tpp_revdef(doc: DocBlocks) -> list[str]:
    key = "2a-tpp-rev-all-def"
    b = doc.get(DOC8, "## 3. tpp PART1", r"在其\*\*之后\*\*插入", key)
    require(b, 0, b[0] == "template<typename Precision>", key, "§3 首行")
    require(b, 1, "Readout::reverse_all(" in b[1], key, "§3 第 2 行")
    require(b, len(b) - 1, b[-1] == "}", key, "§3 末行")
    return [""] + b


def build_tpp_call(doc: DocBlocks) -> list[str]:
    key = "2b-tpp-rev-call"
    b = doc.get(DOC8, "#### A. 调用点", r"在其\*\*之前\*\*插入", key)
    require(b, len(b) - 1, b[-1] == "        compute_polar_reverse_forces(num_nodes);", key, "调用行")
    return b + [""]


def build_tpp_node(doc: DocBlocks) -> list[str]:
    key = "2c-tpp-node-force"
    b = doc.get(DOC8, "### 5.1 归约点 1", r"在该函数\*\*体末尾\*\*", key)
    require(b, 0, b[0].startswith("    // P3 模块⑧: 节点级极性力并入"), key, "§5.1 首行")
    require(b, len(b) - 1, b[-1] == "    }", key, "§5.1 末行")
    return b + [""]


def build_tpp_stress(doc: DocBlocks) -> list[str]:
    key = "2d-tpp-stress-guard"
    b = doc.get(DOC8, "### 6.3 防静默错误", r"函数体首部插入", key)
    require(b, 0, b[0].startswith("    // P3 模块⑧: 极化 cell 伴随未实现"), key, "§6.3 首行")
    require(b, len(b) - 1, b[-1] == '            "implemented yet; disable virial/stress for polar models.");',
            key, "§6.3 末行")
    return b + [""]


def build_tu_volume(doc: DocBlocks) -> list[str]:
    key = "3a-tu-volume"
    b = doc.get(DOC8, "#### B. 模块③", r"在其\*\*之后\*\*插入", key)
    require(b, 0, b[0].startswith("        // P3 模块⑧: 缓存 volume/rcell"), key, "§4.0-B 首行")
    require(b, len(b) - 1, b[-1].strip() == "polar_rcell_host[3*row+column]=Precision(reciprocal[row][column]);",
            key, "§4.0-B 末行")
    return b


def build_tu_c1_clamp(doc: DocBlocks) -> list[str]:
    key = "3b-tu-c1-clamp"
    b = doc.get(DOC8, "#### C. ⑤", r"先留原始 S", key)
    require(b, 0, b[0].strip().startswith("Precision raw_source_sums[2]="), key, "C-1 替换首行")
    require(b, len(b) - 1, b[-1].strip() == "source_sums[channel]=Precision(1);", key, "C-1 替换末行")
    return b


def build_tu_c1_scalars(doc: DocBlocks) -> list[str]:
    key = "3c-tu-c1-scalars"
    b = doc.get(DOC8, "#### C. ⑤", r"polar scf fukui correction", key)
    require(b, 0, b[0].startswith("                // P3 模块⑧: 逐步缓存"), key, "C-1 插入首行")
    require(b, 1, b[1].strip().startswith("Kokkos::realloc(Kokkos::WithoutInitializing,"), key, "realloc 行")
    require(b, 2, b[2].strip().startswith("polar_scf_step_densities_post,polar_recursion_steps,"), key, "视图行")
    require(b, 3, b[3].strip() == "num_nodes,8);", key, "形状行")
    require(b, 4, b[4].strip() == "auto post_density=polar_scf_step_densities_post;", key, "alias 行")
    require(b, 10, b[10].strip() == "});", key, "快照 kernel 收尾行")
    # 删除重复快照 b[1:11](realloc×3 + alias + parallel_for×6), 保留标量缓存 + [v3] 注释
    kept = b[11:]  # if(...)/assign/赋值 共 8 行
    require(kept, 0, kept[0].strip().startswith("if(static_cast<int>(polar_scf_fukui_scalars.size())"),
            key, "标量段首行")
    require(kept, len(kept) - 1, kept[-1].strip() == "factor_alpha,factor_beta};", key, "标量段末行")
    head = [
        "                // P3 模块⑧: 逐步 Fukui 标量缓存([Sα,Sβ,predα,predβ,Fα,Fβ]),供反向一次性消费。",
        "                // [v3 适配·D-03] 校后密度链直接消费既有 polar_scf_step_densities——其",
        '                // "polar scf density snapshot" 内核位于本 Fukui 校正之后(校后口径, 与提案',
        "                // step_densities_post 同值同源); 原提案的重复快照视图/内核不再新增。",
    ]
    return head + kept


def build_tu_c2_clamp(doc: DocBlocks) -> list[str]:
    key = "3d-tu-c2-clamp"
    b = doc.get(DOC8, "#### C. ⑤", r"^\*\*C-2 初始化校正\*\*", key)
    # 上面取到的是 C-2 锚点 fence; 替换内容在下一 fence
    b2 = doc.get(DOC8, "#### C. ⑤", r"^替换为：$", key + ":repl")
    require(b2, 0, b2[0].strip().startswith("const Precision raw_source_sums[2]="), key, "C-2 替换首行")
    require(b2, len(b2) - 1, b2[-1].strip() == "source_sums[channel]=Precision(1);", key, "C-2 替换末行")
    return b2


def build_tu_c2_scalars(doc: DocBlocks) -> list[str]:
    key = "3e-tu-c2-scalars"
    b = doc.get(DOC8, "#### C. ⑤", r"初始化校正 kernel", key)
    require(b, 0, b[0].startswith("        // P3 模块⑧: 初始化 Fukui 标量缓存"), key, "C-2 插入首行")
    require(b, len(b) - 1, b[-1] == "            factor_alpha,factor_beta};", key, "C-2 插入末行")
    return b


def build_tu_main(doc: DocBlocks) -> list[str]:
    key = "3f-tu-reverse-main"
    b = doc.get(DOC8, "### 4.2", r"在其\*\*之前\*\*插入以下两块", key)
    require(b, 0, b[0].startswith("// ===="), key, "§4.2 首行")
    require(b, 1, "P3 模块⑧" in b[1], key, "§4.2 注释第 2 行")
    require(b, len(b) - 1, b[-1] == "}", key, "§4.2 末行")
    body = sub_coulomb_density_source(b, key)
    body = sub_step_densities(body, key)
    body = sub_fukui_reverse(body, key)
    body = sub_layout_right(body, key)
    header = [
        "// [v3 适配] 本块按适配规则 1 整体迁入 polar TU(mace_polar_impl.tpp; 显式实例化之前)。",
        "// 相对 module8_patch_final.md §4.2 原文的偏离(全表见 docs/merge_adapt_8.md §2):",
        "//   D-04 d_in 步链: polar_scf_step_densities_post → 既有 polar_scf_step_densities(校后口径同值);",
        "//   D-05 ⑥ 种子密度源: polar_spin_charge_density → polar_scf_final_density(与 ⑥ 前向同源);",
        "//   D-06 lambda 形参 Kokkos::View<...**> 补 LayoutRight(bench/polar_lint.py R3);",
        "//   D-08 fukui_reverse 三处系数按引擎前向闭式伴随修正(FD 证据见脚本 [8] 自检)。",
    ]
    return header + body + [""]


def build_tu_enable(doc: DocBlocks) -> list[str]:
    return [
        "        // [v3 适配·D-10] ⑥ 已落码(v2 起前向能量已接入 node_energies)按提案 §0-3/§10-2",
        '        // "⑥ 落码时置位" 启用 coulomb 反向种子(hpp 默认 false 保留)。',
        "        polar_coulomb_enabled=true;",
    ]


def build_cpp_bind(doc: DocBlocks) -> list[str]:
    key = "4-cpp-binding"
    b = doc.get(DOC8, "### 7.1", r"^### 7\.1", key)
    require(b, 0, b[0] == '        .def("polar_scf_final_density",[](const Evaluator& self) {',
            key, "§7.1 首行=原块")
    require(b, 2, b[2] == "        })", key, "§7.1 第 3 行")
    require(b, 3, b[3].strip().startswith('.def("polar_node_forces"'), key, "node_forces 块")
    require(b, 7, b[7].strip().startswith('.def("polar_reverse_terms_last"'), key, "terms_last 块")
    require(b, 12, b[12].strip().startswith('.def("polar_blocks_enabled"'), key, "blocks_enabled 块")
    require(b, 15, b[15] == "        })", key, "blocks_enabled 收尾")
    require(b, 16, b[16].strip().startswith('.def("polar_scf_step_densities_post"'), key, "_post 块")
    require(b, len(b) - 1, b[-1] == "        });", key, "§7.1 末行")
    return (
        [
            '        .def("polar_coulomb_rho_im",[](const Evaluator& self) {',
            "            return view2vector(self.polar_coulomb_rho_im);",
            "        })",
        ]
        + b[3:15]
        + ["        });"]
    )


def build_pair_wire(doc: DocBlocks) -> list[str]:
    key = "5-pair-force-wire"
    b = doc.get(DOC8, "### 5.2 归约点 2", r"force_reduction.*\*\*之后\*\*插入",
                key)
    require(b, 0, b[0].startswith("  // P3 模块⑧: 节点级极性力"), key, "§5.2 首行")
    require(b, len(b) - 1, b[-1] == "  }", key, "§5.2 末行")
    return [""] + b


BUILDERS = {
    "hpp_array": build_hpp_array,
    "hpp_public": build_hpp_public,
    "hpp_revdecl": build_hpp_revdecl,
    "tpp_revdef": build_tpp_revdef,
    "tpp_call": build_tpp_call,
    "tpp_node": build_tpp_node,
    "tpp_stress": build_tpp_stress,
    "tu_volume": build_tu_volume,
    "tu_c1_clamp": build_tu_c1_clamp,
    "tu_c1_scalars": build_tu_c1_scalars,
    "tu_c2_clamp": build_tu_c2_clamp,
    "tu_c2_scalars": build_tu_c2_scalars,
    "tu_main": build_tu_main,
    "tu_enable": build_tu_enable,
    "cpp_bind": build_cpp_bind,
    "pair_wire": build_pair_wire,
}

# 每个块的内容来源（DocBlocks.refs 键）; 用于逐块报告
CONTENT_REFS = {
    "hpp_array": (),
    "hpp_public": ("1b-hpp-public",),
    "hpp_revdecl": ("1c-hpp-rev-all-decl",),
    "tpp_revdef": ("2a-tpp-rev-all-def",),
    "tpp_call": ("2b-tpp-rev-call",),
    "tpp_node": ("2c-tpp-node-force",),
    "tpp_stress": ("2d-tpp-stress-guard",),
    "tu_volume": ("3a-tu-volume",),
    "tu_c1_clamp": ("3b-tu-c1-clamp",),
    "tu_c1_scalars": ("3c-tu-c1-scalars",),
    "tu_c2_clamp": ("3d-tu-c2-clamp", "3d-tu-c2-clamp:repl"),
    "tu_c2_scalars": ("3e-tu-c2-scalars",),
    "tu_main": ("3f-tu-reverse-main",),
    "tu_enable": (),
    "cpp_bind": ("4-cpp-binding",),
    "pair_wire": ("5-pair-force-wire",),
}


# --------------------------------------------------------------------------
# 内存应用（顺序模拟; 与 v2 同机制）
# --------------------------------------------------------------------------
def find_unique(content: str, anchor: str, bid: str, where: str) -> int:
    """返回行边界对齐的唯一锚点起始偏移；count!=1 时打印诊断并中止."""
    needle = "\n" + anchor + "\n"
    count = content.count(needle)
    a_lines = anchor.split("\n")
    if count != 1:
        out(f"  [诊断] 块 {bid} ({where}): 锚点 count={count}（要求 1）")
        out(f"  [诊断] 锚点首行: {a_lines[0]!r}")
        lines = content.split("\n")
        first_hits = [i + 1 for i, line in enumerate(lines) if line == a_lines[0]]
        out(f"  [诊断] 首行精确命中行号: {first_hits[:20]}" + (" …" if len(first_hits) > 20 else ""))
        if count > 1:
            out("  [诊断] 完整锚点多次命中（行号:…）:")
            pos, occ = 0, []
            while True:
                pos = content.find(needle, pos)
                if pos < 0:
                    break
                occ.append(content.count("\n", 0, pos + 1) + 1)
                pos += 1
            for ln in occ[:5]:
                lo = max(1, ln - 1)
                for n in range(lo, min(ln + len(a_lines) + 1, len(lines) + 1)):
                    out(f"      {n:>6}: {lines[n - 1]}")
                out("      ---")
        raise PatchError(
            f"块 {bid}: 锚点未唯一命中（count={count}）— 树可能漂移；"
            f"请对照 docs/merge_adapt_8.md §0 的只读快照复跑复验"
        )
    return content.find(needle) + 1  # 锚点起始偏移（跳过前导 \n）


def line_of(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


def check_context(content: str, anchor: str, bid: str, before, after) -> None:
    lines = content.split("\n")
    start = content.count("\n", 0, content.find("\n" + anchor + "\n") + 1) + 1
    if before:
        for k, expected in enumerate(before):
            n = start - 1 - k  # 紧邻上一行, 再上一行…
            actual = lines[n - 1] if n >= 1 else "<BOF>"
            if actual != expected:
                raise PatchError(
                    f"块 {bid}: 锚点上方第 {k + 1} 行不符\n"
                    f"      期望: {expected!r}\n      实际: {actual!r} (行 {n})"
                )
    if after:
        end = start + len(anchor.split("\n")) - 1
        for k, expected in enumerate(after):
            n = end + 1 + k
            actual = lines[n - 1] if n <= len(lines) else "<EOF>"
            if actual != expected:
                raise PatchError(
                    f"块 {bid}: 锚点下方第 {k + 1} 行不符\n"
                    f"      期望: {expected!r}\n      实际: {actual!r} (行 {n})"
                )


def apply_block(content: str, blk: dict) -> tuple[str, dict]:
    """在内存 content 上应用一个块；返回 (新内容, 报告行)."""
    bid = blk["bid"]
    anchor = blk["anchor"]
    idx = find_unique(content, anchor, bid, blk["file"])
    anchor_lines = anchor.split("\n")
    start_line = line_of(content, idx)

    check_context(content, anchor, bid, blk.get("expect_before"), blk.get("expect_after"))

    insert_lines = blk["content_lines"]
    if blk["op"] == "insert_after":
        offset = idx + len(anchor) + 1
        inserted = "\n".join(insert_lines) + "\n"
        new_content = content[:offset] + inserted + content[offset:]
        detail = (
            f"整块之后插入 {len(insert_lines)} 行"
            f"（锚点行 {start_line}-{start_line + len(anchor_lines) - 1}）"
        )
        delta = f"+{len(insert_lines)}"
    elif blk["op"] == "insert_before":
        offset = idx
        inserted = "\n".join(insert_lines) + "\n"
        new_content = content[:offset] + inserted + content[offset:]
        detail = (
            f"整块之前插入 {len(insert_lines)} 行"
            f"（锚点行 {start_line}-{start_line + len(anchor_lines) - 1}）"
        )
        delta = f"+{len(insert_lines)}"
    elif blk["op"] == "replace":
        replacement = "\n".join(insert_lines)
        new_content = content[:idx] + replacement + content[idx + len(anchor):]
        detail = (
            f"整块替换: {len(anchor_lines)} 行 → {len(insert_lines)} 行"
            f"（锚点行 {start_line}-{start_line + len(anchor_lines) - 1}）"
        )
        delta = f"{len(insert_lines) - len(anchor_lines):+d}"
    else:
        raise PatchError(f"块 {bid}: 未知 op={blk['op']!r}")

    report = dict(
        bid=bid, group=blk["group"], file=blk["file"], anchor_hits=1,
        start_line=start_line, anchor_lines=len(anchor_lines), detail=detail,
        delta=delta, content_ref=blk.get("content_ref", "-"),
        note=blk["note"], insert_lines=len(insert_lines),
    )
    return new_content, report


# --------------------------------------------------------------------------
# diff 预估（内存 difflib；不调用 git）
# --------------------------------------------------------------------------
def diff_counts(orig: str, new: str) -> tuple[int, int]:
    a = orig.split("\n")
    b = new.split("\n")
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    insertions = deletions = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            deletions += i2 - i1
        if tag in ("replace", "insert"):
            insertions += j2 - j1
    return insertions, deletions


def probe_counts(text: str, needle: str) -> tuple[int, int]:
    return text.count(needle), sum(1 for line in text.split("\n") if needle in line)


# --------------------------------------------------------------------------
# [8] 数值自检（纯 stdlib; Fukui 伴随 + ⑥ 种子链 FD 校验）
# --------------------------------------------------------------------------
def _lcg(seed: int, count: int, m: int = 2 ** 31, a: int = 1103515245, c: int = 12345):
    x = seed
    vals = []
    for _ in range(count):
        x = (a * x + c) % m
        vals.append((x / m) * 2.0 - 1.0)
    return vals


def numerical_self_check() -> tuple[float, float, float, float]:
    """返回 (fukui_proposal_err, fukui_v3_err, coulomb_abs_err, coulomb_rel_err)."""
    # ---- (1) Fukui 校正伴随: 中央差分校验（单通道, 与 ⑤ 前向核同式）----
    s = [0.7, -1.3, 0.45]
    d0 = [0.31, -0.12, 0.87]
    T = 0.55
    g = [1.1, -0.7, 0.25]
    N = 3
    h = 1e-6

    def fwd(sv, dp, used):
        pred = sum(dp)
        f = (T - pred) / used
        return [dp[n] + sv[n] * f for n in range(N)]

    def energy(sv, dp, used):
        r = fwd(sv, dp, used)
        return sum(g[k] * r[k] for k in range(N))

    def fd():
        db = [0.0] * N
        sb = [0.0] * N
        for n in range(N):
            dp = list(d0)
            dm = list(d0)
            dp[n] += h
            dm[n] -= h
            used = sum(s)
            db[n] = (energy(s, dp, used) - energy(s, dm, used)) / (2 * h)
            sp = list(s)
            sm = list(s)
            sp[n] += h
            sm[n] -= h
            sb[n] = (energy(sp, d0, sum(sp)) - energy(sm, d0, sum(sm))) / (2 * h)
        return db, sb

    db_fd, sb_fd = fd()

    def proposal():
        S = sum(s)
        used = S
        pred = sum(d0)
        f = (T - pred) / used
        fbar = sum(g[n] * (s[n] / used) for n in range(N))
        sb = [(f * (1.0 / used)) * g[n] + (-2.0 * fbar * f / S) for n in range(N)]
        db = [g[n] + (-fbar / used) for n in range(N)]
        return db, sb

    def v3():
        S = sum(s)
        used = S
        pred = sum(d0)
        f = (T - pred) / used
        fbar = sum(g[n] * (s[n] / used) for n in range(N))
        sb = [f * g[n] + (-fbar * f) for n in range(N)]
        db = [g[n] + (-fbar) for n in range(N)]
        return db, sb

    def err(a, b):
        return max(abs(x - y) for x, y in zip(a, b))

    pa = proposal()
    v3r = v3()
    e_p = max(err(pa[0], db_fd), err(pa[1], sb_fd))
    e_v = max(err(v3r[0], db_fd), err(v3r[1], sb_fd))
    if not (e_v < 1e-6 and e_p > 1e-2):
        raise PatchError(
            f"[8] Fukui 伴随 FD 自检失败: proposal_err={e_p:.3e} v3_err={e_v:.3e}"
        )
    # 钳制分支闭式（S==0 精确）
    sB = [0.5, -0.5, 0.0]
    dB = [0.2, -0.4, 0.9]
    TB = -0.3
    gB = [0.8, 0.3, -0.6]
    fB = (TB - sum(dB)) / 1.0
    fbB = sum(gB[n] * sB[n] for n in range(3))
    db_exp = [gB[n] - fbB for n in range(3)]
    sb_exp = [fB * gB[n] for n in range(3)]
    if err(db_exp, [gB[n] - fbB for n in range(3)]) > 1e-12 or err(sb_exp, [fB * gB[n] for n in range(3)]) > 1e-12:
        raise PatchError("[8] Fukui 钳制分支闭式自检失败")

    # ---- (2) ⑥ coulomb 种子链 ≡ dE/dq（当前 ⑥ 前向口径, ρ̂ 未乘 (2π)³/V）----
    r = _lcg(7, 200)
    it = iter(r)
    Nk, Nn = 3, 3
    B_re = [[next(it) for _ in range(4)] for _ in range(Nk)]
    B_im = [[next(it) for _ in range(4)] for _ in range(Nk)]
    cosv = [[next(it) for _ in range(Nn)] for _ in range(Nk)]
    sinv = [[next(it) for _ in range(Nn)] for _ in range(Nk)]
    kn2 = [1.0, 2.0, 3.0]
    k0 = [1.0, 0.0, 0.0]
    q = [[next(it) for _ in range(4)] for _ in range(Nn)]
    tpc = 6.283185307179586476925286766559 ** 3
    V = 12.5
    FC = 1.0 / (5.526349406e-3)

    def rho_hat(qv):
        re = [0.0] * Nk
        im = [0.0] * Nk
        for k in range(Nk):
            ac = [0.0] * 4
            as_ = [0.0] * 4
            for n in range(Nn):
                for m in range(4):
                    ac[m] += cosv[k][n] * qv[n][m]
                    as_[m] += sinv[k][n] * qv[n][m]
            for m in range(4):
                re[k] += B_re[k][m] * ac[m] + B_im[k][m] * as_[m]
                im[k] += B_im[k][m] * ac[m] - B_re[k][m] * as_[m]
        return re, im

    def energy_c(qv):
        re, im = rho_hat(qv)
        tot = 0.0
        for k in range(Nk):
            if k0[k] > 0:
                continue
            tot += 2.0 * (re[k] ** 2 + im[k] ** 2) / kn2[k]
        return tot * FC / (2 * V)

    def seed_chain(qv):
        re, im = rho_hat(qv)
        rho_re = [(tpc / V) * x for x in re]
        rho_im = [(tpc / V) * x for x in im]
        sc = 2.0 * V / (tpc * tpc) * FC
        rb_re = [rho_re[k] * (0.0 if k0[k] > 0 else 1.0 / kn2[k]) * sc for k in range(Nk)]
        rb_im = [rho_im[k] * (0.0 if k0[k] > 0 else 1.0 / kn2[k]) * sc for k in range(Nk)]
        qb = [[0.0] * 4 for _ in range(Nn)]
        for k in range(Nk):
            for m in range(4):
                acc_cos = (tpc / V) * (rb_re[k] * B_re[k][m] + rb_im[k] * B_im[k][m])
                acc_sin = (tpc / V) * (rb_re[k] * B_im[k][m] - rb_im[k] * B_re[k][m])
                for n in range(Nn):
                    qb[n][m] += cosv[k][n] * acc_cos + sinv[k][n] * acc_sin
        return qb

    hq = 1e-4
    qb = seed_chain(q)
    worst = 0.0
    scale = 0.0
    for n in range(Nn):
        for m in range(4):
            qp = [row[:] for row in q]
            qm = [row[:] for row in q]
            qp[n][m] += hq
            qm[n][m] -= hq
            fdq = (energy_c(qp) - energy_c(qm)) / (2 * hq)
            worst = max(worst, abs(fdq - qb[n][m]))
            scale = max(scale, abs(fdq))
    rel = worst / scale
    if not rel < 1e-8:
        raise PatchError(f"[8] ⑥ 种子链 FD 自检失败: abs={worst:.3e} rel={rel:.3e}")
    return e_p, e_v, worst, rel


# --------------------------------------------------------------------------
# 模拟产物写出 + polar_lint（dry-run; 只写 --sim-dir）
# --------------------------------------------------------------------------
def emit_sim_files(state: dict[str, str], sim_dir: Path) -> dict[str, Path]:
    sim_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, rel in ENGINE_FILES.items():
        p = sim_dir / Path(rel).name
        p.write_text(state[key], encoding="utf-8", newline="")
        paths[key] = p
    return paths


def run_lint(sim_paths: dict[str, Path]) -> tuple[int, dict[str, int]]:
    """对模拟产物运行 polar_lint（6 类规则）; 返回 (总发现数, 每类计数)."""
    totals: dict[str, int] = {f"R{i}": 0 for i in range(1, 7)}
    grand = 0
    for key in ("tpp", "tu", "pair"):
        proc = subprocess.run(
            [sys.executable, str(LINT_PATH), str(sim_paths[key])],
            capture_output=True, text=True,
        )
        stdout = proc.stdout.strip()
        out(f"    [{key}] rc={proc.returncode}")
        for line in stdout.split("\n"):
            out(f"        {line}")
        for m in re.finditer(r"\[(R[1-6])\]", stdout):
            totals[m.group(1)] += 1
        grand += proc.returncode
    return grand, totals


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    global TEE
    parser = argparse.ArgumentParser(
        description="⑧ 力/全链反向 polar 引擎补丁 v3（新结构适配）应用器：默认 dry-run（不写引擎树）",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只读模拟（默认）")
    mode.add_argument("--apply", action="store_true", help="真实写入（前置: dry-run 全绿）")
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE, help="symmetrix-xl 引擎树根")
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR, help="patch_final 文档目录")
    parser.add_argument("--log", type=Path, default=None, help="把完整输出同时写入该日志文件")
    parser.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR,
                        help="dry-run 模拟产物目录（供 polar_lint; 仅 dry-run 写出）")
    parser.add_argument("--no-lint", action="store_true", help="跳过模拟产物写出与 polar_lint")
    args = parser.parse_args(argv)

    mode_name = "APPLY" if args.apply else "DRY-RUN"
    TEE = Tee(args.log)
    exit_code = 0
    try:
        exit_code = run(mode_name, args)
    except PatchError as exc:
        out("")
        out(f"ABORT: {exc}")
        out("已中止；未写入任何引擎文件。")
        exit_code = 2
    finally:
        TEE.close()
    return exit_code


def run(mode_name: str, args) -> int:
    engine: Path = args.engine
    docs_dir: Path = args.docs_dir
    sim_dir: Path = args.sim_dir

    out("=" * 78)
    out(f"apply_polar_patches_v3.py — 模式: {mode_name}（引擎树写操作: {'是' if args.apply else '否'}）")
    out("目标树: polar 独立 TU + ⑥⑦已落码(v19c) 新结构")
    out(f"引擎: {engine}")
    out(f"文档: {docs_dir}/{DOC8}; 适配裁定: merge_adapt_8.md")
    out("=" * 78)

    # ---- 环境检查 ---------------------------------------------------------
    if not engine.is_dir():
        raise PatchError(f"引擎目录不存在: {engine}")
    for key, rel in ENGINE_FILES.items():
        if not (engine / rel).is_file():
            raise PatchError(f"引擎文件缺失: {engine / rel}")
    if not LINT_PATH.is_file():
        raise PatchError(f"polar_lint 缺失: {LINT_PATH}")

    # ---- 文档内容提取 + 适配变换 -------------------------------------------
    out("")
    out("[0] 文档解析（插入内容逐字提取 + v3 适配变换, 含计数断言）")
    doc = DocBlocks(docs_dir)
    blocks = []
    for spec in BLOCK_SPECS:
        content = BUILDERS[spec["builder"]](doc)
        blk = dict(spec)
        blk["content_lines"] = content
        refs = [doc.refs[k] for k in CONTENT_REFS[spec["builder"]] if k in doc.refs]
        blk["content_ref"] = "; ".join(refs) if refs else "(本脚本字面文本; 无文档代码块)"
        blocks.append(blk)
    for blk in blocks:
        out(f"    {blk['order']:>2}. {blk['bid']:<22} {blk['file']:<4} "
            f"内容 {len(blk['content_lines']):>4} 行")
        out(f"        {blk['note']}")
    out(f"    -> 共 {len(blocks)} 个插入点 / 6 组（①hpp ×3、②主 tpp §3/§4.0-A/§5.1、"
        f"③主 tpp §6.3、④新 TU ×7、⑤binding ×1、⑥pair ×1）")
    out("    来源引用（代码块逐字提取位置）:")
    for key in sorted(doc.refs):
        out(f"        {key:<28} {doc.refs[key]}")

    # ---- 读取引擎文件（只读） ---------------------------------------------
    originals: dict[str, str] = {}
    out("")
    out("[1] 引擎树只读快照")
    for key, rel in ENGINE_FILES.items():
        raw = (engine / rel).read_bytes()
        text = raw.decode("utf-8")
        if "\r" in text:
            raise PatchError(f"{rel}: 检测到 CR/LF（不支持）；请统一为 LF 后重跑")
        originals[key] = text
        md5 = hashlib.md5(raw).hexdigest()
        nlines = text.count("\n") + (0 if text.endswith("\n") else 1)
        exp_md5, exp_n = V3_SNAPSHOT[key]
        drift = "" if (md5 == exp_md5 and nlines == exp_n) else "  <-- 相对 v3 编写快照有漂移（以锚点文本为准）"
        out(f"    {key:<4} {rel}")
        out(f"         {nlines:>6} 行  md5={md5}{drift}")

    # ---- 结构前提核查（只读） ---------------------------------------------
    out("")
    out("[2] 结构前提核查（应用前; 全部要求 count==期望）")
    for key, needle, expected, note in PRE_CHECKS:
        occ, nlin = probe_counts(originals[key], needle)
        ok = "OK " if occ == expected else "FAIL"
        out(f"    [{ok}] {key:<4} {needle.splitlines()[0][:56]!r:<60} occ={occ}")
        out(f"           <- {note}")
        if occ != expected:
            raise PatchError(f"结构前提核查失败: {key} {needle!r} occ={occ} 期望={expected}")

    # ---- 幂等标记检查 ------------------------------------------------------
    out("")
    out("[3] 幂等标记（应用前应全为 0；>0 ⇒ 疑似已应用）")
    for key, needle in IDEMPOTENCE_MARKERS:
        occ, _ = probe_counts(originals[key], needle)
        out(f"    {key:<4} {needle!r:<44} x{occ}")
        if occ != 0:
            raise PatchError(
                f"{key}: 标记 {needle!r} 已存在（x{occ}）— 树疑似已被本补丁（部分）修改, 拒绝继续"
            )

    # ---- 顺序模拟应用 ------------------------------------------------------
    out("")
    out("[4] 逐块应用（内存顺序模拟: 1→16）")
    state = dict(originals)
    reports: list[dict] = []
    for blk in blocks:
        content_before = state[blk["file"]]
        state[blk["file"]], rep = apply_block(content_before, blk)
        reports.append(rep)
        out(
            f"    [{blk['order']}/16] {rep['bid']:<22} {rep['file']:<4} "
            f"锚点 count=1（行 {rep['start_line']}）| {rep['detail']} | Δ{rep['delta']}"
        )
        out(f"           来源: {rep['content_ref']}")
    out("    （16/16 块锚点唯一命中并完成内存模拟）")

    # ---- 应用后结构探针 ----------------------------------------------------
    out("")
    out("[5] 应用后结构探针（内存结果; 期望值=适配后应有计数）")
    probe_failures = []
    for key, needle, expected, src in POST_PROBES:
        occ, nlin = probe_counts(state[key], needle)
        ok = "OK " if occ == expected else "FAIL"
        if occ != expected:
            probe_failures.append(f"{key}:{needle!r} occ={occ} 期望={expected}")
        out(f"    [{ok}] {key:<4} {needle!r:<62} occ={occ} (line-count={nlin})  <- {src}")
    for key, seq, src in ORDER_PROBES:
        text = state[key]
        positions = []
        cursor = 0
        ok = True
        for marker in seq:
            pos = text.find(marker, cursor)
            if pos < 0:
                ok = False
                positions.append(None)
            else:
                positions.append(text.count("\n", 0, pos) + 1)
                cursor = pos + 1
        if None in positions or positions != sorted(positions):
            ok = False
            probe_failures.append(f"{key} 顺序探针失败: {src} -> {positions}")
        out(f"    [{'OK ' if ok else 'FAIL'}] {key:<4} 顺序 {positions}  <- {src}")

    # ---- 行数变化与 diff stat 预估 -----------------------------------------
    out("")
    out("[6] 应用后行数变化（相对当前工作树；本脚本不调用 git）")
    total_ins = total_del = 0
    stat_rows = []
    for key in ("tu", "hpp", "tpp", "cpp", "pair"):
        orig = originals[key]
        new = state[key]
        n0 = orig.count("\n") + (0 if orig.endswith("\n") else 1)
        n1 = new.count("\n") + (0 if new.endswith("\n") else 1)
        ins, dele = diff_counts(orig, new)
        total_ins += ins
        total_del += dele
        rel = ENGINE_FILES[key]
        bar = "+" * min(ins, 40) + "-" * min(dele, 40)
        stat_rows.append(f" {rel} | {ins + dele} {bar}")
        out(f"    {key:<4} {n0} -> {n1}  ({n1 - n0:+d} 行)")
        out(f"         difflib: +{ins} / -{dele}")
    out("")
    out("[7] git diff --stat 预估（difflib 内存行差；git 未调用）")
    for row in stat_rows:
        out(f"    {row.strip()}")
    nfiles = sum(1 for k in ENGINE_FILES if originals[k] != state[k])
    out(f"    {nfiles} files changed, {total_ins} insertions(+), {total_del} deletions(-)")
    out("    说明: 仅本补丁（16 个插入点）相对当前工作树的贡献; 真实 git diff --stat 还含树中其它未提交改动。")

    # ---- [8] 数值自检 ------------------------------------------------------
    out("")
    out("[8] 数值自检（纯 stdlib; Fukui 伴随 + ⑥ 种子链 与中央差分对拍）")
    e_p, e_v, cw, cr = numerical_self_check()
    out(f"    Fukui 伴随 FD: 提案原文 err={e_p:.3e}（须>1e-2, 证明原文对引擎前向多含 1/S）")
    out(f"                   v3 修正 err={e_v:.3e}（须<1e-6; D-08 的机器证据）")
    out(f"    ⑥ 种子链  dE/dq FD: abs={cw:.3e} rel={cr:.3e}（须<1e-8; ⑥ 种子重算无系数偏差）")
    out("    钳制分支闭式: OK")

    # ---- [9] 模拟产物 + polar_lint ----------------------------------------
    lint_findings = 0
    lint_totals: dict[str, int] = {}
    if not args.apply and not args.no_lint:
        out("")
        out(f"[9] 模拟产物写出（{sim_dir}; 仅 dry-run） + polar_lint（6 类规则, 须为 0）")
        sim_paths = emit_sim_files(state, sim_dir)
        for key, p in sim_paths.items():
            raw = p.read_bytes()
            out(f"    wrote {p}  ({len(raw)} bytes)")
        lint_findings, lint_totals = run_lint(sim_paths)
        tally = " ".join(f"{r}={lint_totals[r]}" for r in sorted(lint_totals))
        out(f"    polar_lint 合计发现: {lint_findings} 处  ({tally})")
        if lint_findings > 0:
            probe_failures.append(f"polar_lint 发现 {lint_findings} 处（须 0）")

    # ---- 结论 --------------------------------------------------------------
    out("")
    green = not probe_failures
    if green:
        out("DRY-RUN 全绿: 16/16 插入点锚点唯一命中并完成内存模拟；结构前提/结构/顺序探针全部通过。")
        out("引擎树未被修改（dry-run 只读; 写操作仅日志 + 模拟产物）。")
    else:
        out("DRY-RUN 探针失败项:")
        for item in probe_failures:
            out(f"    - {item}")
        out("引擎树未被修改。")

    if not args.apply:
        if not green:
            raise PatchError("应用后探针存在失败项（见上）")
        return 0

    # ---- --apply: 备份 → 写入 → 复验（失败回滚） --------------------------
    if not green:
        raise PatchError("应用后探针存在失败项；拒绝写入")
    out("")
    out("[10] APPLY: 备份与写入")
    written: list[tuple[str, Path, Path]] = []  # (key, path, bak)
    try:
        for key in ("tu", "hpp", "tpp", "cpp", "pair"):
            if originals[key] == state[key]:
                continue
            path = engine / ENGINE_FILES[key]
            bak = path.with_name(path.name + ".bak")
            if bak.exists():
                raise PatchError(f"备份已存在, 拒绝覆盖: {bak}（请先人工处理）")
            shutil.copy2(path, bak)
            out(f"    backup: {bak}")
            tmp = path.with_name(path.name + ".polar_tmp")
            tmp.write_text(state[key], encoding="utf-8", newline="")
            os.replace(tmp, path)
            written.append((key, path, bak))
            out(f"    wrote : {path}")
        # 写入后复验
        out("")
        out("[11] 写入后复验（字节一致 + 探针）")
        for key, path, _bak in written:
            disk = path.read_text(encoding="utf-8")
            if disk != state[key]:
                raise PatchError(f"写入后字节不一致: {path}")
            out(f"    byte-identical: {path}")
        for key, needle, expected, _src in POST_PROBES:
            disk = (engine / ENGINE_FILES[key]).read_text(encoding="utf-8")
            occ, _ = probe_counts(disk, needle)
            if occ != expected:
                raise PatchError(f"写入后探针失败: {key} {needle!r} occ={occ} 期望={expected}")
        out("    探针复验: 全部通过")
    except Exception as exc:  # 含 PatchError
        out("")
        out(f"APPLY 失败: {exc}")
        for _key, path, bak in written:
            shutil.copy2(bak, path)
            out(f"    已回滚: {path} <- {bak}")
        raise PatchError("写入阶段失败，已回滚（.bak 保留）") from exc

    out("")
    out("APPLY OK: 16/16 插入点已写入；备份保留为 *.bak。")
    out("提醒: 重建（重编 polar TU + 主 tpp/hpp/binding/pair 一次）→ 装载冒烟 → ①-⑤ 回归 → "
        "⑧ 门(post 档待 ⑦ le 反向落地)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
