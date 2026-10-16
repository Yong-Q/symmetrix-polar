#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""apply_polar_patches_v2.py —— ⑥/⑦ polar 引擎补丁（v2 适配：polar 独立 TU 新结构）应用器.

用途
----
把 ``docs/module6_patch_final.md``（⑥ coulomb_energy）与 ``docs/module7_patch_final.md``
（⑦ local_electron_energy）两份提案**适配到新结构**（polar 已拆分为独立编译单元
``libsymmetrix/source/mace_polar_impl.tpp``，由 ``mace_polar_kokkos.cpp`` 单独 include），
并把适配后的结果机械应用到 symmetrix-xl 引擎树。适配裁定与逐处偏离见
``docs/merge_adapt_67.md``。

v2 块清单（4 组 / 7 个插入点）：

- ① hpp 增补（``mace_nonlinear_kokkos.hpp``，3 个插入点）：
  - ①a 公开区（⑤暂存块之后）：⑦-4.1a 视图 + ⑥-A 视图（binding 访问）；
  - ①b 私有区（``polar_updates`` 之后）：⑦-4.1b 表与配置；
  - ①c 公开区（``run_polar_pipeline`` 声明之后）：``parse_polar_extension_tables()`` 声明。
- ② 主 tpp 1 行（``mace_nonlinear_kokkos_impl.tpp``）：``} else polar_enabled=false;`` 之后
  插入 ``if(polar_enabled) parse_polar_extension_tables();``。
- ③ 新 TU 两个插入段（``mace_polar_impl.tpp``）：
  - ③a ``parse_polar_extension_tables()`` 定义（显式实例化之前；⑥-B + ⑦-4.2a 合并，
    ``pb.*`` → ``polar_json.*``，``epd`` 重建，整体缩进 -4）；
  - ③b 运行期合计段（``run_polar_pipeline`` 内 ⑤ SCF 块之后、函数尾 ``}`` 之前：
    ⑥-C1 原文 + ⑦-4.3 + ⑦-4.4 + ⑥-C2；⑦/⑥ 后三块整体 +4 缩进，⑥-C2 ``energies`` →
    成员 ``node_energies``）。
- ④ binding 链（``symmetrix/source/cpp/mace_nonlinear_kokkos.cpp``）：链尾
  ``polar_scf_final_density`` 块展开为 ⑦-4.5（9 条）+ ⑥-D（3 条）合计新链
  （⑦ 链尾 ``});`` → ``})`` 后接 ⑥ 块）。

内容提取纪律（同 v1）：插入内容从两份 patch_final 文档的代码块**程序化逐字提取**
（节标题 + 代码块引言正则唯一定位；结构断言失败即中止），v2 的被迫适配变换
（``pb.→polar_json.``、``energies→node_energies``、缩进 ±4、[v2] 注释头）逐项显式
实现并计数断言，任何一处不满足即中止。锚点全部为引擎树**字面文本**（本树读取快照
2026-10-15；锚点 count==1 强断言）。

应用模式与纪律（同 v1）：

- ``--dry-run``（默认）：只读引擎树；逐块打印 锚点 count/插入行数；输出应用后结构
  探针与行数变化预估（内存 difflib，本脚本不调用 git）。
- ``--apply``：真实写入。写入**前**重跑全部断言（锚点唯一 + 幂等标记），要求 ``.bak``
  不存在，备份后写入；写入后复验（字节一致 + 探针），失败自动回滚。仅应在 dry-run
  全绿后由用户/主线程执行——本脚本交付时**未**在真实树执行 ``--apply``。
- dry-run 绝不写引擎树（唯一写操作 = ``--log`` 日志文件）；不调用 git、不构建。
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 路径与常量
# --------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
MLIP_ROOT = SCRIPT_PATH.parents[1]  # .../task3_sampling_database/mlip
DOCS_DIR = MLIP_ROOT / "docs"
DEFAULT_ENGINE = Path(
    os.environ.get(
        "SYMMETRIX_XL_ENGINE",
        "/workspace/group/shared/bza/external-repos/symmetrix-xl",
    )
)

# 目标树 = 当前新结构（polar 独立 TU）
ENGINE_FILES = {
    "tu": "libsymmetrix/source/mace_polar_impl.tpp",
    "hpp": "libsymmetrix/source/mace_nonlinear_kokkos.hpp",
    "tpp": "libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp",
    "cpp": "symmetrix/source/cpp/mace_nonlinear_kokkos.cpp",
}

# v2 编写时的只读快照（仅漂移提示；锚点文本为准）
# 注: tu 于 2026-10-15 12:13 被并行进程更新（工作密度引入, 994→1000 行）；
# 本节快照取更新后读数, 锚点与插入点不随行号漂移（见 merge_adapt_67.md §0/§4）。
V2_SNAPSHOT = {
    "tu": ("139bf0d40b960ea812b0ec674c66a186", 1000),
    "hpp": ("a01de8b84524fdf6f50b03d4b6f78ecd", 821),
    "tpp": ("c767274a54506063d46dc564f10c86ff", 7358),
    "cpp": ("ba096f9a333db0df3b8e2fdd719f4d8f", 587),
}

# --------------------------------------------------------------------------
# 引擎树锚点（字面文本；全部要求 count==1）
# --------------------------------------------------------------------------
HPP_ANCHOR_PUBLIC = (
    "    Kokkos::View<Precision**,Kokkos::LayoutRight> polar_mixer_temp,\n"
    "        polar_mixer_temp_storage,polar_scratch_a,polar_scratch_a_storage,\n"
    "        polar_scratch_b,polar_scratch_b_storage,polar_nl_input,\n"
    "        polar_nl_input_storage,polar_nl_a,polar_nl_a_storage,polar_nl_b,\n"
    "        polar_nl_b_storage;                                             // ⑤暂存"
)
HPP_ANCHOR_PRIVATE = (
    "    std::vector<E3LinearKokkosT<Precision>> polar_mixer_linears;\n"
    "    std::vector<PolarUpdate> polar_updates;"
)
HPP_ANCHOR_DECL = (
    "    void run_polar_pipeline(int num_nodes,\n"
    "        Kokkos::View<Precision**,Kokkos::LayoutRight> node_attrs,\n"
    "        Kokkos::View<const double*> node_positions,\n"
    "        Kokkos::View<const double*> cell,\n"
    "        double total_charge,double total_spin);"
)
TPP_ANCHOR_LINE = "    } else polar_enabled=false;"
TU_ANCHOR_INSTANTIATION = "template class MaceNonlinearKokkosT<float>;"
TU_ANCHOR_RUNTIME_TAIL = (
    "            Kokkos::parallel_for(\"polar scf final density\",\n"
    "                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},\n"
    "                    {num_nodes,8}),\n"
    "                KOKKOS_LAMBDA(int node,int column) {\n"
    "                    final_density(node,column)=density_all(node,column);\n"
    "                });\n"
    "        }\n"
    "    }"
)
CPP_ANCHOR_CHAIN_TAIL = (
    "        .def(\"polar_scf_final_density\",[](const Evaluator& self) {\n"
    "            return view2vector(self.polar_scf_final_density);\n"
    "        });"
)

# --------------------------------------------------------------------------
# 块表：4 组 / 7 个插入点（v2 新结构）
#
# op:
#   "insert_after"  : 整块锚点之后插入
#   "insert_before" : 整块锚点之前插入
#   "insert_between": 锚点第 prefix_lines 行与第 prefix_lines+1 行**之间**插入
#   "replace"       : 整块锚点替换为 content
# --------------------------------------------------------------------------
BLOCK_SPECS = [
    dict(
        order=1, bid="1a-hpp-public", group="① hpp 增补",
        file="hpp", op="insert_after", anchor=HPP_ANCHOR_PUBLIC,
        builder="hpp_public",
        expect_after=(
            "    void compute_node_energies_forces(int num_nodes,Kokkos::View<const int*> node_types,",
        ),
        note="module7 §4.1a + module6 §1（补丁 A）合并；公开区（binding 访问）",
    ),
    dict(
        order=2, bid="1b-hpp-private", group="① hpp 增补",
        file="hpp", op="insert_after", anchor=HPP_ANCHOR_PRIVATE,
        builder="hpp_private",
        expect_after=("    std::vector<LayerState> states;",),
        note="module7 §4.1b；私有区（表与配置, 仅成员函数消费）",
    ),
    dict(
        order=3, bid="1c-hpp-decl", group="① hpp 增补",
        file="hpp", op="insert_after", anchor=HPP_ANCHOR_DECL,
        builder="hpp_decl",
        expect_after=("", "private:"),
        note="parse_polar_extension_tables() 声明（run_polar_pipeline 声明之后）",
    ),
    dict(
        order=4, bid="2-maintpp-line", group="② 主 tpp 1 行",
        file="tpp", op="insert_after", anchor=TPP_ANCHOR_LINE,
        builder="main_one_line",
        expect_before=(
            "                \"polar_blocks: charges_maps count must match num_recursion_steps.\");",
            "            throw std::invalid_argument(",
        ),
        expect_after=("    if constexpr(std::is_same_v<Precision,float>",),
        note="ctor polar 解析块收尾之后调用 parse_polar_extension_tables()",
    ),
    dict(
        order=5, bid="3a-tu-parse-fn", group="③ 新 TU 两段",
        file="tu", op="insert_before", anchor=TU_ANCHOR_INSTANTIATION,
        builder="parse_fn",
        expect_before=("", "}"),
        expect_after=("template class MaceNonlinearKokkosT<double>;",),
        note="parse_polar_extension_tables() 定义（⑥-B + ⑦-4.2a 合并；显式实例化之前）",
    ),
    dict(
        order=6, bid="3b-tu-runtime", group="③ 新 TU 两段",
        file="tu", op="insert_between", anchor=TU_ANCHOR_RUNTIME_TAIL,
        prefix_lines=7,
        builder="runtime_segment",
        expect_before=(
            "            auto final_density=polar_scf_final_density;",
            "                polar_scf_final_density,num_nodes,8);",
        ),
        expect_after=("}", ""),
        note=("⑥-C1 + ⑦-4.3 + ⑦-4.4 + ⑥-C2（⑤ SCF 块之后、if 块闭括号之前 = 函数尾插点; "
              "含 4.3 末态密度代名与 ⑥-C2 energies→node_energies 两处名字适配）"),
    ),
    dict(
        order=7, bid="4-binding-chain", group="④ binding 链",
        file="cpp", op="replace", anchor=CPP_ANCHOR_CHAIN_TAIL,
        builder="binding_chain",
        expect_before=("        })",),
        expect_after=("}", ""),
        note="链尾展开：⑦-4.5（9 条 .def）+ ⑥-D（3 条 .def）；⑦ 链尾 }); -> }) 后接 ⑥",
    ),
]

# --------------------------------------------------------------------------
# 结构前提核查（应用前；只读断言）
# --------------------------------------------------------------------------
PRE_CHECKS = [
    ("hpp", "    Kokkos::View<double*> atomic_energies,node_energies,node_forces,atom_forces,", 1,
     "node_energies 成员存在且位于公开区（⑥-C2/⑦-4.4 能量写入目标）"),
    ("hpp", "    nlohmann::json polar_json;", 1,
     "polar_json 成员存在（parse_polar_extension_tables 数据源）"),
    ("tpp", "polar_enabled=true;polar_json=pb;", 1,
     "ctor 已把 polar_blocks 存入 polar_json（解析函数可达性前提）"),
    ("tu", "MaceNonlinearKokkosT<Precision>::run_polar_pipeline", 1,
     "run_polar_pipeline 定义唯一（插入目标函数）"),
    ("tu", "auto density_all=polar_working_density;", 2,
     "④/⑤ 均指向工作密度（12:13 版结构; ⑦-4.3 末态改用快照成员的依据）"),
    ("tu", "final_density(node,column)=density_all(node,column);\n"
           "                });\n        }\n    }\n", 1,
     "⑤ SCF 块之后恰无其他内容（3b 插点在函数尾, 9 行锚点唯一性佐证）"),
    ("cpp", CPP_ANCHOR_CHAIN_TAIL, 1,
     "polar 访问器链尾（polar_scf_final_density 块）唯一命中"),
]

# --------------------------------------------------------------------------
# 应用后结构探针（内存结果）
# --------------------------------------------------------------------------
POST_PROBES = [
    ("hpp", "polar_le_q_in_storage", 1, "⑦-4.1a 视图已入"),
    ("hpp", "polar_le_enabled", 2, "⑦-4.1a 注释 + §4.1b 声明"),
    ("hpp", "P3 模块⑦", 2, "§4.1a + §4.1b 原文注释"),
    ("hpp", "P3 模块⑥", 1, "⑥-A 原文注释"),
    ("hpp", "polar_coulomb_si_select", 1, "⑥-A si_select(View<int*>)"),
    ("hpp", "polar_include_self_interaction", 1, "⑥-A include 开关"),
    ("hpp", "    void parse_polar_extension_tables();", 1, "新增声明"),
    ("tpp", "    if(polar_enabled) parse_polar_extension_tables();", 1, "② 主 tpp 1 行"),
    ("tpp", "    } else polar_enabled=false;", 1, "原 polar 块收尾行未动"),
    ("tu", "MaceNonlinearKokkosT<Precision>::parse_polar_extension_tables()", 1, "③a 定义"),
    ("tu", "const auto& epd=polar_json.at(\"electric_potential_descriptor\");", 1, "③a epd 重建"),
    ("tu", "\"polar coulomb rho\"", 1, "⑥-C1 ρ̂ kernel"),
    ("tu", "auto coulomb_density=polar_scf_final_density;", 1, "⑥-C1 密度源=SCF 末态"),
    ("tu", "auto charges_final=polar_scf_final_density;", 1, "⑦-4.3 末态密度(树 12:13 适配)"),
    ("tu", "polar_scf_final_density", 5,
     "⑤ realloc/alias + ⑥-C1 读 + ⑦-4.3 读 + [v2] 注释 1 处"),
    ("tu", "auto coulomb_energies=node_energies;", 1, "⑥-C2 写成员 node_energies"),
    ("tu", "\"add polar coulomb energy\"", 1, "⑥-C2 装配 kernel"),
    ("tu", "\"polar le energy add\"", 1, "⑦-4.4 装配 kernel"),
    ("tu", "launch_le_uvu", 3, "⑦-4.3 uvu 内核（定义 + 2 调用）"),
    ("tu", "auto q_in=polar_le_q_in;", 1, "⑦-4.3 q_in kernel"),
    ("tu", "polar_le_mlp=AffineMLPKokkosT<Precision>(mlp);", 1, "⑦-4.2a mlp 解析"),
    ("tu", "polar_le_total=le_sum;", 1, "⑦-4.3 总量回写"),
    ("tu", "polar_le_enabled", 4, "4.2a 置位/清位 + 4.3/4.4 守卫"),
    ("tu", "P3 模块⑦", 2, "4.2a + 4.3 原文注释"),
    ("tu", "P3 模块⑥", 3, "6-B + C1 + C2 原文注释"),
    ("cpp", ".def(\"polar_le_", 8, "⑦-4.5 九条访问器（8 条 polar_le_* + local_electron）"),
    ("cpp", ".def(\"polar_local_electron_energy\"", 1, "⑦-4.5 逐原子访问器"),
    ("cpp", "polar_le_energy", 1, "⑦-4.5 链尾前"),
    ("cpp", "self.polar_le_total", 1, "⑦ 总量访问器体"),
    ("cpp", ".def(\"polar_coulomb", 3, "⑥-D 三条访问器"),
    ("cpp", "self.polar_coulomb_", 3, "⑥ view2vector 引用"),
]

# 装配序探针：文本中按顺序出现（同 run_polar_pipeline 内）
ORDER_PROBES = [
    ("tu",
     ("Kokkos::parallel_for(\"polar scf final density\"",
      "auto coulomb_density=polar_scf_final_density;"),
     "⑤ 写末态密度 < ⑥-C1 读取"),
    ("tu",
     ("auto coulomb_density=polar_scf_final_density;",
      "\"polar le energy add\"",
      "\"add polar coulomb energy\""),
     "⑥-C1 < ⑦-4.4(add le) < ⑥-C2(add coulomb)（le→coulomb 参考装配序）"),
]

# 幂等标记：应用前必须为 0（>0 说明树已被本补丁（部分）修改）
IDEMPOTENCE_MARKERS = [
    ("hpp", "P3 模块⑦"),
    ("hpp", "P3 模块⑥"),
    ("hpp", "polar_le_q_in_storage"),
    ("tpp", "parse_polar_extension_tables"),
    ("tu", "P3 模块⑥"),
    ("tu", "parse_polar_extension_tables"),
    ("cpp", "polar_le_energy"),
    ("cpp", ".def(\"polar_coulomb"),
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
# 文档解析与内容提取（同 v1 机制）
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
# v2 适配变换（逐项显式; 全部带计数断言）
# --------------------------------------------------------------------------
def sub_pb_to_polar_json(lines: list[str], expected: int, key: str) -> list[str]:
    """`pb.at(...)`/`pb.contains(...)` → `polar_json.at(...)`/`polar_json.contains(...)`."""
    total = sum(line.count("pb.") for line in lines)
    if total != expected:
        raise PatchError(
            f"{key}: `pb.` 出现 {total} 次（预期 {expected}）— 文档可能已变更"
        )
    return [line.replace("pb.", "polar_json.") for line in lines]


def sub_c2_energies(lines: list[str], key: str) -> list[str]:
    """⑥-C2: `auto coulomb_energies=energies;` → 成员 `node_energies`（任务规则 1）."""
    old = "        auto coulomb_energies=energies;"
    new = "        auto coulomb_energies=node_energies;"
    n = sum(1 for line in lines if line == old)
    if n != 1:
        raise PatchError(f"{key}: 预期恰好 1 行 {old!r}, 实际 {n}")
    return [new if line == old else line for line in lines]


def sub_le_charges_final(lines: list[str], key: str) -> list[str]:
    """⑦-4.3: 末态密度代名 `polar_spin_charge_density` → `polar_scf_final_density`.

    [v2 适配·02] 目标树自 2026-10-15 12:13（工作密度引入）起，②/⑤ 改为在函数内
    ``polar_working_density`` 上改写；成员 ``polar_spin_charge_density`` 恢复为只读的
    ① 原始累积（不再是 SCF 末态）。⑤ 末态已发布为快照成员 ``polar_scf_final_density``
    （与工作密度同值；⑥-C1 同源读取），故代名替换。其余行逐字保留。
    """
    old = "            auto charges_final=polar_spin_charge_density;"
    new_lines = [
        "            // [v2 适配] ⑤ 末态密度: 原提案取 polar_spin_charge_density, 该视图自",
        "            // 树 2026-10-15 12:13(工作密度引入) 起为只读①原始累积; 改用同值快照成员。",
        "            auto charges_final=polar_scf_final_density;",
    ]
    n = sum(1 for line in lines if line == old)
    if n != 1:
        raise PatchError(f"{key}: 预期恰好 1 行 {old!r}, 实际 {n}")
    out_lines: list[str] = []
    for line in lines:
        if line == old:
            out_lines.extend(new_lines)
        else:
            out_lines.append(line)
    return out_lines


def reindent(lines: list[str], delta: int, key: str) -> list[str]:
    """整体缩进 ±4（空行保持空; 强制作用域适配的纯空白变换）."""
    out_lines: list[str] = []
    for line in lines:
        if line.strip() == "":
            out_lines.append(line)
            continue
        if delta > 0:
            out_lines.append(" " * delta + line)
        else:
            if not line.startswith(" " * (-delta)):
                raise PatchError(f"{key}: 缩进不足以 -{-delta}: {line!r}")
            out_lines.append(line[-delta:])  # 去掉前 -delta 个空格
    return out_lines


def require(lines: list[str], idx: int, cond: bool, key: str, what: str) -> None:
    if not cond:
        raise PatchError(f"{key}: 结构断言失败（{what}）: 行[{idx}]={lines[idx]!r}")


# --------------------------------------------------------------------------
# 各块内容构建
# --------------------------------------------------------------------------
def build_hpp_public(doc: DocBlocks) -> list[str]:
    key = "①a-hpp"
    a = doc.get("module7_patch_final.md", "### 4.1a hpp", r"^插入", key + ":7-4.1a")
    b = doc.get("module6_patch_final.md", "## 1. 补丁 A", r"^\*\*插入代码全文", key + ":6-A")
    require(a, 0, a[0].startswith("    // P3 模块⑦:"), key, "4.1a 首行")
    require(b, 0, b[0].startswith("    // --- P3 模块⑥:"), key, "6-A 首行")
    return (
        [
            "    // ===== [v2 适配] 模块⑥⑦ hpp 公开增补(binding 与 run_polar_pipeline 共用) =====",
            "    // [公开(供 binding)] ⑦ 运行期视图(module7_patch_final.md §4.1a 原文; 注释 (N,512) 按预检 §8-1 保留):",
        ]
        + a
        + [
            "    // [公开(供 binding)] ⑥ 视图与配置(module6_patch_final.md §1(补丁 A)原文; si 表/bool 随提案同区):",
        ]
        + b
    )


def build_hpp_private(doc: DocBlocks) -> list[str]:
    key = "①b-hpp"
    b = doc.get("module7_patch_final.md", "### 4.1b hpp", r"^插入", key + ":7-4.1b")
    require(b, 0, b[0].startswith("    // --- P3 模块⑦:"), key, "4.1b 首行")
    return [
        "    // ===== [v2 适配] 模块⑥⑦ hpp 私有增补(表与配置; 仅本类成员函数消费) =====",
        "    // [私有] ⑦ 表与配置(module7_patch_final.md §4.1b 原文):",
    ] + b


def build_hpp_decl(doc: DocBlocks) -> list[str]:
    return [
        "    // [v2 适配] 模块⑥⑦ 扩展表解析(定义于独立 TU mace_polar_impl.tpp; ctor 于 polar 解析块收尾调用)。",
        "    void parse_polar_extension_tables();",
    ]


def build_main_one_line(doc: DocBlocks) -> list[str]:
    return ["    if(polar_enabled) parse_polar_extension_tables();"]


def build_parse_fn(doc: DocBlocks) -> list[str]:
    key = "③a-tu-parse"
    b6 = doc.get("module6_patch_final.md", "## 2. 补丁 B", r"^\*\*插入代码全文", key + ":6-B")
    b7 = doc.get("module7_patch_final.md", "### 4.2a tpp", r"^插入", key + ":7-4.2a")
    require(b6, 0, b6[0].strip().startswith("// ---- P3 模块⑥:"), key, "6-B 首行")
    require(b6, len(b6) - 1, b6[-1] == "        }", key, "6-B 末行")
    require(b7, 0, b7[0].strip().startswith("// ---- P3 模块⑦:"), key, "7-4.2a 首行")
    require(b7, len(b7) - 1, b7[-1].strip().startswith("} else polar_le_enabled=false;"),
            key, "7-4.2a 末行")
    b6 = sub_pb_to_polar_json(b6, 1, key + ":6-B")
    b7 = sub_pb_to_polar_json(b7, 3, key + ":7-4.2a")
    b6 = reindent(b6, -4, key + ":6-B")
    b7 = reindent(b7, -4, key + ":7-4.2a")
    return (
        [
            "template<typename Precision>",
            "void MaceNonlinearKokkosT<Precision>::parse_polar_extension_tables()",
            "{",
            "    // [v2 适配] ⑥-B(module6 §2) + ⑦-4.2a(module7 §4.2a) 合并迁入(原 ctor 解析段):",
            "    // 数据源=成员 polar_json(ctor 已赋值); epd 按主 tpp ctor ③ 段同一 JSON 路径重建;",
            "    // 两段整体缩进 -4(函数体作用域); `pb.` 已替换为 `polar_json.`。",
            "    const auto& epd=polar_json.at(\"electric_potential_descriptor\");",
        ]
        + b6
        + [""]
        + b7
        + ["}", ""]
    )


def build_runtime_segment(doc: DocBlocks) -> list[str]:
    key = "③b-tu-runtime"
    c1 = doc.get("module6_patch_final.md", "### 3.2 C1", r"^\*\*插入代码全文", key + ":6-C1")
    f43 = doc.get("module7_patch_final.md", "### 4.3 tpp", r"^插入", key + ":7-4.3")
    f44 = doc.get("module7_patch_final.md", "### 4.4 tpp", r"^插入", key + ":7-4.4")
    c2 = doc.get("module6_patch_final.md", "### 3.3 C2", r"^\*\*插入代码全文", key + ":6-C2")
    require(c1, 0, c1[0].strip().startswith("// ---- P3 模块⑥:"), key, "C1 首行")
    require(c1, len(c1) - 1, c1[-1] == "        }", key, "C1 末行")
    require(f43, 0, f43[0] == "    if(polar_enabled&&polar_le_enabled) {", key, "4.3 首行")
    require(f43, len(f43) - 1, f43[-1] == "    }", key, "4.3 末行")
    require(f44, 0, f44[0].startswith("    if(polar_enabled&&polar_le_enabled&&polar_le_add) {"),
            key, "4.4 首行")
    require(f44, len(f44) - 1, f44[-1] == "    }", key, "4.4 末行")
    require(c2, 0, c2[0] == "    if(polar_enabled) {", key, "C2 首行")
    require(c2, len(c2) - 1, c2[-1] == "    }", key, "C2 末行")
    c2 = sub_c2_energies(c2, key + ":6-C2")
    f43 = sub_le_charges_final(f43, key + ":7-4.3")
    return (
        [
            "        // [v2 适配] ⑥⑦ 运行期合计段: ⑥-C1(module6 §3.2 原文, 8 空格缩进不变)",
            "        // + ⑦-4.3/4.4(module7 原文, 整体 +4 缩进; 4.3 末态密度代名 spin_charge",
            "        //   _density -> polar_scf_final_density, 见行内注释) + ⑥-C2(module6 §3.3",
            "        // 原文, +4 缩进; energies 局部别名 -> 成员 node_energies)。插入点=⑤ 块之后、函数尾前。",
        ]
        + c1
        + [""]
        + reindent(f43, 4, key + ":7-4.3")
        + [""]
        + reindent(f44, 4, key + ":7-4.4")
        + [""]
        + reindent(c2, 4, key + ":6-C2")
    )


def build_binding_chain(doc: DocBlocks) -> list[str]:
    key = "④-cpp"
    r7 = doc.get("module7_patch_final.md", "### 4.5 binding", r"^替换为", key + ":7-4.5")
    d6 = doc.get("module6_patch_final.md", "## 4. 补丁 D", r"^\*\*插入代码全文", key + ":6-D")
    anchor_lines = CPP_ANCHOR_CHAIN_TAIL.split("\n")
    require(r7, 0, r7[0] == anchor_lines[0], key, "4.5 第 1 行=链尾块")
    require(r7, 1, r7[1] == anchor_lines[1], key, "4.5 第 2 行=链尾块")
    require(r7, 2, r7[2] == "        })", key, "4.5 第 3 行=`        })`")
    require(r7, len(r7) - 1, r7[-1] == "        });", key, "4.5 末行=`        });`")
    require(d6, 0, d6[0].startswith("        .def(\"polar_coulomb_energy\""), key, "6-D 首行")
    require(d6, len(d6) - 1, d6[-1] == "        });", key, "6-D 末行")
    # v2 合并：⑦ 链尾 `        });` → `        })`，其后接 ⑥ 3 条 .def（含最终 `;`）
    return r7[:-1] + ["        })"] + d6


BUILDERS = {
    "hpp_public": build_hpp_public,
    "hpp_private": build_hpp_private,
    "hpp_decl": build_hpp_decl,
    "main_one_line": build_main_one_line,
    "parse_fn": build_parse_fn,
    "runtime_segment": build_runtime_segment,
    "binding_chain": build_binding_chain,
}

# 每个块的内容来源（DocBlocks.refs 键）; 用于逐块报告
CONTENT_REFS = {
    "hpp_public": ("①a-hpp:7-4.1a", "①a-hpp:6-A"),
    "hpp_private": ("①b-hpp:7-4.1b",),
    "hpp_decl": (),
    "main_one_line": (),
    "parse_fn": ("③a-tu-parse:6-B", "③a-tu-parse:7-4.2a"),
    "runtime_segment": ("③b-tu-runtime:6-C1", "③b-tu-runtime:7-4.3",
                        "③b-tu-runtime:7-4.4", "③b-tu-runtime:6-C2"),
    "binding_chain": ("④-cpp:7-4.5", "④-cpp:6-D"),
}


# --------------------------------------------------------------------------
# 内存应用（顺序模拟）
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
            f"请对照 docs/merge_adapt_67.md 的只读快照复跑复验"
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
    elif blk["op"] == "insert_between":
        k = blk["prefix_lines"]
        if not (0 <= k <= len(anchor_lines)):
            raise PatchError(f"块 {bid}: prefix_lines={k} 越界（锚点 {len(anchor_lines)} 行）")
        offset = idx + sum(len(line) + 1 for line in anchor_lines[:k])
        inserted = "\n".join(insert_lines) + "\n"
        new_content = content[:offset] + inserted + content[offset:]
        detail = (
            f"锚点第 {k}/{len(anchor_lines)} 行（`{anchor_lines[k - 1].strip()}`）与"
            f"第 {k + 1} 行（`{anchor_lines[k].strip()}`）之间插入 {len(insert_lines)} 行"
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
        anchor_ref=blk.get("anchor_ref", "-"), note=blk["note"],
        insert_lines=len(insert_lines),
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
# 主流程
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    global TEE
    parser = argparse.ArgumentParser(
        description="⑥/⑦ polar 引擎补丁 v2（新结构适配）应用器：默认 dry-run（不写引擎树）",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只读模拟（默认）")
    mode.add_argument("--apply", action="store_true", help="真实写入（前置: dry-run 全绿）")
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE, help="symmetrix-xl 引擎树根")
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR, help="patch_final 文档目录")
    parser.add_argument("--log", type=Path, default=None, help="把完整输出同时写入该日志文件")
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

    out("=" * 78)
    out(f"apply_polar_patches_v2.py — 模式: {mode_name}（引擎树写操作: {'是' if args.apply else '否'}）")
    out("目标树: polar 独立 TU 新结构（mace_polar_impl.tpp + 主 tpp/hpp/binding）")
    out(f"引擎: {engine}")
    out(f"文档: {docs_dir}/module6_patch_final.md, module7_patch_final.md; 适配裁定: merge_adapt_67.md")
    out("=" * 78)

    # ---- 环境检查 ---------------------------------------------------------
    if not engine.is_dir():
        raise PatchError(f"引擎目录不存在: {engine}")
    for key, rel in ENGINE_FILES.items():
        if not (engine / rel).is_file():
            raise PatchError(f"引擎文件缺失: {engine / rel}")

    # ---- 文档内容提取 + 适配变换 -------------------------------------------
    out("")
    out("[0] 文档解析（插入内容逐字提取 + v2 适配变换, 含结构断言）")
    doc = DocBlocks(docs_dir)
    blocks = []
    for spec in BLOCK_SPECS:
        content = BUILDERS[spec["builder"]](doc)
        blk = dict(spec)
        blk["content_lines"] = content
        refs = [doc.refs[k] for k in CONTENT_REFS[spec["builder"]]]
        blk["content_ref"] = "; ".join(refs) if refs else "(本脚本字面文本; 无文档代码块)"
        blocks.append(blk)
    for blk in blocks:
        out(f"    {blk['order']:>2}. {blk['bid']:<16} {blk['file']:<4} "
            f"内容 {len(blk['content_lines']):>4} 行")
        out(f"        {blk['note']}")
    out(f"    -> 共 {len(blocks)} 个插入点 / 4 组（①hpp 增补 ×3、②主 tpp 1 行、"
        f"③新 TU ×2、④binding 链 ×1）")
    out("    来源引用（代码块逐字提取位置）:")
    for key in sorted(doc.refs):
        out(f"        {key:<24} {doc.refs[key]}")

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
        exp_md5, exp_n = V2_SNAPSHOT[key]
        drift = "" if (md5 == exp_md5 and nlines == exp_n) else "  <-- 相对 v2 编写快照有漂移（以锚点文本为准）"
        out(f"    {key:<4} {rel}")
        out(f"         {nlines:>6} 行  md5={md5}{drift}")

    # ---- 结构前提核查（只读） ---------------------------------------------
    out("")
    out("[2] 结构前提核查（应用前; 全部要求 count==1）")
    for key, needle, expected, note in PRE_CHECKS:
        occ, nlin = probe_counts(originals[key], needle)
        ok = "OK " if occ == expected else "FAIL"
        out(f"    [{ok}] {key:<4} {needle.splitlines()[0][:46]!r:<50} occ={occ} (lines={nlin})")
        out(f"           <- {note}")
        if occ != expected:
            raise PatchError(f"结构前提核查失败: {key} {needle!r} occ={occ} 期望={expected}")

    # ---- 幂等标记检查 ------------------------------------------------------
    out("")
    out("[3] 幂等标记（应用前应全为 0；>0 ⇒ 疑似已应用）")
    for key, needle in IDEMPOTENCE_MARKERS:
        occ, _ = probe_counts(originals[key], needle)
        out(f"    {key:<4} {needle!r:<40} x{occ}")
        if occ != 0:
            raise PatchError(
                f"{key}: 标记 {needle!r} 已存在（x{occ}）— 树疑似已被本补丁（部分）修改, 拒绝继续"
            )

    # ---- 顺序模拟应用 ------------------------------------------------------
    out("")
    out("[4] 逐块应用（内存顺序模拟: ①→②→③→④）")
    state = dict(originals)
    reports: list[dict] = []
    for blk in blocks:
        content_before = state[blk["file"]]
        state[blk["file"]], rep = apply_block(content_before, blk)
        reports.append(rep)
        out(
            f"    [{blk['order']}/7] {rep['bid']:<16} {rep['file']:<4} "
            f"锚点 count=1（行 {rep['start_line']}）| {rep['detail']} | Δ{rep['delta']}"
        )
        out(f"           来源: {rep['content_ref']}")
    out("    （7/7 块锚点唯一命中并完成内存模拟）")

    # ---- 应用后结构探针 ----------------------------------------------------
    out("")
    out("[5] 应用后结构探针（内存结果; 期望值见脚本块表注释）")
    probe_failures = []
    for key, needle, expected, src in POST_PROBES:
        occ, nlin = probe_counts(state[key], needle)
        ok = "OK " if occ == expected else "FAIL"
        if occ != expected:
            probe_failures.append(f"{key}:{needle!r} occ={occ} 期望={expected}")
        out(f"    [{ok}] {key:<4} {needle!r:<44} occ={occ} (line-count={nlin})  <- {src}")
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
    for key in ("tu", "hpp", "tpp", "cpp"):
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
    nfiles = sum(1 for k in ("tu", "hpp", "tpp", "cpp") if originals[k] != state[k])
    out(f"    {nfiles} files changed, {total_ins} insertions(+), {total_del} deletions(-)")
    out("    说明: 仅本补丁（7 个插入点）相对当前工作树的贡献; 真实 git diff --stat 还含树中其它未提交改动。")

    # ---- 结论 --------------------------------------------------------------
    out("")
    green = not probe_failures
    if green:
        out("DRY-RUN 全绿: 7/7 插入点锚点唯一命中并完成内存模拟；结构前提/结构/顺序探针全部通过。")
        out("引擎树未被修改（dry-run 只读）。")
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
    out("[8] APPLY: 备份与写入")
    written: list[tuple[str, Path, Path]] = []  # (key, path, bak)
    try:
        for key in ("tu", "hpp", "tpp", "cpp"):
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
        out("[9] 写入后复验（字节一致 + 探针）")
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
    out("APPLY OK: 7/7 插入点已写入；备份保留为 *.bak。")
    out("提醒: 按 docs/merge_adapt_67.md §后续步骤 进行 重建 -> 装载冒烟 -> ⑥门 -> ⑦门 -> 端到端。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
