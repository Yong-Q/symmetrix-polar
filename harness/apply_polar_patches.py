#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""apply_polar_patches.py —— ⑥/⑦ polar 引擎补丁（文本锚点）应用器.

用途
----
把两份"文本锚点 + 逐行插入代码块"形式的引擎补丁提案机械应用到 symmetrix-xl:

- ``docs/module7_patch_final.md``（⑦ local_electron_energy，8 个插入点）
- ``docs/module6_patch_final.md``（⑥ coulomb_energy，5 个插入点）
- 应用顺序依据 ``docs/merge_preflight_67.md`` §5.1/§6: **先 ⑦(8 步)、后 ⑥(5 步)**，
  共 13 步；预检已证明全部锚点唯一（count==1）。

本脚本的插入内容**程序化提取**自两份 patch_final 文档的代码块（逐字），锚点文本
同样从文档的"锚点"代码块提取（少数以文档行内原文/预检 §6 定位文本显式抄录，带来源
标注）。应用为**内存顺序模拟**：

- ``--dry-run``（默认）: 只读引擎树；逐块打印 文件/锚点命中状态/插入行数；给出各文件
  行数变化与 ``git diff --stat`` 的**预估**（内存 difflib 行差——本脚本不调用 git）。
- ``--apply``: 真实写入。写入**前**重跑全部断言（锚点唯一 + 幂等标记检查），创建
  ``<file>.bak`` 备份；写入后复验（字节一致 + 结构探针 + 装配序探针），任一失败自动
  回滚。仅应在 dry-run 全绿后由用户/主线程执行。

纪律
----
- dry-run 绝不写引擎树（唯一写操作 = ``--log`` 指定的日志文件）；
- 不调用 git、不构建、不改动任何其它文件;
- 锚点断言失败 ⇒ 打印诊断（命中行号/首行位置）并中止（退出码 2）。

来源标注约定：每个块带 ``[doc §节]`` 与 preflight 步骤号；⑥-A/B/D 三块因 ⑦ 先应用而
锚点位移，按 merge_preflight_67.md §5.2(a) 的"顺延规则"定位（详见块表 note）。
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

ENGINE_FILES = {
    "hpp": "libsymmetrix/source/mace_nonlinear_kokkos.hpp",
    "tpp": "libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp",
    "cpp": "symmetrix/source/cpp/mace_nonlinear_kokkos.cpp",
}

# merge_preflight_67.md §1 "本预检定稿读数"（仅作漂移提示；锚点文本为准, 见 §9）
PREFLIGHT_SNAPSHOT = {
    "hpp": ("fa20b4fd45302d3b5384dc3980307526", 815),
    "tpp": ("a4c3b5a5f4202819b74db2ba670aabc6", 8292),
    "cpp": ("ba096f9a333db0df3b8e2fdd719f4d8f", 587),
}

# --------------------------------------------------------------------------
# 块表：13 个插入点，按 merge_preflight_67.md §5.1 顺序（先 ⑦ 后 ⑥）
#
# 字段说明：
#   doc/section        : 来源文档 + 节标题前缀（内容与锚点代码块的提取范围）
#   content_intro      : "插入内容"代码块识别正则（对代码块上方引言行）
#   anchor_source      : "fence"  = 从文档"锚点"代码块提取（intro 匹配 anchor_intro）
#                        "literal"= 显式抄录（文档行内/预检定位原文，见 note）
#                        "prior_tail"= 取前一已解析块插入内容/替换文本的末 n 行（顺延）
#   op                 : "insert"（prefix_lines 表示插入点越过锚点前 k 行；None=整块之后,
#                                    0=整块之前）、"replace"（整块替换为文档"替换为"块）、
#                        "chain_extend"（.def 链尾展开: 锚点末行 `});` -> `})` + 新链）
#   expect_before/after: 锚点块紧邻上/下若干行（近→远/近→远），不符即中止（定位防线）
#   absent_at_apply    : 需在应用时刻"消失"的文档原锚点（证明顺延前提成立）
#   note               : 来源与顺延说明
# --------------------------------------------------------------------------
BLOCKS = [
    dict(
        order=1, bid="7-4.1a",
        doc="module7_patch_final.md", section="### 4.1a hpp",
        content_intro=r"^插入", anchor_source="fence", anchor_intro=r"锚点",
        file="hpp", op="insert", prefix_lines=None,
        expect_after=(
            "    void compute_node_energies_forces(int num_nodes,Kokkos::View<const int*> node_types,",
        ),
        note="module7_patch_final.md §4.1a = merge_preflight §6 step 1（hpp public 视图区）",
    ),
    dict(
        order=2, bid="7-4.1b",
        doc="module7_patch_final.md", section="### 4.1b hpp",
        content_intro=r"^插入", anchor_source="fence", anchor_intro=r"锚点",
        file="hpp", op="insert", prefix_lines=None,
        expect_after=("    std::vector<LayerState> states;",),
        note="module7_patch_final.md §4.1b = merge_preflight §6 step 2",
    ),
    dict(
        order=3, bid="7-4.2a",
        doc="module7_patch_final.md", section="### 4.2a tpp",
        content_intro=r"^插入", anchor_source="fence", anchor_intro=r"锚点",
        file="tpp", op="insert", prefix_lines=3,  # throw…; 与 `} else polar_enabled=false;` 之间
        expect_after=None,
        note="module7_patch_final.md §4.2a = merge_preflight §6 step 3（插于锚点第 3/4 行之间）",
    ),
    dict(
        order=4, bid="7-4.2b",
        doc="module7_patch_final.md", section="### 4.2b tpp",
        content_intro=r"^插入", anchor_source="fence", anchor_intro=r"锚点",
        file="tpp", op="insert", prefix_lines=2,  # if(polar_enabled){…} 闭括号之前
        expect_after=("}", ""),
        note="module7_patch_final.md §4.2b = merge_preflight §6 step 4（闭括号之前）",
    ),
    dict(
        order=5, bid="7-4.2c",
        doc="module7_patch_final.md", section="### 4.2c tpp",
        content_intro=r"^插入", anchor_source="fence", anchor_intro=r"锚点",
        file="tpp", op="insert", prefix_lines=2,
        expect_after=(
            "    if(interactions.size()!=products.size()||interactions.size()!=readouts.size())",
        ),
        note="module7_patch_final.md §4.2c = merge_preflight §6 step 5（闭括号之前）",
    ),
    dict(
        order=6, bid="7-4.3",
        doc="module7_patch_final.md", section="### 4.3 tpp",
        content_intro=r"^插入", anchor_source="fence", anchor_intro=r"锚点",
        file="tpp", op="insert", prefix_lines=0,  # readout 累加循环之前
        expect_before=("", "    }"),
        note="module7_patch_final.md §4.3 = merge_preflight §6 step 6（插于锚点之前）",
    ),
    dict(
        order=7, bid="7-4.4",
        doc="module7_patch_final.md", section="### 4.4 tpp",
        content_intro=r"^插入", anchor_source="fence", anchor_intro=r"锚点",
        file="tpp", op="insert", prefix_lines=None,
        expect_after=(
            "",
            "    ensure_view(harmonic_adjoints,harmonic_adjoints_storage,edges,num_lm);",
        ),
        note="module7_patch_final.md §4.4 = merge_preflight §6 step 7（scale 块之后）",
    ),
    dict(
        order=8, bid="7-4.5",
        doc="module7_patch_final.md", section="### 4.5 binding",
        content_intro=r"^替换为", anchor_source="fence", anchor_intro=r"锚点",
        file="cpp", op="replace",
        expect_after=("}", ""),
        note="module7_patch_final.md §4.5 = merge_preflight §6 step 8（整块替换: 原块+9 个 .def）",
    ),
    dict(
        order=9, bid="6-A",
        doc="module6_patch_final.md", section="## 1. 补丁 A",
        content_intro=r"^\*\*插入", anchor_source="literal",
        # 顺延规则(merge_preflight_67.md §5.2(a)): 原锚点=⑤暂存块末行(polar_nl_b_storage;)
        # 之后; ⑦-4.1a 先应用后, 插入点顺延为 ⑦-4.1a 块之后、声明行之前 ⇒
        # 以声明行为锚 insert-before (prefix_lines=0)。
        anchor_text=(
            "    void compute_node_energies_forces(int num_nodes,Kokkos::View<const int*> node_types,"
        ),
        file="hpp", op="insert", prefix_lines=0,
        expect_before=("    Precision polar_le_total=Precision(0);                              // 末次求值总量",),
        note="module6_patch_final.md §1 = merge_preflight §6 step 9; 锚点顺延见 §5.2(a)（⑦-4.1a 块尾紧邻其上）",
    ),
    dict(
        order=10, bid="6-B",
        doc="module6_patch_final.md", section="## 2. 补丁 B",
        content_intro=r"^\*\*插入", anchor_source="literal",
        # 顺延规则 §5.2(a): 原锚点=throw…charges_maps 计数校验之后; ⑦-4.2a 先应用后,
        # 推荐插到 ⑦-4.2a 块之后、`} else polar_enabled=false;` 之前 ⇒ 以后者为锚。
        anchor_text="    } else polar_enabled=false;",
        file="tpp", op="insert", prefix_lines=0,
        expect_before=("        } else polar_le_enabled=false;",),
        note="module6_patch_final.md §2 = merge_preflight §6 step 10; 锚点顺延见 §5.2(a)",
    ),
    dict(
        order=11, bid="6-C1",
        doc="module6_patch_final.md", section="### 3.2 C1",
        content_intro=r"^\*\*插入", anchor_source="fence", anchor_intro=r"锚点",
        file="tpp", op="insert", prefix_lines=7,  # ⑤ 块尾 `        }` 与 `    }` 之间
        expect_after=("",),
        note="module6_patch_final.md §3.2 = merge_preflight §6 step 11（整段锚点, 插于末行 `    }` 前）",
    ),
    dict(
        order=12, bid="6-C2",
        doc="module6_patch_final.md", section="### 3.3 C2",
        content_intro=r"^\*\*插入", anchor_source="literal",
        anchor_text="    if(has_zbl) {",
        file="tpp", op="insert", prefix_lines=0,
        expect_before=("", "    });"),
        note="module6_patch_final.md §3.3 = merge_preflight §6 step 12（文档行内锚点原文）",
    ),
    dict(
        order=13, bid="6-D",
        doc="module6_patch_final.md", section="## 4. 补丁 D",
        content_intro=r"^\*\*插入", anchor_source="prior_tail", prior=("7-4.5", 4),
        file="cpp", op="chain_extend",
        expect_after=("}", ""),
        absent_at_apply=dict(doc="module6_patch_final.md", section="## 4. 补丁 D", intro=r"锚点"),
        note=(
            "module6_patch_final.md §4 = merge_preflight §6 step 13; 锚点顺延为 ⑦-4.5 后的"
            "当前链尾终端块（§5.2(a)）; 原锚点（polar_scf_final_density 块）应已消失"
        ),
    ),
]

# --------------------------------------------------------------------------
# 应用后结构探针（merge_preflight_67.md §6/§6.1 的预期 grep 的机器化）
# 计数 = 子串出现次数（occurrence）; line_count 一并报告（grep -c 语义）。
# --------------------------------------------------------------------------
POST_PROBES = [
    ("hpp", "polar_le_q_in_storage", 1, "merge_preflight §6 step 1"),
    ("hpp", "P3 模块⑦", 2, "merge_preflight §6 step 2 / module7 §5-B#11"),
    ("hpp", "P3 模块⑥", 1, "merge_preflight §6.1"),
    ("hpp", "polar_coulomb_si_select", 1, "merge_preflight §6 step 9"),
    ("hpp", "polar_include_self_interaction", 1, "merge_preflight §6 step 9"),
    ("tpp", "polar_le_mlp=AffineMLPKokkosT<Precision>(mlp);", 1, "merge_preflight §6 step 3"),
    ("tpp", "polar_le_linear_up_q.set_backend(selected);", 1, "merge_preflight §6 step 4"),
    ("tpp", "polar_le_linear_up_q.set_workspace(e3_linear_workspace);", 1, "merge_preflight §6 step 5"),
    ("tpp", "launch_le_uvu", 3, "merge_preflight §6 step 6（定义 + 2 调用）"),
    ("tpp", "auto q_in=polar_le_q_in;", 1, "merge_preflight §6 step 6"),
    ("tpp", '"polar le energy add"', 1, "merge_preflight §6 step 7"),
    ("tpp", "polar_le_enabled", 6, "module7 §5-B#10"),
    ("tpp", "P3 模块⑦", 2, "module7 §5-B#11"),
    ("tpp", '"polar coulomb rho"', 1, "merge_preflight §6 step 11"),
    ("tpp", '"add polar coulomb energy"', 1, "merge_preflight §6 step 12"),
    ("tpp", "P3 模块⑥", 3, "merge_preflight §6.1"),
    ("tpp", "auto coulomb_density=polar_scf_final_density;", 1, "module6 §5 #7（密度源=SCF 末态）"),
    ("cpp", "polar_le_energy", 1, "merge_preflight §6 step 8"),
    ("cpp", '.def("polar_coulomb', 3, "merge_preflight §6 step 13"),
    ("cpp", "self.polar_coulomb_", 3, "自加（访问器 view2vector 引用计数）"),
]

# 装配序探针：文本中按顺序出现（preflight §6.1）
ORDER_PROBES = [
    ("tpp",
     ('Kokkos::parallel_for("scale nonlinear energies"',
      '"polar le energy add"',
      "    if(has_zbl) {"),
     "装配序 scale < le < zbl (merge_preflight §6.1)"),
    ("tpp",
     ('Kokkos::parallel_for("polar scf final density"',
      "auto coulomb_density=polar_scf_final_density;"),
     "⑤ 写入 < ⑥ 读取 (merge_preflight §6.1)"),
]

# 幂等标记：应用前必须为 0（>0 说明树已被本补丁（部分）修改）
IDEMPOTENCE_MARKERS = [
    ("hpp", "P3 模块⑦"),
    ("hpp", "P3 模块⑥"),
    ("tpp", "P3 模块⑦"),
    ("tpp", "P3 模块⑥"),
    ("cpp", "polar_le_energy"),
    ('cpp', '.def("polar_coulomb'),
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
# 文档解析：代码块提取
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
            # intro = 代码块上方紧邻的整段：先跳过空行, 再由下至上收集连续非空行
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


# --------------------------------------------------------------------------
# 块解析（文档 → 锚点文本 + 插入内容）
# --------------------------------------------------------------------------
def resolve_blocks(docs_dir: Path) -> list[dict]:
    cache: dict[str, tuple[list[str], list[dict]]] = {}
    resolved: list[dict] = []
    by_bid: dict[str, dict] = {}
    for spec in BLOCKS:
        doc_name = spec["doc"]
        if doc_name not in cache:
            cache[doc_name] = split_fences(docs_dir / doc_name)
        doc_lines, fences = cache[doc_name]
        sec_fences = section_fences(doc_name, doc_lines, fences, spec["section"])

        content_fence = pick_fence(
            doc_name, spec["section"], sec_fences, spec["content_intro"],
            "插入内容" if spec["op"] != "replace" else "替换内容",
        )
        entry = dict(spec)
        entry["content_lines"] = content_fence["body"]
        entry["content_ref"] = f"{doc_name} L{content_fence['start']}-{content_fence['end'] - 1}"

        if spec["anchor_source"] == "fence":
            anchor_fence = pick_fence(
                doc_name, spec["section"], sec_fences, spec["anchor_intro"], "锚点"
            )
            entry["anchor"] = "\n".join(anchor_fence["body"])
            entry["anchor_ref"] = f"{doc_name} L{anchor_fence['start']}-{anchor_fence['end'] - 1}"
        elif spec["anchor_source"] == "literal":
            entry["anchor"] = spec["anchor_text"]
            entry["anchor_ref"] = f"{doc_name} {spec['section']} 行内原文（预检定位）"
        elif spec["anchor_source"] == "prior_tail":
            prior_bid, n_tail = spec["prior"]
            prior = by_bid.get(prior_bid)
            if prior is None:
                raise PatchError(
                    f"{spec['bid']}: 依赖块 {prior_bid} 必须先行解析（块表顺序错误）"
                )
            tail = prior["content_lines"][-n_tail:]
            entry["anchor"] = "\n".join(tail)
            entry["anchor_ref"] = (
                f"顺延锚点: {prior_bid} 内容末 {n_tail} 行（merge_preflight §5.2(a)）"
            )
        else:
            raise PatchError(f"{spec['bid']}: 未知 anchor_source={spec['anchor_source']!r}")

        # 定稿校验：文档"替换为"块与锚点关系（⑦-4.5: 原块前 2 行 + `})` + 9 个 .def）
        if spec["op"] == "replace":
            a = entry["anchor"].split("\n")
            r = entry["content_lines"]
            if not (r[0] == a[0] and r[1] == a[1] and r[2] == "        })" and r[-1].strip() == "});"):
                raise PatchError(
                    f"{spec['bid']}: 替换块与锚点块形态不符（预期 原块前 2 行 + 收起括号 + 新链）"
                )
        if spec["op"] == "chain_extend":
            if not entry["anchor"].split("\n")[-1].strip().endswith("});"):
                raise PatchError(f"{spec['bid']}: 链尾锚点末行必须以链尾收起括号结尾")
            if not entry["content_lines"][0].lstrip().startswith('.def("'):
                raise PatchError(f"{spec['bid']}: 插入内容需以 `.def(` 开头")
        if spec["op"] == "insert":
            if not entry["content_lines"]:
                raise PatchError(f"{spec['bid']}: 插入内容为空")

        entry["absent_anchor"] = None
        if spec.get("absent_at_apply"):
            aa = spec["absent_at_apply"]
            aa_lines, aa_fences = cache[aa["doc"]]
            aa_sec = section_fences(aa["doc"], aa_lines, aa_fences, aa["section"])
            f = pick_fence(aa["doc"], aa["section"], aa_sec, aa["intro"], "原锚点")
            entry["absent_anchor"] = "\n".join(f["body"])
            entry["absent_anchor_ref"] = f"{aa['doc']} L{f['start']}-{f['end'] - 1}"

        resolved.append(entry)
        by_bid[spec["bid"]] = entry
    return resolved


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
            f"请对照 merge_preflight_67.md §9 复跑复验"
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
    if blk.get("absent_anchor"):
        needle = "\n" + blk["absent_anchor"] + "\n"
        cnt = content.count(needle)
        if cnt != 0:
            raise PatchError(
                f"块 {bid}: 顺延前提不成立 — 原锚点（{blk['absent_anchor_ref']}）"
                f"在应用时刻仍存在 count={cnt}（应为 0）"
            )

    insert_lines = blk["content_lines"]
    if blk["op"] == "insert":
        k = blk["prefix_lines"]
        if k is None:
            k = len(anchor_lines)
        if not (0 <= k <= len(anchor_lines)):
            raise PatchError(f"块 {bid}: prefix_lines={k} 越界（锚点 {len(anchor_lines)} 行）")
        offset = idx + sum(len(line) + 1 for line in anchor_lines[:k])
        inserted = "\n".join(insert_lines) + "\n"
        new_content = content[:offset] + inserted + content[offset:]
        detail = (
            f"插入 {len(insert_lines)} 行 @ 行 {line_of(content, offset)}"
            f"（锚点 {start_line}-{start_line + len(anchor_lines) - 1}，越过前 {k}/{len(anchor_lines)} 行）"
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
    elif blk["op"] == "chain_extend":
        if not anchor_lines[-1].strip().endswith("});"):
            raise PatchError(f"块 {bid}: 链尾锚点末行不以链尾收起括号结尾")
        replacement_lines = anchor_lines[:-1] + ["        })"] + insert_lines
        replacement = "\n".join(replacement_lines)
        new_content = content[:idx] + replacement + content[idx + len(anchor):]
        detail = (
            f"链尾展开: {len(anchor_lines)} 行 → {len(replacement_lines)} 行"
            "（链尾 `});` 改为 `})` + 新链；"
            f"锚点行 {start_line}）"
        )
        delta = f"{len(replacement_lines) - len(anchor_lines):+d}"
    else:
        raise PatchError(f"块 {bid}: 未知 op={blk['op']!r}")

    report = dict(
        bid=bid, file=blk["file"], anchor_hits=1, start_line=start_line,
        anchor_lines=len(anchor_lines), detail=detail, delta=delta,
        content_ref=blk["content_ref"], anchor_ref=blk["anchor_ref"],
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
# 主流程
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    global TEE
    parser = argparse.ArgumentParser(
        description="⑥/⑦ polar 引擎补丁（文本锚点）应用器：默认 dry-run（不写引擎树）",
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
    out(f"apply_polar_patches.py — 模式: {mode_name}（引擎树写操作: {'是' if args.apply else '否'}）")
    out(f"日期: 2026-10-15（脚本自述；无副作用）")
    out(f"引擎: {engine}")
    out(f"文档: {docs_dir}/module6_patch_final.md, module7_patch_final.md, merge_preflight_67.md")
    out("=" * 78)

    # ---- 环境检查 ---------------------------------------------------------
    if not engine.is_dir():
        raise PatchError(f"引擎目录不存在: {engine}")
    for key, rel in ENGINE_FILES.items():
        if not (engine / rel).is_file():
            raise PatchError(f"引擎文件缺失: {engine / rel}")

    # ---- 解析全部 13 块（内容程序化提取 + 锚点解析） -----------------------
    out("")
    out("[0] 文档解析（插入内容与锚点文本）")
    blocks = resolve_blocks(docs_dir)
    if len(blocks) != 13:
        raise PatchError(f"块表应为 13 块, 实际 {len(blocks)}")
    for blk in blocks:
        out(f"    {blk['order']:>2}. {blk['bid']:<7} {blk['file']:<4} 内容 {blk['content_ref']}")
        out(f"        锚点来源: {blk['anchor_ref']}")
    out(f"    -> 共 {len(blocks)} 块（⑦ 8 步 + ⑥ 5 步；顺序 = merge_preflight_67.md §5.1/§6）")

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
        exp_md5, exp_n = PREFLIGHT_SNAPSHOT[key]
        drift = "" if (md5 == exp_md5 and nlines == exp_n) else "  <-- 相对 §1 定稿读数有漂移（以锚点文本为准, §9）"
        out(f"    {key:<4} {rel}")
        out(f"         {nlines:>6} 行  md5={md5}{drift}")

    # ---- 幂等标记检查 ------------------------------------------------------
    out("")
    out("[2] 幂等标记（应用前应全为 0；>0 ⇒ 疑似已应用）")
    for key, needle in IDEMPOTENCE_MARKERS:
        occ, _ = probe_counts(originals[key], needle)
        out(f"    {key:<4} {needle!r:<40} x{occ}")
        if occ != 0:
            raise PatchError(
                f"{key}: 标记 {needle!r} 已存在（x{occ}）— 树疑似已被本补丁（部分）修改, 拒绝继续"
            )

    # ---- 顺序模拟应用 ------------------------------------------------------
    out("")
    out("[3] 逐块应用（内存顺序模拟: ⑦ 先, ⑥ 后）")
    state = dict(originals)
    reports: list[dict] = []
    for blk in blocks:
        content_before = state[blk["file"]]
        state[blk["file"]], rep = apply_block(content_before, blk)
        reports.append(rep)
        out(
            f"    [{blk['order']:>2}/13] {rep['bid']:<7} {rep['file']:<4} "
            f"锚点 count=1（行 {rep['start_line']}）| {rep['detail']} | Δ{rep['delta']}"
        )
        out(f"           来源: {rep['content_ref']}；{rep['note']}")

    # ---- 应用后结构探针 ----------------------------------------------------
    out("")
    out("[4] 应用后结构探针（内存结果; 期望值 = merge_preflight_67.md §6/§6.1）")
    probe_failures = []
    for key, needle, expected, src in POST_PROBES:
        occ, nlin = probe_counts(state[key], needle)
        ok = "OK " if occ == expected else "FAIL"
        if occ != expected:
            probe_failures.append(f"{key}:{needle!r} occ={occ} 期望={expected}")
        out(f"    [{ok}] {key:<4} {needle!r:<42} occ={occ} (line-count={nlin})  <- {src}")
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
    out("[5] 应用后行数变化（相对当前工作树；本脚本不调用 git）")
    total_ins = total_del = 0
    stat_rows = []
    for key in ("hpp", "tpp", "cpp"):
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
    out("[6] git diff --stat 预估（difflib 内存行差；git 未调用）")
    for row in stat_rows:
        out(f"    {row.strip()}")
    nfiles = sum(1 for k in ("hpp", "tpp", "cpp") if originals[k] != state[k])
    out(f"    {nfiles} files changed, {total_ins} insertions(+), {total_del} deletions(-)")
    out("    说明: 仅本补丁（13 块）相对当前工作树的贡献; 真实 git diff --stat 还含树中其它未提交改动。")

    # ---- 结论 --------------------------------------------------------------
    out("")
    green = not probe_failures
    if green:
        out("DRY-RUN 全绿: 13/13 块锚点唯一命中并完成内存模拟；结构/顺序/探针全部通过。")
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
    out("[7] APPLY: 备份与写入")
    written: list[tuple[str, Path, Path]] = []  # (key, path, bak)
    try:
        for key in ("hpp", "tpp", "cpp"):
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
        out("[8] 写入后复验（字节一致 + 探针）")
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
    out("APPLY OK: 13/13 块已写入；备份保留为 *.bak。")
    out("提醒: 按 merge_preflight_67.md §7 进行 重建 -> 装载冒烟 -> ⑥门 -> ⑦门 -> 端到端。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
