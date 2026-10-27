#!/usr/bin/env python3
"""构建蒸馏技术报告 PDF（中文，按团队 research-tech-report 规范组织）。

页面职责：摘要与判断 → 数据与模型 → 六分子精度 → 速度 → 重加权边界 →
稳定性与数据 → 方法 → 局限与复现。图由 distill_figures.py 生成，与
README 共用同一份结果。
"""
import json
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np

from distill_figures import C, accuracy_fig, architecture_fig, speed_fig

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "data/full_valid_head_ewald.json").read_text())
OUT = HERE / "DISTILLATION_THEORY_PERFORMANCE_2026-10-27.pdf"

FONT = Path("/workspace/group/shared/bza/.fonts/NotoSansSC.ttf")
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 300,
})
PAGE = (8.5, 11)


def _width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def cjk_wrap(text: str, width: int = 108) -> list[str]:
    """按显示宽度换行，中文按 2 计。"""
    lines, cur, w = [], "", 0
    for ch in text:
        cw = _width(ch)
        if w + cw > width and cur:
            lines.append(cur)
            cur, w = "", 0
        cur += ch
        w += cw
    if cur:
        lines.append(cur)
    return lines


def text_page(pdf, title, paragraphs, fontsize=9.2):
    fig = plt.figure(figsize=PAGE)
    y = 0.95
    fig.text(0.07, y, title, fontsize=15, fontweight="bold", ha="left")
    y -= 0.045
    for para in paragraphs:
        if para.startswith("## "):
            y -= 0.008
            fig.text(0.07, y, para[3:], fontsize=11, fontweight="bold",
                     ha="left")
            y -= 0.024
            continue
        for line in cjk_wrap(para, width=96):
            if y < 0.055:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=PAGE)
                y = 0.95
            fig.text(0.07, y, line, fontsize=fontsize, ha="left")
            y -= 0.0182
        y -= 0.012
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def figure_page(pdf, fig, title, caption, width_frac=0.88):
    fig.set_dpi(300)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    page = plt.figure(figsize=PAGE)
    page.text(0.06, 0.925, title, fontsize=13.5, fontweight="bold", ha="left")
    aspect = img.shape[1] / img.shape[0]
    w = width_frac * PAGE[0]
    h = w / aspect
    top = 0.895
    ax = page.add_axes([0.06, top - h / PAGE[1], width_frac, h / PAGE[1]])
    ax.imshow(img)
    ax.axis("off")
    y = top - h / PAGE[1] - 0.035
    for para in caption:
        for line in cjk_wrap(para, width=100):
            page.text(0.06, y, line, fontsize=8.6, ha="left")
            y -= 0.019
        y -= 0.008
    pdf.savefig(page, bbox_inches="tight")
    plt.close(page)


# ------------------------------------------------------------------ page 1
def cover_page(pdf):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.06, 0.955, "从 MACE-Polar 蒸馏线性 MACE", fontsize=20,
             fontweight="bold")
    fig.text(0.06, 0.922,
             "六分子精度、速度，以及一步重加权为什么在 1,152 原子不可用",
             fontsize=12)
    fig.text(0.06, 0.900, "2026 年 10 月 27 日 · Symmetrix-Polar / 蒸馏线",
             fontsize=9, color=C["grey"])
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.890, 0.890],
                              color=C["grey"], lw=1.0))

    cards = [
        ("能量 / 力（3,465 帧）", "1.12 / 7.75", "meV/atom / meV/Å"),
        ("六分子气相", "1.17 / 1.00", "meV/atom / meV/Å"),
        ("原子电荷 q", "0.0117 e", "4π 标度，867 帧"),
        ("l=1 偶极 μ", "0.0475", "4π 标度，867 帧"),
        ("预测多极的 Ewald 能", "0.311", "meV/atom"),
        ("速度（5,184 原子，A100）", "4.47 / ~20.2",
         "仅骨架 FP32 / 完整链路 FP64（投影）"),
    ]
    x0, y0, w, h = 0.06, 0.735, 0.275, 0.100
    dx, dy = 0.30, 0.118
    for i, (lab, val, unit) in enumerate(cards):
        r, c = divmod(i, 3)
        x, y = x0 + c * dx, y0 - r * dy
        fig.add_artist(FancyBboxPatch((x, y), w, h,
                                      boxstyle="round,pad=0.008,rounding_size=0.012",
                                      transform=fig.transFigure,
                                      fc="#EAF2F8", ec=C["grey"], lw=0.7))
        fig.text(x + 0.012, y + h - 0.024, lab, fontsize=8.4, color=C["grey"])
        fig.text(x + 0.012, y + 0.038, val, fontsize=16, fontweight="bold",
                 color=C["blue"])
        fig.text(x + 0.012, y + 0.016, unit, fontsize=7.6, color=C["grey"])

    body = [
        "128 通道线性 MACE 学生从 MACE-Polar-1-M 教师蒸馏，在全部 23,376 帧"
        "（3,465 帧留出）上达到 1.12 meV/atom 与 7.75 meV/Å；电荷与 l=1 "
        "多极读出在 867 帧上给出 0.0117 e 与 0.0475（4π 标度），把预测多极送进"
        "教师同一套 GTO/Ewald 核，静电能与教师相差 0.311 meV/atom。",
        "速度分三档，不能混用：仅骨架的 FP32 引擎路径 4.47 µs/atom/step"
        "（实测，5,184 原子），同一骨架 FP64 为 12.3；完整链路（骨架 + q/μ "
        "读出 + Ewald）在 FP64 下约 20.2 µs/atom/step，这是投影值，因为"
        "Symmetrix 标准 MACE pair style 还没有电荷/偶极输出。",
        "重加权边界：1,152 原子的一步重要性采样 σ_β=8.02、ESS=0.24 %，"
        "单帧占 91.5 % 权重，不能用于生产；96 原子 ESS≈44 %、气相 98 %。"
        "需要 POLAR 的 c1/c2 时应直接学生采样，修正则用 λ 中间势 + BAR/MBAR。",
    ]
    y = 0.545
    for para in body:
        for line in cjk_wrap(para, width=88):
            fig.text(0.06, y, line, fontsize=9.2, ha="left")
            y -= 0.0195
        y -= 0.016
    fig.text(0.06, 0.115,
             "范围：q/μ 读出只训练了 6,809 帧（覆盖 1,100 个状态），不是全部 23K；"
             "CO2/H2O/NH3 的 Ewald 残差 0.4–0.6 meV/atom，0.1 meV/atom 未达到。",
             fontsize=8.4, color=C["grey"])
    fig.text(0.06, 0.090,
             "同一骨架在 FP32 用 4.47 只是为了与论文的线性 MACE-OFF23（4.50）对齐，"
             "不代表完整模型的成本；完整模型按 FP64 的 ~20 µs 计。",
             fontsize=8.4, color=C["grey"])
    fig.text(0.06, 0.045, "Symmetrix-Polar · 2026-10-27", fontsize=8,
             color=C["grey"])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


with PdfPages(OUT) as pdf:
    cover_page(pdf)

    figure_page(
        pdf, architecture_fig(), "数据、模型与读出：一条从教师到学生的链路", [
            "教师是 512 通道非线性 polar MACE，带 l=1 原子多极、GTO/Ewald 长程和两步"
            "电荷密度迭代；它给 23,376 帧 E/F 标签和 6,809 帧 q/μ 标签，覆盖六种分子、"
            "1,100 个状态。学生是标准线性 MACE：128 通道、两层相互作用、r_cut 5 Å、"
            "848,528 个参数，用教师的 REF_energy/REF_forces 训练 60 个 epoch，力权重 20。",
            "电荷由 0e 特征的 MLP 给出；三个 l=1 多极系数由 1o 特征的完整线性映射"
            "（384→3）给出。这个映射混合通道，在原始坐标系里不满足旋转一致性"
            "（0.825 meV/atom），所以每帧先做 cell QR 旋转到规范系，读出后再转回，"
            "旋转一致性到 4.6e-14 meV/atom；正交晶胞本来就接近规范系，无需重训。",
            "GTO/Ewald 核与教师完全相同（l=1、1.5 Å 涂抹、自能与 PBC 修正）。"
            "预测的 q、μ 送入该核算出的静电能只作诊断；E_elec 不能加到 MACE 总能量上，"
            "因为骨架训练目标已经包含长程项。",
        ], width_frac=0.92)

    figure_page(
        pdf, accuracy_fig(), "六分子精度：能量 1.12 meV/atom、力 7.75 meV/Å", [
            "A、B 分别是 3,465 帧留出集上每个分子的能量与力 MAE，按气相与致密/体相分开；"
            "C 是 867 帧带教师多极标签的电荷与偶极误差（4π 标度，虚线为全帧值）；"
            "D 是把预测 q、μ 送进教师 GTO/Ewald 核与教师系数结果之差。所有数值都是"
            "对教师的 MAE，不是与实验的比较。",
            "CO2 密相的力误差 23.62 meV/Å 看着大，但它的参考力本身约 1.34 eV/Å，"
            "是其他致密分子的两倍，相对误差 1.8 %，与 H2O（1.7 %）、NH3（1.3 %）同档；"
            "C2H4、C2H6、CH4 都在 0.3–0.6 %。其中一个 0.02 mol/Å^3、150 K 的 CO2 状态"
            "仍在弛豫（E/atom 在轨迹上降 1.1 eV，早期帧力达 3.6 eV/Å），它单独把绝对"
            "F 从约 11.5 拉到 23.6 meV/Å；做 c1/c2 时该状态不能当作平衡态使用。",
            "Ewald 残差集中在 CO2、H2O、NH3（0.38–0.63 meV/atom），"
            "C2H4/C2H6/CH4 只有 0.01–0.05；这是局域多极基组的表示限制，"
            "不是帧数不够，0.1 meV/atom 在本表示下未达到。",
        ], width_frac=0.92)

    figure_page(
        pdf, speed_fig(), "速度：骨架 4.47、完整链路约 20 µs/atom（投影）", [
            "所有数值都是 5,184 原子、单张 A100 上的 µs/atom/step。A 图里 4.47 是"
            "仅骨架、FP32、无 q/μ 读出、无 Ewald 项的引擎实测；它只用于说明与论文的"
            "线性 MACE-OFF23（4.50）同档。同一骨架 FP64 是 12.3。完整链路（骨架 + "
            "q/μ + Ewald）按 12.3 + 7.9 + <0.1 ≈ 20.2 计，是投影而不是端到端实测，"
            "因为 Symmetrix 标准 MACE pair style 还没有电荷/偶极输出。",
            "B 图把精度和速度放在同一张图上：蓝色点（骨架 FP32/FP64）用的是骨架"
            "自身的 E/F 精度，橙色点（完整链路投影）假设 q/μ/Ewald 不改变 E/F，"
            "绿色和紫色分别是 128L1 非线性 polar 学生与精确教师的引擎实测。"
            "在 20 µs 预算下，当前方案是唯一把 E/F、q/μ 和长程都放进同一张图的路线。",
        ], width_frac=0.92)

    text_page(pdf, "重加权：1,152 原子的一步重要性采样不可用", [
        "## 结论与条件",
        "用蒸馏学生做源采样、再按 w ∝ exp(−βΔU) 重加权到 POLAR，"
        "在 1,152 原子、300 K、501 帧（500 ps，1 ps 一帧）上得到 σ_β=8.02、"
        "ESS=0.24 %，501 帧里等效只有 1.2 帧；单帧（第 277 帧）占 91.5 % 权重。"
        "96 原子学生自采轨迹（19 帧）σ_β=1.87、ESS≈44 %，气相 336 帧 σ_β=0.16、"
        "ESS=98 %。对照 OFF23→POLAR 同尺寸为 σ_β=13.76、ESS=0.47 %。",
        "## 为什么学生更准、ESS 反而更低",
        "学生整体更接近 POLAR（σ_β 8.0 对 13.8），但 ESS 由最大几个权重决定，"
        "不由 σ_β 单调决定。1,152 原子这批帧是 OFF23 采的，对学生是分布外；"
        "第 277 帧 ΔU 比均值低约 0.59 eV，单帧拿走 91.5 % 权重。OFF23 不在自己的"
        "轨迹上遇到这种离群，所以最大单帧只有 59.6 %。ESS 是尾部敏感诊断，"
        "不能单独作为“可用”判据。",
        "## 尺寸效应与可行路线",
        "ΔU 是广延量：1,152 原子时逐帧标准差 0.207 eV（0.18 meV/atom），"
        "βΔU 的 σ 就是 8；要把 σ_β 压到 3 以下，每原子能量涨落需从 0.18 降到"
        "约 0.07 meV/atom，还要再准 2.5–3 倍。因此生产尺寸不跑一步重加权："
        "直接用学生采样（不加权），偏差由模型误差决定；必须修正时按 test 仓库"
        "Task 6 采样 λ 中间势并用 BAR/MBAR，混合势的能量和力必须来自同一次采样。",
        "## 文献为什么能重加权",
        "Herzog 等（JCTC 2022）的 MLPT 用 200 个训练构型学 ΔU 再配权，MLMC 用"
        "校正势重新采样后只对残差配权；Chehaibou 等（JCTC 2019）把同类方法用于"
        "RPA 吸附焓；Hilpert 与 Kresse（JCP 2026）在 63 个水分子、292 K、5,000 个"
        "去相关构型上对 DFT−MLIP 能量差配权，MACE 的 ESS≈4,200；Thaler 与"
        "Zavadlav（Nat Commun 2021）把重加权放进可微训练；Shirts 与 Chodera"
        "（JCP 2008）用 MBAR 合并多个平衡态。它们的共同点是：扰动小或目标相同、"
        "用中间态/校正势重新采样、样本大，或观测量不是整盒广延量。",
    ])

    fig = plt.figure(figsize=(8.3, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.38)
    ax = fig.add_subplot(gs[0, 0])
    labels = ["教师\n力旋转", "学生\n力旋转", "原始线性头\n旋转", "规范系\n旋转"]
    vals = [1.5e-8, 4.0e-5, 0.825, 4.6e-14]
    cols = [C["green"], C["blue"], C["orange"], C["purple"]]
    ax.bar(np.arange(4), [max(v, 1e-16) for v in vals], color=cols, width=0.65)
    ax.set_yscale("log"); ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylabel("旋转误差（混合单位）", fontsize=8.5)
    ax.set_title("旋转一致性", fontsize=9.5)
    for i, v in enumerate(vals):
        ax.text(i, max(v, 1e-16) * 2.2, f"{v:.1e}", ha="center", fontsize=7.2)
    ax = fig.add_subplot(gs[0, 1])
    labels = ["E/F 训练", "E/F 留出", "q/μ 训练", "q/μ 留出", "状态数"]
    vals = [23376, 3465, 6809, 867, 1100]
    ax.bar(np.arange(5), vals, color=[C["blue"], C["sky"], C["green"],
                                      C["yellow"], C["grey"]], width=0.65)
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(labels, rotation=25,
                                                    ha="right", fontsize=7.6)
    ax.set_ylabel("帧数 / 状态数", fontsize=8.5)
    ax.set_title("数据覆盖", fontsize=9.5)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:,}", ha="center", fontsize=7.2)
    ax = fig.add_subplot(gs[1, :])
    names = ["E/F 骨架", "q 头", "μ 头", "Ewald", "投影总计"]
    us = [12.3, 0.03, 0.02, 7.9, 20.2]
    ax.barh(np.arange(len(names)), us,
            color=[C["blue"], C["orange"], C["green"], C["sky"], C["grey"]],
            height=0.6)
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=7.8)
    ax.set_xlabel("µs/(atom·step)（FP64，单张 A100）", fontsize=8.5)
    ax.set_title("速度预算拆分", fontsize=9.5)
    for i, v in enumerate(us):
        ax.text(v + 0.2, i, f"{v:.2f}", va="center", fontsize=7.4)
    ax.set_xlim(0, 24)
    ax.invert_yaxis()
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.16, axis="y", lw=0.6)
    fig.tight_layout()
    figure_page(pdf, fig, "稳定性、数据覆盖与速度预算", [
        "学生力场旋转误差 4.0e-5，NVE 漂移 0.245 meV/atom/ps（1 ps）；"
        "原始完整线性头不满足旋转一致性（0.825 meV/atom），规范系变换后到 "
        "4.6e-14，且正交晶胞无需重训。",
        "数据是 23,376/3,465 帧 E/F 与 6,809/867 帧 q/μ，覆盖 1,100 个状态；"
        "ACF 估计约 4,450 个有效独立样本，帧数不等于独立样本数。",
        "速度预算把实测骨架（12.3 FP64）、实测 PyTorch Ewald（1,152 原子 7.9）"
        "和实测头（<0.1）相加得到 20.2；这是投影，不是端到端引擎测量。",
    ])

    text_page(pdf, "方法与记账", [
        "## 教师与数据",
        "教师是 MACE-Polar-1-M：512 通道非线性 polar MACE，带 l=1 原子多极、"
        "精确 GTO/Ewald 长程和两步电荷密度迭代。蒸馏语料含 23,376 帧训练与"
        "3,465 帧留出，六种分子（H2O、NH3、CH4、C2H4、C2H6、CO2）、"
        "1,100/176 个状态，按状态划分。ACF 分析给出约 4,450 个有效独立样本，"
        "23K 不是 23K 个独立样本。",
        "## 骨架与读出",
        "学生是标准线性 MACE：128 通道、两层相互作用、l_max=3、r_cut=5.0 Å、"
        "848,528 个参数，训练 60 个 epoch，力权重 20，Adam + ReduceLROnPlateau。"
        "模型导出为 Symmetrix JSON，在 FP32 direct/JIT 路径上以 4.47 µs/atom/step "
        "运行（5,184 原子）；FP64 物化路径 12.3 µs。",
        "q/μ 语料是 stride-4 训练集加 965 帧状态补充，覆盖全部 1,100 个状态。"
        "0e 与 1o 节点特征取自能量读出；两层 MLP 把 0e 映射到每原子电荷，"
        "完整线性映射（384→3）把 1o 映射到教师的三 l=1 多极系数。该线性映射"
        "不是逐通道等变映射：教师系数来自非线性电荷密度头，需要通道/基组混合；"
        "物理能量和力仍等变，因为 GTO/Ewald 核是协变的。",
        "## 规范系",
        "完整线性头在原始坐标系里依赖取向。每帧先用 3×3 cell QR 旋转到规范"
        "（上三角）取向，再过读出与 Ewald 核，最后把输出转回；每帧一次 3×3 QR，"
        "旋转一致性从 0.825 meV/atom 到 4.6e-14。当前训练晶胞是正交的，本来"
        "就是规范系，因此无需重训；三斜晶胞需要按规范系重训。",
        "## Ewald 诊断与记账",
        "预测 q、μ 送入教师同一套 graph_longrange GTOElectrostaticEnergy 核"
        "（l=1、1.5 Å 涂抹、k 空间因子 1.5、自能与 PBC 修正）。867 帧上预测"
        "系数的静电能与教师系数相差 0.311 meV/atom。MACE 骨架训练目标是教师"
        "总能量，已含长程项；q/μ 与 E_elec 是输出/诊断量，E_elec 不能加到 MACE "
        "总能量上，除非骨架改为对短程残差训练。",
    ])

    text_page(pdf, "局限、未完成与复现", [
        "## 精度边界",
        "0.1 meV/atom 未达到：CO2/H2O/NH3 的 Ewald 残差 0.4–0.6 meV/atom，"
        "属于局域多极基组的表示限制，不是帧数限制。q/μ 只训练了 6,809 帧，"
        "剩余 16,567 帧多为同一状态内的相关帧，标不标对精度影响有限，"
        "但对一致性有用。",
        "## 速度边界",
        "20.2 µs 是投影：引擎骨架 12.3 + PyTorch Ewald 7.9 + 头 <0.1。"
        "Symmetrix 标准 MACE pair style 没有电荷/偶极输出，端到端混合 pair style "
        "仍未实现；在此之前，4.47 只能作为“仅骨架 FP32”的实测值，"
        "不能当作完整模型的成本。",
        "## 重加权与 c1/c2",
        "1,152 原子一步重要性采样不可用（σ_β=8.02、ESS=0.24 %）；"
        "96 原子和气相可用。生产 c1/c2 应直接用学生采样，偏差由模型误差决定；"
        "需要修正时按 λ 中间势 + BAR/MBAR，且先在小体系验收。"
        "学生轨迹给出的 c1/c2 是学生模型的关联函数，不是 POLAR 的精确值，"
        "必须用教师短轨迹做同状态对照。",
        "## 复现",
        "脚本与 JSON 在仓库内：scripts/charge_dipole_student.py、"
        "eval_full_valid.py、test_package_forces_nve.py、"
        "docs/results/data/；ESS 审计在 docs/results/"
        "ESS_STUDENT_TO_POLAR_2026-10-27.md，输入哈希与逐帧数据在同目录。"
        "图表由 docs/results/distill_figures.py 统一生成，README 与 PDF 共用。",
    ])

print("wrote", OUT)
