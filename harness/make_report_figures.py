#!/usr/bin/env python3
"""Regenerate the three report figures from committed data.

Ladder and position values are the rows of docs/results/OPTIMIZATION_LEDGER.md;
the bottleneck figure is computed from docs/results/data/cupti_b4team_1296_summary.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FONT = REPO / "docs/technical/fonts/NotoSansSC-Medium-static.ttf"
font_manager.fontManager.addfont(str(FONT))
OUT = REPO / "docs/report/figures"
OUT.mkdir(parents=True, exist_ok=True)
INK = "#24292f"
BLUE = "#1f6feb"
GREEN = "#1a7f37"
GRAY = "#8c8c8c"
RED = "#cf222e"
plt.rcParams.update({
    "font.family": "Noto Sans SC", "axes.unicode_minus": False,
    "font.size": 10, "axes.edgecolor": "#d0d7de", "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

# Ledger §1 (1,296 atoms, full PolarMACE, FP64, one A100).
ladder = [
    ("fused MH-1 基线", 462.6, GRAY),
    ("+ FP64 team MH-1", 311.3, GRAY),
    ("+ B4 gather", 269.3, BLUE),
    ("+ gather tile", 257.1, BLUE),
    ("+ edge-feature tile", 247.7, BLUE),
    ("+ path-grouped forward", 230.5, GREEN),
    ("+ edge-up/scatter", 224.1, GREEN),
]
fig, ax = plt.subplots(figsize=(10, 4.8))
y = np.arange(len(ladder))[::-1]
ax.barh(y, [v for _, v, _ in ladder], color=[c for _, _, c in ladder],
        height=0.58)
for yi, (_, v, _) in zip(y, ladder):
    ax.text(v * 1.06, yi, f"{v:.1f}", va="center", fontsize=9.5,
            fontweight="bold")
ax.axvline(4.5, color=RED, ls="--", lw=1.2)
ax.text(4.5 * 1.05, y[-1] - 0.55, "目标 4.5 µs/atom", color=RED,
        fontsize=8.5, va="top")
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels([n for n, _, _ in ladder])
ax.set_xlabel("µs/atom（1,296 原子，FP64，单张 A100）")
ax.set_xlim(3, 900)
ax.grid(axis="x", alpha=0.3)
ax.set_title("优化阶梯：462.6 → 224.1 µs/atom（−51.6%）",
             loc="left", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "ladder_us_per_atom.png", dpi=170)
plt.close(fig)

# Ledger §4.
position = [
    ("PyTorch PolarMACE\n（128 原子）", 734.4, GRAY),
    ("本引擎 full polar\n（1,296 原子）", 224.1, GREEN),
    ("本引擎 linear MACE\nFP32 streamed（1,296）", 9.69, BLUE),
    ("论文 linear MACE\nFP32 streamed（5,184）", 4.5, BLUE),
    ("目标", 4.5, RED),
]
fig, ax = plt.subplots(figsize=(10, 4.2))
y = np.arange(len(position))[::-1]
for yi, (_, v, c) in zip(y, position):
    ax.barh(yi, v, color=c, height=0.55)
    ax.text(v * 1.08, yi, f"{v:g}", va="center", fontsize=9.5,
            fontweight="bold")
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels([n for n, _, _ in position], fontsize=8.5)
ax.set_xlabel("µs/atom（对数坐标；不同模型/尺寸已标注）")
ax.set_xlim(3, 1400)
ax.grid(axis="x", alpha=0.3)
ax.set_title("位置：同模型 224.1 µs/atom；线性引擎与论文持平",
             loc="left", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "position_us_per_atom.png", dpi=170)
plt.close(fig)

# Ledger §2 / CUPTI family summary.
summary = json.loads(
    (REPO / "docs/results/data/cupti_b4team_1296_summary.json").read_text())
fams = [f for f in summary["families"] if f["total_ms"] > 0.05]
fams.sort(key=lambda f: f["total_ms"])
fig, ax = plt.subplots(figsize=(9.6, 4.4))
name_map = {
    "polar pipeline": "polar pipeline", "backbone nonlinear": "非线性骨干",
    "tensor product": "张量积", "cuBLAS/external": "cuBLAS/外部",
    "MLP/dense": "MLP/dense", "e3nn linear": "e3nn linear",
    "other": "其他", "radial/spline": "radial/spline",
    "Kokkos init/fill": "Kokkos 初始化",
}
names = [name_map.get(f["family"], f["family"]) for f in fams]
vals = [f["total_ms"] * 1000 / 1296 for f in fams]
ax.barh(names, vals, color=[BLUE if v >= 50 else GRAY for v in vals],
        height=0.6)
for i, v in enumerate(vals):
    ax.text(v + 1.2, i, f"{v:.1f}", va="center", fontsize=9,
            fontweight="bold")
ax.set_xlabel("µs/atom（1,296 原子；opt3 前 CUPTI trace）")
ax.set_xlim(0, max(vals) * 1.16)
ax.grid(axis="x", alpha=0.3)
ax.set_title("GPU 时间去向：张量积与 polar pipeline 占主导",
             loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "bottlenecks_us_per_atom.png", dpi=170)
plt.close(fig)
print("wrote", OUT)
