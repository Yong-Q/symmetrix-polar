#!/usr/bin/env python3
"""Architecture evolution figure: paper/upstream baseline vs current PolarMACE engine."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("/workspace/group/shared/bza/symmetrix-polar/docs/results/figures")

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "figure.dpi": 150,
})

fig = plt.figure(figsize=(14.0, 9.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                      width_ratios=[1.05, 0.95], hspace=0.16, wspace=0.16)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])


def setup(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_box(ax, x, y, w, h, text, fc, ec, fs=7.6, tc="#111111"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.0, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, linespacing=1.15)


def draw_arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
        linewidth=0.9, color="#666666", shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- panel (a)
setup(ax_a)
ax_a.set_title("(a) Baseline: paper / upstream", loc="left", fontsize=10.5,
               fontweight="bold")
ax_a.text(0.03, 0.90, "linear MACE, FP32, no global SCF, no exact Ewald",
          fontsize=8.0, color="#555555")

boxes_a = [
    ("LAMMPS", "#e8e8e8"),
    ("Symmetrix-XL\nKokkos / CUDA", "#e8e8e8"),
    ("linear MACE\nFP32", "#d9e3f0"),
    ("direct / JIT\nstreamed edges", "#d9e3f0"),
    ("A100", "#e8e8e8"),
]
w, h, y = 0.16, 0.24, 0.50
xs = [0.03, 0.225, 0.42, 0.615, 0.81]
for i, (text, fc) in enumerate(boxes_a):
    draw_box(ax_a, xs[i], y, w, h, text, fc, "#777777", fs=7.8)
    if i < len(xs) - 1:
        draw_arrow(ax_a, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

ax_a.text(0.03, 0.33, "Measured on the paper's own workload:",
          fontsize=8.2, color="#333333")
ax_a.text(0.03, 0.24, "4.50 µs/atom/step at 5,184 atoms",
          fontsize=10.0, color="#1f4e79", fontweight="bold")
ax_a.text(0.03, 0.14, "This is the matched streaming baseline; "
          "it does not run the polar model.",
          fontsize=7.8, color="#666666")

# ---------------------------------------------------------------- panel (b)
setup(ax_b)
ax_b.set_title("(b) Current: PolarMACE engine", loc="left", fontsize=10.5,
               fontweight="bold")
ax_b.text(0.03, 0.90, "same trained model; different execution path",
          fontsize=8.0, color="#555555")

boxes_b = [
    ("LAMMPS\npair_style", "#e8e8e8"),
    ("Symmetrix-XL\nKokkos/CUDA", "#e8e8e8"),
    ("A100 /\nqueue004", "#e6d7f0"),
]
w2, h2, y2 = 0.17, 0.32, 0.46
xs2 = [0.015, 0.215, 0.815]
for i, (text, fc) in enumerate(boxes_b):
    draw_box(ax_b, xs2[i], y2, w2, h2, text, fc, "#777777", fs=6.8)
# stacked model + polar layer
draw_box(ax_b, 0.415, 0.62, w2, 0.16,
         "nonlinear\nPolarMACE FP64", "#d9e3f0", "#777777", fs=6.8)
draw_box(ax_b, 0.415, 0.46, w2, 0.14,
         "2-step SCF\n+ exact Ewald", "#fde2cf", "#777777", fs=6.8)
# kernels
draw_box(ax_b, 0.615, 0.46, w2, 0.32,
         "optimized\nkernels\nS1.1·A1·N1·B4\nR1·node-fwd\nfused2", "#d9ead3", "#777777", fs=6.2)
# arrows: 1->2, 2->model, model->kernels, kernels->A100
draw_arrow(ax_b, xs2[0] + w2, 0.62, xs2[1], 0.62)
draw_arrow(ax_b, xs2[1] + w2, 0.62, 0.415, 0.62)
draw_arrow(ax_b, 0.415 + w2, 0.62, 0.615, 0.62)
draw_arrow(ax_b, 0.615 + w2, 0.62, 0.815, 0.62)

ax_b.text(0.02, 0.30, "Measured end-to-end (clean q4 A/B):",
          fontsize=8.2, color="#333333")
ax_b.text(0.02, 0.21, "133.04 µs/atom/step at 1,296 atoms",
          fontsize=10.0, color="#1f4e79", fontweight="bold")
ax_b.text(0.02, 0.11, "138.91 baseline → 133.04 (−4.2%); 140.91 at 5,184",
          fontsize=7.8, color="#666666")

# ---------------------------------------------------------------- panel (c)
setup(ax_c)
ax_c.set_title("(c) Implementation changes (model unchanged)", loc="left",
               fontsize=10.5, fontweight="bold")
ax_c.text(0.02, 0.92, "Change", fontsize=8.0, fontweight="bold", color="#333333")
ax_c.text(0.66, 0.92, "Measured effect", fontsize=8.0, fontweight="bold",
          color="#333333")

rows = [
    ("S1.1 node-contiguous streamed-edge tiles + fused reverse",
     "253.84 ms @1,296; 8.55 GiB"),
    ("A1 UVU 32×16 tile, chunked weight staging",
     "290.20 → 266.21 ms (−8.26%)"),
    ("N1 grouped FP64 path-GEMM (6 tables, 20 planes)",
     "255.35 → 189.30 ms (−25.87%)"),
    ("Team MH-1 + B4 + message/edge/path tiles",
     "462.6 → 224.1 µs/atom"),
    ("V3 prepared path-plan cache",
     "~24 D2H syncs/eval removed"),
    ("R1 reverse MH-1 register accumulation",
     "−3.05% same-binary"),
    ("Node-owned forward dataflow",
     "−2.92 µs/atom (q4)"),
    ("Fused2 reverse dataflow",
     "−2.83 µs/atom (q4)"),
    ("q4 clean A/B, both default-on",
     "138.91 → 133.04 (−4.2%)"),
]
row_y = [0.82 - 0.088 * i for i in range(len(rows))]
for i, ((left, right), yy) in enumerate(zip(rows, row_y)):
    ax_c.add_patch(FancyBboxPatch(
        (0.01, yy - 0.040), 0.98, 0.080,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=0.0, facecolor="#f4f7fb" if i % 2 == 0 else "#ffffff"))
    ax_c.text(0.02, yy, left, fontsize=6.4, va="center", color="#222222")
    ax_c.text(0.66, yy, right, fontsize=6.4, va="center", color="#1f4e79",
              fontweight="bold")
ax_c.text(0.02, 0.018,
          "Weights, equations, FP64, exact Ewald, and the LAMMPS interface are unchanged.",
          fontsize=7.0, color="#666666")

# ---------------------------------------------------------------- panel (d)
ladder = [462.6, 311.3, 269.3, 257.1, 247.7, 230.5, 224.1, 205.4, 195.9,
          146.1, 138.9, 133.0]
labels = ["fused\nMH-1", "+FP64\nteam", "+B4\ngather", "+msg\ntile",
          "+edge\nfeat", "+path\ngrouped", "+edge-\nup", "+A1\nUVU",
          "+S1.1", "+N1\nGEMM", "+R1", "+node+\nfused2"]
x = list(range(len(ladder)))
ax_d.plot(x, ladder, "-o", color="#4c78a8", lw=1.8, ms=5.5, zorder=3)
ax_d.fill_between(x, ladder, 0, color="#4c78a8", alpha=0.08, zorder=1)
for xi, v in zip(x, ladder):
    ax_d.text(xi, v + 8, f"{v:g}", ha="center", va="bottom", fontsize=6.6,
              fontweight="bold" if xi in (0, len(ladder) - 1) else "normal")
ax_d.axhline(80, color="#d62728", ls="--", lw=1.0)
ax_d.text(11.35, 84, "80 target", ha="right", va="bottom", fontsize=7.4,
          color="#d62728")
ax_d.axhline(40, color="#d62728", ls=":", lw=1.0)
ax_d.text(11.35, 44, "40 stretch", ha="right", va="bottom", fontsize=7.4,
          color="#d62728")
ax_d.text(5.6, 420, "3.48× total\n462.6 → 133.0 µs/atom", ha="center",
          va="top", fontsize=7.8,
          bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbbbbb",
                    alpha=0.95))
ax_d.set_xlim(-0.5, 11.5)
ax_d.set_ylim(0, 500)
ax_d.set_xticks(x)
ax_d.set_xticklabels(labels, fontsize=6.8)
ax_d.set_ylabel("µs / atom / step")
ax_d.set_title("(d) Measured 1,296-atom ladder", loc="left", fontsize=10.5,
               fontweight="bold")
ax_d.grid(axis="y", alpha=0.25)

fig.suptitle("Symmetrix-Polar architecture evolution and measured speed "
             "(2026-10-22)", fontsize=13, fontweight="bold")
fig.text(0.5, 0.012,
         "Sources: docs/results/OPTIMIZATION_LEDGER.md; "
         "docs/results/LATEST_SPEED_AND_PLAN_2026-10-22.md; "
         "q4 A/B t-20261022180640-dslc9. "
         "The polar model's weights and equations are unchanged; all gains are execution changes.",
         ha="center", va="bottom", fontsize=7.6, color="#444444")
fig.tight_layout(rect=[0, 0.045, 1, 0.955])
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"ARCHITECTURE_EVOLUTION.{ext}", dpi=140,
                bbox_inches="tight")
print("wrote", OUT / "ARCHITECTURE_EVOLUTION.png")
