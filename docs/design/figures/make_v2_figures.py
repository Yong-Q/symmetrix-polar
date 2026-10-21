#!/usr/bin/env python3
"""Figures for docs/design/ARCHITECTURE_V2_PLAN.md.

No new measurement is made here.  Measured points are the accepted values in
docs/results/*; projected values are explicitly labelled [P] and are derived
from the measured additive ladder in the plan (section 2.3).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
DPI = 150

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
PURPLE = "#6A3D9A"
GRAY = "#7F7F7F"
LIGHT = "#F3F6FA"

# ==========================================================================
# Figure 1 - architecture block diagram
# ==========================================================================
fig = plt.figure(figsize=(13.15, 8.05), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 15)
ax.set_ylim(0, 9.2)
ax.axis("off")


def box(x, y, w, h, title, lines, fc, ec, title_size=7.2, body_size=6.7,
        lw=1.1, z=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.07",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color="white", zorder=z + 1)
    if lines:
        ax.text(x + 0.10, y + h - 0.40, "\n".join(lines), ha="left", va="top",
                fontsize=body_size, color="#111111", linespacing=1.38,
                zorder=z + 1)


def arrow(x0, y0, x1, y1, color="#3a3a3a", lw=1.4, ls="-", z=5):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color, linestyle=ls, zorder=z))


ax.text(0.30, 9.02, "Symmetrix-Polar V2 architecture", fontsize=15,
        fontweight="bold", va="top")
ax.text(0.30, 8.74, "Single-GPU execution (one persistent stream) with the "
        "per-rank multi-GPU contract on the right.  FP64, exact GTO-Ewald, "
        "S = 2 SCF steps, fixed ascending reduction order.", fontsize=8.4,
        color="#333333", va="top")
ax.text(0.30, 8.50, "Design intent only.  Time/memory numbers are in "
        "v2_comm_memory_budget.png; [M] measured, [D] derived from measured "
        "deltas, [P] projection.", fontsize=7.6, color="#555555", va="top",
        style="italic")

# ---- host strip
ax.add_patch(FancyBboxPatch((0.30, 7.62), 11.55, 0.60,
                            boxstyle="round,pad=0.02,rounding_size=0.07",
                            linewidth=1.0, edgecolor="#B0B0B0",
                            facecolor="#ECECEC", zorder=1))
ax.text(0.42, 8.06, "HOST", fontsize=8.2, fontweight="bold", color=GRAY,
        va="center")
host = [
    (0.95, "LAMMPS pair style", "rank-local graph | r_cut 6 A | CSR"),
    (3.42, "node-contiguous tile plan", "CSR-aligned | FNV keyed"),
    (6.18, "prepared plans", "backbone + polar UVU paths"),
    (9.18, "graph + reduce cache", "CUDA Graph execs | energy reduce"),
]
for x, t, sub in host:
    ax.add_patch(FancyBboxPatch((x, 7.70), 2.02, 0.43,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=0.8, edgecolor="#9a9a9a",
                                facecolor="#FFFFFF", zorder=3))
    ax.text(x + 1.01, 7.99, t, ha="center", va="center", fontsize=7.4,
            fontweight="bold", color="#222222")
    ax.text(x + 1.01, 7.82, sub, ha="center", va="center", fontsize=6.4,
            color="#444444")

# ---- device region
ax.add_patch(FancyBboxPatch((0.30, 1.00), 11.55, 6.48,
                            boxstyle="round,pad=0.02,rounding_size=0.09",
                            linewidth=1.2, edgecolor="#2F4B6E",
                            facecolor=LIGHT, zorder=0))
ax.text(0.42, 7.18, "GPU DEVICE  /  one persistent stream  /  no host sync or "
        "MPI inside a captured phase", fontsize=7.9, fontweight="bold",
        color="#2F4B6E", va="center")
ax.text(11.72, 7.18, "FORWARD ->", fontsize=7.9, color=BLUE,
        fontweight="bold", ha="right", va="center")

phases = [
    (0.44, 1.78, "F0 geometry + embed", BLUE,
     ["node embed 512", "radial 8 / harmonics 16", "k-grid 678 / 2748 [M]",
      "cos/sin [Nk, N_owned]", "resident; recomputed"]),
    (2.34, 1.84, "F1 backbone fwd L0>L1", BLUE,
     ["node linears: owned rows", "tile loop: receiver-owned UVU",
      "phi [C,64], register acc.", "fused cutoff + density",
      "message written once", "readout / sources / mixer"]),
    (4.30, 1.84, "P1 polar SCF x2", ORANGE,
     ["assemble rho(k) + allreduce", "project (node, k-chunk)",
      "dot UVU -> MLP -> tp UVU", "readout, density update",
      "Fukui renormalize", "device scalars only"]),
    (6.26, 1.78, "P2 Coulomb + LE", ORANGE,
     ["exact GTO-Ewald E(k)", "same k, basis, 1/k^2, SI",
      "LE q+v fused UVU", "LE total: device scalar", "no approximation"]),
    (8.16, 1.84, "R1 polar reverse", GREEN,
     ["load replay cache / replay", "source-owned X1 / X2",
      "register 4-chunk fold", "no global partials",
      "MLP / readout / Fukui rev.", "no new atomics"]),
    (10.12, 1.60, "R2 backbone rev L1>L0", GREEN,
     ["source-owned up_adj", "deterministic conv rev.",
      "node chain: owned only", "force / virial reduce"]),
]
for x, w, t, c, lines in phases:
    box(x, 3.80, w, 2.90, t, lines, fc=c, ec=c)

for x0, x1 in [(2.22, 2.34), (4.18, 4.30), (6.14, 6.26), (8.04, 8.16),
               (10.00, 10.12)]:
    arrow(x0, 5.25, x1 - 0.01, 5.25, lw=1.5)

arrow(11.55, 3.52, 0.55, 3.52, color=GREEN, lw=1.3, ls=(0, (5, 4)))
ax.text(0.50, 3.28, "<- REVERSE (dashed)", fontsize=7.9, color=GREEN,
        fontweight="bold", ha="left")
ax.text(0.45, 2.42, "solid = forward   |   dashed = reverse", fontsize=6.7,
        color="#555555", style="italic")

# ---- memory band
ax.add_patch(FancyBboxPatch((0.44, 1.18), 5.60, 0.66,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            linewidth=1.0, edgecolor="#B07A00",
                            facecolor="#FFF2CC", zorder=3))
ax.text(0.56, 1.68, "RESIDENT (owned nodes)", fontsize=7.5,
        fontweight="bold", color="#7A5300")
ax.text(0.56, 1.45, "model weights | node state | Ewald cos/sin | SCF densities | plans",
        fontsize=6.4, color="#4d3500")
ax.add_patch(FancyBboxPatch((6.14, 1.18), 5.58, 0.66,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            linewidth=1.0, edgecolor="#8C4A00",
                            facecolor="#FCE4D6", zorder=3))
ax.text(6.26, 1.68, "SHARED TILE ARENA (grow-only)", fontsize=7.5,
        fontweight="bold", color="#7A3D00")
ax.text(6.26, 1.45, "one max-set arena; C = 16,384 / 32,768 / 65,536 edges by size",
        fontsize=6.4, color="#4d2600")

# ---- multi-GPU panel
ax.add_patch(FancyBboxPatch((12.05, 3.55), 2.75, 4.68,
                            boxstyle="round,pad=0.02,rounding_size=0.09",
                            linewidth=1.2, edgecolor=PURPLE,
                            facecolor="#F5F0FA", zorder=0))
ax.text(13.42, 8.00, "MULTI-GPU (per rank)", fontsize=8.8, fontweight="bold",
        color="#4B2A72", ha="center")
ax.text(12.16, 7.73, "\n".join([
    "- one rank per GPU",
    "- rank graph = owned receivers",
    "  + ghost source rows",
    "- halo: 2 fwd + 2 rev,",
    "  width 512 (up / up_adj)",
    "  = 16 KB/ghost/eval",
    "- 2 packed allreduces / SCF:",
    "  [rho 4Nk]  [Fukui 4]",
    "- k-space replicated / rank",
    "  (owned nodes only)",
    "- no root-rank gather",
    "- energy allreduce 1x / eval",
    "- per-rank memory 5-13 GiB [P]",
]), ha="left", va="top", fontsize=6.6, color="#2d1a45", linespacing=1.38)

# ---- invariants
ax.add_patch(FancyBboxPatch((12.05, 1.00), 2.75, 2.36,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            linewidth=0.9, edgecolor="#999999",
                            facecolor="#FFFFFF", zorder=3))
ax.text(12.17, 3.20, "INVARIANTS", fontsize=7.5, fontweight="bold",
        color="#333333")
ax.text(12.17, 2.98, "\n".join([
    "FP64 only; no FP16/TF32",
    "exact Ewald; S = 2 fixed",
    "ascending edge / k order",
    "no atomics in new paths",
    "default off until gates pass",
    "128/1,296: dE <= 1e-9 eV,",
    "  dF <= 1e-8 eV/A",
    "2-rank: dE <= 5e-5 eV,",
    "  dF <= 1e-4 eV/A",
]), ha="left", va="top", fontsize=6.5, color="#333333", linespacing=1.35)

fig.savefig(OUT / "v2_architecture.png", dpi=DPI, facecolor="white",
            bbox_inches="tight", pad_inches=0.06)
plt.close(fig)

# ==========================================================================
# Figure 2 - time / memory / communication budget
# ==========================================================================
fig = plt.figure(figsize=(13.15, 5.05), dpi=DPI)
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.10, 1.05], wspace=0.30,
                      left=0.055, right=0.985, top=0.84, bottom=0.14)

# ---- panel A
axA = fig.add_subplot(gs[0, 0])
cats = [
    ("polar (SCF/Ewald/UVU/LE)", "#F58518"),
    ("backbone nonlinear", "#4C78A8"),
    ("tensor product / MH-1", "#72B7B2"),
    ("cuBLAS / external", "#E45756"),
    ("MLP / dense", "#B279A2"),
    ("e3nn linear", "#FF9DA6"),
    ("other", "#9D755D"),
    ("host / launch gap", "#7F7F7F"),
]
pre = [116.95, 75.38, 71.02, 35.83, 16.57, 13.67, 10.90, 8.65]
cur = [87.51, 28.85, 66.09, 22.85, 20.88, 13.33, 0.24, 14.09]
v2 = [70.0, 22.0, 45.0, 18.0, 17.0, 10.0, 0.2, 2.0]
xs = np.arange(3)
bottoms = np.zeros(3)
for i, (name, c) in enumerate(cats):
    vals = np.array([pre[i], cur[i], v2[i]])
    axA.bar(xs, vals, bottom=bottoms, color=c, edgecolor="white",
            linewidth=0.7, width=0.62, label=name)
    bottoms += vals
for x, total in zip(xs, bottoms):
    axA.text(x, total + (6 if x < 2 else 26), f"{total:.0f}", ha="center",
             va="bottom", fontsize=9.2, fontweight="bold")
axA.errorbar([2], [184], yerr=[[16], [11]], fmt="none", ecolor="#222222",
             capsize=7, lw=1.2, zorder=6)
axA.set_xticks(xs)
axA.set_xticklabels(["measured\ndirect trace\n348.97 ms",
                     "current 6582bc6\nfresh trace [M/D]\n253.84 ms",
                     "V2 target [P]\n168-195 ms"], fontsize=7.3)
axA.set_ylabel("ms / eval  (1,296 H2O, FP64, 1x A100)", fontsize=8.2)
axA.set_title("(a) Time budget at 1,296 atoms", fontsize=9.4, loc="left",
              fontweight="bold")
axA.grid(axis="y", alpha=0.25, lw=0.6)
axA.set_axisbelow(True)
axA.set_ylim(0, 400)
h, l = axA.get_legend_handles_labels()
axA.legend(h, l, fontsize=5.9, loc="upper right", frameon=False, ncol=1,
           handlelength=1.0, borderaxespad=0.2)

# ---- panel B
axB = fig.add_subplot(gs[0, 1])
mem_cats = [
    ("tile arena", "#E45756"),
    ("node / backbone", "#4C78A8"),
    ("polar", "#F58518"),
    ("Ewald", "#54A24B"),
    ("e3/product/readout/base", "#7F7F7F"),
]
mem_bars = {
    "1,296\ncurrent [M]": [4.45, 1.94, 0.64, 0.01, 1.51],
    "1,296\nV2 [P]": [2.46, 1.94, 0.46, 0.01, 1.31],
    "5,184\ncurrent [M]": [4.69, 7.77, 2.66, 0.20, 3.89],
    "5,184\nV2 C=16k [P]": [2.46, 7.77, 1.80, 0.21, 3.40],
    "5,184\nV2 C=32k [P]": [4.93, 7.77, 1.80, 0.21, 3.40],
}
labels = list(mem_bars.keys())
xs = np.arange(len(labels))
bottoms = np.zeros(len(labels))
for i, (name, c) in enumerate(mem_cats):
    vals = np.array([mem_bars[k][i] for k in labels])
    axB.bar(xs, vals, bottom=bottoms, color=c, edgecolor="white",
            linewidth=0.7, width=0.66, label=name)
    bottoms += vals
for x, total in zip(xs, bottoms):
    axB.text(x, total + 0.85, f"{total:.1f}", ha="center", va="bottom",
             fontsize=8.2, fontweight="bold")
axB.scatter([0], [8.55], marker="_", s=110, color="black", zorder=6, lw=2)
axB.scatter([2], [19.21], marker="_", s=110, color="black", zorder=6, lw=2)
axB.set_xticks(xs)
axB.set_xticklabels(labels, fontsize=6.5)
axB.set_ylabel("peak VRAM (GiB)", fontsize=8.2)
axB.set_title("(b) Peak memory", fontsize=9.4, loc="left", fontweight="bold")
axB.grid(axis="y", alpha=0.25, lw=0.6)
axB.set_axisbelow(True)
axB.set_ylim(0, 23)
axB.text(0.02, 0.97, "black ticks = measured 8.55 / 19.21 GiB;\n"
         "bars are the calibrated ledger model.", transform=axB.transAxes,
         fontsize=6.1, va="top", color="#555555")
h, l = axB.get_legend_handles_labels()
axB.legend(h, l, fontsize=5.9, loc="upper left", frameon=False,
           bbox_to_anchor=(0.0, 0.90), handlelength=1.0)

# ---- panel C
axC = fig.add_subplot(gs[0, 2])
events = ["kernel launches", "host<->device syncs", "global allreduce calls"]
cur_ev = [1762, 24, 20]
v2_ev = [120, 2, 8]
y = np.arange(len(events))
hgt = 0.34
axC.barh(y + hgt / 2, cur_ev, height=hgt, color="#B0B0B0",
         edgecolor="white", label="current [M/D]")
axC.barh(y - hgt / 2, v2_ev, height=hgt, color=PURPLE,
         edgecolor="white", label="V2 target [P]")
for yy, v in zip(y + hgt / 2, cur_ev):
    axC.text(v * 1.12, yy, f"{v}", va="center", fontsize=7.4)
for yy, v in zip(y - hgt / 2, v2_ev):
    axC.text(v * 1.12, yy, f"{v}", va="center", fontsize=7.4,
             color="#4B2A72", fontweight="bold")
axC.set_xscale("log")
axC.set_xlim(1, 6000)
axC.set_yticks(y)
axC.set_yticklabels(events, fontsize=7.4)
axC.set_xlabel("count per eval (log scale); halo 37 -> 16 KB/ghost [P]",
               fontsize=7.4)
axC.set_title("(c) Control / communication events", fontsize=9.4, loc="left",
              fontweight="bold")
axC.grid(axis="x", alpha=0.25, lw=0.6)
axC.set_axisbelow(True)
axC.legend(fontsize=6.3, loc="lower right", frameon=False)

fig.suptitle("Symmetrix-Polar V2 communication and memory budget — measured "
             "anchors and projections", fontsize=11.0, fontweight="bold",
             x=0.055, ha="left", y=0.975)
fig.savefig(OUT / "v2_comm_memory_budget.png", dpi=DPI, facecolor="white",
            bbox_inches="tight", pad_inches=0.06)
plt.close(fig)

# ==========================================================================
# Figure 3 - multi-GPU scaling projection
# ==========================================================================
fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=DPI)
ranks = np.array([1, 2, 4])
low1296 = np.array([1.0, 1.10, 1.50])
high1296 = np.array([1.0, 1.22, 2.00])
low5184 = np.array([1.0, 1.22, 1.90])
high5184 = np.array([1.0, 1.35, 2.60])
ax.fill_between(ranks, low1296, high1296, color=BLUE, alpha=0.22,
                label="1,296 H2O  [P]")
ax.fill_between(ranks, low5184, high5184, color=ORANGE, alpha=0.22,
                label="5,184 H2O  [P]")
ax.plot(ranks, (low1296 + high1296) / 2, "-o", color=BLUE, lw=1.6, ms=5)
ax.plot(ranks, (low5184 + high5184) / 2, "-o", color=ORANGE, lw=1.6, ms=5)
ax.plot(ranks, ranks, "--", color="#888888", lw=1.0, label="linear ideal")
ax.scatter([2], [1.0], marker="*", s=170, color="#333333", zorder=6)
ax.annotate("measured anchor: 2 ranks on ONE A100\n"
            "= no speedup (shared GPU; MPI_H1.md sec. 4)",
            xy=(2, 1.0), xytext=(2.30, 1.10), fontsize=7.2, color="#333333",
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"))
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4])
ax.set_xticklabels(["1 GPU", "2 GPUs", "4 GPUs"])
ax.set_xlim(0.9, 4.6)
ax.set_ylim(0.8, 4.15)
ax.set_ylabel("speedup vs V2 single GPU", fontsize=9)
ax.set_title("Multi-GPU scaling projection (one rank per GPU; all values "
             "except the anchor are estimates)", fontsize=10.2, loc="left",
             fontweight="bold")
ax.grid(alpha=0.25, lw=0.6)
ax.set_axisbelow(True)
ax.legend(fontsize=7.4, loc="upper left", frameon=False)
ax.text(1.02, 0.84, "ghost/local: 2-GPU slab g ~ 1.4 / 0.9; 4-GPU cube g ~ 2.8 / 1.5\n"
        "(CUDA_SPEEDUP_PAPERS_2026.md sec. 3.3); 4x A100 only on the remote "
        "Volc queue.", fontsize=6.6, color="#555555")
fig.tight_layout()
fig.savefig(OUT / "v2_scaling_projection.png", dpi=DPI, facecolor="white",
            bbox_inches="tight", pad_inches=0.06)
plt.close(fig)

print("wrote", OUT / "v2_architecture.png")
print("wrote", OUT / "v2_comm_memory_budget.png")
print("wrote", OUT / "v2_scaling_projection.png")
