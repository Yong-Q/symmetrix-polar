#!/usr/bin/env python3
"""Schematic and result figures for the Symmetrix-Polar technical report."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/technical/figures"
OUT.mkdir(parents=True, exist_ok=True)
ROUND2 = REPO / "bench/kernel_round2_2026-10-19/analysis_exact/r5_full_labels.csv"
ROUND3 = REPO / "bench/kernel_round3_2026-10-19/analysis_exact/r3c_full_labels.csv"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                     "axes.titleweight": "bold", "savefig.dpi": 160})


def box(ax, xy, w, h, text, color):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=1.2, facecolor=color, edgecolor="#30363d"))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=8.5, wrap=True)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color="#57606a"))


# 1. pipeline
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")
box(ax, (0.2, 1.5), 2.0, 1.6, "Atoms + cell\n(NH3, periodic)", "#dbeafe")
box(ax, (2.6, 1.5), 2.7, 1.6,
    "MACE nonlinear backbone\nedge/node messages, e3 linear,\nMH-1 tensor products",
    "#fef3c7")
box(ax, (5.7, 1.5), 2.8, 1.6,
    "Polar long-range layer\nFukui init -> GTO density ->\nk-space Ewald -> SCF fixed point",
    "#dcfce7")
box(ax, (8.9, 1.5), 2.0, 1.6, "Energy + forces\nFP64 reverse mode", "#e0e7ff")
box(ax, (11.2, 1.5), 1.6, 1.6, "LAMMPS\npair style", "#f3f4f6")
for a, b in [((2.2, 2.3), (2.6, 2.3)), ((5.3, 2.3), (5.7, 2.3)),
             ((8.5, 2.3), (8.9, 2.3)), ((10.9, 2.3), (11.2, 2.3))]:
    arrow(ax, a, b)
ax.text(6.5, 0.75,
        "All stages run on one A100 in FP64; the polar layer is self-consistent "
        "and is differentiated through for forces.",
        ha="center", fontsize=9, color="#57606a")
ax.set_title("PolarMACE execution pipeline in the Symmetrix-XL Kokkos engine")
fig.tight_layout(); fig.savefig(OUT / "fig_01_pipeline.png"); plt.close(fig)

# 2. kernel attribution
def load(p):
    return {r["label"]: r for r in csv.DictReader(open(p))}
r3 = load(ROUND3)
skip = {"(unlabeled)"}
rows = [(k, float(v["total_ms"]), float(v["sm_coverage"]), v["family"])
        for k, v in r3.items() if k not in skip]
rows.sort(key=lambda x: -x[1])
rows = rows[:14]
fig, ax = plt.subplots(figsize=(10.5, 5.2))
labels = [r[0].replace("e3 tensor ", "").replace("polar ", "") for r in rows][::-1]
vals = [r[1] for r in rows][::-1]
colors = {"tensor product": "#1f6feb", "polar pipeline": "#2da44e",
          "backbone nonlinear": "#bf8700", "cuBLAS/external": "#8250df",
          "MLP/dense": "#cf222e", "e3nn linear": "#0969da"}.get
bar_colors = [colors(rows[::-1][i][3], "#8c8c8c") for i in range(len(rows))]
ax.barh(np.arange(len(rows)), vals, color=bar_colors, height=0.62)
for i, v in enumerate(vals):
    ax.text(v + 0.06, i, f"{v:.2f}", va="center", fontsize=8.5)
ax.set_yticks(np.arange(len(rows))); ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel("CUPTI kernel time (ms / eval, 128 atoms = 32 NH3)")
ax.set_title("Round-3 kernel attribution (build 5 trace; GPU busy 43.91 ms)")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig_02_kernels.png"); plt.close(fig)

# 3. speed ladder
fig, ax = plt.subplots(figsize=(10, 4.6))
names = ["v28 initial\npolar", "v29 Kokkos\nengine", "round 1", "round 2",
         "round 3\n(latest, gates PASS)"]
vals = [6951.7, 183.8, 122.2, 56.50, 48.74]
cols = ["#8c8c8c", "#8c8c8c", "#8c8c8c", "#8c8c8c", "#1f6feb"]
y = np.arange(len(names))[::-1]
ax.barh(y, vals, color=cols, height=0.55)
for yi, v in zip(y, vals):
    ax.text(v * 1.07, yi, f"{v:.2f} ms", va="center", fontsize=10, fontweight="bold")
ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlabel("LAMMPS eval time, 1 rank (ms, log scale)")
ax.set_title("128 atoms (32 NH3), full PolarMACE, FP64, one A100")
ax.set_xlim(20, 15000); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig_03_speed_ladder.png"); plt.close(fig)

# 4. RBE
fig, ax = plt.subplots(figsize=(9.5, 4.8))
p = np.array([50, 200, 800], dtype=float)
rms = np.array([0.0409879, 0.0214259, 0.0102798])
ax.loglog(p, rms, "o-", color="#cf222e", label="RBE RMS(ΔF), measured")
ax.loglog(p, rms[0] * np.sqrt(50 / p), "--", color="#8c8c8c",
          label="slope -0.5 reference")
ax.axhline(1e-4, color="#1f6feb", ls=":", linewidth=1.5,
           label="production gate 1e-4 eV/Å")
ax.annotate("p ≈ 8.5e6 needed\n(≈17 TB cos/sin)", xy=(800, 0.0103),
            xytext=(120, 0.00022), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#57606a"))
ax.set_xlabel("RBE batch size p"); ax.set_ylabel("RMS force error (eV/Å)")
ax.set_title("RBE: 1/√p variance, 100-400× above the production gate")
ax.legend(frameon=False, fontsize=9); ax.grid(True, which="both", alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "fig_04_rbe.png"); plt.close(fig)

# 5. gate checklist
fig, ax = plt.subplots(figsize=(10, 4.3))
ax.axis("off")
items = [
    ("module 5: density / stages", "PASS", "12 + 11 items"),
    ("module 6: Coulomb / SCF", "PASS", "ΔE rel 1.02e-15"),
    ("module 7: local electron", "PASS", "16/16, out 4.55e-7"),
    ("module 8: forces", "PASS", "FD 5.78e-4, parity 7.19e-7"),
    ("end-to-end 32 frames", "PASS", "32/32"),
    ("LAMMPS 1-rank parity", "PASS", "dE 2.91e-11 eV, dF 2.63e-14 eV/Å"),
    ("RBE production force gate", "REJECTED", "RMS 6.1e-3 eV/Å, 70.4 ms/step"),
]
for i, (name, verdict, detail) in enumerate(items):
    y = 0.92 - i * 0.13
    ax.text(0.02, y, name, fontsize=10.5, va="center")
    color = "#2da44e" if verdict == "PASS" else "#cf222e"
    ax.text(0.52, y, verdict, fontsize=10.5, va="center", color=color,
            fontweight="bold")
    ax.text(0.64, y, detail, fontsize=9, va="center", color="#57606a")
ax.set_title("Numerical gate status of the latest round-3 build", loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig_05_gates.png"); plt.close(fig)

# 6. system-size scaling
fig, ax = plt.subplots(figsize=(9.5, 4.6))
names = ["128 atoms\n(32 NH3)", "1024 atoms\n(256 NH3)", "4096 atoms\n(~1024 NH3, est.)"]
vals = [48.74, 312.1, 1250.0]
cols = ["#1f6feb", "#1f6feb", "#9ec5fe"]
y = np.arange(len(names))[::-1]
ax.barh(y, vals, color=cols, height=0.55)
for yi, v in zip(y, vals):
    ax.text(v * 1.05, yi, f"{v:.0f} ms" if v >= 100 else f"{v:.2f} ms",
            va="center", fontsize=10, fontweight="bold")
ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlabel("ms / eval (log scale, full PolarMACE, one A100)")
ax.set_title("System-size scaling")
ax.set_xlim(30, 2500); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig_06_scaling.png"); plt.close(fig)

print("wrote technical figures to", OUT)
