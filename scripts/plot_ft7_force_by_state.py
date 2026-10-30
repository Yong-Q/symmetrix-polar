#!/usr/bin/env python3
"""Force/energy error by coverage state: production vs ft6 vs ft7.

Reads the three dense-pool evaluations written on 2026-10-29 and draws the
per-state force MAE (top) and energy MAE (bottom), so the residual Plan D
distillation gap can be read directly against density.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
RES = REPO / "bench/plan_d_ft7_20261029/results"
FIG = REPO / "docs/results/figures/FORCE_ERROR_BY_STATE_2026-10-29.png"

MODELS = [
    ("base (production)", "base_on_dense_20261029a.json", "#4d4d4d"),
    ("ft6 (density-matched)", "ft6_on_dense_20261029a.json", "#1f77b4"),
    ("ft7 (+36 dense CO2)", "ft7_on_dense_20261029a.json", "#d62728"),
]

# (json key, label, density annotated on the axis)
STATES = [
    ("tag:site-C2H6-64mol-r0.01-700K-f921", "C2H6\n0.010", "0.010"),
    ("tag:site-C2H4-64mol-r0.015-500K-f911", "C2H4\n0.015", "0.015"),
    ("tag:site-CH4-64mol-r0.02-500K-f931", "CH4\n0.020", "0.020"),
    ("tag:site-NH3-128mol-r0.025-500K-f953", "NH3\n0.025", "0.025"),
    ("tag:site-CO2-64mol-r0.025-500K-f901", "CO2\n0.025", "0.025"),
    ("tag:site-H2O-128mol-r0.04-500K-f941", "H2O\n0.040", "0.040"),
    ("tag:site-CO2-64mol-r0.03-500K-f901", "CO2\n0.030", "0.030"),
]

data = {}
for _, fname, _ in MODELS:
    data[fname] = json.loads((RES / fname).read_text())

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                         gridspec_kw={"height_ratios": [3, 2]})
x = np.arange(len(STATES))
w = 0.26
for ax, metric, ylabel in (
    (axes[0], "mae_f_meV_per_A", "Force MAE (meV Å$^{-1}$)"),
    (axes[1], "mae_e_meV_per_atom", "Energy MAE (meV atom$^{-1}$)"),
):
    for i, (label, fname, color) in enumerate(MODELS):
        vals = [data[fname].get(k, {}).get(metric, np.nan) for k, _, _ in STATES]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=color,
                      edgecolor="white", linewidth=0.4)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}" if metric.startswith("mae_f") else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

axes[0].set_title("Plan D student error on the 2026-10-29 coverage pool "
                  "(133 frames, PolarMACE labels)")
axes[0].legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
axes[1].set_xticks(x)
axes[1].set_xticklabels([lbl for _, lbl, _ in STATES], fontsize=8)
axes[1].set_xlabel("State (species / density, molecules Å$^{-3}$)")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150)
print(f"wrote {FIG}")
