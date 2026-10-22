#!/usr/bin/env python3
"""Figure: paper Symmetrix-XL speed vs current PolarMACE engine."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/workspace/group/shared/bza/symmetrix-polar")
DATA = ROOT / "docs/results/data/paper_vs_current_speed_2026-10-22.json"
OUT = ROOT / "docs/results/figures"

d = json.loads(DATA.read_text())
rows = d["rows"]
rows = sorted(rows, key=lambda r: r["value"])

colors = {
    "paper": "#9aa0a6",
    "this-linear-fp32": "#4c78a8",
    "paper-baseline": "#f58518",
    "this-linear-fp64": "#54a24b",
    "current-polar": "#d62728",
    "earlier-polar": "#e8a0a0",
    "pytorch": "#9467bd",
}
legend_labels = {
    "paper": "paper (linear MACE, FP32)",
    "this-linear-fp32": "this engine (linear MACE, FP32)",
    "paper-baseline": "paper baseline (ML-IAP+cuEq)",
    "this-linear-fp64": "this engine (linear MACE, FP64)",
    "current-polar": "this engine (full PolarMACE, FP64)",
    "earlier-polar": "earlier PolarMACE",
    "pytorch": "PyTorch PolarMACE reference",
}

plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 11.5,
                     "axes.labelsize": 10, "figure.dpi": 150})
fig, ax = plt.subplots(figsize=(11.5, 6.0))
y = np.arange(len(rows))
vals = [r["value"] for r in rows]
bar_colors = [colors[r["group"]] for r in rows]
ax.barh(y, vals, left=3.0, color=bar_colors, height=0.62,
        edgecolor="black", linewidth=0.4, alpha=0.9)
ax.set_xscale("log")
ax.set_xlim(3.0, 1400.0)
ax.set_yticks(y)
ax.set_yticklabels([r["label"] for r in rows], fontsize=7.8)
ax.set_xlabel("µs / atom / step (log scale)")
ax.set_title("Paper's linear-MACE speed vs full PolarMACE")
ax.grid(axis="x", which="major", alpha=0.3)
ax.grid(axis="x", which="minor", alpha=0.12)

for yi, v in zip(y, vals):
    ax.text(v * 1.08, yi, f"{v:g}", va="center", ha="left",
            fontsize=8.4, fontweight="bold")

ax.axvline(4.50, color="#333333", ls="--", lw=1.1, alpha=0.8)
ax.axvline(145.75, color="#d62728", ls="--", lw=1.1, alpha=0.7)

handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in legend_labels]
ax.legend(handles, [legend_labels[k] for k in legend_labels],
          loc="lower right", fontsize=7.3, frameon=True, ncol=1)

fig.text(0.5, 0.015,
         "Paper: arXiv:2610.01036. 4.50 = linear MACE-OFF23, FP32, standard streaming, 5,184 atoms. "
         "Full PolarMACE adds 2-step SCF/Fukui, exact Ewald, polar UVU/LE, and FP64; not like-for-like.",
         ha="center", va="bottom", fontsize=7.8, color="#444444")
fig.tight_layout(rect=[0, 0.05, 1, 0.97])
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"PAPER_VS_CURRENT.{ext}", dpi=150, bbox_inches="tight")
print("wrote", OUT / "PAPER_VS_CURRENT.png")
