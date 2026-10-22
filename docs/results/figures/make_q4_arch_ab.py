#!/usr/bin/env python3
"""q4 clean A/B bar chart: baseline vs node-owned forward vs fused2 reverse."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/workspace/group/shared/bza/symmetrix-polar")
d = json.load(open(ROOT / "docs/results/data/q4_arch_ab_2026-10-22.json"))
rm = d["round_medians"]
labels = ["Baseline\n(N1+R1)", "Node-owned\nforward", "Fused2\nreverse",
          "Node + fused2\n(default)"]
keys = ["base", "node", "rev", "both"]
vals = [rm[k]["us_per_atom"] for k in keys]
colors = ["#8c8c8c", "#1f77b4", "#2ca02c", "#d62728"]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, vals, color=colors, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
    if v != vals[0]:
        ax.text(b.get_x() + b.get_width() / 2, v / 2,
                f"−{vals[0] - v:.2f}\n({(vals[0] - v) / vals[0] * 100:.1f}%)",
                ha="center", va="center", color="white", fontsize=10)
ax.axhline(100, ls="--", lw=1.2, color="k", alpha=0.7)
ax.text(3.42, 100.8, "100 target", ha="right", va="bottom", fontsize=9)
ax.set_ylabel("µs/atom/step", fontsize=12)
ax.set_ylim(0, 155)
ax.set_title("q4 clean A/B, 1,296 water, 1×A100-80GB\n"
             "(2 rounds × 4 arms × 5 steps; exact FP64 PolarMACE)", fontsize=12)
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(ROOT / "docs/results/figures/Q4_ARCH_AB.png", dpi=150)
