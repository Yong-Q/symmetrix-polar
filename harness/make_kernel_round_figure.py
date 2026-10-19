#!/usr/bin/env python3
"""Final kernel-round speed summary: before vs after on the same systems."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/workspace/group/shared/bza/symmetrix-polar/docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#1f6feb"
GREEN = "#2da44e"
GRAY = "#8c96a0"

panels = [
    ("NH3, 128 atoms (ms/eval)", [183.50, 87.24], [122.21, 40.71],
     ["full PolarMACE", "polar-off"], (1.50, 2.14)),
    ("N2, 432 atoms (ms/step)", [477.83, 191.9], [293.34, 47.65],
     ["full PolarMACE", "polar-off"], (1.63, 4.03)),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, (title, before, after, labels, speedups) in zip(axes, panels):
    x = np.arange(len(labels))
    w = 0.34
    b1 = ax.bar(x - w / 2, before, w, label="v29 baseline", color=GRAY)
    b2 = ax.bar(x + w / 2, after, w, label="final kernel round", color=BLUE)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h * 1.03,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9)
    for i, s in enumerate(speedups):
        ax.text(i, max(before[i], after[i]) * 1.18, f"{s:.2f}x",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(title)
    ax.set_ylim(0, max(before) * 1.32)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=8, loc="upper right")

fig.suptitle("Kernel round: same-model before/after (A100, f64, one rank)",
             fontsize=13, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
path = OUT / "12_kernel_round_speed.png"
fig.savefig(path, bbox_inches="tight")
print("wrote", path)
