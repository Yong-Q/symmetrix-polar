#!/usr/bin/env python3
"""Clean Plan B-micro figure: 30.68 -> 30.33 ms and slice attribution."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle

figstyle.apply()

accepted = 30.557
final = 30.325
target = 30.0

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0),
                         gridspec_kw={"width_ratios": [1.0, 1.0]})

ax = axes[0]
labels = ["accepted\n(30.56)", "B-micro\n(final)"]
values = [accepted, final]
ax.bar(range(2), values, color=[figstyle.GRAY, figstyle.GREEN], width=0.55)
for i, v in enumerate(values):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
ax.axhline(target, color=figstyle.RED, ls="--", lw=1.0)
ax.text(1.32, target + 0.015, "target 30.0", color=figstyle.RED,
        ha="left", va="bottom", fontsize=8)
ax.set_xticks(range(2)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("ms / eval"); ax.set_title("NH3 128 atoms, full PolarMACE (FP64)")
ax.set_ylim(29.9, 30.75)
ax.set_axisbelow(True)

ax = axes[1]
names = ["reverse Fukui", "forward Fukui", "total"]
savings = [0.209, 0.143, 0.352]
ax.barh(range(3), savings, color=figstyle.GREEN, height=0.5)
for i, v in enumerate(savings):
    ax.text(v + 0.008, i, f"+{v:.3f} ms", va="center", ha="left", fontsize=8.5)
ax.set_yticks(range(3)); ax.set_yticklabels(names, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("wall-time saving (ms)")
ax.set_title("One-rank device Fukui scalars")
ax.set_xlim(0, 0.48)
ax.set_axisbelow(True)

fig.tight_layout()
figstyle.save(fig, "/workspace/group/shared/bza/symmetrix-polar/docs/figures/25_plan_b_micro_speed.png")
figstyle.save(fig, "/workspace/group/shared/bza/symmetrix-polar/docs/technical/figures/fig_11_plan_b_micro.png")
print("wrote Plan B-micro figure")
