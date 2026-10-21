#!/usr/bin/env python3
"""Clean Plan B-full speed/slice figure for the technical report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle

figstyle.apply()

build8 = 48.74
bcore = 38.73
final = 31.33

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2),
                         gridspec_kw={"width_ratios": [1.1, 1.0]})

# Panel A: step-down
ax = axes[0]
labels = ["Round-3\nbuild 8", "B-core\n(38.73)", "B-full\n(final)"]
values = [build8, bcore, final]
colors = [figstyle.GRAY, figstyle.BLUE, figstyle.GREEN]
ax.bar(range(len(values)), values, color=colors, width=0.6)
for i, v in enumerate(values):
    ax.text(i, v + 0.6, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(range(len(values))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("ms / eval"); ax.set_title("NH3 128 atoms, full PolarMACE (FP64)")
ax.set_ylim(26, 53)
ax.set_axisbelow(True)
ax.annotate("", xy=(2, final + 2.0), xytext=(0, build8 + 2.0),
            arrowprops=dict(arrowstyle="->", color=figstyle.GREEN, lw=1.2))
ax.text(1.0, 50.0, "1.556x vs build 8", ha="center", va="bottom",
        color=figstyle.GREEN, fontweight="bold", fontsize=8.5)

# Panel B: slice savings
ax = axes[1]
names = ["MH-1 FP64\nteam", "UVU reverse\nd1 spec", "cutoff\nteam", "UVU fwd/LE\nd1 (rejected)"]
savings = [3.10, 3.83, 2.03, 0.74]
cols = [figstyle.GREEN, figstyle.GREEN, figstyle.GREEN, figstyle.GRAY]
ax.barh(range(len(names)), savings, color=cols, height=0.55)
for i, v in enumerate(savings):
    ax.text(v + 0.12, i, f"{v:+.2f} ms", va="center", ha="left", fontsize=8.5)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("wall-time saving (ms)")
ax.set_title("Accepted slices (same-binary A/B)")
ax.axvline(0, color=figstyle.INK, lw=0.8)
ax.set_xlim(0, 4.9)
ax.set_axisbelow(True)

fig.tight_layout()
figstyle.save(fig, "/workspace/group/shared/bza/symmetrix-polar/docs/figures/23_plan_b_full_speed.png")
figstyle.save(fig, "/workspace/group/shared/bza/symmetrix-polar/docs/technical/figures/fig_09_plan_b_full.png")
print("wrote Plan B-full figure")
