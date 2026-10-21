#!/usr/bin/env python3
"""Clean Plan B-lite speed/slice figure for the technical report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle

figstyle.apply()

# Post-write-through-fix re-measure (B4) + pre-fix B1 A/B.
# Source: bench/plan_b_lite_2026-10-20/ab_wtfix/wtfix_b4_full.json
build8 = 48.74
b1 = 42.79
final = 38.66
prior_b = 49.69

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2),
                         gridspec_kw={"width_ratios": [1.15, 1.0]})

# Panel A: step-down
ax = axes[0]
labels = ["B tree prior", "Round-3\nbuild 8", "B1\nreplay cache", "B1+B4\nfinal"]
values = [prior_b, build8, b1, final]
colors = [figstyle.GRAY, figstyle.GRAY, figstyle.BLUE, figstyle.GREEN]
bars = ax.bar(range(len(values)), values, color=colors, width=0.62)
for i, v in enumerate(values):
    ax.text(i, v + 0.6, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(range(len(values))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("ms / eval"); ax.set_title("NH3 128 atoms, full PolarMACE (FP64)")
ax.set_ylim(34, 52)
ax.set_axisbelow(True)
ax.annotate("", xy=(3, final + 2.4), xytext=(1, build8 + 2.4),
            arrowprops=dict(arrowstyle="->", color=figstyle.GREEN, lw=1.2))
ax.text(2.0, 49.2, "1.261x vs build 8", ha="center", va="bottom",
        color=figstyle.GREEN, fontweight="bold", fontsize=8.5)

# Panel B: slice savings
ax = axes[1]
names = ["B1 replay\ncache (pre-fix)", "B4 gather\nfusion (post-fix)", "B3 merged\nunpack", "B5 small\npolar"]
savings = [3.39, 4.13, -5.97, 0.01]
cols = [figstyle.GREEN, figstyle.GREEN, figstyle.RED, figstyle.GRAY]
ax.barh(range(len(names)), savings, color=cols, height=0.55)
for i, v in enumerate(savings):
    x = v + (0.12 if v >= 0 else -0.12)
    ax.text(x, i, f"{v:+.2f} ms", va="center",
            ha="left" if v >= 0 else "right", fontsize=8.5)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("wall-time saving (ms)")
ax.set_title("Slice attribution (same binary)")
ax.axvline(0, color=figstyle.INK, lw=0.8)
ax.set_xlim(-7.2, 5.2)
ax.set_axisbelow(True)

fig.tight_layout()
figstyle.save(fig, "/workspace/group/shared/bza/symmetrix-polar/docs/figures/21_plan_b_lite_speed.png")
figstyle.save(fig, "/workspace/group/shared/bza/symmetrix-polar/docs/technical/figures/fig_07_plan_b_lite.png")
print("wrote Plan B-lite figure")
