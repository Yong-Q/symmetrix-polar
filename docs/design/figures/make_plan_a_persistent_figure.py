#!/usr/bin/env python3
"""Plan A persistent-execution budget figure (2026-10-23).

Left: current-state reconstructed exact budget at 1,296 atoms, with the
measured removable memory-staging slices hatched.
Right: waterfall from the measured 133.04 to the conservative / base /
aggressive exact landings.

All speed values are us/atom/step.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Current-state reconstruction (us/atom/step). See PLAN_A_PERSISTENT_2026-10-23.
slices = [
    ("tensor (MH-1, post-fused2)", 42.4, "kernel"),
    ("backbone nonlinear", 19.0, "kernel"),
    ("cuBLAS / external GEMM", 19.0, "kernel"),
    ("polar pipeline", 15.3, "kernel"),
    ("MLP/dense (epilogue + LN)", 14.9, "staging"),
    ("e3 linear pack/unpack", 10.0, "staging"),
    ("edge D2D tape copies", 9.6, "staging"),
    ("other copies / memsets", 1.3, "staging"),
    ("launch / host-sync gap", 1.6, "host"),
]
total = sum(v for _, v, _ in slices)
colors = {"kernel": "#4C72B0", "staging": "#DD8452", "host": "#55A868"}

fig = plt.figure(figsize=(13.2, 4.6), dpi=150)
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.30)

# ---------------- left: stacked budget ----------------
ax = fig.add_subplot(gs[0, 0])
left = 0.0
for name, val, kind in slices:
    ax.barh(0, val, left=left, height=0.5, color=colors[kind],
            edgecolor="white", linewidth=0.8,
            hatch="//" if kind == "staging" else None)
    if val >= 8:
        ax.text(left + val / 2, 0, f"{val:.1f}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
    left += val
ax.set_xlim(0, total * 1.01)
ax.set_ylim(-0.6, 1.35)
ax.set_yticks([])
ax.set_xlabel("µs/atom/step (1,296 atoms, FP64, exact)")
ax.set_title(f"Current exact budget: {total:.1f} µs/atom/step\n"
             "(reconstructed from the post-N1 CUPTI trace and measured A/B deltas)",
             fontsize=10)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)

legend = [Patch(facecolor=colors["kernel"], label="kernel compute / memory"),
          Patch(facecolor=colors["staging"], hatch="//",
                label="removable staging traffic (copies, epilogue, pack/unpack)"),
          Patch(facecolor=colors["host"], label="launch / host-sync gap")]
fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.06),
           ncol=3, frameon=False, fontsize=8)

# ---------------- right: waterfall ----------------
ax2 = fig.add_subplot(gs[0, 1])
start = 133.04
scenarios = [
    ("conservative", [4.5, 4.0, 3.5, 2.5, 2.0, 1.5, 1.0], "#8C8C8C"),
    ("base case", [6.5, 6.5, 5.5, 3.0, 3.0, 2.5, 1.0, 3.0], "#4C72B0"),
    ("aggressive\nbound", [8.5, 9.0, 6.2, 3.4, 5.2, 3.5, 5.0, 1.4, 4.0],
     "#C44E52"),
]
colors2 = ["#DD8452", "#DD8452", "#4C72B0", "#4C72B0", "#55A868",
           "#4C72B0", "#4C72B0", "#55A868", "#C44E52"]
n = len(scenarios)
for si, (name, deltas, edge) in enumerate(scenarios):
    run = start
    for di, d in enumerate(deltas):
        ax2.bar(si, d, bottom=run - d, width=0.62, color=colors2[di],
                edgecolor="white", linewidth=0.5)
        run -= d
    ax2.text(si, start + 1.8, f"→ {run:.0f}", fontsize=11, fontweight="bold",
             ha="center", color=edge)
ax2.axhline(100, color="k", linestyle="--", linewidth=1.2)
ax2.text(n - 0.55, 100.6, "target 100", fontsize=8.5, ha="right")
ax2.axhline(start, color="#999999", linestyle=":", linewidth=1.0)
ax2.text(-0.45, start + 0.7, f"measured {start:.2f}", fontsize=8,
         color="#666666")
ax2.set_ylim(85, 137)
ax2.set_xlim(-0.55, n - 0.45)
ax2.set_yticks([90, 100, 110, 120, 130])
ax2.set_ylabel("µs/atom/step")
ax2.set_xticks(range(n))
ax2.set_xticklabels([s[0] for s in scenarios], fontsize=9)
ax2.set_title("Exact landing scenarios", fontsize=10)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.savefig("/workspace/group/shared/bza/symmetrix-polar/docs/design/figures/"
            "PLAN_A_PERSISTENT_BUDGET.png", bbox_inches="tight")
print("wrote PLAN_A_PERSISTENT_BUDGET.png")
