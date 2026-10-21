#!/usr/bin/env python3
"""Generate figures for POLAR_PAPER_OPTIMIZATION_PLAN.md.

All values are derived from the source-code tensor enumeration in the plan
(section 2) and from the accepted measured baselines.  Nothing here is a new
measurement; the projection panels are explicitly labelled estimates.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
NK = {1296: 678, 5184: 2748}
S = 2

# doubles per node, grouped by lifetime/ownership
FWD = {
    "density state (4x8 + 2)": 34,
    "scf_step_densities [S,N,8]": S * 8,
    "node_feats_e3nn [N,2048]": 2048,
    "features_mixed [N,2048]": 2048,
    "mixer_temp [N,2048]": 2048,
    "potential/field feats [N,16]+32": 48,
    "pe [N,2048]": 2048,
    "scratch a/b [N,2048]": 4096,
    "dp+se [N,512]": 1024,
    "nl_input [N,1024]+nl_a/b": 1152,
    "nl_out [N,512]": 512,
    "tp_out [N,2048]": 2048,
    "ro1/ro2 [N,160]": 320,
    "gate1/2 [N,128]": 256,
    "update_out [N,10]": 10,
    "LE q_in/up/dot [N,8+512x4]": 2056,
    "LE invar [N,1024]": 1024,
    "LE mlp_out+atom": 2,
    "fwd uvu partial [4,N,512]": 4 * 512,
}
REV = {
    "density/channel/source/feats adj [N,8]": 32,
    "potential+position adj [N,16+3]": 19,
    "nfeat adj [N,2048]": 2048,
    "fmix adj [N,2048]": 2048,
    "uvu mixed_bar [N,512]": 512,
    "LE bar set (invar+4x512+nf)": 4097,
    "LE q_in/p_last bar": 24,
    "fukui+out adj": 12,
    "gate/ro adj [N,128/160]": 576,
    "tp/nl/h/nlin adj": 3200,
    "dp/se/pe adj": 3072,
    "replay step_potential/pe/dp/se": 3088,
    "replay nl_in/h/a/out": 1920,
    "replay tp/ro/gate/out/nlout": 2996,
}
CACHE = {
    "replay cache pe [S,N,2048]": S * 2048,
    "replay cache nl_h+nl_a [S,N,192]": S * 384,
    "replay cache nl_out [S,N,512]": S * 512,
    "replay cache ro1+ro2 [S,N,160]": S * 320,
    "replay cache out [S,N,10]": S * 10,
}
PART = {
    "rev uvu partial x1 [4,N,1536]": 4 * 512 * 3,
    "rev uvu partial x2 [4,N,1536]": 4 * 512 * 3,
}

gib = lambda d, n: d * n * 8 / 2**30

groups = {
    "forward activations": (FWD, "#4C78A8"),
    "reverse activations": (REV, "#F58518"),
    "replay cache": (CACHE, "#54A24B"),
    "UVU reduction partials": (PART, "#E45756"),
}

fig = plt.figure(figsize=(11.5, 5.2), dpi=150)
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.33)

# ---- panel a: top named buffers at 5,184 -------------------------------
ax = fig.add_subplot(gs[0, 0])
rows = []
for name, (d, color) in groups.items():
    for key, w in d.items():
        rows.append((key, w, color))
rows.sort(key=lambda r: r[1], reverse=True)
top = rows[:16][::-1]
y = np.arange(len(top))
ax.barh(y, [gib(w, 5184) * 1024 for _, w, _ in top],
        color=[c for _, _, c in top], height=0.72)
ax.set_yticks(y)
ax.set_yticklabels([k for k, _, _ in top], fontsize=7.5)
ax.set_xlabel("MiB at 5,184 atoms (named buffers, FP64)", fontsize=8)
ax.tick_params(axis="x", labelsize=7.5)
ax.grid(axis="x", lw=0.4, alpha=0.35)
ax.set_title("(a) Largest polar-layer tensors at 5,184 atoms", fontsize=9, loc="left")
for yi, (_, w, _) in zip(y, top):
    ax.text(gib(w, 5184) * 1024 + 2, yi, f"{gib(w, 5184)*1024:.0f}",
            va="center", fontsize=6.5, color="#333333")
handles = [plt.Rectangle((0, 0), 1, 1, color=g[1]) for g in groups.values()]
ax.legend(handles, list(groups.keys()), fontsize=7, loc="lower right",
          frameon=False)

# ---- panel b: totals by group -----------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
labels = list(groups.keys()) + ["Ewald cos/sin"]
width = 0.36
x = np.arange(len(labels))
for i, n in enumerate((1296, 5184)):
    vals = [gib(sum(d.values()), n) for d, _ in groups.values()]
    vals.append(2 * NK[n] * n * 8 / 2**30)
    off = -width / 2 + i * width
    ax2.bar(x + off, vals, width, label=f"{n:,} atoms",
            color=("#9ecae1", "#4C78A8")[i],
            edgecolor="black", lw=0.4)
ax2.set_xticks(x)
ax2.set_xticklabels(["fwd", "rev", "cache", "partials", "Ewald"], fontsize=7.5)
ax2.set_ylabel("GiB", fontsize=8)
ax2.tick_params(axis="y", labelsize=7.5)
ax2.grid(axis="y", lw=0.4, alpha=0.35)
ax2.legend(fontsize=7.5, frameon=False, loc="upper right")
ax2.set_title("(b) Totals by lifetime group", fontsize=9, loc="left")
ax2.set_ylim(0, 1.15)
ax2.text(0.03, 0.97, "named total: 0.66 / 2.78 GiB",
         transform=ax2.transAxes, ha="left", va="top", fontsize=7.2,
         bbox=dict(fc="white", ec="#bbbbbb", lw=0.4, pad=2))

fig.savefig(OUT / "polar_memory_ledger.png", bbox_inches="tight")
plt.close(fig)

# ---- figure 2: projection ----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), dpi=150,
                         gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})

# (a) VRAM scenarios at 5,184
ax = axes[0]
scen = [
    ("S1.1 measured", 19.21, "#777777"),
    ("Tier 1: P1-P3 + cache", 18.43, "#4C78A8"),
    ("Tier 1 + P5 (low)", 18.03, "#72B7B2"),
    ("Tier 1 + P5 (high)", 17.23, "#54A24B"),
]
y = np.arange(len(scen))[::-1]
for yi, (lab, v, col) in zip(y, scen):
    ax.hlines(yi, 17.0, v, color=col, lw=5, alpha=0.9)
    ax.plot([v], [yi], "o", color=col, ms=7, mec="black", mew=0.4)
    ax.text(v + 0.03, yi, f"{v:.2f}", va="center", fontsize=7.5)
ax.axvline(18.0, color="crimson", lw=1.0, ls="--")
ax.text(18.05, 3.30, "target < 18.0", color="crimson", fontsize=7.5)
ax.set_yticks(y)
ax.set_yticklabels([s_[0] for s_ in scen], fontsize=7.8)
ax.set_ylim(-0.6, 3.6)
ax.set_xlim(17.0, 19.5)
ax.set_xlabel("peak VRAM at 5,184 atoms (GiB)", fontsize=8)
ax.grid(axis="x", lw=0.4, alpha=0.35)
ax.tick_params(axis="x", labelsize=7.5)
ax.set_title("(a) VRAM projection (estimate)", fontsize=9, loc="left")
ax.text(0.60, 0.12,
        "Tier 1 removes UVU partials,\n"
        "mixed_bar, debug copies,\n"
        "cache step 0. P5 = node-tiled\n"
        "reverse replay + arena.",
        transform=ax.transAxes, fontsize=6.4, va="bottom", ha="left",
        bbox=dict(fc="white", ec="#bbbbbb", lw=0.4, pad=2))

# (b,c) speed ranges
speed = [
    ("(b) 1,296 atoms", 278.15, 266.21, 220, 292,
     [(-1, -3), (-3, -8), (-4, -10), (-8, -15), (-2, -5), (-1, -2)]),
    ("(c) 5,184 atoms", 1084.25, 1000.0, 930, 1110,
     [(-1, -3), (-6, -15), (-8, -20), (-30, -55), (-8, -20), (-3, -6)]),
]
for ax, (title, base, target, lo_y, hi_y, ranges) in zip(axes[1:], speed):
    lo = base + sum(d[0] for d in ranges)
    hi = base + sum(d[1] for d in ranges)
    ax.plot([0, 0], [lo, hi], color="#4C78A8", lw=9,
            solid_capstyle="butt", alpha=0.85)
    ax.plot([0], [base], "o", color="#777777", ms=7, mec="black", mew=0.4)
    ax.text(0.09, base, f"base {base:.0f}", fontsize=7, va="center",
            color="#555555")
    ax.text(0.09, lo, f"{lo:.0f}", fontsize=7.5, va="center",
            color="#1f4e79")
    ax.text(0.09, hi, f"{hi:.0f}", fontsize=7.5, va="center",
            color="#1f4e79")
    ax.axhline(target, color="crimson", lw=1.0, ls="--")
    ax.text(-0.34, target + (2 if base < 500 else 6), f"target {target:.0f}",
            fontsize=6.8, color="crimson", va="bottom")
    ax.set_xlim(-0.35, 0.55)
    ax.set_ylim(lo_y, hi_y)
    ax.set_xticks([])
    ax.set_ylabel("ms / eval", fontsize=8)
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_title(title + ": speed range", fontsize=9, loc="left")
    ax.text(0.03, 0.97,
            "additive P1-P3, K1-K3, P5;\nall projections, P0-gated",
            transform=ax.transAxes, fontsize=6.4, va="top",
            bbox=dict(fc="white", ec="#bbbbbb", lw=0.4, pad=2))

fig.savefig(OUT / "polar_projection.png", bbox_inches="tight")
plt.close(fig)

print("wrote", OUT / "polar_memory_ledger.png")
print("wrote", OUT / "polar_projection.png")
