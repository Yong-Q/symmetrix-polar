#!/usr/bin/env python3
"""Figure: full-rank UVU weights vs low-rank polar activations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/workspace/group/shared/bza/symmetrix-polar")
FR = json.loads((ROOT / "bench/theory_rank_2026-10-22/feature_rank.json").read_text())
OUT = ROOT / "docs/results/figures"

rows = {r["module"]: r for r in FR["rows"]}
fd = rows["fdcm0_node_feats"]
le = rows["le_q"]

plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5,
                     "axes.labelsize": 9.5, "figure.dpi": 150})
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

# (a) activation energy rank
ax = axes[0]
labels = ["90%", "99%", "99.9%", "99.99%"]
fd_v = [fd[f"energy_rank_{k}"] for k in ("90", "99", "999", "9999")]
le_v = [le[f"energy_rank_{k}"] for k in ("90", "99", "999", "9999")]
x = np.arange(len(labels))
w = 0.36
ax.bar(x - w / 2, fd_v, w, label="field map input", color="#4c78a8")
ax.bar(x + w / 2, le_v, w, label="local-electron input", color="#f58518")
for xi, v in zip(x - w / 2, fd_v):
    ax.text(xi, v + 8, str(v), ha="center", va="bottom", fontsize=7.6)
for xi, v in zip(x + w / 2, le_v):
    ax.text(xi, v + 8, str(v), ha="center", va="bottom", fontsize=7.6)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("rank needed (of 512)")
ax.set_title("(a) Polar input activations are low-rank")
ax.set_ylim(0, 450)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=7.5)
ax.text(0.03, 0.95, "512-atom NH3 snapshot, 512×2048 input",
        transform=ax.transAxes, va="top", fontsize=7.4, color="#555555")

# (b) cost ratio vs projection rank
ax = axes[1]
r = np.array([32, 64, 128, 256, 512])
linear_ratio = 409.6 / r       # 2048 -> 512 linear
uvu_ratio = 256.0 / r          # 512 x 512 square UVU
ax.plot(r, linear_ratio, "-o", color="#4c78a8", ms=5,
        label="2048→512 linear: 409.6/r")
ax.plot(r, uvu_ratio, "-s", color="#f58518", ms=5,
        label="512×512 UVU: 256/r")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(r)
ax.set_xticklabels([str(v) for v in r])
ax.set_xlabel("projection rank r")
ax.set_ylabel("FLOP reduction factor")
ax.set_title("(b) The rank/error trade-off")
ax.grid(which="both", alpha=0.22)
ax.legend(fontsize=7.3)
for rr, lab in ((62, "r99=62"), (76, "r99=76"), (202, "r99.9=202"), (238, "r99.9=238")):
    ax.axvline(rr, color="#999999", ls=":", lw=0.9)
    ax.text(rr * 1.03, 1.05, lab, rotation=90, va="bottom", fontsize=6.6,
            color="#666666")
ax.set_ylim(0.7, 20)

fig.suptitle("PolarMACE: full-rank weights, low-rank activations", fontsize=12)
fig.text(0.5, 0.015,
         "UVU weights: rank 512/512 at 1e-4 (no weight compression). "
         "Activations: 99% energy in 62–76 directions; 99.9% in 202–238. "
         "The output E/F error at rank r is the unmeasured gate.",
         ha="center", va="bottom", fontsize=7.6, color="#444444")
fig.tight_layout(rect=[0, 0.06, 1, 0.93])
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"FEATURE_LOWRANK.{ext}", dpi=150, bbox_inches="tight")
print("wrote", OUT / "FEATURE_LOWRANK.png")
