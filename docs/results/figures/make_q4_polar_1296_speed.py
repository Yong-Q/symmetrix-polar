#!/usr/bin/env python3
"""Figure: queue004 vs local PolarMACE 1,296-atom speed (2026-10-22)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/workspace/group/shared/bza/symmetrix-polar")
DATA = ROOT / "docs/results/data/q4_polar_1296_2026-10-22.json"
OUT = ROOT / "docs/results/figures"

d = json.loads(DATA.read_text())
lg = d["local_reference"]["generic"]
lm = d["local_reference"]["materialized"]
qg = d["q4"]["generic"]
qm = d["q4"]["materialized"]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10.8,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))

# (a) paired dot plot: local vs q4 for both paths
ax = axes[0]
rows = [
    ("generic\n(fastest)", lg["us_per_atom"], qg["us_per_atom"], "#4c78a8", 1.0),
    ("materialized", lm["us_per_atom"], qm["us_per_atom"], "#f58518", 0.0),
]
for label, local_v, q4_v, color, y in rows:
    y_local = y + 0.14
    y_q4 = y - 0.14
    ax.plot([local_v, q4_v], [y_local, y_q4], color="#999999", lw=1.3, zorder=1)
    ax.scatter([local_v], [y_local], s=85, facecolors="white", edgecolors="#333333",
               linewidths=1.3, zorder=3, label="local" if y == 1.0 else None)
    ax.scatter([q4_v], [y_q4], s=85, marker="s", facecolors=color,
               edgecolors=color, zorder=3, label="queue004" if y == 1.0 else None)
    ax.annotate(f"{local_v:.1f}", (local_v, y_local + 0.07), ha="center", va="bottom",
                fontsize=8.6, color="#333333")
    ax.annotate(f"{q4_v:.1f}", (q4_v, y_q4 - 0.07), ha="center", va="top",
                fontsize=8.6, color=color, fontweight="bold")

ax.set_yticks([1.0, 0.0])
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(143.0, 158.5)
ax.set_ylim(-0.55, 1.60)
ax.set_xlabel("µs / atom / eval")
ax.set_title("(a) 1,296 atoms, one A100-80GB", fontsize=10.8)
ax.grid(axis="x", alpha=0.25)
ax.legend(loc="upper right", fontsize=8.3, frameon=True)
ax.text(0.03, 0.96,
        "q4 matches local within 0.4% (no single-GPU speedup)",
        transform=ax.transAxes, fontsize=8.4, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9))

# (b) q4 per-eval samples
ax = axes[1]
xg = np.arange(1, len(qg["samples_ms"]) + 1)
xm = np.arange(1, len(qm["samples_ms"]) + 1)
ax.plot(xg, qg["samples_ms"], "-o", color="#4c78a8", ms=6,
        label=f"generic: {qg['median_ms']:.1f} ms/eval")
ax.plot(xm, qm["samples_ms"], "-s", color="#f58518", ms=6,
        label=f"materialized: {qm['median_ms']:.1f} ms/eval")
ax.axhline(qg["median_ms"], color="#4c78a8", ls="--", lw=1.0, alpha=0.7)
ax.axhline(qm["median_ms"], color="#f58518", ls="--", lw=1.0, alpha=0.7)
ax.set_xlim(0.6, 5.4)
ax.set_ylim(186.5, 203.5)
ax.set_xticks(xg)
ax.set_xlabel("run 1 sample")
ax.set_ylabel("ms / eval")
ax.set_title("(b) q4 samples, 5 × run 1", fontsize=10.8)
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="center right", fontsize=8.4, frameon=True)
ax.text(0.03, 0.04,
        f"PE stable at {qg['pe_eV']:.8f} eV",
        transform=ax.transAxes, fontsize=8.2, va="bottom")

fig.suptitle("queue004 vs local: current N1 PolarMACE, 1,296 water atoms",
             fontsize=11.8)
fig.text(0.5, 0.015,
         "q4: t-20261022132315-b97jc (materialized, 64 s) and t-20261022132500-94hhz (generic, 69 s); "
         "lmp_gemm sha256 093982d5…; model sha256 0e98f0ae…",
         ha="center", va="bottom", fontsize=8.0, color="#444444")
fig.text(0.5, -0.015,
         "LAMMPS wall time 16 s in both q4 runs, of which model load is ~15 s; "
         "five measured evals take ~0.95–1.0 s.",
         ha="center", va="bottom", fontsize=8.0, color="#666666")
fig.tight_layout(rect=[0, 0.07, 1, 0.92])
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"Q4_POLAR_1296_SPEED.{ext}", dpi=150, bbox_inches="tight")
print("wrote", OUT / "Q4_POLAR_1296_SPEED.png")
