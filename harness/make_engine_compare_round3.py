#!/usr/bin/env python3
"""Latest measured engine cost on the same 128-atom NH3 box, one A100.

One bar per configuration, using the latest measured value.  No historical
version columns: the full PolarMACE bar is the round-3 final (LAMMPS parity
PASS), the backbone is the latest polar-off measurement, and the linear
MACE-OFF23 bars are a different, smaller model shown for scale only.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs/examples/data/engine_compare_round3.json"
comp = json.load(open(DATA))

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "savefig.dpi": 150,
})

rows = [
    ("Full PolarMACE\n(latest, 128 atoms = 32 NH3)", comp["polar_mace"]["full_polar_ms"], "#1f6feb"),
    ("Nonlinear backbone\n(polar off, latest)", comp["polar_mace"]["backbone_polar_off_ms"], "#bf8700"),
    ("Linear MACE-OFF23 medium\ngeneric streamed (Symmetrix path)", comp["linear_mace_off23_medium"]["generic_streamed_ms"], "#2da44e"),
    ("Linear MACE-OFF23 medium\nmaterialized (Symmetrix path)", comp["linear_mace_off23_medium"]["materialized_ms"], "#1a7f37"),
]

fig, ax = plt.subplots(figsize=(10, 4.8))
y = np.arange(len(rows))[::-1]
vals = [v for _, v, _ in rows]
ax.barh(y, vals, color=[c for _, _, c in rows], height=0.55)
for yi, v in zip(y, vals):
    ax.text(v * 1.07, yi, f"{v:.1f} ms", va="center", fontsize=11,
            fontweight="bold")
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels([l for l, _, _ in rows])
ax.set_xlabel("LAMMPS pair evaluation, 1 rank (ms, log scale)")
ax.set_title("Same 128-atom NH$_3$ box, one A100 — latest measured")
ax.set_xlim(4, max(vals) * 2.2)
ax.grid(axis="y", visible=False)

fig.text(
    0.5, 0.01,
    "Full PolarMACE: latest 48.74 ms (complete gate suite PASS: module 5-8, "
    "32/32 end-to-end, LAMMPS parity). Backbone polar-off: latest 27.7 ms "
    "(full suite PASS). Linear MACE-OFF23 is a smaller, different model; shown for scale only.",
    ha="center", va="bottom", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f6f8fa",
              edgecolor="#d0d7de"),
)
fig.tight_layout(rect=[0, 0.08, 1, 0.96])
for out in (REPO / "results/polar_engine_compare.png",
            REPO / "docs/figures/05_engine_compare.png"):
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
plt.close(fig)
