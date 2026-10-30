#!/usr/bin/env python3
"""Per-state student error on the 2026-10-29 q4 on-policy pool.

Bars: production student vs ft7.  The hatch marks the states whose initial
packing was not minimised cleanly (min interatomic distance < 1.0 A), which
inflate the aggregate without representing equilibrium 0.035 CO2.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
RES = REPO / "bench/active_cov_20261029"
FIG = REPO / "docs/results/figures/Q4_DENSE_ERROR_2026-10-29.png"

base = json.loads((RES / "eval_base_q4dense.json").read_text())
ft7 = json.loads((RES / "eval_ft7_q4dense.json").read_text())

states = [k for k in sorted(base) if k.startswith("tag:")]
labels = []
for k in states:
    raw = k.split("tag:")[1]                       # CO2-64mol-r0.035-500K-s43
    species, _, rho, temp = raw.split("-")[:4]
    labels.append(f"{species} {rho.lstrip('r')} {temp} {raw.rsplit('-', 1)[-1]}")
stressed = {"CO2-64mol-r0.035-500K-s43"}  # min_d 0.93 A, FIRE did not converge

x = np.arange(len(states))
w = 0.38
fig, ax = plt.subplots(figsize=(11, 5.5))
fb = [base[k]["mae_f_meV_per_A"] for k in states]
ff = [ft7[k]["mae_f_meV_per_A"] for k in states]
b1 = ax.bar(x - w / 2, fb, w, label="base (production)", color="#4d4d4d")
b2 = ax.bar(x + w / 2, ff, w, label="ft7", color="#d62728")
for i, s in enumerate(states):
    if s.split("tag:")[1] in stressed:
        for b in (b1[i], b2[i]):
            b.set_hatch("///")
ax.set_ylabel("Force MAE (meV Å$^{-1}$)")
ax.set_title("Student vs MACE-Polar on the q4 on-policy pool (138 frames, 12 states)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
ax.grid(axis="y", alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.99, 0.95, "hatched: packing not minimised\n(min d < 1.0 Å), excluded from ft8",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#555555")
fig.tight_layout()
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150)
print(f"wrote {FIG}")
