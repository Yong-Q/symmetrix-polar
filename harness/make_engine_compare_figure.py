#!/usr/bin/env python3
"""Engine-cost and MPI-overhead comparison figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/workspace/group/shared/bza/task3_sampling_database/mlip")
OUT = ROOT / "bench"

comp = json.load(open(OUT / "lammps_parity/engine_compare/engine_compare.json"))
lin_mpi = json.load(open(OUT / "lammps_parity/linear_mpi/linear_mpi.json"))
p1 = json.load(open(OUT / "lammps_parity/polar_time_mpi1.json"))
p2 = json.load(open(OUT / "lammps_parity/polar_time_mpi2.json"))
p2f = json.load(open(OUT / "lammps_parity/polar_time_mpi2_onehop.json"))

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "savefig.dpi": 150,
})

left = [
    ("Linear MACE\n(MACEKokkos)", comp[0]["ms_per_eval"], "#2da44e"),
    ("Linear MACE\n+m MPI msg", lin_mpi[0]["ms_per_eval"], "#1a7f37"),
    ("Nonlinear backbone\n(polar off)", comp[1]["ms_per_eval"], "#bf8700"),
    ("Full PolarMACE\nv29", comp[2]["ms_per_eval"], "#1f6feb"),
]
right = [
    ("Upstream linear MPI\n(1→2 ranks)", lin_mpi[1]["ms_per_eval"] / lin_mpi[0]["ms_per_eval"], "#2da44e"),
    ("Our 1-hop ghost\n(1→2 ranks)", p2f["ms_per_eval"] / p1["ms_per_eval"], "#bf8700"),
    ("Our exact 2-hop ghost\n(1→2 ranks)", p2["ms_per_eval"] / p1["ms_per_eval"], "#1f6feb"),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2),
                         gridspec_kw={"width_ratios": [1.5, 1.0]})

ax = axes[0]
y = np.arange(len(left))[::-1]
vals = [v for _, v, _ in left]
ax.barh(y, vals, color=[c for _, _, c in left], height=0.55)
for yi, v in zip(y, vals):
    ax.text(v * 1.06, yi, f"{v:.1f} ms", va="center", fontsize=10,
            fontweight="bold")
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels([l for l, _, _ in left])
ax.set_xlabel("LAMMPS pair evaluation, 1 rank (ms, log scale)")
ax.set_title("Same 128-atom NH$_3$ box, one A100")
ax.set_xlim(4, max(vals) * 2.2)
ax.grid(axis="y", visible=False)

ax = axes[1]
y = np.arange(len(right))[::-1]
ratios = [v for _, v, _ in right]
ax.barh(y, ratios, color=[c for _, _, c in right], height=0.55)
for yi, v in zip(y, ratios):
    ax.text(v + 0.6, yi, f"{v:.1f}×", va="center", fontsize=11,
            fontweight="bold")
ax.axvline(1.0, color="#57606a", ls="--", lw=1.0)
ax.set_yticks(y)
ax.set_yticklabels([l for l, _, _ in right])
ax.set_xlabel("2-rank time / 1-rank time (lower is better)")
ax.set_title("MPI overhead on one shared GPU")
ax.set_xlim(0, max(ratios) * 1.25)
ax.grid(axis="y", visible=False)
fig.text(
    0.5, 0.01,
    "Message passing is ~113× faster than exact ghost replication at 2 ranks; "
    "on a single shared GPU even upstream MPI is 3.4× slower than 1 rank.",
    ha="center", va="bottom", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f6f8fa",
              edgecolor="#d0d7de"),
)

fig.suptitle("Engine cost and MPI overhead: linear MACE vs PolarMACE",
             fontsize=14, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig(OUT / "polar_engine_compare.png", bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "polar_engine_compare.png")
