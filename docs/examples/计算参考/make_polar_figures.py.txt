#!/usr/bin/env python3
"""Generate polished performance and MPI figures from the gate JSONs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/workspace/group/shared/bza/task3_sampling_database/mlip")
OUT = ROOT / "bench"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

v28 = json.load(open(OUT / "polar_perf_v28.json"))["evals"]
v29 = json.load(open(OUT / "polar_perf.json"))["evals"]

keys = [
    ("nh3-f0 n=128 det", "128 atoms\n deterministic", "128 atoms · deterministic"),
    ("nh3-f0 n=128 rbe p=200", "128 atoms\n RBE p=200", "128 atoms · RBE p=200"),
    ("nh3-f0x2x2x2 n=1024 det", "1024 atoms\n deterministic", "1024 atoms · deterministic"),
]
labels = [k[1] for k in keys]
a = np.array([v28[k[0]]["median_ms"] for k in keys])
b = np.array([v29[k[0]]["median_ms"] for k in keys])
speedup = a / b

# --- single-rank v28 vs v29 -------------------------------------------------
fig, axes = plt.subplots(
    1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.55, 1.0]}
)
ax = axes[0]
y = np.arange(len(keys))[::-1]
for yi, va, vb in zip(y, a, b):
    ax.plot([va, vb], [yi, yi], color="#c9d1d9", lw=3, zorder=1)
ax.scatter(a, y, s=90, color="#8c96a0", edgecolor="white", zorder=3,
           label="v28 baseline")
ax.scatter(b, y, s=90, color="#1f6feb", edgecolor="white", zorder=4,
           label="v29 optimized")
for yi, va, vb in zip(y, a, b):
    ax.annotate(f"{va:,.0f} ms", (va, yi), xytext=(8, 12),
                textcoords="offset points", fontsize=9, color="#57606a")
    ax.annotate(f"{vb:,.0f} ms", (vb, yi), xytext=(-10, -18),
                textcoords="offset points", fontsize=9, color="#1f6feb",
                fontweight="bold", ha="right")
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels([k[2] for k in keys])
ax.set_xlabel("wall time per energy+force evaluation (ms, log scale)")
ax.set_title("Single-rank evaluation cost (A100, f64)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17),
          ncol=2, frameon=False)
ax.set_xlim(min(b.min(), a.min()) * 0.55, a.max() * 1.65)
ax.set_ylim(-0.6, 2.65)
ax.grid(axis="y", visible=False)

ax = axes[1]
colors = ["#1f6feb", "#1f6feb", "#2da44e"]
bars = ax.barh(y, speedup, color=colors, height=0.55, edgecolor="white")
for yi, value in zip(y, speedup):
    ax.text(value + 0.8, yi, f"{value:.1f}×", va="center",
            fontsize=12, fontweight="bold", color="#24292f")
ax.set_yticks(y)
ax.set_yticklabels([])
ax.set_xlabel("speedup vs v28")
ax.set_title("Speedup")
ax.set_xlim(0, max(speedup) * 1.25)
ax.grid(axis="y", visible=False)

fig.suptitle("Symmetrix-Polar v29: 37.8× faster at 128 atoms", y=0.99,
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "polar_perf_v28_vs_v29.png", bbox_inches="tight")
plt.close(fig)

# --- MPI status -------------------------------------------------------------
def load_ms(path):
    return json.load(open(path))["ms_per_eval"]

mpi_labels = [
    "1 rank\noptimized",
    "2 ranks\nexact (2-hop ghost)",
    "2 ranks\nfast (1-hop ghost)",
]
mpi_values = np.array([
    load_ms(OUT / "lammps_parity/polar_time_mpi1.json"),
    load_ms(OUT / "lammps_parity/polar_time_mpi2.json"),
    load_ms(OUT / "lammps_parity/polar_time_mpi2_onehop.json"),
])
mpi_colors = ["#2da44e", "#1f6feb", "#bf8700"]

fig, ax = plt.subplots(figsize=(10, 4.8))
y = np.arange(len(mpi_labels))[::-1]
ax.barh(y, mpi_values, color=mpi_colors, height=0.55, edgecolor="white")
for yi, value in zip(y, mpi_values):
    ax.text(value * 1.06, yi, f"{value:,.0f} ms", va="center",
            fontsize=11, fontweight="bold")
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels(mpi_labels)
ax.set_xlabel("wall time per energy+force evaluation (ms, log scale)")
ax.set_title("MPI on a single shared A100: correctness achieved, no speedup yet")
ax.set_xlim(mpi_values.min() * 0.6, mpi_values.max() * 2.2)
ax.grid(axis="y", visible=False)
ax.text(
    0.98, 0.98,
    "2-hop graph has ~20× more edges than the 1-rank graph\n"
    "(periodic ghost replication in a 14.7 Å box).\n"
    "1-hop mode is approximate: max |ΔF| = 4.1e-3 eV/Å (gate 1e-4).\n"
    "Scalable speedup requires local-graph message passing + multiple GPUs.",
    transform=ax.transAxes, ha="right", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f6f8fa",
              edgecolor="#d0d7de"),
)
fig.tight_layout()
fig.savefig(OUT / "polar_mpi_status.png", bbox_inches="tight")
plt.close(fig)
print("figures written:", OUT / "polar_perf_v28_vs_v29.png", OUT / "polar_mpi_status.png")
