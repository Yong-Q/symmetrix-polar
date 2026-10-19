#!/usr/bin/env python3
"""Figures for kernel round 2, the performance model, and the MPI H1 diagnosis.

All numbers are read from repository artifacts; nothing is re-estimated.
Run with the project Python (matplotlib + numpy).
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import mathtext
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

import figstyle

PROJ = Path("/workspace/group/shared/bza/symmetrix-polar")
BENCH = PROJ / "bench" / "kernel_round2_2026-10-19"
FIG = PROJ / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

figstyle.apply()

summary = json.loads((BENCH / "summary.json").read_text())
h1 = json.loads(Path(
    "/workspace/group/shared/bza/task3_sampling_database/mlip/bench/"
    "lammps_parity/h1_validation/h1_validation_results.json").read_text())


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# 13: round-2 speed
# ---------------------------------------------------------------------------
sp = summary["speed"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0),
                         gridspec_kw={"width_ratios": [1.55, 1.0]})
ax = axes[0]
labels = ["full", "polar-off"]
before = [sp["baseline_ms"], sp["polaroff_before_ms"]]
after = [sp["final_ms"], sp["polaroff_after_ms"]]
x = list(range(len(labels)))
b1 = ax.bar([i - 0.18 for i in x], before, width=0.36,
            color=figstyle.GRAY, label="before round 2")
b2 = ax.bar([i + 0.18 for i in x], after, width=0.36,
            color=figstyle.BLUE, label="round 2")
for i, (b, a) in enumerate(zip(before, after)):
    top = max(b, a)
    ax.text(i - 0.18, b + 2, f"{b:.1f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + 0.18, a + 2, f"{a:.1f}", ha="center", va="bottom", fontsize=8)
    ax.text(i, top * 1.12, f"{b/a:.2f}x", ha="center", va="bottom",
            color=figstyle.BLUE, fontweight="bold", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("ms / eval"); ax.set_title("NH3, 128 atoms (energy + forces)")
ax.set_ylim(0, max(before) * 1.22)
ax.set_axisbelow(True)
ax = axes[1]
before = [sp["n2_432_before_ms_per_step"]]
after = [sp["n2_432_after_ms_per_step"]]
ax.bar([-0.18], before, width=0.36, color=figstyle.GRAY)
ax.bar([0.18], after, width=0.36, color=figstyle.BLUE)
ax.text(-0.18, before[0] + 4, f"{before[0]:.1f}", ha="center", fontsize=8)
ax.text(0.18, after[0] + 4, f"{after[0]:.1f}", ha="center", fontsize=8)
ax.text(0, before[0] * 1.12, f"{before[0]/after[0]:.2f}x", ha="center",
        va="bottom", color=figstyle.BLUE, fontweight="bold", fontsize=9)
ax.set_xticks([0]); ax.set_xticklabels(["full"])
ax.set_ylabel("ms / step"); ax.set_title("N2, 432 atoms (first 10-step loop)")
ax.set_ylim(0, before[0] * 1.22)
ax.set_axisbelow(True)
fig.legend([b1, b2], ["before round 2", "round 2"], loc="upper center",
           ncol=2, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.94))
figstyle.save(fig, FIG / "13_kernel_round2_speed.png")

# ---------------------------------------------------------------------------
# 14: waterfall from 122.46 to 56.50 ms
# ---------------------------------------------------------------------------
items = [
    ("stale dense-reverse\nrebuilt", 25.588),
    ("fused gate\nnormalization", 13.034),
    ("reverse UVU\nx2 + x1", 7.346 + 2.245),
    ("forward / replay /\nLE UVU", 4.562 + 4.566 + 3.685),
    ("project\nkernels", 2.005 + 1.991 + 1.008),
]
start = sp["baseline_ms"]
final = sp["final_ms"]
fig, ax = plt.subplots(figsize=(10, 4.6))
running = start
ax.bar(0, start, color="#9aa4b2", width=0.6)
ax.text(0, start + 2, f"{start:.2f}", ha="center", fontsize=8)
for i, (label, delta) in enumerate(items, start=1):
    ax.bar(i, delta, bottom=running - delta, color="#d1495b", width=0.6)
    ax.text(i, running + 2, f"-{delta:.1f}", ha="center", fontsize=8,
            color="#a11")
    running -= delta
residual = running - final
ax.bar(len(items) + 1, residual, bottom=final, color="#c9d1d9", width=0.6)
ax.bar(len(items) + 2, final, color="#1f6feb", width=0.6)
ax.text(len(items) + 2, final + 2, f"{final:.2f}", ha="center", fontsize=8,
        fontweight="bold")
ax.set_xticks(range(len(items) + 3))
ax.set_xticklabels(["baseline\n(accepted lmp)"] + [x[0] for x in items]
                   + ["residual", "round 2\n(final)"], fontsize=7.5)
ax.set_ylabel("ms / eval (128-atom NH3, full polar)")
ax.set_title("Kernel round 2: where the 65.96 ms went")
ax.set_ylim(0, start * 1.08)
fig.tight_layout()
fig.savefig(FIG / "14_kernel_round2_waterfall.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 15: family before/after
# ---------------------------------------------------------------------------
fam = summary["families_ms_per_eval"]
order = ["polar pipeline", "backbone nonlinear", "tensor product",
         "MLP/dense", "cuBLAS/external", "e3nn linear", "other"]
before = [fam[k]["before"] for k in order]
after = [fam[k]["after"] for k in order]
fig, ax = plt.subplots(figsize=(10, 4.4))
y = range(len(order))
ax.barh([i + 0.19 for i in y], before, height=0.36, color="#9aa4b2",
        label="before")
ax.barh([i - 0.19 for i in y], after, height=0.36, color="#1f6feb",
        label="round 2")
for i, (b, a) in enumerate(zip(before, after)):
    ax.text(b + 0.8, i + 0.19, f"{b:.1f}", va="center", fontsize=7.5)
    ax.text(a + 0.8, i - 0.19, f"{a:.1f}", va="center", fontsize=7.5)
    if b > 0:
        ax.text(max(b, a) + 9, i, f"{b/a:.2f}x", va="center", fontsize=7.5,
                color="#0b3d91")
ax.set_yticks(list(y)); ax.set_yticklabels(order)
ax.set_xlabel("GPU kernel time (ms / eval)")
ax.set_title("Kernel families, 128-atom NH3 full model")
ax.set_xlim(0, 85)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "15_kernel_round2_families.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 16: remaining top kernels and SM coverage
# ---------------------------------------------------------------------------
rows = read_csv(BENCH / "analysis_exact/r5_full_labels.csv")
rows = [r for r in rows if float(r["total_ms"]) > 0.5]
rows.sort(key=lambda r: -float(r["total_ms"]))
top = rows[:14]
labels = [r["label"].replace("polar ", "").replace(" mode0", "")
          for r in top]
times = [float(r["total_ms"]) for r in top]
cov = [100 * float(r["sm_coverage"]) for r in top]
fig, ax = plt.subplots(figsize=(10, 5.2))
colors = ["#d1495b" if c < 50 else "#1f6feb" for c in cov]
bars = ax.barh(range(len(top)), times, color=colors)
ax.set_yticks(range(len(top))); ax.set_yticklabels(labels, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("GPU time (ms / eval, round 2)")
ax.set_title("Remaining top kernels: red = SM coverage below 50%")
for i, (t, c) in enumerate(zip(times, cov)):
    ax.text(t + 0.05, i, f"{t:.2f} ms / {c:.0f}%", va="center", fontsize=7.5)
ax.set_xlim(0, max(times) * 1.28)
fig.tight_layout()
fig.savefig(FIG / "16_kernel_round2_coverage.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 17: MPI H1 diagnosis
# ---------------------------------------------------------------------------
prod_old = 183.593
prod_new = sp["final_ms"]
h1_1 = h1["timing_run1"]["h1_1rank_ms"]
h1_2 = h1["timing_run1"]["h1_2rank_ms"]
exact = h1["timing_run1"]["exact_2hop_2rank_ms"]
lin_1 = 13.586
lin_2 = 45.878
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
ax = axes[0]
labels = ["production\n1 rank", "H1\n1 rank", "H1\n2 ranks", "exact 2-hop\n2 ranks"]
vals = [prod_old, h1_1[0], h1_2[0], exact]
ax.bar(range(4), vals, color=["#1f6feb", "#d1495b", "#d1495b", "#9aa4b2"])
for i, v in enumerate(vals):
    ax.text(i, v * 1.08, f"{v:.0f}", ha="center", fontsize=8)
ax.set_yscale("log")
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("ms / eval (log scale)")
ax.set_title("PolarMACE on one A100")
ax.annotate("1042 periodic\nghosts", xy=(1, h1_1[0]), xytext=(1.45, 300),
            fontsize=7.5, color="#a11",
            arrowprops=dict(arrowstyle="->", color="#a11", lw=0.8))
ax = axes[1]
ax.bar([0, 1], [lin_1, lin_2], color=["#1f6feb", "#d1495b"])
for i, v in enumerate([lin_1, lin_2]):
    ax.text(i, v * 1.08, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["1 rank", "2 ranks"])
ax.set_ylabel("ms / eval")
ax.set_title("Upstream linear MACE-OFF23 H1 MPI\n(same one-A100 topology)")
ax.text(0.5, max(lin_1, lin_2) * 0.45, "3.38x slower", ha="center",
        color="#0b3d91", fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "17_mpi_h1_diagnosis.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 18: H1 MPI schematic
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")
for x0, name in [(0.4, "rank 0"), (5.4, "rank 1")]:
    ax.add_patch(FancyBboxPatch((x0, 0.5), 4.2, 3.3,
                 boxstyle="round,pad=0.05,rounding_size=0.12",
                 fc="#f6f8fa", ec="#8b949e", lw=1))
    ax.text(x0 + 0.15, 3.55, name, fontsize=10, fontweight="bold")
    for i in range(6):
        ax.add_patch(Circle((x0 + 0.7 + (i % 3) * 0.85,
                             2.35 - (i // 3) * 0.85), 0.18,
                            fc="#1f6feb", ec="none"))
    for i in range(4):
        ax.add_patch(Circle((x0 + 3.0 + (i % 2) * 0.55,
                             2.35 - (i // 2) * 0.85), 0.15,
                            fc="white", ec="#d1495b", lw=1.2, ls="--"))
    ax.text(x0 + 0.55, 0.85, "local atoms", fontsize=7.5, color="#1f6feb")
    ax.text(x0 + 2.85, 0.85, "ghost rows", fontsize=7.5, color="#a11")
ax.add_patch(FancyArrowPatch((4.65, 2.85), (5.35, 2.85), arrowstyle="-|>",
             mutation_scale=12, color="#0b3d91", lw=1.4))
ax.text(4.45, 3.05, "H1 features\n(forward)", fontsize=7.5, color="#0b3d91")
ax.add_patch(FancyArrowPatch((5.35, 1.75), (4.65, 1.75), arrowstyle="-|>",
             mutation_scale=12, color="#a11", lw=1.4))
ax.text(4.42, 1.15, "adjoints\n(reverse)", fontsize=7.5, color="#a11")
ax.text(0.4, 0.1, "Each layer: exchange H1 (N x C) once; ghosts only need H1, "
        "not the full readout/polar pipeline.", fontsize=8, color="#57606a")
fig.tight_layout()
fig.savefig(FIG / "18_mpi_h1_schematic.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Formula images (for the PDF / offline viewers without MathJax)
# ---------------------------------------------------------------------------
FORMULAS = {
    "eq_ewald": r"$E_{\mathrm{lr}} = \frac{2\pi}{V}\sum_{\mathbf{k}\neq 0}"
                r"\frac{|\tilde{\rho}(\mathbf{k})|^2}{k^2},\quad "
                r"\tilde{\rho}(\mathbf{k})=\sum_j \tilde{\rho}_j(\mathbf{k})"
                r"e^{i\mathbf{k}\cdot\mathbf{r}_j}$",
    "eq_scf": r"$\rho^{(t+1)} = (1-\beta)\rho^{(t)} + \beta\,"
              r"\mathcal{F}(\rho^{(t)}),\quad "
              r"\|\rho^{(t+1)}-\rho^{(t)}\|<\epsilon$",
    "eq_linear": r"$y_j = \sum_i W_{ji}x_i + b_j$",
    "eq_tensor": r"$z^{(l)}_m = \sum_{l_1m_1l_2m_2} "
                 r"C^{lm}_{l_1m_1l_2m_2}\,x^{(l_1)}_{m_1}\,"
                 r"y^{(l_2)}_{m_2}$",
    "eq_occupancy": r"$O_i=\min\!\left(1,\frac{G_i}{N_{\mathrm{SM}}B_i}\right),"
                    r"\quad T_{\mathrm{SM}}=\sum_i t_iO_i$",
    "eq_amdahl": r"$S_{\mathrm{tot}}=\frac{1}{(1-p)+p/s},\quad "
                 r"S_{\mathrm{max}}=\frac{1}{1-p}$",
    "eq_comm": r"$V_{\mathrm{comm}}=N_g\,C\,b,\quad "
               r"T_r\approx\frac{T_{\mathrm{local}}+T_{\mathrm{comm}}}{r}"
               r"+T_{\mathrm{global}}$",
}
for name, formula in FORMULAS.items():
    mathtext.math_to_image(formula, str(FIG / f"{name}.png"), dpi=200,
                           format="png")

print("wrote figures to", FIG)
