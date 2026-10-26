#!/usr/bin/env python3
"""Accuracy-vs-speed map for the 4G vs linear-distillation decision (2026-10-26).

All accuracy numbers are on held-out bulk frames (253 frames). Speeds are
microseconds per atom per step at 5,184 atoms unless labelled otherwise.

Sources:
  4G fixed        docs/results/PLAN_C_4G_STATUS_2026-10-25.md
  4G + local AL   docs/results/PLAN_C_4G_LOCAL_AL_2026-10-26.md
  4G wide+ee4G    docs/results/PLAN_C_4G_WIDE_EE4G_2026-10-26.md
  Plan D linear   bench/plan_d_bulk_2026-10-25/logs/plan_d_bulk_run-42_debug.log
  Plan D speeds   bench/plan_d_*_2026-10-2*/speed*, docs/results/PAPER_COMPARABILITY.md
  polar students  README.md (q4 tasks 2026-10-23)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)

# ---------------------------------------------------------------- 4G family
for ax, col, gate, ylim in [
    (axes[0], "E", 5.0, (0.03, 150)),
    (axes[1], "F", 30.0, (0.03, 400)),
]:
    # 4G family (same speed, three accuracy points)
    y_wide = 31.01 if col == "E" else 75.06
    y_fix = 39.44 if col == "E" else 77.48
    y_al = 38.79 if col == "E" else 76.70
    ax.plot(13.45, y_wide, "D", color="#d62728", ms=9, mfc="white",
            mew=1.5, zorder=5)
    ax.plot([13.45, 13.45], [y_fix, y_al], "|", color="#d62728", ms=13,
            mew=1.6, zorder=4)
    ax.annotate("4G wide+ee4G\n31.0 / 75.1 (speed pending)",
                (13.45, y_wide), xytext=(10, 6),
                textcoords="offset points", fontsize=7.5,
                color="#d62728")
    ax.annotate("4G fixed / AL\n38.8–39.4 / 76.7–77.5", (13.45, y_al),
                xytext=(10, -26), textcoords="offset points",
                fontsize=7.5, color="#d62728")

    # Plan D linear (one accuracy, three speed paths)
    y_d = 1.2 if col == "E" else 7.8
    ax.plot(4.47, y_d, "^", color="#1f77b4", ms=9, mfc="white",
            mew=1.5, zorder=5)
    ax.plot(12.3, y_d, "P", color="#1f77b4", ms=8, mfc="white",
            mew=1.4, zorder=5)
    ax.plot(284.8, y_d, "v", color="#1f77b4", ms=9, zorder=5)
    ax.annotate("Plan D linear\n1.2 / 7.8", (4.47, y_d), xytext=(-8, 14),
                textcoords="offset points", fontsize=8,
                color="#1f77b4", ha="left")
    ax.annotate("FP32 direct 4.5", (4.47, y_d), xytext=(0, -14),
                textcoords="offset points", fontsize=6.5,
                color="#1f77b4", ha="center")
    ax.annotate("FP64 ref 12.3", (12.3, y_d), xytext=(0, -14),
                textcoords="offset points", fontsize=6.5,
                color="#1f77b4", ha="center")
    ax.annotate("FP64 generic 285", (284.8, y_d), xytext=(0, 10),
                textcoords="offset points", fontsize=6.5,
                color="#1f77b4", ha="center")

    # polar students
    y_128 = 3.8 if col == "E" else 6.1
    y_192 = 2.1 if col == "E" else 6.7
    ax.plot(91.64, y_128, "o", color="#2ca02c", ms=8, zorder=5)
    ax.plot(116.84, y_192, "s", color="#2ca02c", ms=8, zorder=5)
    ax.annotate("128L1 polar\n3.8 / 6.1", (91.64, y_128), xytext=(-8, 12),
                textcoords="offset points", fontsize=7.5,
                color="#2ca02c", ha="center")
    ax.annotate("192L1 polar\n2.1 / 6.7", (116.84, y_192), xytext=(10, -20),
                textcoords="offset points", fontsize=7.5,
                color="#2ca02c")

    # exact teacher
    ax.plot(133.04, 0.05 * gate, "x", color="#555555", ms=9, mew=1.6,
            zorder=5)
    ax.annotate("exact PolarMACE", (133.04, 0.05 * gate), xytext=(-6, 6),
                textcoords="offset points", fontsize=7.5,
                color="#555555", ha="right")

    ax.axhline(gate, color="#444444", ls="--", lw=1.1, zorder=1)
    ax.annotate("gate = %.0f" % gate, (0.02, gate * 1.06),
                xycoords=("axes fraction", "data"), fontsize=8,
                color="#444444")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3.2, 420)
    ax.set_ylim(*ylim)
    ax.set_xlabel("speed [µs atom$^{-1}$ step$^{-1}$, 5,184 atoms]")
    ax.grid(alpha=0.25, which="both")

axes[0].set_title("Bulk energy MAE [meV atom$^{-1}$]")
axes[1].set_title("Bulk force MAE [meV Å$^{-1}$]")
axes[0].set_ylabel("error (log scale)")

fig.suptitle(
    "Fast-model decision map (held-out bulk): 4G-HDNNP vs linear distillation\n"
    "filled = measured FP64 path; open = FP32 path or FP64 materialized reference",
    fontsize=11)

fig.savefig(OUT / "MODEL_ROUTE_DECISION_2026-10-26.png", dpi=150)
print("wrote", OUT / "MODEL_ROUTE_DECISION_2026-10-26.png")
