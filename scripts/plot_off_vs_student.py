#!/usr/bin/env python3
"""MACE-OFF23 vs the distilled student, both scored against the PolarMACE
teacher on the same frames."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
R = REPO / "bench/plan_d_ft6_20261029/results"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                     "axes.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False})

CASES = [
    ("CO$_2$ dense\n(no field, 0.00-0.02)", "off_vs_student_valid_co2.json"),
    ("CO$_2$ field\n0.01", "off_vs_student_co2_001.json"),
    ("CO$_2$ field\n0.02", "off_vs_student_co2_002.json"),
    ("CO$_2$ field\n0.03", "off_vs_student_co2_003_full.json"),
]


def main():
    labels, off, stu = [], [], []
    for lab, f in CASES:
        labels.append(lab)
        d = json.loads((R / f).read_text())["ALL"]
        off.append(d["off23_vs_polar_f"])
        stu.append(d["student_vs_polar_f"])
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    b1 = ax.bar(x - w / 2, off, w, color="#D55E00", edgecolor="k",
                linewidth=0.5, label="MACE-OFF23-medium vs teacher")
    b2 = ax.bar(x + w / 2, stu, w, color="#0072B2", edgecolor="k",
                linewidth=0.5, label="distilled student vs teacher")
    for bars, vals in ((b1, off), (b2, stu)):
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v * 1.06, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_yscale("log")
    ax.set_ylabel("force MAE vs PolarMACE teacher (meV Å$^{-1}$)")
    ax.set_title("The student is 6-7x closer to the teacher than MACE-OFF23 "
                 "at every density", fontsize=10)
    ax.legend(frameon=False, fontsize=8.4, loc="upper left")
    ax.grid(axis="y", lw=0.35, alpha=0.4, which="both")
    fig.tight_layout()
    out = REPO / "docs/results/figures/OFF_VS_STUDENT_2026-10-29.png"
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), dpi=400, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
