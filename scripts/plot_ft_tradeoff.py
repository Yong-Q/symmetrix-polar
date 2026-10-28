#!/usr/bin/env python3
"""Pareto view of the three combined-distillation attempts against the base
student, on the old held-out gate and on the newly sampled field holdout."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
FIG = REPO / "docs/results/figures"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

POINTS = [
    ("base student (production)", "bench/plan_d_bulk_2026-10-25/results/valid_by_species.json",
     "bench/plan_d_ft3_20261028/results/base_on_holdout.json", "#000000", "o", 90),
    ("ft3  lr 1e-5, ep2", "bench/plan_d_ft3_20261028/results/ft3_ep2_valid_full.json",
     "bench/plan_d_ft3_20261028/results/ft3_ep2_on_holdout.json", "#0072B2", "s", 70),
    ("ft4  lr 1e-5, ep0", "bench/plan_d_ft4_20261028/results/ft4_ep0_valid_full.json",
     "bench/plan_d_ft4_20261028/results/ft4_ep0_on_holdout.json", "#D55E00", "^", 70),
    ("ft5  lr 2e-6, final", "bench/plan_d_ft5_20261028/results/ft5_valid_full.json",
     "bench/plan_d_ft5_20261028/results/ft5_on_holdout.json", "#009E73", "D", 70),
]


def load(rel):
    return json.loads((REPO / rel).read_text())["ALL"]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    for ax, idx, title in ((axes[0], 0, "(a) held-out bulk/gas, 3,465 frames"),
                           (axes[1], 1, "(b) new whole-seed field holdout, 182 frames")):
        for name, vrel, hrel, col, mk, ms in POINTS:
            d = load(vrel if idx == 0 else hrel)
            x = d["mae_f_meV_per_A"]
            y = d["mae_e_meV_per_atom"]
            ax.scatter(x, y, s=ms, color=col, marker=mk, edgecolor="k",
                       linewidth=0.6, zorder=3, label=name)
            ax.annotate(name.split("  ")[0] if name.startswith(("ft",)) else "base",
                        (x, y), textcoords="offset points", xytext=(7, 6),
                        fontsize=7.6, color=col)
        ax.set_xlabel("force MAE (meV Å$^{-1}$)")
        ax.set_ylabel("energy MAE (meV atom$^{-1}$)")
        ax.set_title(title, fontsize=9.8)
        ax.grid(lw=0.35, alpha=0.4)
    axes[0].set_xlim(7.60, 7.82)
    axes[0].set_ylim(1.08, 1.34)
    axes[1].set_xlim(26.4, 30.4)
    axes[1].set_ylim(3.9, 4.8)
    axes[0].annotate("better", xy=(7.61, 1.09), fontsize=7.5, color="0.35")
    fig.suptitle("Combined distillation keeps the new region better but pays on "
                 "the old energy gate", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = FIG / f"FT_TRADEOFF_2026-10-28.{ext}"
        fig.savefig(out, dpi=400, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
