#!/usr/bin/env python3
"""Base student vs combined distillation fine-tune (ft3, epoch 2).

Two panels: held-out bulk/gas set (what the README reports) and the new
whole-seed field holdout (what the fine-tune is supposed to learn).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")
FIG = REPO / "docs/results/figures"
BASE = REPO / "bench/plan_d_bulk_2026-10-25/results/valid_by_species.json"
FT = REPO / "bench/plan_d_ft3_20261028/results"
HOLD_BASE = FT / "base_on_holdout.json"
HOLD_FT = FT / "ft3_ep2_on_holdout.json"

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
})


def panel(ax, labels, base, ft, ylabel, title, ylim=None):
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax.bar(x - w / 2, base, w, color=OKABE[0], edgecolor="k",
                linewidth=0.5, label="base student (2026-10-25)")
    b2 = ax.bar(x + w / 2, ft, w, color=OKABE[1], edgecolor="k",
                linewidth=0.5, label="+ combined distill (ft3, ep2)")
    for bars, vals in ((b1, base), (b2, ft)):
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v * 1.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", lw=0.4, alpha=0.35)


def main():
    b = json.loads(BASE.read_text())
    f = json.loads((FT / "ft3_ep2_valid_full.json").read_text())
    hb = json.loads(HOLD_BASE.read_text())
    hf = json.loads(HOLD_FT.read_text())
    keys = ["ALL", "species:CO2", "species:H2O", "species:NH3",
            "species:CH4", "species:C2H4", "species:C2H6"]
    labs = ["ALL", "CO$_2$", "H$_2$O", "NH$_3$", "CH$_4$", "C$_2$H$_4$",
            "C$_2$H$_6$"]
    hkeys = ["ALL", "species:CO2", "species:H2O", "species:NH3",
             "species:CH4", "species:C2H4"]
    hlabs = ["ALL", "CO$_2$", "H$_2$O", "NH$_3$", "CH$_4$", "C$_2$H$_4$"]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    panel(axes[0, 0], labs,
          [b[k]["mae_f_meV_per_A"] for k in keys],
          [f[k]["mae_f_meV_per_A"] for k in keys],
          "force MAE (meV Å$^{-1}$)",
          "(a) held-out bulk/gas, 3,465 frames", ylim=(0, 28))
    panel(axes[0, 1], hlabs,
          [hb[k]["mae_f_meV_per_A"] for k in hkeys],
          [hf[k]["mae_f_meV_per_A"] for k in hkeys],
          "force MAE (meV Å$^{-1}$)",
          "(b) new whole-seed field holdout, 182 frames", ylim=(0, 52))
    panel(axes[1, 0], labs,
          [b[k]["mae_e_meV_per_atom"] for k in keys],
          [f[k]["mae_e_meV_per_atom"] for k in keys],
          "energy MAE (meV atom$^{-1}$)",
          "(c) held-out bulk/gas, 3,465 frames", ylim=(0, 2.7))
    panel(axes[1, 1], hlabs,
          [hb[k]["mae_e_meV_per_atom"] for k in hkeys],
          [hf[k]["mae_e_meV_per_atom"] for k in hkeys],
          "energy MAE (meV atom$^{-1}$)",
          "(d) new whole-seed field holdout, 182 frames", ylim=(0, 11.5))
    axes[0, 0].legend(frameon=False, fontsize=8.6, loc="upper right")
    fig.suptitle("Combined distillation fine-tune: gains on the newly sampled "
                 "region, cost on the old energy gate", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = FIG / f"FT3_COMPARE_2026-10-28.{ext}"
        fig.savefig(out, dpi=400, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
