#!/usr/bin/env python3
"""Shared PNAS-style figures for the distillation README and PDF.

Each builder returns a matplotlib Figure; callers save PNG or embed in a PDF.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
DATA = json.loads((HERE / "data/full_valid_head_ewald.json").read_text())
VF = HERE.parent.parent / "bench/plan_d_bulk_2026-10-25/results/valid_by_species.json"
if not VF.exists():
    VF = HERE / "data/valid_by_species.json"
VALID = json.loads(VF.read_text()) if VF.exists() else {}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 180,
})
C = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73",
     "purple": "#CC79A7", "yellow": "#E69F00", "sky": "#56B4E9",
     "grey": "#4D4D4D"}
SPECIES = ["C2H4", "C2H6", "CH4", "CO2", "H2O", "NH3"]
FOURPI = 4.0 * np.pi


def _panel(ax, lab):
    ax.text(-0.20, 1.05, lab, transform=ax.transAxes, fontsize=11,
            fontweight="bold")


def _tidy(ax, grid="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.16, axis=grid, lw=0.6)


# ---------------------------------------------------------------- architecture
def architecture_fig():
    fig = plt.figure(figsize=(10.6, 5.9))
    ax = fig.add_axes([0.0, 0.08, 1.0, 0.84])
    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    boxes = {
        "teacher": (0.25, 6.5, 3.0, 1.25,
                    "MACE-Polar-1-M teacher\n512 ch nonlinear, l=1 multipoles\nGTO/Ewald, 2 SCF steps"),
        "data": (3.55, 6.5, 3.0, 1.25,
                 "Distillation corpus\n23,376 E/F frames + 6,809 q/$\\mu$ frames\n1,100 states, six molecules"),
        "backbone": (6.85, 6.5, 3.0, 1.25,
                     "Linear MACE backbone\n128 ch, 2 interactions, $r_c$=5 A\n848,528 params, FP32/FP64"),
        "heads": (6.85, 4.15, 3.0, 1.4,
                  "Readout heads\nq: MLP 128-256-128-1\n$\\mu$: linear 384-3 (canonical frame)"),
        "ewald": (3.55, 4.15, 3.0, 1.4,
                  "GTO / Ewald kernel\nl=1 multipoles, 1.5 A smearing\nself + PBC corrections"),
        "out": (0.25, 4.15, 3.0, 1.4,
                "Outputs\nE, F (23K backbone)\nq, $\\mu$, $E_{elec}$ (diagnostic)"),
        "speed": (0.25, 1.55, 4.65, 1.45,
                  "Speed budget (projected)\n12.3 backbone + 7.9 Ewald\n+ <0.1 heads = ~20.2 $\\mu$s/atom"),
        "limit": (5.2, 1.55, 4.65, 1.45,
                  "Known limits\n0.1 meV/atom not reached for CO$_2$/H$_2$O/NH$_3$\nengine hybrid pair style pending"),
    }
    for k, (x, y, w, h, txt) in boxes.items():
        color = "#EAF2F8" if k in ("teacher", "data", "backbone") else "#FDF2E9"
        if k in ("speed", "limit"):
            color = "#F4F6F7"
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.09,rounding_size=0.14",
                                    fc=color, ec=C["grey"], lw=0.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=8.4, family="DejaVu Serif")
    arrows = [((3.30, 7.12), (3.55, 7.12)), ((6.60, 7.12), (6.85, 7.12)),
              ((8.35, 6.50), (8.35, 5.55)), ((6.85, 4.85), (6.60, 4.85)),
              ((3.55, 4.85), (3.30, 4.85))]
    for (x1, y1), (x2, y2) in arrows:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, lw=0.9, color=C["grey"]))
    ax.annotate("", xy=(2.6, 3.05), xytext=(2.6, 4.15),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=C["grey"],
                                linestyle=(0, (3, 3))))
    ax.text(2.75, 3.55, "sum of the three stages", fontsize=7.4,
            color=C["grey"], style="italic")
    ax.text(0.25, 0.55,
            "Teacher distillation ($E$, F, q, $\\mu$), charge/dipole readout heads, "
            "and the canonical-frame + GTO/Ewald path.",
            fontsize=9.5, fontweight="bold")
    return fig


# -------------------------------------------------------------------- accuracy
def accuracy_fig():
    q = [DATA[s]["q_mae"] * FOURPI for s in SPECIES]
    mu = [DATA[s]["mu_mae"] * FOURPI for s in SPECIES]
    ee = [DATA[s]["E_abs_diff_meV_atom"] for s in SPECIES]
    q_all = DATA["ALL"]["q_mae"] * FOURPI
    mu_all = DATA["ALL"]["mu_mae"] * FOURPI
    ee_all = DATA["ALL"]["E_abs_diff_meV_atom"]
    have_ef = all(f"cross:{s}|bulk" in VALID and f"cross:{s}|gas" in VALID
                  for s in SPECIES)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0))
    x = np.arange(len(SPECIES))

    ax = axes[0, 0]
    if have_ef:
        e_bulk = [VALID[f"cross:{s}|bulk"]["mae_e_meV_per_atom"] for s in SPECIES]
        e_gas = [VALID[f"cross:{s}|gas"]["mae_e_meV_per_atom"] for s in SPECIES]
        ax.bar(x - 0.19, e_gas, width=0.38, color=C["sky"], label="gas")
        ax.bar(x + 0.19, e_bulk, width=0.38, color=C["blue"], label="dense / bulk")
        top = max(e_gas + e_bulk) * 1.22
    else:
        e_all = [VALID.get(f"species:{s}", {}).get("mae_e_meV_per_atom", np.nan)
                 for s in SPECIES]
        ax.bar(x, e_all, width=0.66, color=C["blue"])
        top = np.nanmax(e_all) * 1.25
    ax.set_xticks(x); ax.set_xticklabels(SPECIES)
    ax.set_ylabel("E MAE [meV atom$^{-1}$]")
    ax.set_title("Energy accuracy on held-out frames")
    ax.set_ylim(0, top)
    ax.legend(frameon=False, loc="upper left", ncol=2)

    ax = axes[0, 1]
    if have_ef:
        f_bulk = [VALID[f"cross:{s}|bulk"]["mae_f_meV_per_A"] for s in SPECIES]
        f_gas = [VALID[f"cross:{s}|gas"]["mae_f_meV_per_A"] for s in SPECIES]
        ax.bar(x - 0.19, f_gas, width=0.38, color=C["yellow"], label="gas")
        ax.bar(x + 0.19, f_bulk, width=0.38, color=C["orange"], label="dense / bulk")
        top = max(f_gas + f_bulk) * 1.22
    else:
        f_all = [VALID.get(f"species:{s}", {}).get("mae_f_meV_per_A", np.nan)
                 for s in SPECIES]
        ax.bar(x, f_all, width=0.66, color=C["orange"])
        top = np.nanmax(f_all) * 1.25
    ax.set_xticks(x); ax.set_xticklabels(SPECIES)
    ax.set_ylabel("F MAE [meV A$^{-1}$]")
    ax.set_title("Force accuracy on held-out frames")
    ax.set_ylim(0, top)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    for ax, lab in zip([axes[0, 0], axes[0, 1]], ["A", "B"]):
        _panel(ax, lab)

    ax = axes[1, 0]
    ax.bar(x - 0.18, q, width=0.36, color=C["orange"], label="q MAE [e]")
    ax.bar(x + 0.18, mu, width=0.36, color=C["green"], label="$\\mu$ MAE [e A]")
    ax.axhline(q_all, color=C["orange"], ls=":", lw=0.9)
    ax.axhline(mu_all, color=C["green"], ls=":", lw=0.9)
    ax.text(-0.42, q_all * 1.18, f"all {q_all:.4f}", ha="left", fontsize=7,
            color=C["orange"])
    ax.text(-0.42, mu_all * 1.18, f"all {mu_all:.4f}", ha="left", fontsize=7,
            color=C["green"])
    ax.set_xticks(x); ax.set_xticklabels(SPECIES)
    ax.set_yscale("log")
    ax.set_ylabel("MAE (4$\\pi$ scale)")
    ax.set_title("Charge and dipole readout")
    ax.set_ylim(5e-4, 4e-1)
    ax.legend(frameon=False, loc="upper left", ncol=2)

    ax = axes[1, 1]
    bars = ax.bar(x, ee, width=0.66, color=C["blue"])
    for s in ("CO2", "H2O", "NH3"):
        bars[SPECIES.index(s)].set_color(C["purple"])
    ax.axhline(ee_all, color=C["orange"], ls="--", lw=1.0)
    ax.text(-0.42, ee_all + 0.025, f"all {ee_all:.3f}", ha="left", fontsize=7,
            color=C["orange"])
    ax.set_xticks(x); ax.set_xticklabels(SPECIES)
    ax.set_ylabel("Ewald |$\\Delta E$| [meV atom$^{-1}$]")
    ax.set_title("Electrostatic energy from predicted $q$, $\\mu$")
    ax.set_ylim(0, max(ee) * 1.30)
    for i, v in enumerate(ee):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
    for ax, lab in zip([axes[1, 0], axes[1, 1]], ["C", "D"]):
        _panel(ax, lab)

    for ax in axes.ravel():
        _tidy(ax)
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    return fig


# ------------------------------------------------------------------ speed
def speed_fig():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    names = ["linear MACE-OFF23\nFP32 · reference",
             "backbone only\nFP32 · measured",
             "backbone only\nFP64 · measured",
             "full package (+ q/$\\mu$ + Ewald)\nFP64 · projected",
             "128L1 polar student\nFP64 · measured",
             "exact PolarMACE\nFP64 · measured"]
    speeds = [4.50, 4.47, 12.3, 20.2, 95.11, 128.99]
    colors = [C["grey"], C["blue"], C["sky"], C["orange"], C["green"], C["purple"]]
    bars = ax.barh(np.arange(len(names)), speeds, color=colors, height=0.62)
    bars[3].set_hatch("//")
    bars[3].set_edgecolor("white")
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=7.6)
    ax.set_xscale("log"); ax.set_xlabel("$\\mu$s atom$^{-1}$ step$^{-1}$ (log)")
    ax.set_title("Speed at 5,184 atoms, one A100\n(backbone-only vs full package)")
    for i, v in enumerate(speeds):
        ax.text(v * 1.09, i, f"{v:.2f}", va="center", fontsize=7.8)
    ax.set_xlim(3, 320)
    ax.axvline(20.0, color=C["grey"], ls=":", lw=0.7)
    ax.invert_yaxis()

    ax = axes[1]
    pts = [("backbone FP32 (E/F)", 4.47, 1.2, C["blue"]),
           ("backbone FP64 (E/F)", 12.3, 1.2, C["sky"]),
           ("+ q/$\\mu$ + Ewald (proj.)", 20.2, 1.2, C["orange"]),
           ("128L1 polar student", 95.11, 3.8, C["green"]),
           ("exact PolarMACE", 128.99, 0.05, C["purple"])]
    for i, (name, sp, acc, col) in enumerate(pts):
        ax.plot(sp, max(acc, 0.02), "o", color=col, ms=7, zorder=3)
        ax.annotate(str(i + 1), (sp, max(acc, 0.02)), xytext=(7, -10),
                    textcoords="offset points", fontsize=7.5, color=col,
                    fontweight="bold")
    ax.axhline(5.0, color=C["grey"], ls="--", lw=0.8)
    ax.text(3.2, 5.7, "E gate 5 meV/atom", fontsize=7.5, color=C["grey"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("speed [$\\mu$s atom$^{-1}$ step$^{-1}$]")
    ax.set_ylabel("bulk E MAE [meV atom$^{-1}$]")
    ax.set_title("Speed vs accuracy")
    ax.set_xlim(3, 320); ax.set_ylim(0.02, 60)
    leg = [f"{i+1}.  {p[0]}" for i, p in enumerate(pts)]
    ax.text(0.02, 0.03, "\n".join(leg), transform=ax.transAxes, fontsize=7.2,
            va="bottom", ha="left",
            bbox=dict(fc="white", ec="none", alpha=0.85))
    for ax, lab in zip(axes, ["A", "B"]):
        _panel(ax, lab)
        _tidy(ax, grid="both")
    fig.tight_layout(w_pad=2.0)
    return fig
