#!/usr/bin/env python3
"""Figures for docs/design/MATH_ARCH_2026-10-22.md.

All speed labels are in us/atom/step.  No millisecond-based speed units.
Measured inputs are the committed N1/CUPTI budget at 1,296 atoms.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#059669"
GREY = "#6b7280"
RED = "#dc2626"


# ----------------------------------------------------------------------------
# Fig 1: measured budget and the candidate-1 component gates
# ----------------------------------------------------------------------------
def fig_budget():
    comps = [
        ("tensor product (MH-1)", 50.81, 27.0),
        ("host gap", 12.44, 1.5),
        ("polar pipeline", 15.62, 11.0),
        ("backbone nonlinear", 22.28, 22.28),
        ("cuBLAS / external", 19.38, 19.38),
        ("MLP / dense", 15.19, 15.19),
        ("e3 linear", 10.15, 10.15),
    ]
    comps = sorted(comps, key=lambda x: x[1])
    labels = [c[0] for c in comps]
    cur = np.array([c[1] for c in comps])
    tgt = np.array([c[2] for c in comps])
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    h = 0.36
    ax.barh(y + h / 2, cur, height=h, color=BLUE, label="measured (N1, 145.75)")
    ax.barh(y - h / 2, tgt, height=h, color=ORANGE, alpha=0.9,
            label="candidate-1 target (gates)")
    for yi, c, t in zip(y, cur, tgt):
        ax.text(c + 0.6, yi + h / 2, f"{c:.1f}", va="center", fontsize=7.5, color=BLUE)
        ax.text(t + 0.6, yi - h / 2, f"{t:.1f}", va="center", fontsize=7.5, color=ORANGE)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\mu$s/atom/step")
    ax.set_xlim(0, 58)
    ax.set_title("Measured budget and the two component gates (1,296 atoms, FP64, one A100)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    fig.savefig(os.path.join(OUT, "MATH_ARCH_BUDGET.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 2: roofline position of the dominant kernels
# ----------------------------------------------------------------------------
def fig_roofline():
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    x = np.logspace(-1.2, 2.2, 400)
    ax.plot(x, 2.0 * x, color=GREY, lw=1.4, ls="--", label="HBM roofline (2.0 TB/s)")
    ax.axhline(19.5, color=RED, lw=1.4, ls=":", label="A100 FP64 peak (19.5 TFLOPS)")
    ax.fill_between(x, 1e-3, np.minimum(2.0 * x, 19.5), color=GREY, alpha=0.06)
    ax.fill_between(x, 2.0 * x, 19.5, where=(2.0 * x < 19.5), color=RED, alpha=0.04)

    pts = [
        ("tensor product\n(0.45 flops/B)", 0.45, 0.019, BLUE, (8, 10)),
        ("Ewald / potential\n(~3 flops/B)", 3.2, 0.126, GREEN, (10, -20)),
        ("UVU grouped GEMM\n(54 flops/B)", 54.0, 15.64, ORANGE, (-125, -62)),
    ]
    for lab, xi, yi, c, off in pts:
        ax.scatter([xi], [yi], s=70, color=c, zorder=5, edgecolor="white", linewidth=0.8)
        ax.annotate(lab, (xi, yi), xytext=off, textcoords="offset points",
                    fontsize=8, color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.7))
    ax.annotate("", xy=(0.45, 0.9), xytext=(0.45, 0.019),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.0, ls=":"))
    ax.text(0.50, 0.10, "47x below the\nmemory roofline", fontsize=7.5, color=BLUE)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 160)
    ax.set_ylim(1e-3, 60)
    ax.set_xlabel("arithmetic intensity (flops/byte)")
    ax.set_ylabel("achieved performance (TFLOPS)")
    ax.set_title("The card is busy but inefficient: measured kernels on the FP64 roofline")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(which="both", alpha=0.18, lw=0.4)
    fig.savefig(os.path.join(OUT, "MATH_ARCH_ROOFLINE.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 3: candidate-1 projection waterfall
# ----------------------------------------------------------------------------
def fig_projection():
    steps = [
        ("current", 146.1, None),
        ("tensor 50.8 -> 27", 146.1, 122.3),
        ("host 12.4 -> 1.5", 122.3, 111.4),
        ("polar 15.6 -> 11.0", 111.4, 106.8),
        ("grouped GEMM/MLP/e3", 106.8, 97.3),
        ("gate-only total", 113.3, None),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs = np.arange(len(steps))
    for i, (lab, a, b) in enumerate(steps):
        if b is None:
            ax.bar(i, a, color=GREY if lab == "current" else ORANGE, width=0.62)
            ax.text(i, a + 2, f"{a:.1f}", ha="center", fontsize=8)
        else:
            ax.bar(i, a - b, bottom=b, color=GREEN, width=0.62)
            ax.text(i, a + 2, f"{b:.1f}", ha="center", fontsize=8)
            ax.plot([i - 0.5, i + 0.5], [b, b], color="white", lw=0.6, alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in steps], rotation=25, ha="right")
    ax.set_ylabel(r"$\mu$s/atom/step")
    ax.set_ylim(0, 175)
    ax.set_title("Candidate-1 projection: 146.1 -> 97.3 (central 100-110, not 90)")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.text(0.02, 0.97, "green = saving\norange = formal gate sum", transform=ax.transAxes,
            fontsize=7.5, color=GREY, ha="left", va="top")
    fig.savefig(os.path.join(OUT, "MATH_ARCH_PROJECTION.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 4: compression break-even
# ----------------------------------------------------------------------------
def fig_compression():
    r = np.logspace(np.log10(16), np.log10(512), 300)
    s_2048 = 2048 * 512 / (r * (2048 + 512))
    s_512 = 512 * 512 / (r * (512 + 512))
    s_joint = 512 * 512 / (2 * 512 * r + r ** 2)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.loglog(r, s_2048, color=BLUE, lw=1.8, label="2048->512 layer (r* = 410)")
    ax.loglog(r, s_512, color=ORANGE, lw=1.8, label="512x512 UVU plane (r* = 256)")
    ax.loglog(r, s_joint, color=GREEN, lw=1.8, label="shared-basis joint (r* = 212)")
    ax.axhline(1.0, color=GREY, lw=1.0, ls="--")
    for rr, c in [(64, BLUE), (128, BLUE), (256, BLUE)]:
        ax.scatter([rr], [2048 * 512 / (rr * 2560)], color=c, s=28, zorder=5)
    ax.axvspan(256, 512, color=RED, alpha=0.07)
    ax.text(320, 0.16, "loss region:\nr > r*", fontsize=7.5, color=RED)
    ax.scatter([256], [1.0], marker="x", s=45, color=RED, zorder=6)
    ax.annotate("measured E/F gate\nfailed at r = 256", (256, 1.0), xytext=(-78, -38),
                textcoords="offset points", fontsize=7.0, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.text(22, 1.08, "break-even", fontsize=7.5, color=GREY)
    ax.set_xlim(16, 512)
    ax.set_ylim(0.1, 12)
    ax.set_xlabel("rank r (shared channel basis)")
    ax.set_ylabel("speedup of the compressed layer set")
    ax.set_title("Task-aware compression: the rank must be below the break-even, not below the variance rank")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(which="both", alpha=0.18, lw=0.4)
    fig.savefig(os.path.join(OUT, "MATH_ARCH_COMPRESSION.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Fig 5: SCF execution graph and synchronization
# ----------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc, ec, fs=7.2, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       fc=fc, ec=ec, lw=0.9)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")
    return p


def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, color=color))


def barrier(ax, x, y, label=None):
    d = 0.018
    ax.add_patch(Polygon([[x, y + d], [x + d, y], [x, y - d], [x - d, y]],
                         closed=True, fc=RED, ec="white", lw=0.5, zorder=6))
    if label:
        ax.text(x, y + 0.032, label, ha="center", fontsize=6.2, color=RED)


def fig_scf_graph():
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 7 phase boxes, equal width, 6 barriers in the gaps
    w, gap = 0.115, 0.025
    x0 = 0.02
    xs = [x0 + i * (w + gap) for i in range(7)]
    y_f = 0.70
    h = 0.145
    fwd = [
        ("backbone\nforward\n(streamed)", "#dbeafe", BLUE),
        ("prologue\nd0, k-geom\ncos/sin", "#dbeafe", BLUE),
        ("SCF t=0\nEwald\nrho(k)", "#fef3c7", ORANGE),
        ("SCF t=0\nupdate\n20 UVU + MLP", "#fef3c7", ORANGE),
        ("SCF t=1\nEwald\nrho(k)", "#fef3c7", ORANGE),
        ("SCF t=1\nupdate\n20 UVU + MLP", "#fef3c7", ORANGE),
        ("LE +\nCoulomb", "#dcfce7", GREEN),
    ]
    for i, (txt, fc, ec) in enumerate(fwd):
        box(ax, xs[i], y_f, w, h, txt, fc, ec, fs=6.8)
    # labels above the 6 gaps
    for j, lab in enumerate(["B1", "B2", "B3", "B4", "B5", "B6"]):
        xb = (xs[j] + w + xs[j + 1]) / 2
        barrier(ax, xb, y_f + h / 2)
        ax.text(xb, y_f + h + 0.025, lab, ha="center", fontsize=6.5, color=RED)
    for j in range(6):
        arrow(ax, xs[j] + w, y_f + h / 2, xs[j + 1], y_f + h / 2)

    # resident state box
    box(ax, 0.10, 0.535, 0.80, 0.105,
        "resident across the SCF unit:  d (N x 8), P (N x 16), s (N x 2), "
        "cos/sin (2 x Nk x N),  Z tiles (20 x 512 x node tile)\n"
        "one grow-only arena; Z kept at N <= 5,184 (424 MB), recomputed at N >= 10,368 "
        "(20 vs 40 GEMMs)",
        "#f1f5f9", "#475569", fs=6.4)

    # reverse row: same seven phases, right to left
    y_r = 0.32
    rev = [
        ("force\nreduction", "#f3e8ff", "#7e22ce"),
        ("backbone\nreverse\n(source-owned)", "#f3e8ff", "#7e22ce"),
        ("inject\nnode adjoint", "#f3e8ff", "#7e22ce"),
        ("SCF t=0\nadjoint", "#f3e8ff", "#7e22ce"),
        ("SCF t=1\nadjoint\n(no replay)", "#f3e8ff", "#7e22ce"),
        ("LE + Coulomb\nadjoint", "#f3e8ff", "#7e22ce"),
        ("seed\nE, F", "#f3e8ff", "#7e22ce"),
    ]
    for i, (txt, fc, ec) in enumerate(rev):
        box(ax, xs[i], y_r, w, h, txt, fc, ec, fs=6.8)
    for j, lab in enumerate(["B6'", "B5'", "B4'", "B3'", "B2'", "B1'"]):
        xb = (xs[j] + w + xs[j + 1]) / 2
        barrier(ax, xb, y_r + h / 2)
        ax.text(xb, y_r + h + 0.025, lab, ha="center", fontsize=6.5, color=RED)
    for j in range(6):
        arrow(ax, xs[j + 1], y_r + h / 2, xs[j] + w, y_r + h / 2, "#7e22ce")

    ax.text(0.5, 0.955,
            "SCF fixed point as the execution unit: 6 forward and 6 adjoint phase barriers",
            ha="center", fontsize=9.2, fontweight="bold")
    ax.text(0.5, 0.20,
            "red diamonds = deterministic device-wide reductions (rho(k), Fukui, LE/Coulomb): "
            "1-2 us each, <0.02 us/atom in total",
            ha="center", fontsize=6.9, color=RED)
    ax.text(0.5, 0.13,
            "forward and adjoint run in one persistent region; the reverse is the unrolled 2-step adjoint, "
            "never an implicit fixed-point solve",
            ha="center", fontsize=6.9, color="#7e22ce")
    fig.savefig(os.path.join(OUT, "MATH_ARCH_SCF_GRAPH.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_budget()
    fig_roofline()
    fig_projection()
    fig_compression()
    fig_scf_graph()
    print("wrote figures to", OUT)
