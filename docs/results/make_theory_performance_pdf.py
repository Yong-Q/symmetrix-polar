#!/usr/bin/env python3
"""Build the distillation theory + performance PDF (PNAS-style figures).

Pages: cover, architecture, accuracy, speed, stability/data, methods,
decisions.  Figures come from distill_figures.py so the README and the PDF
never diverge.
"""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np

from distill_figures import C, accuracy_fig, architecture_fig, speed_fig

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "data/full_valid_head_ewald.json").read_text())
VF = HERE.parent.parent / "bench/plan_d_bulk_2026-10-25/results/valid_by_species.json"
VALID = json.loads(VF.read_text()) if VF.exists() else {}
OUT = HERE / "DISTILLATION_THEORY_PERFORMANCE_2026-10-27.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 300,
})
PAGE = (8.5, 11)


def text_page(pdf, title, paragraphs, fontsize=8.5):
    fig = plt.figure(figsize=PAGE)
    y = 0.95
    fig.text(0.07, y, title, fontsize=14, fontweight="bold", ha="left")
    y -= 0.045
    for para in paragraphs:
        if para.startswith("## "):
            y -= 0.008
            fig.text(0.07, y, para[3:], fontsize=10, fontweight="bold", ha="left")
            y -= 0.022
            continue
        for line in textwrap.wrap(para, width=108):
            if y < 0.055:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=PAGE)
                y = 0.95
            fig.text(0.07, y, line, fontsize=fontsize, ha="left")
            y -= 0.0165
        y -= 0.012
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def figure_page(pdf, fig, title, caption):
    """Place a wide figure on a letter page, keeping a uniform page size."""
    fig.set_dpi(300)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    page = plt.figure(figsize=PAGE)
    page.text(0.075, 0.915, title, fontsize=13, fontweight="bold", ha="left")
    aspect = img.shape[1] / img.shape[0]
    w = 0.86 * PAGE[0]
    h = w / aspect
    top = 0.885
    ax = page.add_axes([0.07, top - h / PAGE[1], 0.86, h / PAGE[1]])
    ax.imshow(img)
    ax.axis("off")
    y = top - h / PAGE[1] - 0.035
    for para in caption:
        for line in textwrap.wrap(para, width=112):
            page.text(0.07, y, line, fontsize=8.3, ha="left")
            y -= 0.017
        y -= 0.008
    pdf.savefig(page, bbox_inches="tight")
    plt.close(page)


# ------------------------------------------------------------------ page 1
def cover_page(pdf):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.07, 0.955, "Distilling MACE-Polar into a fast linear student",
             fontsize=18, fontweight="bold")
    fig.text(0.07, 0.925,
             "Architecture, six-molecule accuracy, and speed of the "
             "charge/dipole distillation",
             fontsize=10.5, style="italic")
    fig.text(0.07, 0.905, "27 October 2026  ·  Symmetrix-Polar / Plan D",
             fontsize=8.5, color=C["grey"])
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.893, 0.893],
                              color=C["grey"], lw=1.0))
    fig.text(0.07, 0.858, "KEY NUMBERS", fontsize=7.5, color=C["grey"],
             fontweight="bold")

    cards = [
        ("E / F, 3,465 held-out frames", "1.2 / 7.8",
         "meV atom$^{-1}$  /  meV A$^{-1}$"),
        ("Six-molecule gas", "1.17 / 1.00",
         "meV atom$^{-1}$  /  meV A$^{-1}$"),
        ("Atomic charge q", "0.0117 e", "867 frames · 4$\\pi$ scale"),
        ("l=1 dipole $\\mu$", "0.0475", "e A · 4$\\pi$ scale"),
        ("GTO/Ewald from $q,\\mu$", "0.311", "meV atom$^{-1}$ overall"),
        ("Speed, 5,184 atoms", "4.47 / ~20.2",
         "backbone-only FP32 / full FP64 (proj.)"),
    ]
    x0, y0, w, h = 0.07, 0.725, 0.27, 0.105
    dx, dy = 0.295, 0.122
    for i, (lab, val, unit) in enumerate(cards):
        r, c = divmod(i, 3)
        x = x0 + c * dx
        y = y0 - r * dy
        fig.add_artist(FancyBboxPatch((x, y), w, h,
                                      boxstyle="round,pad=0.008,rounding_size=0.012",
                                      transform=fig.transFigure,
                                      fc="#EAF2F8", ec=C["grey"], lw=0.7))
        fig.text(x + 0.012, y + h - 0.022, lab, fontsize=7.4, color=C["grey"])
        fig.text(x + 0.012, y + 0.036, val, fontsize=15, fontweight="bold",
                 color=C["blue"])
        fig.text(x + 0.012, y + 0.014, unit, fontsize=7.0, color=C["grey"])

    body = [
        "A 128-channel linear MACE backbone distilled from the MACE-Polar-1-M "
        "teacher reaches 1.2 meV/atom and 7.8 meV/A on the full 23,376-frame "
        "corpus (3,465 held-out).  A charge MLP and a linear l=1 multipole "
        "readout, trained on 6,809 frames covering all 1,100 states, predict "
        "q to 0.0117 e and mu to 0.0475 (4pi scale).  Feeding the predicted "
        "multipoles through the teacher's GTO/Ewald kernel reproduces the "
        "teacher electrostatic energy to 0.311 meV/atom overall.",
        "The student is stable in NVE (0.245 meV/atom/ps over 1 ps), its "
        "forces rotate to 4.0e-5, and a canonical-frame transform makes the "
        "full linear head rotation-consistent to 4.6e-14 meV/atom (0.825 "
        "without it).  The projected speed is ~20.2 us/atom/step: 12.3 engine "
        "backbone + 7.9 PyTorch Ewald + <0.1 heads, with a 1.2x k-space "
        "cutoff giving <=20 us.  An end-to-end engine hybrid pair style is "
        "the remaining implementation.",
    ]
    y = 0.555
    for para in body:
        for line in textwrap.wrap(para, width=100):
            fig.text(0.07, y, line, fontsize=8.6, ha="left")
            y -= 0.0175
        y -= 0.014
    fig.text(0.07, 0.10,
             "Scope.  The q/mu readout is trained on 6,809 frames, not the "
             "full 23K; the remaining frames are mostly correlated frames "
             "inside the same 1,100 states.",
             fontsize=7.8, color=C["grey"])
    fig.text(0.07, 0.076,
             "CO2/H2O/NH3 retain 0.4-0.6 meV/atom Ewald error; 0.1 meV/atom is "
             "not claimed.  The ~20 us speed is a projection, not a measured "
             "end-to-end engine run.",
             fontsize=7.8, color=C["grey"])
    fig.text(0.07, 0.035, "Symmetrix-Polar  ·  MatOS", fontsize=7.5,
             color=C["grey"])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


with PdfPages(OUT) as pdf:
    cover_page(pdf)

    figure_page(
        pdf, architecture_fig(), "1 · Distillation architecture", [
            "The teacher (512-channel nonlinear polar MACE, l=1 multipoles, "
            "2-step SCF) labels a 23,376-frame E/F corpus and a 6,809-frame "
            "q/mu corpus over 1,100 states.  The student backbone is a standard "
            "128-channel linear MACE (2 interactions, r_c = 5 A, 848,528 "
            "parameters).  Charges come from a 0e MLP; the l=1 multipoles come "
            "from a full linear 384->3 map in the canonical frame.  The GTO/"
            "Ewald kernel is the teacher's exact long-range operator.",
        ])

    figure_page(
        pdf, accuracy_fig(), "2 · Accuracy on the six molecules", [
            "A/B: energy and force MAE of the distilled backbone on the 3,465 "
            "held-out frames, split into gas and dense/bulk states for each "
            "molecule.  C: charge and l=1 dipole MAE (4pi scale) on the 867 "
            "frames that carry teacher multipoles; dotted lines are the "
            "all-frame values.  D: electrostatic energy computed from the "
            "predicted q and mu with the teacher GTO/Ewald kernel, compared "
            "with the same kernel on teacher coefficients.  CO2, H2O, and NH3 "
            "carry the residual (purple bars); the representation, not the "
            "frame count, is the limit.  CO2's larger absolute force error "
            "tracks its ~2x larger reference forces (1.34 vs 0.65-0.76 eV/A): "
            "its relative error is 1.8%, similar to H2O (1.7%) and NH3 (1.3%).",
        ])

    figure_page(
        pdf, speed_fig(), "3 · Speed and the speed-accuracy map", [
            "All measurements are us/atom/step at 5,184 atoms on one A100.  "
            "The FP32 direct/JIT engine path runs the bare backbone at 4.47 "
            "(no q/mu heads, no Ewald), matching the paper's linear "
            "MACE-OFF23 range; the same backbone in FP64 is 12.3.  The full "
            "package adds the q/mu heads and the exact GTO/Ewald kernel and is "
            "projected to land at ~20.2 in FP64 (hatched), because the engine "
            "hybrid pair style is not yet implemented.  The 4.47 and the ~20.2 "
            "are therefore different scopes, not two measurements of the same "
            "quantity.  The 128L1 "
            "nonlinear polar student (95.1) and the exact teacher (129.0) "
            "bracket the current exact-PolarMACE cost.",
        ])

    # ---------------- page 5: stability + data coverage ----------------
    fig = plt.figure(figsize=(8.3, 6.4))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    labels = ["teacher\nforce rot.", "student\nforce rot.", "head raw\nrotation",
              "canonical\nrotation"]
    vals = [1.5e-8, 4.0e-5, 0.825, 4.6e-14]
    cols = [C["green"], C["blue"], C["orange"], C["purple"]]
    ax.bar(np.arange(4), [max(v, 1e-16) for v in vals], color=cols, width=0.65)
    ax.set_yscale("log"); ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("rotation error (mixed units)")
    ax.set_title("Rotation consistency")
    for i, v in enumerate(vals):
        ax.text(i, max(v, 1e-16) * 2.2, f"{v:.1e}", ha="center", fontsize=6.2)

    ax = fig.add_subplot(gs[0, 1])
    labels = ["E/F train", "E/F valid", "q/mu train", "q/mu valid", "states"]
    vals = [23376, 3465, 6809, 867, 1100]
    ax.bar(np.arange(5), vals, color=[C["blue"], C["sky"], C["green"],
                                      C["yellow"], C["grey"]], width=0.65)
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(labels, rotation=25,
                                                    ha="right", fontsize=6.5)
    ax.set_ylabel("frames / states")
    ax.set_title("Data coverage")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:,}", ha="center", fontsize=6.2)

    ax = fig.add_subplot(gs[1, :])
    names = ["E/F\nbackbone", "q head", "mu head", "Ewald", "projected\ntotal"]
    us = [12.3, 0.03, 0.02, 7.9, 20.2]
    ax.barh(np.arange(len(names)), us,
            color=[C["blue"], C["orange"], C["green"], C["sky"], C["grey"]],
            height=0.6)
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=6.8)
    ax.set_xlabel("us atom$^{-1}$ step$^{-1}$ (FP64, one A100)")
    ax.set_title("Speed budget breakdown")
    for i, v in enumerate(us):
        ax.text(v + 0.2, i, f"{v:.2f}", va="center", fontsize=6.6)
    ax.set_xlim(0, 24)
    ax.invert_yaxis()
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.16, axis="y", lw=0.6)
    fig.tight_layout()
    figure_page(pdf, fig, "4 · Stability, data coverage, and the speed budget", [
        "The student force field rotates to 4.0e-5 and the NVE drift is "
        "0.245 meV/atom/ps over 1 ps.  The raw full linear head is not "
        "rotation-consistent (0.825 meV/atom); the canonical-frame transform "
        "fixes it to 4.6e-14 without retraining.",
        "Data coverage: 23,376 / 3,465 E/F train/valid frames and 6,809 / 867 "
        "q/mu frames over 1,100 states.  The ACF analysis gives ~4,450 "
        "effective independent samples, so frame counts are not "
        "independent-sample counts.",
        "The speed budget sums the measured engine backbone (12.3 FP64), the "
        "measured PyTorch Ewald kernel (7.9 at 1,152 atoms) and the measured "
        "heads (<0.1); the 20.2 total is a projection, not an end-to-end "
        "engine measurement.",
    ])

    # ---------------- page 6: theory/methods ----------------
    text_page(pdf, "Theory and methods", [
        "## Teacher and data",
        "The teacher is MACE-Polar-1-M: a 512-channel nonlinear polar MACE with "
        "l=1 atomic multipoles, exact GTO/Ewald long-range electrostatics, and "
        "two charge-density recursion steps.  The distillation corpus contains "
        "23,376 training and 3,465 validation frames of six molecules (H2O, "
        "NH3, CH4, C2H4, C2H6, CO2) across 1,100/176 states, with a state-level "
        "split.  The ACF analysis gives about 4,450 effective independent "
        "samples for the full 26,841-frame corpus; the 23K frame count is not a "
        "23K independent-sample count.",
        "## Backbone",
        "The student is a standard linear MACE: 128 channels, two interactions, "
        "l_max=3, r_cut=5.0 A, 848,528 parameters.  It is trained on the "
        "teacher's total energy and forces (REF_energy/REF_forces) for 60 "
        "epochs, force weight 20, Adam with ReduceLROnPlateau.  The model is "
        "exported to the Symmetrix JSON contract and runs on the FP32 "
        "direct/JIT engine path at 4.47 us/atom/step (5,184 atoms); the FP64 "
        "materialized reference is 12.3 us.",
        "## Charge and multipole readout",
        "The 6,809-frame q/mu set is the stride-4 train set plus a 965-frame "
        "state supplement, covering all 1,100 states.  The 0e and 1o node "
        "features are extracted at the energy readout.  A two-layer MLP maps "
        "0e features to per-atom charge; a full linear map (384->3) maps the "
        "1o features to the teacher's three l=1 multipole coefficients.  The "
        "linear map is not a per-channel equivariant map: the teacher's "
        "coefficients come from the nonlinear charge-density head and require "
        "channel/basis mixing.  The physical energy and forces remain "
        "equivariant because the GTO/Ewald kernel is covariant.",
        "## Canonical frame",
        "The full linear head is orientation-dependent in the raw frame.  A "
        "3x3 cell QR rotates every frame to a canonical (upper-triangular) "
        "orientation before the head and the Ewald kernel, and the outputs are "
        "rotated back.  The transform costs one 3x3 QR per frame and restores "
        "rotation consistency from 0.825 meV/atom to 4.6e-14 meV/atom.  The "
        "current orthorhombic training cells are already canonical, so no "
        "retraining is required; triclinic cells would need canonicalized "
        "retraining.",
        "## Ewald diagnostic",
        "The predicted q and mu are fed to the same graph_longrange "
        "GTOElectrostaticEnergy kernel used by the teacher (l=1, 1.5 A "
        "smearing, k-space factor 1.5, self and PBC corrections).  On 867 "
        "held-out frames the predicted-coefficient electrostatic energy "
        "matches the teacher-coefficient energy to 0.311 meV/atom overall.",
        "## Important bookkeeping",
        "The MACE backbone was trained on the teacher's total energy, which "
        "already contains the long-range term.  The q/mu head and E_elec are "
        "output/diagnostic quantities; E_elec must not be added to the MACE "
        "total energy unless the backbone is retrained on the short-range "
        "residual.  This distinction is what makes the current package "
        "physically consistent without double counting.",
    ])

    # ---------------- page 7: improvements and caveats ----------------
    text_page(pdf, "Design decisions, improvements, and open items", [
        "## Decisions",
        "1. The 4G-HDNNP accuracy line was closed: wide+ee4G remains at 31.0 "
        "meV/atom and 75.1 meV/A (6.2x/2.5x over the 5/30 gate), active "
        "learning moves it by ~1%, and the overfit probe floors the force "
        "error near 60 meV/A.  The missing l=1 multipoles and thin angular "
        "basis, not the data volume, are the limit.",
        "2. The linear MACE distillation is the E/F main line: 1.2/7.8 on the "
        "full 23K corpus, 1.17/1.00 on the six-molecule gas set, and 4.47 "
        "us/atom FP32 on the engine.",
        "3. The q/mu head is post-hoc on frozen features.  The strict "
        "per-channel equivariant linear head is limited to ~0.227; the full "
        "linear map reaches 0.045 but is not rotation-consistent.  The "
        "canonical frame resolves the rotation issue without a tensor product "
        "or a slower equivariant network.",
        "4. Tensor-product tests: a single bilinear 1o x 1o layer is exactly "
        "equivariant (3.6e-6) but only reaches mu MAE 1.004; 2e x 1o reaches "
        "0.905 at 1.5 us/atom; 0e+1o+2e reaches 0.905 at 3.0 us/atom.  A deep "
        "equivariant network would be needed for 0.045 but would exceed the "
        "20 us budget.  Ewald is the preferred long-range backend: 7.9 us/atom "
        "median vs 9.3 for PME at 1,152 atoms, and it is the teacher's exact "
        "kernel.",
        "## Open items",
        "- The 20.2 us total is a projection (engine backbone 12.3 + PyTorch "
        "Ewald 7.9 + heads); the Symmetrix standard MACE pair style has no "
        "charge/dipole output.  An engine hybrid (standard MACE + q/mu readout "
        "+ Ewald) is the remaining implementation.",
        "- q/mu labels cover 6,809 frames, not 23K.  The remaining 16,567 "
        "frames are mostly correlated frames inside the same 1,100 states; "
        "labeling them is optional for accuracy and mainly a consistency "
        "choice.",
        "- CO2/H2O/NH3 retain 0.4-0.6 meV/atom Ewald error; 0.1 meV/atom is "
        "not reached.  This is a representation (multipole/local-basis) limit, "
        "not a frame-count limit.",
        "- The site-field (JCTC-style four-sine scalar potential) and "
        "inhomogeneous-density tranches are being sampled separately; the "
        "MACE-Polar teacher remains the label source.  Electric-field response "
        "requires the MACE-Polar external-field block, not MACE-OFF23.",
        "## Reproducibility",
        "All scripts and JSON diagnostics are in the repository: "
        "scripts/charge_dipole_student.py, eval_full_valid.py, "
        "test_package_forces_nve.py, and docs/results/data/.  Model hashes are "
        "recorded in the 500 ps audit package and in the Plan D log.  "
        "Figures are generated by docs/results/distill_figures.py.",
    ])

print("wrote", OUT)
