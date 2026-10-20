#!/usr/bin/env python3
"""Render the comparison report to a clean, standardized PDF.

Layout:
  1. Cover
  2. How to read + key results + report map
  3. Report body (Markdown -> HTML -> PDF)
  4. Figure index
  5. Figure gallery, grouped by theme, with numbered captions and a
     "Read:" line that states what the reader should look for.

The built-in Helvetica font is Latin-1 only; the body is English. The
Chinese reading guide stays in docs/阅读说明.md and the offline HTML.
"""
import re
from pathlib import Path

import markdown
from fpdf import FPDF
from fpdf.fonts import FontFace
from PIL import Image

PROJ = Path("/workspace/group/shared/bza/symmetrix-polar")
DOC = PROJ / "docs"
MD = DOC / "COMPARISON_REPORT_2026-10-19.md"
OUT = DOC / "COMPARISON_REPORT_2026-10-19.pdf"

BLUE = (31, 111, 235)
INK = (36, 41, 47)
MUTED = (87, 96, 106)
RULE = (210, 215, 222)
GREEN = (45, 164, 78)

# ---------------------------------------------------------------------------
# Markdown -> HTML
# ---------------------------------------------------------------------------
text = MD.read_text()
# Normalize common Unicode punctuation and drop anything outside Latin-1.
for _src, _dst in {
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
    "\u2192": "->", "\u2264": "<=", "\u2265": ">=",
    "\u00d7": "x", "\u00b1": "+/-", "\u03c1": "rho",
    "\u03a3": "Sigma", "\u03c3": "sigma", "\u03bc": "mu",
}.items():
    text = text.replace(_src, _dst)

# Resolve relative image paths. Inline images are removed from the body;
# the numbered gallery below is the canonical figure location.
text = re.sub(r"\]\((figures/[^)]+)\)", lambda m: f"]({DOC / m.group(1)})", text)
text = re.sub(r"\]\((\.\./results/[^)]+)\)", lambda m: f"]({PROJ / m.group(1)[3:]})", text)
html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])

_INLINE = re.compile(r"</?(?:strong|em|code|span|a|b|i|u|sub|sup)\b[^>]*>", re.I)


def _flatten_cell(match):
    return _INLINE.sub("", match.group(0))


html = re.sub(r"<(td|th)\b[^>]*>.*?</\1>", _flatten_cell, html, flags=re.S)
html = re.sub(r"<p>\s*<img [^>]*>\s*</p>", "", html)
html = html.replace(">阅读说明<", ">reading guide<")
html = html.replace("阅读说明.md", "%E9%98%85%E8%AF%BB%E8%AF%B4%E6%98%8E.md")

# Long paths were shortened in the source Markdown and long shell lines use
# backslash continuations, so no HTML-level line breaking is needed. Inserting
# <br> inside fpdf2 <td> cells is unsupported and would abort the render.


class PDF(FPDF):
    def header(self):
        # Keep the page top clean: fpdf2's HTML table continuation does not
        # always honor a header text band, so the report identity lives in the
        # footer instead. This avoids any header/table overlap.
        return

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, f"MatOS compute  |  {self.page_no()}", align="C")


# ---------------------------------------------------------------------------
# Figure catalog: (path, title, "Read:" line)
# ---------------------------------------------------------------------------
F = DOC / "figures"
R = PROJ / "results"

FIG_GROUPS = [
    ("Speed and scaling", [
        (F / "01_cost_structure.png",
         "Cost structure at 128 and 432 atoms",
         "Where the time goes: backbone, long-range k-space, and the polar non-k kernels."),
        (R / "polar_perf_v28_vs_v29.png",
         "v28 -> v29: P5 LAMMPS wiring + UVU rewrite",
         "The 37.9x single-rank jump that started the optimization campaign."),
        (F / "04_homog_cost.png",
         "Homogeneous MD per-step cost and 1 ns extrapolation",
         "Production cost for N2/CO2/CH4/H2O/NH3 at 216 molecules."),
        (F / "05_engine_compare.png",
         "Linear MACE vs nonlinear backbone vs full PolarMACE",
         "The 12x gap is model size/architecture, not the same-model engine overhead."),
        (F / "12_kernel_round_speed.png",
         "Kernel round 1: e3 linear packed GEMM",
         "The e3 linear kernel itself became 12.07x faster; the full model gained 1.50x (Amdahl)."),
        (F / "13_kernel_round2_speed.png",
         "Kernel round 2: 122.46 -> 56.50 ms/eval",
         "Left: full and polar-off at 128 atoms. Right: the 432-atom step. Both include energy+forces."),
        (F / "14_kernel_round2_waterfall.png",
         "Kernel round 2 waterfall: where the 65.96 ms went",
         "The two stale-object activations alone account for 38.6 ms; the rest is UVU/project kernels."),
        (F / "15_kernel_round2_families.png",
         "Kernel-family roll-up before/after round 2",
         "Polar 73.9 -> 20.9 ms, backbone 19.4 -> 6.4 ms; tensor products stay at 10.8 ms."),
    ]),
    ("Accuracy, gates, and dynamics", [
        (F / "02_rbe_tradeoff.png",
         "RBE speed-accuracy tradeoff",
         "RBE is faster but 100-400x over the 1e-4 eV/A force gate; rejected as production."),
        (R / "polar_rbe_gate_v29.png",
         "RBE force error vs batch size",
         "Every displayed batch size is above the hard force gate."),
        (F / "03_gate_margins.png",
         "Accuracy gate margins",
         "All accepted gates pass with large margin; the plot is the measured/tolerance ratio."),
        (F / "07_lammps_parity.png",
         "LAMMPS single-point force parity",
         "dE 2.9e-11 eV, max dF 2.6e-14 eV/A; the residual distribution is machine noise."),
        (F / "08_nvt_md.png",
         "NVT MD stability (50 steps)",
         "No lost atoms; Etot span 0.21 eV; temperature/pressure remain stable."),
    ]),
    ("MPI", [
        (F / "06_mpi_status.png",
         "MPI status: correctness achieved, no one-GPU speedup",
         "Exact 2-hop and H1 pass the force gate; both are slower on one shared A100."),
        (F / "17_mpi_h1_diagnosis.png",
         "MPI H1 pre-fix diagnosis: the 1042-ghost problem",
         "Historical: H1 1-rank ran 1170 nodes vs 128 (4.87x). The one-rank fold removed it: 418.64 -> 50.20 ms."),
        (F / "18_mpi_h1_schematic.png",
         "H1 forward/adjoint exchange schematic",
         "Forward fills ghost rows with H1; reverse sums ghost adjoints back to the owner."),
        (F / "19_mpi_h1_final.png",
         "MPI H1 final: one-rank fix and two-rank blocker",
         "H1 1-rank 50.20 ms vs default 50.84 (0.99x); 2-rank is blocked by six 2.0-5.4 GiB/rank workspaces (peak 80,966 MiB)."),
    ]),
    ("Kernel profile", [
        (F / "09_kernel_top_labels.png",
         "Top GPU kernels by total time (pre-round-2 profile)",
         "The starting point: polar pipeline and e3 scalar linear dominated."),
        (F / "10_kernel_stage_stack.png",
         "Host stage spans vs GPU busy",
         "How the eval's host stages map to measured GPU kernel time."),
        (F / "11_kernel_polar_delta.png",
         "Polar-specific kernel delta",
         "The kernels the polar layer adds relative to the polar-off backbone."),
        (F / "16_kernel_round2_coverage.png",
         "Remaining kernels after round 2 and their SM coverage",
         "Red = below 50% static coverage. Tensor products are 100% covered but instruction/store-bound; see the round-3 plan."),
    ]),
]

EQ_GROUP = [
    (F / "eq_ewald.png", "GTO-Ewald reciprocal-space energy",
     "The Gaussian is inside the GTO form factor; do not apply it twice."),
    (F / "eq_scf.png", "Self-consistent charge/spin density fixed point",
     "The engine iterates the Fukui-balanced update to a density tolerance."),
    (F / "eq_linear.png", "e3nn linear layer",
     "A batched matrix multiply; round 1 routed it through packed FP64 GEMM."),
    (F / "eq_tensor.png", "e3nn tensor product",
     "The Clebsch-Gordan contraction implemented by the tensor-product kernels."),
    (F / "eq_occupancy.png", "Static SM coverage and SM-weighted busy time",
     "The definition behind every occupancy number in this report; it is a static estimate."),
    (F / "eq_amdahl.png", "Amdahl speedup and ceiling",
     "Explains why a 12x kernel gain becomes a 1.5x end-to-end gain."),
    (F / "eq_comm.png", "H1 communication volume and multi-rank wall-time model",
     "Bytes per exchange are small; the 1-rank overhead is ghost compute, not communication."),
]


def section_title(pdf, text, size=16, color=BLUE, space=2):
    pdf.ln(space)
    pdf.set_font("Helvetica", "B", size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, size * 0.55, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)


def sub_title(pdf, text, size=11, color=INK):
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, size * 0.5, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.5)


def body(pdf, text, size=9.2, color=INK, line=5.0, italic=False):
    pdf.set_font("Helvetica", "I" if italic else "", size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, line, text, new_x="LMARGIN", new_y="NEXT")


def kv_rows(pdf, rows, label_w=70, size=8.8, line=4.8):
    """Two-column key/value rows with correct wrapping and row height.

    The value column is drawn first at the right x; the label is then drawn at
    the saved left x. Both start at the same y, and the cursor is advanced to
    the maximum of the two heights. This avoids the overlapping rows produced
    by fpdf2's new_y="TOP" when a value wraps.
    """
    for label, value in rows:
        y0 = pdf.get_y()
        x0 = pdf.get_x()
        pdf.set_xy(14 + label_w, y0)
        pdf.set_font("Helvetica", "", size)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(196 - (14 + label_w), line, value,
                       new_x="LEFT", new_y="NEXT")
        y_value = pdf.get_y()
        pdf.set_xy(x0, y0)
        pdf.set_font("Helvetica", "B", size)
        pdf.set_text_color(*INK)
        pdf.multi_cell(label_w, line, label,
                       new_x="LEFT", new_y="NEXT")
        y_label = pdf.get_y()
        pdf.set_y(max(y_value, y_label) + 0.8)


def bullets(pdf, items, size=9.0, line=4.8):
    for item in items:
        pdf.set_font("Helvetica", "", size)
        pdf.set_text_color(*INK)
        pdf.multi_cell(0, line, "  -  " + item, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def add_figure(pdf, number, path, title, read, kind="Figure"):
    if not path.exists():
        return
    with Image.open(path) as im:
        w_px, h_px = im.size
    aspect = h_px / w_px

    # Draw the caption first on a fresh page if the caption alone does not fit.
    if pdf.get_y() + 22 > 281:
        pdf.add_page()
    cap_y = pdf.get_y()
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5, f"{kind} {number}. {title}",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 4.4, "Read: " + read, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # Fit the image into the remaining space, reserving room for the source
    # line and the bottom margin. If it cannot fit, move the caption to a new
    # page and retry once.
    source_h = 14.0
    available = 281 - pdf.get_y() - source_h
    width_mm = 174.0
    height_mm = width_mm * aspect
    if height_mm > available:
        width_mm = max(90.0, width_mm * available / height_mm)
        height_mm = width_mm * aspect
    if height_mm > available + 0.5:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(*INK)
        pdf.multi_cell(0, 5, f"{kind} {number}. {title}",
                       new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(0, 4.4, "Read: " + read,
                       new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        available = 281 - pdf.get_y() - source_h
        width_mm = 174.0
        height_mm = width_mm * aspect
        if height_mm > available:
            width_mm = max(90.0, width_mm * available / height_mm)
            height_mm = width_mm * aspect
    y0 = pdf.get_y()
    pdf.image(str(path), x=14 + (180 - width_mm) / 2, y=y0, w=width_mm)
    pdf.set_y(y0 + height_mm + 2)
    pdf.set_font("Helvetica", "", 6.5)
    pdf.set_text_color(140, 148, 158)
    pdf.multi_cell(0, 3.6, f"Source: {path.relative_to(PROJ)}",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
pdf = PDF(format="A4")
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_margins(14, 18, 14)
pdf.set_title("Symmetrix-Polar: consolidated speed and accuracy comparison")
pdf.set_author("MatOS compute")
pdf.set_creator("MatOS")

# --- Cover -----------------------------------------------------------------
pdf.add_page()
pdf.set_font("Helvetica", "B", 26)
pdf.set_text_color(*BLUE)
pdf.ln(22)
pdf.multi_cell(0, 13, "Symmetrix-Polar", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "B", 16)
pdf.set_text_color(*INK)
pdf.multi_cell(0, 9, "Consolidated speed and accuracy comparison",
               new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)
pdf.set_font("Helvetica", "", 10.5)
pdf.set_text_color(*MUTED)
pdf.multi_cell(0, 6.5,
    "MACE-Polar (PolarMACE) on the Symmetrix-XL nonlinear Kokkos engine\n"
    "LAMMPS pair style  |  one A100-80GB  |  FP64  |  energy + forces",
    new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)
pdf.set_draw_color(*BLUE)
pdf.set_line_width(0.9)
pdf.line(14, pdf.get_y(), 196, pdf.get_y())
pdf.ln(6)
pdf.set_font("Helvetica", "", 9.2)
pdf.set_text_color(*INK)
pdf.multi_cell(0, 5.4,
    "Version: 2026-10-19 (kernel round 2 accepted)\n"
    "Model: polar_symmetrix_probe.json (MACE_Nonlinear + polar, 316 MB)\n"
    "Repositories: Yong-Q/symmetrix-polar  |  starlessinred-interns/qiuyong\n"
    "Source of truth: docs/COMPARISON_REPORT_2026-10-19.md",
    new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)
pdf.set_font("Helvetica", "B", 9.5)
pdf.set_text_color(*BLUE)
pdf.multi_cell(0, 5.2, "What is inside", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(*INK)
pdf.multi_cell(0, 5.0,
    "  -  The accepted kernel-round-2 result: 122.46 -> 56.50 ms/eval (2.17x)\n"
    "  -  Every numerical gate, with input, metric, tolerance, and command\n"
    "  -  The MPI H1 diagnosis (1042 periodic ghosts) and the honest pending state\n"
    "  -  Modular theory and performance-model formulas, with a figure index\n"
    "  -  A reproducible speed protocol and artifact map",
    new_x="LMARGIN", new_y="NEXT")
pdf.ln(7)
pdf.set_font("Helvetica", "I", 8.2)
pdf.set_text_color(*MUTED)
pdf.multi_cell(0, 4.8,
    "All numbers come from repository JSON/log artifacts. In-flight results are "
    "marked pending. The force gate (1e-4 eV/A) is a hard minimum: RBE and 1-hop "
    "ghost MPI are rejected as production alternatives.",
    new_x="LMARGIN", new_y="NEXT")

# Bottom anchor so the cover is not top-heavy.
pdf.set_y(-46)
pdf.set_draw_color(*RULE)
pdf.set_line_width(0.3)
pdf.line(14, pdf.get_y(), 196, pdf.get_y())
pdf.ln(2)
pdf.set_font("Helvetica", "", 8.2)
pdf.set_text_color(*MUTED)
pdf.multi_cell(0, 4.6,
    "Prepared by MatOS compute  |  2026-10-19  |  one A100-80GB, FP64\n"
    "Chinese reading guide: docs/reading-guide.md  |  Theory and tests: docs/THEORY_AND_METHODS.md",
    new_x="LMARGIN", new_y="NEXT")

# --- How to read + key results + report map --------------------------------
pdf.add_page()
section_title(pdf, "How to read this report", 15)
body(pdf,
     "This report separates what was measured from what is proposed. Every "
     "accepted number has a gate status and an artifact path. The body is the "
     "consolidated comparison; the modular theory and test documentation lives "
     "in docs/THEORY_AND_METHODS.md and is linked from the reading guide.")
pdf.ln(2)
sub_title(pdf, "Key results at a glance", 11)
kv_rows(pdf, [
    ("128 atoms, full PolarMACE", "56.50 ms/eval (round 1: 122.46; 2.17x; v29: 183.78; 3.25x)"),
    ("128 atoms, polar-off", "27.66 ms/eval (round 1: 40.58; 1.47x)"),
    ("432 atoms, N2 full", "126.59 ms/step (round 1: 294.39; 2.33x)"),
    ("PyTorch reference", "94.0 ms/eval -> engine is 1.66x faster"),
    ("Accuracy gates", "module 5-8, 32/32 end-to-end, LAMMPS 1-rank: all PASS"),
    ("MPI H1", "2-rank parity PASS on the pre-fix binary; 1-rank 1042-ghost defect, fix in progress"),
], label_w=66)
pdf.ln(2)
sub_title(pdf, "Report map", 11)
bullets(pdf, [
    "Section 0: summary and scope. Section 1: the model and what is compared.",
    "Section 2: verification and gate status. Section 3: speed and MPI measurements.",
    "Section 4: findings (cost structure, MH-1, RBE/1-hop rejection, one-GPU MPI).",
    "Section 5: kernel rounds 1 and 2, including the stale-object lesson and the MPI H1 diagnosis.",
    "Sections 6-7: reproduction commands and artifact index.",
    "Theory and test modules: docs/THEORY_AND_METHODS.md (formulas, occupancy, gate catalog).",
])
pdf.ln(2)
pdf.set_font("Helvetica", "I", 8.3)
pdf.set_text_color(*MUTED)
pdf.multi_cell(0, 4.6,
    "Conventions: ms/eval = one energy+force evaluation; ms/step = one MD step. "
    "SM coverage is a static estimate from grid/block/register data, not a "
    "hardware-counter measurement. A binary hash alone is not proof of the "
    "source state; every round records the trace kernel names as well.",
    new_x="LMARGIN", new_y="NEXT")

# --- Body ------------------------------------------------------------------
pdf.add_page()
pdf.set_text_color(*INK)
try:
    pdf.write_html(html, tag_styles={
        "h1": FontFace(size_pt=14, color=BLUE, emphasis="BOLD"),
        "h2": FontFace(size_pt=12, color=BLUE, emphasis="BOLD"),
        "h3": FontFace(size_pt=10.5, color=INK, emphasis="BOLD"),
        "h4": FontFace(size_pt=9.8, color=INK, emphasis="BOLD"),
        "p": FontFace(size_pt=9.0, color=INK),
        "li": FontFace(size_pt=9.0, color=INK),
        "code": FontFace(family="Courier", size_pt=7.6),
        "pre": FontFace(family="Courier", size_pt=7.0),
    })
except Exception as exc:  # noqa: BLE001 - keep the report renderable
    print("tag_styles fallback:", exc)
    pdf.write_html(html)

# --- Figure index ----------------------------------------------------------
pdf.add_page()
section_title(pdf, "Figure index", 15)
body(pdf,
     "Figures are numbered in the order they appear in the gallery. The "
     "'Read:' line under each figure states what the reader should look for; "
     "the source path is printed below each image for traceability.",
     size=8.8, line=4.7)
pdf.ln(2)
fig_no = 0
for group, items in FIG_GROUPS:
    pdf.set_font("Helvetica", "B", 8.6)
    pdf.set_text_color(*BLUE)
    pdf.multi_cell(0, 4.6, group, new_x="LMARGIN", new_y="NEXT")
    for _path, title, _read in items:
        fig_no += 1
        pdf.set_font("Helvetica", "", 8.4)
        pdf.set_text_color(*INK)
        pdf.multi_cell(0, 4.4, f"Figure {fig_no}.  {title}",
                       new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
pdf.set_font("Helvetica", "B", 8.6)
pdf.set_text_color(*BLUE)
pdf.multi_cell(0, 4.6, "Theory formulas", new_x="LMARGIN", new_y="NEXT")
for eq_no, (_path, title, _read) in enumerate(EQ_GROUP, start=1):
    pdf.set_font("Helvetica", "", 8.4)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 4.4, f"Equation {eq_no}.  {title}",
                   new_x="LMARGIN", new_y="NEXT")

# --- Gallery ---------------------------------------------------------------
pdf.add_page()
section_title(pdf, "Figure gallery", 15)
body(pdf,
     "The key diagnostic figures from docs/figures/ and results/. Each "
     "caption states what is plotted; each 'Read:' line states the intended "
     "takeaway. The source path under the image points to the exact file.",
     size=8.8, line=4.7)
fig_no = 0
for group, items in FIG_GROUPS:
    pdf.add_page()
    section_title(pdf, group, 13)
    for path, title, read in items:
        fig_no += 1
        add_figure(pdf, fig_no, path, title, read, kind="Figure")

pdf.add_page()
section_title(pdf, "Theory formulas", 13)
body(pdf,
     "The formulas behind the energy model, tensor algebra, performance model, "
     "and MPI design. Full derivations and the per-gate test catalog are in "
     "docs/THEORY_AND_METHODS.md.",
     size=8.8, line=4.7)
pdf.ln(1)
for eq_no, (path, title, read) in enumerate(EQ_GROUP, start=1):
    add_figure(pdf, eq_no, path, title, read, kind="Equation")

pdf.output(str(OUT))
print("wrote", OUT, "pages", pdf.page_no())
