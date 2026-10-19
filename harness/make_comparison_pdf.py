#!/usr/bin/env python3
"""Render the comparison report to a PDF with a cover, figures, and page numbers."""
import re
from pathlib import Path

import markdown
from fpdf import FPDF

PROJ = Path("/workspace/group/shared/bza/symmetrix-polar")
DOC = PROJ / "docs"
MD = DOC / "COMPARISON_REPORT_2026-10-19.md"
OUT = DOC / "COMPARISON_REPORT_2026-10-19.pdf"

text = MD.read_text()
# The built-in Helvetica font is Latin-1 only. Normalize common Unicode
# punctuation and drop anything outside Latin-1 so a single curly quote or
# Chinese character cannot abort the whole PDF render.
for _src, _dst in {
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
    "\u2192": "->", "\u2264": "<=", "\u2265": ">=",
    "\u00d7": "x", "\u00b1": "+/-", "\u03c1": "rho",
    "\u03a3": "Sigma", "\u03c3": "sigma", "\u03bc": "mu",
}.items():
    text = text.replace(_src, _dst)
# Resolve relative image paths to absolute paths for the HTML->PDF renderer.
text = re.sub(r"\]\((figures/[^)]+)\)", lambda m: f"]({DOC / m.group(1)})", text)
text = re.sub(r"\]\((\.\./results/[^)]+)\)", lambda m: f"]({PROJ / m.group(1)[3:]})", text)
html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])

# fpdf2 supports only one text chunk per table cell, so inline emphasis inside
# <td>/<th> raises "Unsupported nested HTML tags". Strip inline tags there and
# keep the plain cell text; the figures are shown in the gallery below.
_INLINE = re.compile(r"</?(?:strong|em|code|span|a|b|i|u|sub|sup)\b[^>]*>", re.I)


def _flatten_cell(match):
    return _INLINE.sub("", match.group(0))


html = re.sub(r"<(td|th)\b[^>]*>.*?</\1>", _flatten_cell, html, flags=re.S)
# Drop inline body images; the figure gallery below carries every figure.
html = re.sub(r"<p>\s*<img [^>]*>\s*</p>", "", html)

# The body uses the built-in Helvetica font, which has no CJK glyphs. Show the
# reading-guide link with an ASCII label and percent-encode its file name; the
# Markdown and HTML versions keep the original Chinese link text.
html = html.replace(">阅读说明<", ">reading guide<")
html = html.replace("阅读说明.md", "%E9%98%85%E8%AF%BB%E8%AF%B4%E6%98%8E.md")


class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110, 118, 129)
        self.cell(0, 6, "Symmetrix-Polar: consolidated speed and accuracy comparison", align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(7)
        self.set_draw_color(210, 215, 222)
        self.line(10, 15, 200, 15)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(110, 118, 129)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


pdf = PDF(format="A4")
pdf.set_auto_page_break(auto=True, margin=16)
pdf.set_margins(14, 14, 14)
pdf.set_title("Symmetrix-Polar: consolidated speed and accuracy comparison")
pdf.set_author("MatOS compute")
pdf.set_creator("MatOS")

# Cover page.
pdf.add_page()
pdf.set_font("Helvetica", "B", 24)
pdf.set_text_color(31, 111, 235)
pdf.ln(35)
pdf.multi_cell(0, 12, "Symmetrix-Polar", align="L", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "B", 16)
pdf.set_text_color(36, 41, 47)
pdf.multi_cell(0, 9, "Consolidated speed and accuracy comparison", align="L", new_x="LMARGIN", new_y="NEXT")
pdf.ln(4)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(87, 96, 106)
pdf.multi_cell(0, 7,
    "MACE-Polar (PolarMACE) on the Symmetrix-XL nonlinear Kokkos engine\n"
    "LAMMPS pair style, single A100, f64\n\n"
    "Version: 2026-10-19\n"
    "Model: polar_symmetrix_probe.json (MACE_Nonlinear + polar)\n"
    "Repositories: Yong-Q/symmetrix-polar · starlessinred-interns/qiuyong", align="L", new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)
pdf.set_draw_color(31, 111, 235)
pdf.set_line_width(0.8)
pdf.line(14, pdf.get_y(), 80, pdf.get_y())
pdf.ln(8)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(36, 41, 47)
pdf.multi_cell(0, 6,
    "This document is generated from docs/COMPARISON_REPORT_2026-10-19.md. "
    "All numbers come from the repository JSON/log artifacts; in-flight results "
    "are marked pending. The force gate (1e-4 eV/Å) is treated as a hard minimum: "
    "RBE and 1-hop ghost MPI are rejected as production alternatives.", align="L", new_x="LMARGIN", new_y="NEXT")

# Report body.
pdf.add_page()
pdf.set_text_color(36, 41, 47)
# Last-resort Latin-1 guard: the built-in font cannot encode arbitrary Unicode.
html = html.encode("latin-1", "replace").decode("latin-1")
pdf.write_html(html)

# Figure appendix, so the key figures are guaranteed to appear even if the
# HTML renderer skips a relative image.
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.set_text_color(31, 111, 235)
pdf.cell(0, 10, "Figure gallery", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(87, 96, 106)
pdf.multi_cell(0, 5,
    "The key diagnostic figures from docs/figures/. Full captions are in the "
    "report body and in the reading guide.", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
figs = [
    (DOC / "figures" / "01_cost_structure.png",
     "Where the time goes: backbone vs long-range vs polar kernels."),
    (PROJ / "results" / "polar_perf_v28_vs_v29.png",
     "v28 vs v29: P5 LAMMPS wiring and the 37.8x single-rank speedup."),
    (DOC / "figures" / "02_rbe_tradeoff.png",
     "RBE is faster but fails the 1e-4 eV/Å force gate."),
    (PROJ / "results" / "polar_rbe_gate_v29.png",
     "RBE force error vs batch size against the hard force gate."),
    (DOC / "figures" / "03_gate_margins.png",
     "All accuracy gates pass with large margin."),
    (DOC / "figures" / "04_homog_cost.png",
     "Homogeneous MD per-step cost and PNAS-length 1 ns extrapolation."),
    (DOC / "figures" / "05_engine_compare.png",
     "Engine/model cost: linear MACE vs nonlinear backbone vs full PolarMACE."),
    (DOC / "figures" / "06_mpi_status.png",
     "MPI on one shared A100: correctness achieved, no speedup yet."),
    (DOC / "figures" / "07_lammps_parity.png",
     "LAMMPS single-point force parity and residual distribution."),
    (DOC / "figures" / "08_nvt_md.png",
     "NVT MD temperature and energy stability."),
    (DOC / "figures" / "09_kernel_top_labels.png",
     "Top GPU kernels by total time in one full PolarMACE eval."),
    (DOC / "figures" / "10_kernel_stage_stack.png",
     "Host stage spans vs measured GPU busy time."),
    (DOC / "figures" / "11_kernel_polar_delta.png",
     "Kernel cost added by the polar blocks relative to the backbone."),
    (DOC / "figures" / "12_kernel_round_speed.png",
     "Kernel round 1: same-model before/after at 128 and 432 atoms."),
    (DOC / "figures" / "13_kernel_round2_speed.png",
     "Kernel round 2: 122.46 -> 56.50 ms/eval, polar-off 40.58 -> 27.66 ms."),
    (DOC / "figures" / "14_kernel_round2_waterfall.png",
     "Where the 65.96 ms went in round 2, including the stale-object activation."),
    (DOC / "figures" / "15_kernel_round2_families.png",
     "Kernel-family before/after: polar 73.9 -> 20.9 ms, backbone 19.4 -> 6.4 ms."),
    (DOC / "figures" / "16_kernel_round2_coverage.png",
     "Remaining top kernels after round 2; red marks SM coverage below 50%."),
    (DOC / "figures" / "17_mpi_h1_diagnosis.png",
     "MPI H1 diagnosis: 1042 periodic ghosts make H1 1-rank 4.87x the production path."),
    (DOC / "figures" / "18_mpi_h1_schematic.png",
     "H1 forward feature exchange and reverse adjoint exchange between two ranks."),
    (DOC / "figures" / "eq_ewald.png",
     "Formula: GTO-Ewald reciprocal-space long-range energy."),
    (DOC / "figures" / "eq_scf.png",
     "Formula: self-consistent charge/spin density fixed point."),
    (DOC / "figures" / "eq_tensor.png",
     "Formula: e3nn tensor product with Clebsch-Gordan coefficients."),
    (DOC / "figures" / "eq_occupancy.png",
     "Formula: static SM coverage and SM-weighted busy time."),
    (DOC / "figures" / "eq_amdahl.png",
     "Formula: Amdahl speedup and the ceiling for a partially optimized kernel."),
    (DOC / "figures" / "eq_comm.png",
     "Formula: H1 communication volume and the multi-rank wall-time model."),
]
from PIL import Image

for i, (path, caption) in enumerate(figs):
    if not path.exists():
        continue
    with Image.open(path) as im:
        width_px, height_px = im.size
    height_mm = 174.0 * height_px / width_px
    if pdf.get_y() + height_mm + 14 > 281:
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(36, 41, 47)
    pdf.multi_cell(0, 5, f"Figure {i+1}. {caption}", new_x="LMARGIN", new_y="NEXT")
    y0 = pdf.get_y()
    pdf.image(str(path), x=18, y=y0, w=174)
    pdf.set_y(y0 + height_mm + 4)

pdf.output(str(OUT))
print("wrote", OUT, "pages", pdf.page_no())
