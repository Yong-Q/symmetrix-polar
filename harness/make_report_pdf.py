#!/usr/bin/env python3
"""Render the presentation report Markdown to PDF with typeset math.

The layout follows the team reading-package conventions: blue/cyan theme,
cover, TOC with page numbers, one section per page, display equations,
figure captions, tables and bookmarks. Display math is rendered with
matplotlib mathtext (STIX) to transparent PNG and embedded, so the PDF does
not depend on a LaTeX installation.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Import the real Pillow before the vendored pytools/PIL can shadow it.
import PIL  # noqa: E402,F401
import PIL.Image  # noqa: E402,F401
PYTOOLS = REPO / "pytools"
if PYTOOLS.exists():
    sys.path.insert(0, str(PYTOOLS))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from fpdf import FPDF  # noqa: E402
from fpdf.enums import XPos, YPos  # noqa: E402

DOC = REPO / "docs/report"
MD = DOC / "SYMMETRIX_POLAR_REPORT_2026-10-21.md"
OUT = DOC / "SYMMETRIX_POLAR_REPORT_2026-10-21.pdf"
FONT = REPO / "docs/technical/fonts/NotoSansSC-Medium-static.ttf"
CACHE = DOC / "math_cache"

BLUE = (31, 111, 235)
CYAN = (14, 165, 233)
INK = (36, 41, 47)
MUTED = (60, 68, 78)
GREEN = (20, 120, 60)
RED = (180, 25, 35)
LIGHT = (246, 249, 255)
RULE = (208, 215, 222)

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
})


def latex_to_text(text: str) -> str:
    text = text.replace("\\(", "").replace("\\)", "")
    text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", text)
    repl = {
        "\\sum": "Σ", "\\sigma": "σ", "\\pi": "π", "\\rho": "ρ",
        "\\Delta": "Δ", "\\epsilon": "ε", "\\mu": "μ", "\\propto": "∝",
        "\\sqrt": "√", "\\ne": "≠", "\\times": "×", "\\le": "≤",
        "\\ge": "≥", "\\approx": "≈", "\\exp": "exp", "\\mathbf": "",
        "\\mathrm": "", "\\tilde": "", "\\hat": "", "\\,": " ",
        "\\;": " ", "\\!": "", "\\quad": "  ", "\\cdot": "·",
        "\\otimes": "⊗", "\\in": "∈", "\\mathbb": "",
    }
    for src, dst in repl.items():
        text = text.replace(src, dst)
    text = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("{", "").replace("}", "")
    return text


def clean_inline(text: str) -> str:
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"\$([^$]+)\$", lambda m: latex_to_text(m.group(1)), text)
    return latex_to_text(text).strip()


def render_math(latex: str, index: int, fontsize: int = 14) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"eq_{index:03d}.png"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=fontsize, color="#24292f")
    fig.savefig(path, dpi=420, transparent=True, bbox_inches="tight",
                pad_inches=0.03)
    plt.close(fig)
    return path


class Report(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 20, 18)
        self.add_font("NotoSansSC", "", str(FONT))
        self.set_font("NotoSansSC", "", 10)
        self.eq_no = 0
        self.fig_no = 0
        self.set_title("Symmetrix-Polar：PolarMACE 的 GPU 引擎优化")
        self.set_author("Yong-Q")
        self.set_subject("PolarMACE GPU engine optimization report")

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("NotoSansSC", "", 7.5)
        self.set_text_color(*MUTED)
        self.cell(120, 5, "Symmetrix-Polar · PolarMACE GPU 引擎 · 汇报版", align="L")
        self.cell(0, 5, "2026-10-21", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*CYAN)
        self.set_line_width(0.5)
        self.line(18, self.get_y(), 192, self.get_y())
        self.ln(2)
        self.set_text_color(*INK)
        self.set_font("NotoSansSC", "", 10)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-13)
        self.set_font("NotoSansSC", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 5, f"{self.page_no()} / {{nb}}", align="C")

    def h2(self, text, new_page=True):
        # Start a new page only when the current one is already well used;
        # this avoids near-empty pages while keeping major sections apart.
        if new_page and self.page_no() > 1 and self.get_y() > 105:
            self.add_page()
        self.start_section(text, level=0)
        self.set_font("NotoSansSC", "", 16)
        self.set_text_color(*BLUE)
        self.multi_cell(0, 9, text)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.8)
        self.line(18, self.get_y(), 192, self.get_y())
        self.ln(3)
        self.set_text_color(*INK)
        self.set_font("NotoSansSC", "", 10)

    def h3(self, text):
        self.start_section(text, level=1)
        self.ln(1)
        self.set_font("NotoSansSC", "", 11.5)
        self.set_text_color(*CYAN)
        self.multi_cell(0, 6, text)
        self.set_text_color(*INK)
        self.ln(1)
        self.set_font("NotoSansSC", "", 10)

    def para(self, text):
        text = clean_inline(text)
        if not text:
            return
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.7, text, align="L")
        self.ln(1.4)

    def bullets(self, items):
        for item in items:
            item = clean_inline(item)
            self.set_x(22)
            self.multi_cell(168, 5.7, "• " + item, align="L")
        self.ln(1.5)

    def quote(self, text):
        text = clean_inline(text)
        self.set_fill_color(*LIGHT)
        self.set_text_color(*MUTED)
        self.set_x(20)
        self.multi_cell(170, 5.4, text, fill=True, align="L")
        self.set_text_color(*INK)
        self.ln(1.5)

    def code(self, lines):
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(245, 247, 250)
        for line in lines:
            self.set_x(self.l_margin)
            self.multi_cell(0, 4.4, line if line else " ", fill=True, align="L")
        self.set_font("NotoSansSC", "", 10)
        self.ln(1.5)

    def math_block(self, latex):
        self.eq_no += 1
        path = render_math(latex, self.eq_no)
        img = PIL.Image.open(path)
        px_w, px_h = img.size
        w_mm = px_w / 420 * 25.4
        h_mm = px_h / 420 * 25.4
        max_w = 158.0
        if w_mm > max_w:
            h_mm *= max_w / w_mm
            w_mm = max_w
        if h_mm > 34:
            w_mm *= 34 / h_mm
            h_mm = 34
        if self.get_y() + h_mm + 6 > self.page_break_trigger:
            self.add_page()
        y = self.get_y() + 1
        self.image(str(path), x=(210 - w_mm) / 2, y=y, w=w_mm)
        self.set_xy(178, y + max(0, h_mm / 2 - 2.5))
        self.set_font("NotoSansSC", "", 9)
        self.set_text_color(*MUTED)
        self.cell(14, 5, f"({self.eq_no})", align="R")
        self.set_text_color(*INK)
        self.set_y(y + h_mm + 2)
        self.set_font("NotoSansSC", "", 10)

    def figure(self, path, caption):
        self.fig_no += 1
        self.set_x(self.l_margin)
        if not path.exists():
            self.para(f"[missing figure: {path.name}]")
            return
        try:
            img = PIL.Image.open(path)
            w_px, h_px = img.size
            width = 166.0
            height = width * h_px / w_px
            if height > 118:
                width *= 118 / height
            if self.get_y() + height + 12 > self.page_break_trigger:
                self.add_page()
            FPDF.image(self, str(path), x=(210 - width) / 2, w=width)
        except Exception as exc:  # noqa: BLE001
            self.para(f"[could not render {path.name}: {exc}]")
            return
        self.set_font("NotoSansSC", "", 8.4)
        self.set_text_color(*MUTED)
        self.multi_cell(0, 4.6, f"图 {self.fig_no}｜{clean_inline(caption)}",
                        align="C")
        self.set_text_color(*INK)
        self.set_font("NotoSansSC", "", 10)
        self.ln(2)

    def _row_heights(self, texts, widths):
        heights = []
        for j, txt in enumerate(texts):
            self.set_font("NotoSansSC", "", 8.6)
            try:
                lines = self.multi_cell(widths[j], 4.6, txt, border=0,
                                        align="L", dry_run=True, output="LINES")
                n = max(1, len(lines))
            except Exception:
                n = max(1, math.ceil(len(txt) / max(4, int(widths[j] / 2.1))))
            heights.append(n * 4.6 + 1.4)
        return heights

    def _row(self, texts, widths, y0, header=False):
        x0 = self.l_margin
        self.set_font("NotoSansSC", "", 8.6)
        if header:
            self.set_fill_color(*BLUE)
            self.set_text_color(255, 255, 255)
        else:
            self.set_text_color(*INK)
        h = max(self._row_heights(texts, widths))
        for j, txt in enumerate(texts):
            self.set_xy(x0 + sum(widths[:j]), y0)
            self.set_fill_color(*(BLUE if header else LIGHT))
            self.multi_cell(widths[j], 4.6, txt, border=0, align="L",
                            fill=True)
        # draw thin rules
        self.set_draw_color(*RULE)
        self.set_line_width(0.25)
        self.line(x0, y0, x0 + sum(widths), y0)
        self.line(x0, y0 + h, x0 + sum(widths), y0 + h)
        return y0 + h

    def table(self, rows):
        if not rows:
            return
        ncols = max(len(r) for r in rows)
        widths = [172 / ncols] * ncols
        if ncols >= 2:
            widths[0] = 172 * 0.32
            rest = (172 - widths[0]) / (ncols - 1)
            widths[1:] = [rest] * (ncols - 1)

        def norm(row):
            row = row + [""] * (ncols - len(row))
            return [clean_inline(c) for c in row]

        header = norm(rows[0])
        body = [norm(r) for r in rows[1:]]
        if self.get_y() + 18 > self.page_break_trigger:
            self.add_page()
        y = self._row(header, widths, self.get_y(), header=True)
        for texts in body:
            h = max(self._row_heights(texts, widths))
            if y + h > self.page_break_trigger:
                self.add_page()
                y = self._row(header, widths, self.get_y(), header=True)
            y = self._row(texts, widths, y)
        self.set_xy(self.l_margin, y + 2)
        self.set_font("NotoSansSC", "", 10)


def render_toc(pdf: Report, outline):
    pdf.set_y(26)
    pdf.set_font("NotoSansSC", "", 18)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 10, "目录", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*CYAN)
    pdf.set_line_width(0.6)
    pdf.line(18, pdf.get_y(), 60, pdf.get_y())
    pdf.ln(6)
    pdf.set_text_color(*INK)
    for section in outline:
        if section.level > 1:
            continue
        title = section.name
        dots = "." * max(3, 64 - len(title))
        pdf.set_x(18)
        pdf.set_font("NotoSansSC", "", 10.5 if section.level == 0 else 9.5)
        pdf.cell(0, 6.2, f"{title} {dots} {section.page_number}",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def cover(pdf: Report):
    pdf.add_page()
    pdf.set_fill_color(*LIGHT)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_fill_color(*BLUE)
    pdf.rect(0, 0, 210, 8, "F")
    pdf.set_fill_color(*CYAN)
    pdf.rect(0, 8, 210, 2, "F")
    pdf.set_y(38)
    pdf.set_x(20)
    pdf.set_font("NotoSansSC", "", 11)
    pdf.set_text_color(*CYAN)
    pdf.cell(0, 6, "SYMMETRIX-POLAR · 技术汇报", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    pdf.set_x(20)
    pdf.set_font("NotoSansSC", "", 27)
    pdf.set_text_color(*BLUE)
    pdf.multi_cell(170, 12, "PolarMACE 的\nGPU 引擎优化", align="L")
    pdf.ln(3)
    pdf.set_x(20)
    pdf.set_font("NotoSansSC", "", 12)
    pdf.set_text_color(*INK)
    pdf.multi_cell(170, 7, "模型理论 · 优化方法 · 实测结果 · 不足与下一步")
    pdf.ln(2)
    pdf.set_x(20)
    pdf.set_font("NotoSansSC", "", 9.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 5, "2026-10-21 · 单张 A100-80GB · FP64", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(12)
    metrics = [
        ("1,296 原子 full PolarMACE", "224.1 µs/atom", GREEN),
        ("相对 PyTorch PolarMACE", "2.43–3.0×", BLUE),
        ("线性 MACE vs 论文", "4.47–4.53 / 4.50", BLUE),
        ("距 4.5 µs/atom 目标", "49.8×", RED),
    ]
    for label, value, color in metrics:
        pdf.set_x(24)
        pdf.set_font("NotoSansSC", "", 9.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(78, 9, label)
        pdf.set_font("NotoSansSC", "", 13)
        pdf.set_text_color(*color)
        pdf.cell(0, 9, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(242)
    pdf.set_draw_color(*RULE)
    pdf.line(24, pdf.get_y(), 186, pdf.get_y())
    pdf.ln(4)
    pdf.set_x(24)
    pdf.set_font("NotoSansSC", "", 8.8)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(160, 5, "口径：所有速度均为 µs/atom，并注明原子数、精度和执行路径。\n"
                           "正确性：单点 dPE 0、max dF 1e-15；三次重复评估一致。\n"
                           "本 PDF 由 Markdown 源文件自动生成，公式为数学排版。")


def main():
    text = MD.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = text.split("\n")
    pdf = Report()
    pdf.alias_nb_pages()
    cover(pdf)
    pdf.add_page()
    pdf.insert_toc_placeholder(render_toc, pages=1)

    i = 0
    para, bullets, table_rows, code_lines = [], [], [], []
    in_code = False

    def flush_para():
        nonlocal para
        if para:
            pdf.para(" ".join(para))
            para = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            pdf.bullets(bullets)
            bullets = []

    def flush_table():
        nonlocal table_rows
        if table_rows:
            pdf.table(table_rows)
            table_rows = []

    def flush_code():
        nonlocal code_lines
        if code_lines:
            pdf.code(code_lines)
            code_lines = []

    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_para(); flush_bullets(); flush_table()
                in_code = True
                code_lines = []
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue
        if line.strip().startswith("\\[") or line.strip().startswith("$$"):
            flush_para(); flush_bullets(); flush_table()
            delim = "\\]" if line.strip().startswith("\\[") else "$$"
            body = line.strip()
            body = body[2:] if body.startswith("\\[") else body[2:]
            parts = [body] if body else []
            while delim not in (parts[-1] if parts else ""):
                i += 1
                if i >= len(lines):
                    break
                parts.append(lines[i].rstrip())
            joined = " ".join(parts)
            joined = joined.split(delim)[0].strip()
            if joined:
                pdf.math_block(joined)
            i += 1
            continue
        if not line.strip():
            flush_para(); flush_bullets(); flush_table()
            i += 1
            continue
        if line.startswith("|"):
            flush_para(); flush_bullets()
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                i += 1
                continue
            table_rows.append(cells)
            i += 1
            continue
        flush_table()
        if line.startswith("# "):
            flush_para(); flush_bullets()
            i += 1
            continue
        if line.startswith("## "):
            flush_para(); flush_bullets()
            pdf.h2(line[3:].strip())
            i += 1
            continue
        if line.startswith("### "):
            flush_para(); flush_bullets()
            pdf.h3(line[4:].strip())
            i += 1
            continue
        m = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if m:
            flush_para(); flush_bullets()
            pdf.figure(DOC / m.group(2), m.group(1))
            i += 1
            continue
        if line.startswith("- "):
            flush_para()
            bullets.append(line[2:].strip())
            i += 1
            continue
        if line.startswith("> "):
            flush_para(); flush_bullets()
            pdf.quote(line[2:].strip())
            i += 1
            continue
        if re.match(r"^\d+\.\s", line.strip()):
            flush_para(); flush_bullets()
            pdf.para(line.strip())
            i += 1
            continue
        para.append(line.strip())
        i += 1
    flush_para(); flush_bullets(); flush_table(); flush_code()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print("wrote", OUT, "pages", pdf.page_no())


if __name__ == "__main__":
    main()
