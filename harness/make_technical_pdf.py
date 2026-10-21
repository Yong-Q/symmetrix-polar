#!/usr/bin/env python3
"""Render docs/technical/TECHNICAL_REPORT.md to an extendable PDF.

The report source is Markdown + PNG figures, so adding a section or a new
result means editing the Markdown (or adding a figure) and re-running this
script.  Chinese text uses Noto Sans SC; tables, code blocks, blockquotes and
images are rendered explicitly.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTOOLS = REPO / "pytools"
# The vendored pytools/PIL is an incomplete copy (no _imaging extension) and
# would shadow the installed Pillow once PYTOOLS is on sys.path. Import the
# real Pillow first so fpdf2 can embed figures.
import PIL  # noqa: E402,F401
import PIL.Image  # noqa: E402,F401
if PYTOOLS.exists():
    sys.path.insert(0, str(PYTOOLS))

from fpdf import FPDF  # noqa: E402
from fpdf.enums import XPos, YPos  # noqa: E402

DOC = REPO / "docs/technical"
MD = DOC / "TECHNICAL_REPORT.md"
# The downloaded Noto Sans SC is a variable font whose default weight is 100
# (Thin); at 10 pt it looks washed out. Instantiate a static Medium (wght=500)
# and use that for all text. Both files are local (docs/technical/fonts/ is
# gitignored and recreated by ensure_font()).
FONT_VF = DOC / "fonts/NotoSansSC-Regular.ttf"
FONT = DOC / "fonts/NotoSansSC-Medium-static.ttf"
OUT = DOC / "SYMMETRIX_POLAR_TECHNICAL_REPORT.pdf"
FONT_URL = ("https://gh-proxy.com/https://raw.githubusercontent.com/"
            "notofonts/noto-cjk/main/Sans/Variable/TTF/Subset/"
            "NotoSansSC-VF.ttf")


def ensure_font() -> None:
    """Download the variable font if needed and instantiate a static Medium."""
    FONT.parent.mkdir(parents=True, exist_ok=True)
    if not FONT_VF.exists():
        import urllib.request
        urllib.request.urlretrieve(FONT_URL, FONT_VF)
    if FONT.exists():
        return
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont
        variable = TTFont(FONT_VF)
        instantiateVariableFont(variable, {"wght": 500}).save(FONT)
    except Exception as exc:  # noqa: BLE001 - keep the report renderable
        print("warning: could not instantiate Medium font:", exc)
        FONT.write_bytes(FONT_VF.read_bytes())

BLUE = (31, 111, 235)
INK = (36, 41, 47)
MUTED = (60, 68, 78)
GREEN = (20, 120, 60)
RED = (180, 25, 35)
LIGHT = (246, 248, 250)
RULE = (208, 215, 222)


def latex_to_text(text: str) -> str:
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", text)
    repl = {
        "\\sum": "Σ", "\\sigma": "σ", "\\pi": "π", "\\rho": "ρ",
        "\\rho": "ρ", "\\Delta": "Δ", "\\epsilon": "ε", "\\mu": "μ",
        "\\propto": "∝", "\\sqrt": "√", "\\ne": "≠", "\\times": "×",
        "\\le": "≤", "\\ge": "≥", "\\approx": "≈", "\\exp": "exp",
        "\\mathbf": "", "\\mathrm": "", "\\tilde": "", "\\hat": "",
        "\\,": " ", "\\;": " ", "\\!": "", "\\quad": "  ",
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
    return latex_to_text(text).strip()


class Report(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)
        self.add_font("NotoSansSC", "", str(FONT))
        self.set_font("NotoSansSC", "", 10)
        self.section_title = ""

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("NotoSansSC", "", 7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 5, "Symmetrix-Polar · 技术报告 · round 3 + Plan B-lite", align="L")
        self.ln(5)
        self.set_draw_color(*RULE)
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
        self.cell(0, 5, f"{self.page_no()}", align="C")

    def h1(self, text):
        self.set_font("NotoSansSC", "", 18)
        self.set_text_color(*BLUE)
        self.multi_cell(0, 9, text)
        self.set_text_color(*INK)
        self.ln(2)

    def h2(self, text, new_page=False):
        if new_page and self.page_no() > 1 and self.get_y() > 30:
            self.add_page()
        self.start_section(text, level=0)
        self.set_font("NotoSansSC", "", 14)
        self.set_text_color(*BLUE)
        self.multi_cell(0, 7, text)
        self.set_draw_color(*BLUE)
        self.line(18, self.get_y(), 192, self.get_y())
        self.ln(3)
        self.set_text_color(*INK)
        self.set_font("NotoSansSC", "", 10)

    def h3(self, text):
        self.start_section(text, level=1)
        self.ln(1)
        self.set_font("NotoSansSC", "", 11.5)
        self.set_text_color(*INK)
        self.multi_cell(0, 6, text)
        self.ln(1)
        self.set_font("NotoSansSC", "", 10)

    def para(self, text):
        text = clean_inline(text)
        if not text:
            return
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.6, text, align="L")
        self.ln(1.2)

    def bullets(self, items):
        for item in items:
            item = clean_inline(item)
            self.set_x(22)
            self.multi_cell(170, 5.6, "• " + item, align="L")
        self.ln(1.5)

    def quote(self, text):
        text = clean_inline(text)
        self.set_fill_color(*LIGHT)
        self.set_text_color(*MUTED)
        self.set_x(22)
        self.multi_cell(168, 5.4, text, fill=True, align="L")
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

    def image(self, path, caption):
        self.set_x(self.l_margin)
        if not path.exists():
            self.para(f"[missing figure: {path.name}]")
            return
        width = 165
        try:
            FPDF.image(self, str(path), x=22, w=width)
        except Exception as exc:
            self.para(f"[could not render {path.name}: {exc}]")
            return
        self.set_font("NotoSansSC", "", 8.5)
        self.set_text_color(*MUTED)
        self.multi_cell(0, 4.6, caption)
        self.set_text_color(*INK)
        self.set_font("NotoSansSC", "", 10)
        self.ln(2)

    def _table_heights(self, texts, widths):
        """Measure the real rendered height of a row (Chinese + English)."""
        heights = []
        for j, txt in enumerate(texts):
            self.set_font("NotoSansSC", "", 8.6)
            try:
                lines = self.multi_cell(
                    widths[j], 4.6, txt, border=0, align="L",
                    dry_run=True, output="LINES")
                n = max(1, len(lines))
            except Exception:
                n = max(1, math.ceil(len(txt) / max(4, int(widths[j] / 2.1))))
            heights.append(n * 4.6 + 1.2)
        return heights

    def _draw_table_row(self, texts, widths, y0):
        x0 = self.l_margin
        self.set_font("NotoSansSC", "", 8.6)
        self.set_text_color(*INK)
        for j, txt in enumerate(texts):
            self.set_xy(x0 + sum(widths[:j]), y0)
            self.multi_cell(widths[j], 4.6, txt, border=1, align="L")
        return y0 + max(self._table_heights(texts, widths))

    def table(self, rows):
        if not rows:
            return
        self.set_x(self.l_margin)
        ncols = max(len(r) for r in rows)
        widths = [172 / ncols] * ncols
        # widen the first column slightly for labels
        if ncols >= 2:
            widths[0] = 172 * 0.38
            rest = (172 - widths[0]) / (ncols - 1)
            widths[1:] = [rest] * (ncols - 1)

        def normalize(row):
            row = row + [""] * (ncols - len(row))
            return [clean_inline(c) for c in row]

        header = normalize(rows[0])
        header_h = max(self._table_heights(header, widths))
        body = [normalize(r) for r in rows[1:]]
        first_h = max(self._table_heights(body[0], widths)) if body else 0
        if self.get_y() + header_h + first_h > self.page_break_trigger:
            self.add_page()
        y = self._draw_table_row(header, widths, self.get_y())
        for texts in body:
            row_h = max(self._table_heights(texts, widths))
            if y + row_h > self.page_break_trigger:
                self.add_page()
                y = self._draw_table_row(header, widths, self.get_y())
            y = self._draw_table_row(texts, widths, y)
        self.set_xy(self.l_margin, y)
        self.ln(2)
        self.set_font("NotoSansSC", "", 10)


def render_toc(pdf: Report, outline):
    pdf.set_y(24)
    pdf.set_font("NotoSansSC", "", 16)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 9, "目录", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*INK)
    pdf.ln(2)
    for section in outline:
        if section.level > 1:
            continue
        indent = 6 if section.level else 0
        pdf.set_x(18 + indent)
        pdf.set_font("NotoSansSC", "", 10 if section.level == 0 else 9)
        title = section.name
        dots = "." * max(3, 70 - len(title))
        pdf.cell(0, 5.4, f"{title} {dots} {section.page_number}",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def cover(pdf: Report):
    pdf.add_page()
    pdf.set_fill_color(*LIGHT)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_y(42)
    pdf.set_font("NotoSansSC", "", 26)
    pdf.set_text_color(*BLUE)
    pdf.multi_cell(0, 13, "Symmetrix-Polar\n技术报告", align="L")
    pdf.ln(4)
    pdf.set_font("NotoSansSC", "", 13)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 7, "非线性极化力场的 GPU 内核优化\n理论 · 实现 · 实测结果 · 局限 · 复现")
    pdf.ln(14)
    pdf.set_font("NotoSansSC", "", 11)
    metrics = [
        ("128 原子（32 NH3），full PolarMACE", "30.56 ms/eval"),
        ("1024 原子（256 NH3），full PolarMACE", "312.1 ms/eval"),
        ("相对 round-3 build 8 / v29 引擎", "1.595× / 6.02×"),
        ("全套数值门（含 2-rank + 重复评估）", "PASS"),
        ("RBE 生产精度门", "REJECTED"),
    ]
    for label, value in metrics:
        pdf.set_x(24)
        pdf.set_font("NotoSansSC", "", 10)
        pdf.set_text_color(*INK)
        pdf.cell(88, 6.5, label)
        pdf.set_font("NotoSansSC", "", 10.5)
        pdf.set_text_color(*(GREEN if value == "PASS" else RED if value == "REJECTED" else BLUE))
        pdf.cell(0, 6.5, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(232)
    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.3)
    pdf.line(24, pdf.get_y(), 186, pdf.get_y())
    pdf.ln(4)
    pdf.set_x(24)
    pdf.set_font("NotoSansSC", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(160, 5, "版本：round 3 + Plan B-30ms · 2026-10-21\n"
                           "模型：MACE_Nonlinear + PolarMACE，FP64，单张 A100-80GB\n"
                           "本 PDF 由 Markdown 源文件自动生成，可直接续写。")


def main():
    ensure_font()
    text = MD.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = text.split("\n")
    pdf = Report()
    pdf.alias_nb_pages()
    cover(pdf)
    pdf.add_page()
    pdf.insert_toc_placeholder(render_toc, pages=1)

    i = 0
    paragraph = []
    bullets = []
    table_rows = []
    code_lines = []
    in_code = False

    def flush_para():
        nonlocal paragraph
        if paragraph:
            pdf.para(" ".join(paragraph))
            paragraph = []

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
        raw = lines[i]
        line = raw.rstrip()
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
        else:
            flush_table()
        if line.startswith("# "):
            flush_para(); flush_bullets()
            # The document title is on the cover.
            i += 1
            continue
        if line.startswith("## "):
            flush_para(); flush_bullets()
            pdf.h2(clean_inline(line[3:]))
            i += 1
            continue
        if line.startswith("### "):
            flush_para(); flush_bullets()
            pdf.h3(clean_inline(line[4:]))
            i += 1
            continue
        if line.startswith("- "):
            flush_para()
            bullets.append(line[2:])
            i += 1
            continue
        if re.match(r"^\d+\.\s", line):
            flush_para()
            bullets.append(line)
            i += 1
            continue
        if line.startswith(">"):
            flush_para(); flush_bullets()
            quote_lines = [line.lstrip("> ").strip()]
            i += 1
            while i < len(lines) and lines[i].startswith(">"):
                quote_lines.append(lines[i].lstrip("> ").strip())
                i += 1
            pdf.quote(" ".join(quote_lines))
            continue
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if m:
            flush_para(); flush_bullets()
            caption, rel = m.group(1), m.group(2)
            path = (DOC / rel).resolve() if not rel.startswith("/") else Path(rel)
            if not path.exists():
                path = (DOC / "figures" / Path(rel).name).resolve()
            pdf.image(path, caption)
            i += 1
            continue
        if line.strip() == "---":
            flush_para(); flush_bullets()
            pdf.ln(2)
            pdf.set_draw_color(*RULE)
            pdf.line(18, pdf.get_y(), 192, pdf.get_y())
            pdf.ln(2)
            i += 1
            continue
        paragraph.append(line.strip())
        i += 1

    flush_para(); flush_bullets(); flush_table(); flush_code()
    pdf.output(str(OUT))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
