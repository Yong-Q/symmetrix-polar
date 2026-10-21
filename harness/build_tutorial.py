#!/usr/bin/env python3
"""Build the Chinese tutorial with vector TeX mathematics and offline MathML.

Run from any directory: python harness/build_tutorial.py
Only disposable TeX intermediates are removed. Build logs and QA evidence remain.
"""
from __future__ import annotations

import base64
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/tutorial"
STEM = "SYMMETRIX_POLAR_TUTORIAL_2026-10-21"
BUILD = DOC / ".build"
QA = DOC / "qa"

HEADER = r"""
\setCJKmainfont[Path=CJK_FONT_DIRECTORY,AutoFakeBold=2]{CJK_FONT_FILENAME}
\setCJKsansfont[Path=CJK_FONT_DIRECTORY,AutoFakeBold=2]{CJK_FONT_FILENAME}
\setCJKmonofont[Path=CJK_FONT_DIRECTORY,AutoFakeBold=2]{CJK_FONT_FILENAME}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{float}
\floatplacement{figure}{H}
\usepackage{etoolbox}
\definecolor{BookBlue}{HTML}{154D76}
\definecolor{BookCyan}{HTML}{008C9E}
\definecolor{BookInk}{HTML}{243544}
\definecolor{BookGray}{HTML}{617484}
\hypersetup{colorlinks=true,linkcolor=BookBlue,urlcolor=BookCyan,bookmarksopen=true,bookmarksnumbered=false}
\ctexset{section={format=\Large\bfseries\color{BookBlue},beforeskip=14pt,afterskip=12pt},subsection={format=\large\bfseries\color{BookCyan},beforeskip=15pt,afterskip=8pt},subsubsection={format=\normalsize\bfseries\color{BookBlue}}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4.5pt plus 0.5pt}
\setlength{\emergencystretch}{3em}
\setlength{\headheight}{16pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\sffamily\color{BookGray}Symmetrix-Polar · 中文教程}
\fancyhead[R]{\small\sffamily\color{BookGray}2026-10-21}
\fancyfoot[C]{\small\sffamily\color{BookGray}\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}
\AtBeginEnvironment{longtable}{\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.18}}
\AtBeginEnvironment{figure}{\small}
\makeatletter
\long\def\@makecaption#1#2{\vskip\abovecaptionskip{\small\color{BookGray}#2\par}\vskip\belowcaptionskip}
\makeatother
\let\OriginalContents\tableofcontents
\renewcommand{\tableofcontents}{\clearpage\thispagestyle{plain}\OriginalContents\clearpage}
\renewcommand{\maketitle}{%
  \begin{titlepage}
  \thispagestyle{empty}
  \vspace*{16mm}
  {\sffamily\color{BookCyan}\large MODEL · PHYSICS · GPU · EVIDENCE\par}
  \vspace{12mm}
  {\sffamily\bfseries\color{BookBlue}\fontsize{32}{42}\selectfont Symmetrix-Polar\par}
  \vspace{6mm}
  {\bfseries\color{BookBlue}\fontsize{23}{34}\selectfont 从模型原理到 GPU 优化\par}
  \vspace{8mm}
  {\Large 中文教程与测试解读\par}
  \vspace{8mm}
  {\color{BookCyan}\rule{\textwidth}{1.2pt}\par}
  \vspace{9mm}
  {\large 从原子、能量和力出发，逐步理解 MACE、PolarMACE、\par
   长程静电、GPU 执行与本项目的性能证据。\par}
  \vfill
  {\color{BookGray}Yong-Q · Symmetrix-Polar\par}
  \vspace{3mm}
  {\color{BookGray}2026-10-21\par}
  \vspace{9mm}
  {\small 性能统一为 µs/atom/step · 数学公式为矢量排版\par}
  \end{titlepage}
}
"""

CSS = """
:root{color-scheme:light;--blue:#154d76;--cyan:#008c9e;--ink:#243544;--muted:#617484}
*{box-sizing:border-box}body{margin:0;background:#edf3f6;color:var(--ink);font-family:'Noto Sans CJK SC','Microsoft YaHei',sans-serif;line-height:1.85;font-size:17px}
body>header,body>nav,body>section,body>h1,body>h2,body>h3,body>p,body>table,body>figure,body>ul,body>ol,body>div,body>blockquote{max-width:1000px;margin-left:auto;margin-right:auto}
body{padding:40px 28px 100px;background:linear-gradient(#e5f2f6 0 400px,#fff 400px)}
header{padding:35px 0 40px;border-bottom:3px solid var(--cyan)}header .title{font-size:38px;line-height:1.35;color:var(--blue);margin:0 0 20px}header p{color:var(--muted);margin:7px 0}
h1{font-size:29px;color:var(--blue);line-height:1.5;margin-top:64px!important;border-bottom:2px solid #bee3e8;padding-bottom:15px}h2{font-size:23px;color:var(--cyan);line-height:1.5;margin-top:35px!important}h3{color:var(--blue)}
#TOC{background:white;border:1px solid #c5dce5;border-radius:10px;padding:25px 30px;margin-top:35px;margin-bottom:45px}#TOC ul{padding-left:25px}#TOC li{margin:5px 0}a{color:var(--blue);text-decoration-thickness:1px;text-underline-offset:3px}
p{margin-top:15px;margin-bottom:15px}strong{color:var(--blue)}img{display:block;max-width:100%;height:auto;margin:28px auto 12px}figure{margin-top:34px;margin-bottom:34px}figcaption{font-size:14px;line-height:1.7;color:var(--muted);padding:10px 15px;border-left:3px solid var(--cyan);background:#f2f8fa}
table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.65;margin-top:24px;margin-bottom:28px}th{background:#e8f3f7;color:var(--blue);text-align:left}th,td{padding:11px 12px;border-bottom:1px solid #d4e0e6;vertical-align:top}tr:nth-child(even){background:#f8fbfc}
code{font-size:.86em;background:#eff4f7;padding:2px 5px;overflow-wrap:anywhere}pre{padding:20px;background:#eff4f7;overflow:auto}math[display='block']{margin:25px auto;overflow-x:auto;overflow-y:hidden;padding:8px;max-width:100%}math{font-family:'STIX Two Math','Cambria Math',math}
blockquote{border-left:4px solid var(--cyan);background:#f1f7fa;padding:10px 25px;color:var(--muted)}.offline-note{font-size:14px;color:var(--muted);margin-top:35px}
@media(max-width:700px){body{font-size:16px;padding:20px 16px}header .title{font-size:29px}h1{font-size:25px}h2{font-size:21px}table{display:block;overflow-x:auto}#TOC{padding:18px}}
@media print{body{background:#fff;padding:0;font-size:11pt}h1{break-before:page}header{break-after:page}#TOC{break-after:page}a{color:inherit}table,figure{break-inside:avoid}}
"""


def run(args: list[str], log_name: str) -> None:
    result = subprocess.run(args, cwd=DOC, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (QA / log_name).write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        print(result.stdout[-16000:])
        raise SystemExit(f"Build failed ({result.returncode}); see {QA / log_name}")


def main() -> None:
    BUILD.mkdir(exist_ok=True)
    QA.mkdir(exist_ok=True)
    src = DOC / f"{STEM}.md"
    raw = src.read_text(encoding="utf-8")
    missing = []
    for image_path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", raw):
        if not (DOC / image_path).is_file():
            missing.append(image_path)
    if missing:
        raise SystemExit("Missing figures: " + ", ".join(missing))
    header = BUILD / "header.tex"
    font_match = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True).strip()
    font_path = Path(font_match)
    if not font_path.is_file():
        raise SystemExit("Install Noto Sans CJK SC and make it discoverable with fc-match.")
    resolved_header = HEADER.replace("CJK_FONT_DIRECTORY", str(font_path.parent) + "/").replace("CJK_FONT_FILENAME", font_path.name)
    header.write_text(resolved_header, encoding="utf-8")
    css = DOC / "tutorial.css"
    css.write_text(CSS, encoding="utf-8")
    lua = BUILD / "layout.lua"
    lua.write_text('function Header(h)\n if h.level == 1 and FORMAT:match("latex") then return {pandoc.RawBlock("latex", "\\\\clearpage"), h} end\n return h\nend\n', encoding="utf-8")
    # High-resolution Python PNGs avoid the source PDF font-subset issue.
    # All TeX mathematics remains vector. Source SVG/PDF chart exports are retained.
    pdf_raw = raw
    pdf_src = BUILD / "pdf_source.md"
    pdf_src.write_text(pdf_raw, encoding="utf-8")
    tex = BUILD / "tutorial.tex"
    common = ["pandoc", "--from=markdown+implicit_figures+tex_math_dollars", "--standalone", "--toc", "--toc-depth=2", "--resource-path=" + str(DOC)]
    run(common + [str(pdf_src), "--to=latex", "--metadata=lang:", "--lua-filter=" + str(lua), "--include-in-header=" + str(header), "-V", "documentclass=ctexart", "-V", "classoption=fontset=none", "-V", "mainfont=DejaVu Serif", "-V", "sansfont=DejaVu Sans", "-V", "monofont=DejaVu Sans Mono", "-V", "CJKsansfont=Noto Sans CJK SC", "-V", "CJKmonofont=Noto Sans CJK SC", "-V", "fontsize=10pt", "-V", "linestretch=1.18", "-V", "geometry=a4paper,top=23mm,bottom=22mm,left=20mm,right=20mm", "-V", "colorlinks=true", "-o", str(tex)], "pandoc_pdf.log")
    tex_content = tex.read_text(encoding="utf-8").replace("MACE\\_Nonlinear/PolarMACE", "MACE\\_\\allowbreak Nonlinear/\\allowbreak PolarMACE")
    tex_content = re.sub(r"(\\section\{附录\s+D：)", lambda m: "\\setstretch{1.10}\n" + m[1], tex_content)
    tex.write_text(tex_content, encoding="utf-8")
    for attempt in range(1, 4):
        run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=" + str(BUILD), str(tex)], f"xelatex_pass{attempt}.log")
    shutil.copy2(BUILD / "tutorial.pdf", DOC / f"{STEM}.pdf")
    # Pandoc 2.12 self-contained embeds image resources and CSS; MathML is native.
    run(common + [str(src), "--to=html5", "--mathml", "--self-contained", "--css=" + str(css), "--metadata=pagetitle:Symmetrix-Polar 中文教程", "-o", str(DOC / "index.html")], "pandoc_html.log")
    import fitz
    pdf = fitz.open(DOC / f"{STEM}.pdf")
    warnings = []
    text_log = (QA / "xelatex_pass3.log").read_text(encoding="utf-8")
    for line in text_log.splitlines():
        if any(key in line for key in ("Overfull", "Missing character", "Warning")):
            warnings.append(line)
    report = {"pages":len(pdf), "bookmarks":len(pdf.get_toc()), "html_bytes":(DOC / "index.html").stat().st_size, "mathml_elements":(DOC / "index.html").read_text(encoding="utf-8").count("<math"), "warnings":warnings}
    (QA / "build_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # These exact paths are task-owned intermediates; logs and source TeX remain.
    for suffix in ("aux", "out", "toc"):
        (BUILD / ("tutorial." + suffix)).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
