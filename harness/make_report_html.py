#!/usr/bin/env python3
"""Build the offline HTML reading page for the presentation report.

Display equations are pre-rendered to PNG with the same mathtext settings as
the PDF, so the page reads offline without MathJax or a network connection.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mistune

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs/report"
MD = DOC / "SYMMETRIX_POLAR_REPORT_2026-10-21.md"
OUT = DOC / "index.html"
MATH = DOC / "math"
MATH.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral"})


def render_math(latex: str, index: int, fontsize: int = 14) -> Path:
    path = MATH / f"eq_{index:03d}.png"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=fontsize, color="#24292f")
    fig.savefig(path, dpi=420, transparent=True, bbox_inches="tight",
                pad_inches=0.03)
    plt.close(fig)
    return path


def plain_inline(text: str) -> str:
    text = re.sub(r"\\\(([^)]*)\\\)", lambda m: m.group(1), text)
    return text


def main() -> None:
    md = MD.read_text(encoding="utf-8")
    md = plain_inline(md)

    equations: list[str] = []
    figures: list[tuple[str, str]] = []

    def eq_repl(m: re.Match) -> str:
        equations.append(" ".join(m.group(1).split()))
        return f"\n\n@@EQ{len(equations):03d}@@\n\n"

    def fig_repl(m: re.Match) -> str:
        cap, path = m.group(1), m.group(2)
        figures.append((cap, path))
        return f"\n\n@@FIG{len(figures):03d}@@\n\n"

    md = re.sub(r"\\\[(.*?)\\\]", eq_repl, md, flags=re.S)
    md = re.sub(r"!\[(.*?)\]\((.*?)\)", fig_repl, md)

    # Drop the first H1 (the cover already shows the title).
    md = re.sub(r"^# .*?\n", "", md, count=1)

    renderer = mistune.create_markdown(plugins=["table"])
    body = renderer(md)

    # Section ids and sidebar TOC.
    headings = re.findall(r"^(#{2,3})\s+(.*)$", MD.read_text(encoding="utf-8"),
                          flags=re.M)
    toc = []
    for i, (level, title) in enumerate(headings):
        toc.append((len(level) - 2, title.strip(), f"sec-{i}"))
    h_index = 0

    def heading_id(m: re.Match) -> str:
        nonlocal h_index
        level, inner = m.group(1), m.group(2)
        if h_index < len(toc):
            ident = toc[h_index][2]
            h_index += 1
            return f"<{level} id=\"{ident}\">{inner}</{level}>"
        return m.group(0)

    body = re.sub(r"<(h[23])>(.*?)</\1>", heading_id, body, flags=re.S)

    for i, latex in enumerate(equations, 1):
        render_math(latex, i)
        body = body.replace(
            f"<p>@@EQ{i:03d}@@</p>",
            f'<div class="equation"><img src="math/eq_{i:03d}.png" '
            f'alt="equation {i}"><span class="eqno">({i})</span></div>')
        body = body.replace(f"@@EQ{i:03d}@@",
                            f'<div class="equation"><img src="math/eq_{i:03d}.png" '
                            f'alt="equation {i}"><span class="eqno">({i})</span></div>')

    for i, (cap, path) in enumerate(figures, 1):
        block = (f'<figure><img src="{html.escape(path)}" '
                 f'alt="{html.escape(cap)}"><figcaption>图 {i}｜'
                 f'{html.escape(cap)}</figcaption></figure>')
        body = body.replace(f"<p>@@FIG{i:03d}@@</p>", block)
        body = body.replace(f"@@FIG{i:03d}@@", block)

    nav_items = []
    for level, title, ident in toc:
        nav_items.append(
            f'<a class="lvl{level}" href="#{ident}">{html.escape(title)}</a>')
    nav = "\n".join(nav_items)

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Symmetrix-Polar：PolarMACE 的 GPU 引擎优化</title>
<style>
:root {{ --blue:#1f6feb; --cyan:#0ea5e9; --ink:#24292f; --muted:#3c4450;
         --rule:#d0d7de; --light:#f6f9ff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--light); color:var(--ink);
        font-family:"Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif;
        line-height:1.75; }}
.layout {{ display:flex; max-width:1240px; margin:0 auto; background:#fff; }}
.sidebar {{ width:270px; position:sticky; top:0; height:100vh; overflow:auto;
            padding:24px 18px; border-right:1px solid var(--rule);
            background:#fff; flex:0 0 270px; }}
.sidebar h2 {{ font-size:15px; color:var(--blue); margin:0 0 12px; }}
.sidebar a {{ display:block; color:var(--muted); text-decoration:none;
              font-size:13px; padding:3px 0; border-left:2px solid transparent;
              padding-left:8px; }}
.sidebar a:hover {{ color:var(--blue); }}
.sidebar a.lvl1 {{ padding-left:20px; font-size:12.5px; }}
.content {{ flex:1; padding:34px 54px 60px; min-width:0; }}
.hero {{ background:linear-gradient(135deg,#eef5ff 0%,#f6f9ff 60%,#eefaff 100%);
         border-top:4px solid var(--blue); padding:30px 32px; margin:-34px -54px 26px; }}
.hero .kicker {{ color:var(--cyan); font-size:12px; letter-spacing:0.08em; }}
.hero h1 {{ color:var(--blue); font-size:34px; line-height:1.25; margin:8px 0 6px; }}
.hero .sub {{ color:var(--muted); font-size:15px; }}
.hero .meta {{ color:var(--muted); font-size:12.5px; margin-top:10px; }}
.metrics {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:18px; }}
.metric {{ background:#fff; border:1px solid var(--rule); border-radius:10px;
           padding:10px 14px; min-width:180px; }}
.metric .v {{ font-size:18px; font-weight:700; color:var(--blue); }}
.metric .l {{ font-size:12px; color:var(--muted); }}
h2 {{ color:var(--blue); font-size:23px; margin:34px 0 8px;
      padding-bottom:6px; border-bottom:2px solid var(--blue); }}
h3 {{ color:var(--cyan); font-size:17px; margin:22px 0 6px; }}
p {{ margin:9px 0; }}
ul {{ margin:8px 0 8px 22px; }}
li {{ margin:4px 0; }}
table {{ border-collapse:collapse; width:100%; margin:16px 0; font-size:13.5px; }}
th {{ background:var(--blue); color:#fff; font-weight:600; }}
th,td {{ border:1px solid var(--rule); padding:6px 9px; text-align:left; }}
tr:nth-child(even) td {{ background:var(--light); }}
.equation {{ display:flex; align-items:center; justify-content:center;
             gap:14px; margin:16px 0; }}
.equation img {{ max-width:86%; height:auto; }}
.eqno {{ color:var(--muted); font-size:13px; }}
figure {{ margin:22px 0; text-align:center; }}
figure img {{ max-width:100%; height:auto; }}
figcaption {{ color:var(--muted); font-size:13px; margin-top:6px; }}
blockquote {{ margin:12px 0; padding:10px 16px; background:var(--light);
              border-left:4px solid var(--cyan); color:var(--muted); }}
code {{ background:#f1f4f8; padding:1px 5px; border-radius:4px;
        font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:90%; }}
pre code {{ display:block; padding:12px; overflow:auto; }}
.footer {{ color:var(--muted); font-size:12px; border-top:1px solid var(--rule);
           margin-top:36px; padding-top:10px; }}
@media print {{
  body {{ background:#fff; }}
  .sidebar {{ display:none; }}
  .layout {{ display:block; }}
  .content {{ padding:0; }}
  .hero {{ margin:0 0 18px; }}
  h2 {{ page-break-before:always; }}
}}
@media (max-width:900px) {{
  .layout {{ display:block; }}
  .sidebar {{ position:static; width:auto; height:auto; border-right:0;
              border-bottom:1px solid var(--rule); }}
  .content {{ padding:24px 20px 40px; }}
  .hero {{ margin:-24px -20px 20px; }}
}}
</style>
</head>
<body>
<div class="layout">
<aside class="sidebar">
<h2>目录</h2>
{nav}
</aside>
<main class="content">
<div class="hero">
<div class="kicker">SYMMETRIX-POLAR · 技术汇报</div>
<h1>PolarMACE 的 GPU 引擎优化</h1>
<div class="sub">模型理论 · 优化方法 · 实测结果 · 不足与下一步</div>
<div class="meta">2026-10-21 · 单张 A100-80GB · FP64 · 所有速度单位 µs/atom</div>
<div class="metrics">
<div class="metric"><div class="v">224.1 µs/atom</div><div class="l">1,296 原子 full PolarMACE</div></div>
<div class="metric"><div class="v">2.43–3.0×</div><div class="l">相对 PyTorch PolarMACE</div></div>
<div class="metric"><div class="v">4.47–4.53 / 4.50</div><div class="l">线性 MACE vs 论文</div></div>
<div class="metric"><div class="v">49.8×</div><div class="l">距 4.5 µs/atom 目标</div></div>
</div>
</div>
{body}
<div class="footer">Symmetrix-Polar · 汇报版 · 2026-10-21 · 公式为离线数学排版（PNG），无需 MathJax。</div>
</main>
</div>
</body>
</html>
"""
    OUT.write_text(page, encoding="utf-8")
    print("wrote", OUT, "equations", len(equations), "figures", len(figures),
          "headings", len(toc))


if __name__ == "__main__":
    main()
