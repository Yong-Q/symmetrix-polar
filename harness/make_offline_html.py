#!/usr/bin/env python3
"""Render the consolidated comparison report to a standalone offline HTML page."""
from pathlib import Path

import markdown

PROJ = Path("/workspace/group/shared/bza/symmetrix-polar")
DOC = PROJ / "docs"
MD = DOC / "COMPARISON_REPORT_2026-10-19.md"
OUT = DOC / "index.html"

text = MD.read_text()
body = markdown.markdown(
    text, extensions=["tables", "fenced_code", "sane_lists", "toc"]
)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Symmetrix-Polar: consolidated speed and accuracy comparison</title>
<style>
:root {{ --blue:#1f6feb; --ink:#24292f; --muted:#57606a; --line:#d0d7de; --bg:#f6f8fa; }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 0 24px 64px; background: #fff; color: var(--ink);
  font: 16px/1.62 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial,
        "Noto Sans", "PingFang SC", "Microsoft YaHei", sans-serif;
}}
main {{ max-width: 1040px; margin: 0 auto; }}
header.top {{
  max-width: 1040px; margin: 0 auto; padding: 22px 0 14px;
  border-bottom: 1px solid var(--line);
}}
header.top .brand {{ font-weight: 700; color: var(--blue); letter-spacing: .2px; }}
header.top nav {{ margin-top: 8px; font-size: 14px; }}
header.top nav a {{
  color: var(--blue); text-decoration: none; margin-right: 18px;
  border-bottom: 1px solid transparent;
}}
header.top nav a:hover {{ border-bottom-color: var(--blue); }}
h1 {{ font-size: 2.05em; line-height: 1.2; margin: 1.1em 0 .35em; color: #0b3d91; }}
h2 {{
  font-size: 1.5em; margin: 1.7em 0 .55em; padding-bottom: .25em;
  border-bottom: 1px solid var(--line); color: #0b3d91;
}}
h3 {{ font-size: 1.18em; margin: 1.35em 0 .4em; }}
a {{ color: var(--blue); }}
code {{
  font: .9em/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--bg); padding: .12em .38em; border-radius: 4px;
}}
pre {{
  background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
  padding: 12px 14px; overflow-x: auto;
}}
pre code {{ background: none; padding: 0; }}
table {{
  border-collapse: collapse; width: 100%; margin: 1em 0 1.4em; font-size: 14px;
}}
th, td {{ border: 1px solid var(--line); padding: 7px 9px; text-align: left; vertical-align: top; }}
th {{ background: var(--bg); font-weight: 650; }}
tr:nth-child(even) td {{ background: #fbfcfd; }}
img {{ max-width: 100%; height: auto; display: block; margin: 1.1em auto; }}
blockquote {{
  margin: 1em 0; padding: .4em 1em; color: var(--muted);
  border-left: 4px solid var(--line); background: #fbfcfd;
}}
hr {{ border: 0; border-top: 1px solid var(--line); margin: 2em 0; }}
.note {{
  background: #fff8e6; border: 1px solid #f0d58c; border-radius: 8px;
  padding: 10px 14px; margin: 1em 0; font-size: 14px;
}}
footer {{
  max-width: 1040px; margin: 40px auto 0; padding-top: 14px;
  border-top: 1px solid var(--line); color: var(--muted); font-size: 13px;
}}
</style>
</head>
<body>
<header class="top">
  <div class="brand">Symmetrix-Polar</div>
  <nav>
    <a href="COMPARISON_REPORT_2026-10-19.pdf">PDF report</a>
    <a href="COMPARISON_REPORT_2026-10-19.md">Markdown source</a>
    <a href="阅读说明.md">Reading guide</a>
    <a href="figures/">Figures</a>
    <a href="examples/">Data and scripts</a>
  </nav>
</header>
<main>
<div class="note">This offline page is generated from
<code>docs/COMPARISON_REPORT_2026-10-19.md</code>. The paginated
<a href="COMPARISON_REPORT_2026-10-19.pdf">PDF edition</a> contains the same
content with a cover and a figure gallery.</div>
{body}
</main>
<footer>
Symmetrix-Polar · consolidated speed and accuracy comparison · generated 2026-10-19.
</footer>
</body>
</html>
"""

OUT.write_text(HTML)
print("wrote", OUT, len(HTML), "bytes")
