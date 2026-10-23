#!/usr/bin/env python3
"""Lightweight repository checks for the Symmetrix-Polar workflow.

- JSON files under docs/results/data must parse.
- Relative Markdown links in README/docs must resolve.
- Python files under scripts/ and docs/design/figures must compile.
- Public docs are warned, not failed, on ms/eval or ms/atom speed text.
- Tracked files above 10 MB are reported (large binaries should not be in Git).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []
warnings: list[str] = []


def check_json() -> None:
    base = ROOT / "docs/results/data"
    if not base.exists():
        return
    for path in sorted(base.glob("*.json")):
        try:
            json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")


def check_links() -> None:
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    # A link target must look like a path: no spaces/code punctuation, and it
    # must contain a dot or a slash. This skips C++/math fragments that happen
    # to sit inside Markdown brackets.
    path_re = re.compile(r"^[A-Za-z0-9._~/%()+-]+$")
    files = [ROOT / "README.md"]
    files += sorted((ROOT / "docs").rglob("*.md"))
    for path in files:
        if not path.exists():
            continue
        rel = path.relative_to(ROOT)
        if "archive_" in str(rel):
            continue
        text = path.read_text(errors="ignore")
        for target in link_re.findall(text):
            target = target.strip().split("#", 1)[0]
            if (not target or len(target) > 300 or "\n" in target
                    or "://" in target or target.startswith("mailto:")):
                continue
            if not path_re.match(target) or ("." not in target and "/" not in target):
                continue
            try:
                resolved = (path.parent / target).resolve()
                exists = resolved.exists()
            except OSError:
                continue
            if not exists:
                message = f"broken link {rel} -> {target}"
                # Historical narrative docs may point at old artifact names;
                # only README and results data are release-blocking.
                if rel == Path("README.md") or str(rel).startswith("docs/results"):
                    errors.append(message)
                else:
                    warnings.append(message)


def check_python() -> None:
    roots = [ROOT / "scripts", ROOT / "docs/design/figures"]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            try:
                compile(path.read_text(), str(path), "exec")
            except SyntaxError as exc:
                errors.append(f"python syntax {path.relative_to(ROOT)}: {exc}")


def check_units() -> None:
    pat = re.compile(r"\bms\s*/\s*(eval|atom|step)\b", re.IGNORECASE)
    for path in [ROOT / "README.md"] + sorted((ROOT / "docs/design").glob("*.md")):
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if pat.search(line) and "duration" not in line.lower():
                warnings.append(
                    f"ms speed unit {path.relative_to(ROOT)}:{lineno}: {line.strip()[:100]}")


def check_sizes(limit_mb: float = 10.0) -> None:
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT, text=True)
    except Exception:  # noqa: BLE001
        return
    for rel in out.split("\0"):
        if not rel:
            continue
        path = ROOT / rel
        if path.is_file() and path.stat().st_size > limit_mb * 1e6:
            warnings.append(
                f"large tracked file {rel}: {path.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    check_json()
    check_links()
    check_python()
    check_units()
    check_sizes()
    for item in warnings:
        print(f"WARN {item}")
    for item in errors:
        print(f"ERROR {item}")
    print(f"checks: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
