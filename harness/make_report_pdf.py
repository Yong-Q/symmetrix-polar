#!/usr/bin/env python3
"""Compatibility entry: rebuild the current Chinese tutorial."""
from pathlib import Path
import runpy
import shutil
root = Path(__file__).resolve().parents[1]
runpy.run_path(str(root / "harness/build_tutorial.py"), run_name="__main__")
shutil.copy2(root / "docs/tutorial/SYMMETRIX_POLAR_TUTORIAL_2026-10-21.pdf",
             root / "docs/report/SYMMETRIX_POLAR_REPORT_2026-10-21.pdf")
print("Updated tutorial and compatibility report PDF.")
