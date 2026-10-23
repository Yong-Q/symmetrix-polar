#!/usr/bin/env bash
# Pre-push gate: whitespace, JSON, links, Python syntax, unit/format warnings.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== git diff --check"
git diff --check

echo "== repository checks"
PY="${PYTHON:-python3}"
"$PY" scripts/check_repo.py

echo "pre-push checks passed"
