#!/usr/bin/env bash
# One-shot environment for the MACE-OFF23 -> MACE-POLAR sampling / reweighting /
# cDFT-dataset pipeline.
#
#   bash setup.sh                # create ./envs/cdft and download both models
#   ENV_NAME=myenv bash setup.sh # reuse an existing conda env name
#   TORCH_INDEX="" bash setup.sh # skip the torch reinstall (CPU or preinstalled)
#
# What it does:
#   1. creates a conda/micromamba env (python 3.11)
#   2. installs torch + mace-torch==0.3.16 + ase/numpy/scipy/matplotlib/pandas
#   3. installs graph_longrange (pinned commit) for MACE-POLAR electrostatics
#   4. downloads MACE-OFF23-medium and MACE-POLAR-1-M into ./models
#   5. optionally clones the reweighting/cDFT repo (REPO_TEST=1)
#   6. runs a CPU smoke test that loads both potentials and evaluates a water box
#
# Heavy, NOT handled here: the Symmetrix/LAMMPS GPU engine used for the
# production site-field MD.  That build is described in README.md; the Python
# path above is everything needed for labelling, reweighting, cDFT datasets
# and training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-cdft}"
PY_VER="${PY_VER:-3.11}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
MACE_VERSION="${MACE_VERSION:-0.3.16}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models}"
GRAPH_DIR="${GRAPH_DIR:-$ROOT/external/graph_electrostatics}"
GRAPH_COMMIT="${GRAPH_COMMIT:-6a86de5e3ed35fd86a55fc046aa085fe48a72764}"
GRAPH_REPO="${GRAPH_REPO:-https://github.com/WillBaldwin0/graph_electrostatics.git}"
REPO_TEST="${REPO_TEST:-0}"
TEST_REPO_URL="${TEST_REPO_URL:-https://github.com/Yong-Q/test.git}"

OFF23_URL="https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/MACE-OFF23_medium.model"
POLAR_URL="https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-M.model"
OFF23_SHA="4842c52ad210d6e1f84d6cf1ffa70fae25a7e0d755ed55cf223f43913f587db7"
POLAR_SHA="fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"

say() { printf '\n== %s ==\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

say "workspace: $ROOT"

# ---------------------------------------------------------------- env manager
if have micromamba; then
  CONDA=micromamba
elif have mamba; then
  CONDA=mamba
elif have conda; then
  CONDA=conda
else
  die "need conda, mamba or micromamba on PATH"
fi

if ! $CONDA env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  say "creating env $ENV_NAME (python $PY_VER)"
  $CONDA create -y -n "$ENV_NAME" "python=$PY_VER" pip
else
  say "env $ENV_NAME already exists"
fi

if [ "$CONDA" = "micromamba" ]; then
  RUN=("$CONDA" run -n "$ENV_NAME")
else
  RUN=("$CONDA" run -n "$ENV_NAME")
fi
PY=("${RUN[@]}" python)

# ---------------------------------------------------------------- python deps
say "installing python packages"
"${PY[@]}" -m pip install --upgrade pip

if [ -n "$TORCH_INDEX" ]; then
  "${PY[@]}" -m pip install "torch>=2.4" --index-url "$TORCH_INDEX"
else
  "${PY[@]}" -m pip install "torch>=2.4"
fi
"${PY[@]}" -m pip install \
  "mace-torch==${MACE_VERSION}" \
  "ase>=3.23" "numpy>=1.26" "scipy>=1.11" "matplotlib>=3.8" "pandas>=2.1"
# optional GPU speedup for MACE (safe to skip)
"${PY[@]}" -m pip install cuequivariance cuequivariance-torch || true

say "installing graph_longrange (pinned $GRAPH_COMMIT)"
if [ -d "$GRAPH_DIR/.git" ]; then
  git -C "$GRAPH_DIR" fetch --all --quiet || true
  git -C "$GRAPH_DIR" checkout --quiet "$GRAPH_COMMIT" || true
  "${PY[@]}" -m pip install -e "$GRAPH_DIR"
else
  mkdir -p "$(dirname "$GRAPH_DIR")"
  git clone --quiet "$GRAPH_REPO" "$GRAPH_DIR"
  git -C "$GRAPH_DIR" checkout --quiet "$GRAPH_COMMIT"
  "${PY[@]}" -m pip install -e "$GRAPH_DIR"
fi

# ---------------------------------------------------------------- models
say "downloading potentials to $MODEL_DIR"
mkdir -p "$MODEL_DIR/mace-polar"
fetch() {  # url dest sha
  local url="$1" dest="$2" sha="$3"
  if [ -f "$dest" ]; then
    echo "  exists: $dest"
  else
    echo "  fetch: $url"
    curl -fL --retry 3 -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
  fi
  local got
  got="$(sha256sum "$dest" | awk '{print $1}')"
  if [ "$got" != "$sha" ]; then
    echo "  WARNING: sha256 mismatch for $dest"
    echo "           expected $sha"
    echo "           got      $got"
  else
    echo "  sha256 ok: $dest"
  fi
}
fetch "$OFF23_URL" "$MODEL_DIR/MACE-OFF23_medium.model" "$OFF23_SHA"
fetch "$POLAR_URL" "$MODEL_DIR/mace-polar/MACE-POLAR-1-M.model" "$POLAR_SHA"

# ---------------------------------------------------------------- optional repo
if [ "$REPO_TEST" = "1" ] && [ ! -d "$ROOT/external/test" ]; then
  say "cloning reweighting/cDFT repo"
  git clone "$TEST_REPO_URL" "$ROOT/external/test" || \
    echo "  WARNING: could not clone $TEST_REPO_URL (private?) — copy it manually"
fi

# ---------------------------------------------------------------- smoke test
say "smoke test (CPU): load both potentials and evaluate a water box"
export GRAPH_LONG_RANGE_PATH="$GRAPH_DIR"
"${PY[@]}" - "$MODEL_DIR" <<'PY'
import sys
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator

model_dir = Path(sys.argv[1])
# 8 water molecules in a 13 A box (liquid-ish, small enough for CPU)
wat = molecule("H2O")
atoms = Atoms()
for i in range(8):
    m = wat.copy()
    m.translate([(i % 2) * 6.5 + 1.0, ((i // 2) % 2) * 6.5 + 1.0,
                 (i // 4) * 6.5 + 1.0])
    atoms += m
atoms.set_cell([13.0, 13.0, 13.0])
atoms.set_pbc(True)

from mace.calculators import mace_off, mace_polar

off = mace_off(model=str(model_dir / "MACE-OFF23_medium.model"),
               device="cpu", default_dtype="float64")
polar = mace_polar(model=str(model_dir / "mace-polar/MACE-POLAR-1-M.model"),
                   device="cpu", default_dtype="float64")

for name, calc in (("MACE-OFF23-medium", off), ("MACE-POLAR-1-M", polar)):
    a = atoms.copy()
    a.calc = calc
    e = float(a.get_potential_energy())
    f = a.get_forces()
    print(f"  {name:20s} E = {e:.4f} eV   |F|max = {np.abs(f).max():.3f} eV/A")

print("SMOKE OK")
PY

say "done"
cat <<EOF

Env:            $ENV_NAME
Models:         $MODEL_DIR/MACE-OFF23_medium.model
                $MODEL_DIR/mace-polar/MACE-POLAR-1-M.model
graph_longrange:$GRAPH_DIR  (commit $GRAPH_COMMIT)

Next:
  conda run -n $ENV_NAME python -m sampling.cli ...        # from the test repo
  export GRAPH_LONG_RANGE_PATH=$GRAPH_DIR

Production site-field MD additionally needs the Symmetrix/LAMMPS engine
(see README.md).  Remote dispatch uses the MatOS veMLP tools and is not part
of this script.
EOF
