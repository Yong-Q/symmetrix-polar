#!/usr/bin/env python3
"""CLI for a short MACE-OFF NVT trajectory (thin wrapper over sampling)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sampling.trajectory import TrajectoryConfig, read_structure, run_langevin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--timestep-fs", type=float, default=0.5)
    ap.add_argument("--ps", type=float, default=5.0)
    ap.add_argument("--equil-ps", type=float, default=0.5)
    ap.add_argument("--frame-ps", type=float, default=0.1)
    ap.add_argument("--friction", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()

    from mace.calculators import MACECalculator
    atoms = read_structure(args.structure)
    calc = MACECalculator(model_paths=args.model, device=args.device,
                          default_dtype=args.dtype)
    cfg = TrajectoryConfig(
        temperature_K=args.temperature, timestep_fs=args.timestep_fs,
        ps=args.ps, equil_ps=args.equil_ps, frame_ps=args.frame_ps,
        friction=args.friction, seed=args.seed)
    run_langevin(atoms, calc, cfg, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
