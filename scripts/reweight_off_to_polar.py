#!/usr/bin/env python3
"""CLI for MACE-OFF -> MACE-POLAR reweighting (thin wrapper over sampling)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import ase.io
from sampling.off_polar import OffPolarReweighter, ReweightConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--polar-model", required=True)
    ap.add_argument("--source-model", default=None,
                    help="optional; if omitted, frames must carry OFF_energy")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    frames = list(ase.io.iread(args.traj, format="extxyz"))
    cfg = ReweightConfig(temperature_K=args.temperature, stride=args.stride,
                         max_frames=args.max_frames, device=args.device,
                         dtype=args.dtype)
    if args.source_model:
        rw = OffPolarReweighter.from_models(
            source_model=args.source_model, target_model=args.polar_model,
            config=cfg)
    else:
        from sampling.off_polar import GRAPH_LONG_RANGE
        if str(GRAPH_LONG_RANGE) not in sys.path:
            sys.path.insert(0, str(GRAPH_LONG_RANGE))
        from mace.calculators import mace_polar
        target = mace_polar(model=args.polar_model, device=cfg.device,
                            default_dtype=cfg.dtype)
        rw = OffPolarReweighter(None, target, cfg)
        rw.target_model = str(Path(args.polar_model).resolve())

    result = rw.evaluate(frames)
    path = result.write(args.outdir)
    print(json.dumps({k: result.as_dict()[k] for k in (
        "n_frames", "n_atoms", "constant_offset_meV_per_atom",
        "delta_meV_per_atom_std", "sigma_beta", "ess", "ess_fraction",
        "verdict")}, indent=1))
    print(f"[reweight] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
