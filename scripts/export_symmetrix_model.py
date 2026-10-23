#!/usr/bin/env python3
"""Export a MACE/PolarMACE checkpoint to the Symmetrix JSON model format.

This is the model -> engine interface.  It loads the checkpoint with the MACE
calculator (the authoritative PolarMACE class), serializes it with the
Symmetrix extractor, patches the GatedEquivariantBlock activation constants
that the generic extractor does not emit, and validates the dimensions the
engine checks at load time.

Example:
  python scripts/export_symmetrix_model.py \
    --checkpoint path/to/student.model \
    --output bench/my_model/polar_symmetrix.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_SYMMETRIX_ROOT = (
    "/workspace/group/shared/bza/task3_sampling_database/mlip/bench/"
    "lammps_parity/backbone_timing/so_snapshots/v31bb1")
GRAPH_ROOT = "/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0"


def patch_gate_activations(data: dict, model) -> int:
    """Fill readout_gate act_scalars/act_gates for MACE GatedEquivariantBlock.

    The e3nn Gate exposes act_scalars/act_gates; MACE's graph-break-free
    replacement stores _scalar_cst/_gate_cst instead.  The engine only needs
    the constants and the irreps counts.
    """
    blocks = data.get("polar_blocks", {}).get("field_dependent_charges_maps", [])
    patched = 0
    for i, entry in enumerate(blocks):
        gate = entry.get("readout_gate")
        if gate is None or "act_scalars" in gate:
            continue
        cst_s, cst_g = 1.0, 1.0
        try:
            g = model.field_dependent_charges_maps[i].readout.equivariant_nonlin
            cst_s = float(getattr(g, "_scalar_cst", 1.0))
            cst_g = float(getattr(g, "_gate_cst", 1.0))
        except Exception:  # noqa: BLE001 - keep schema defaults
            pass
        gate["act_scalars"] = [{"name": "silu", "cst": cst_s}]
        gate["act_gates"] = [{"name": "sigmoid", "cst": cst_g}]
        patched += 1
    return patched


def validate(data: dict) -> None:
    if "polar_blocks" not in data:
        raise SystemExit("exported model has no polar_blocks")
    interactions = data.get("interactions", [])
    if not interactions:
        raise SystemExit("exported model has no interactions")
    for i, it in enumerate(interactions):
        if "hidden_irreps" not in it or "conv_tp" not in it:
            raise SystemExit(f"interaction {i} is incomplete")
        hidden = it["hidden_irreps"]
        if "x2e" in hidden or "x3o" in hidden or "x2o" in hidden:
            raise SystemExit(
                f"interaction {i} has node l>1 ({hidden}); the exact engine "
                "supports node l<=1 only")
    pb = data["polar_blocks"]
    if "local_electron_energy" in pb:
        le = pb["local_electron_energy"]
        out = le.get("linear_up_q", {}).get("weight", {}).get("shape", [])
        if out:
            width = int(out[0]) // 4
            print(f"local-electron channel width C={width}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--symmetrix-root", default=DEFAULT_SYMMETRIX_ROOT)
    args = ap.parse_args()

    sys.path.insert(0, args.symmetrix_root)
    sys.path.insert(0, GRAPH_ROOT)
    from mace.calculators import mace_polar
    from symmetrix.extract_mace_nonlinear import extract_mace_nonlinear_data

    calc = mace_polar(model=args.checkpoint, device=args.device,
                      default_dtype=args.dtype)
    model = calc.models[0]
    data = extract_mace_nonlinear_data(model)
    patched = patch_gate_activations(data, model)
    validate(data)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB); "
          f"patched {patched} gate activations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
