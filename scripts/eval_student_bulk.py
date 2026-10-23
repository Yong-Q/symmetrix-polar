#!/usr/bin/env python3
"""Evaluate a distilled PolarMACE student on the state-level valid split.

Reports overall and per-stratum MAE:

  E: meV/atom  = mean_frames |E_pred - E_ref| / N_atoms
  F: meV/Angstrom = mean over all atoms and Cartesian components |F_pred - F_ref|

Strata: dataset (bulk/gas), species, bulk density bin, bulk temperature bin,
plus the species x density-bin and species x temperature-bin breakdowns.  The
gas-only row is the reference for the original gas-phase domain.

Speed is reported only when ``--speed`` is given, always in microseconds per
atom per step (one model evaluation = one MD step).  Model load and file I/O
are excluded.  This is a PyTorch calculator measurement, not an engine number.

Usage:
  /opt/matos-tools/conda/envs/matos-compute/bin/python \
    scripts/eval_student_bulk.py --model <student.model> --valid data/valid.xyz \
    --device cuda --dtype float32 --speed
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ase.io

REPO = Path(__file__).resolve().parents[1]
GRAPH_LONG_RANGE = str(REPO.parent / "external-repos/graph_electrostatics-0.4.0")

DENSITY_BINS: List[Tuple[float, float, str]] = [
    (0.0, 1e-3, "dilute (<1e-3)"),
    (1e-3, 1e-2, "1e-3..1e-2"),
    (1e-2, 3.5e-2, "1e-2..3.5e-2"),
    (3.5e-2, float("inf"), "3.5e-2..1e-1"),
]
TEMPERATURE_BINS: List[Tuple[float, float, str]] = [
    (0.0, 250.0, "150-250 K"),
    (250.0, 400.0, "250-400 K"),
    (400.0, 800.0, "400-800 K"),
    (800.0, 1200.0, "800-1200 K"),
    (1200.0, float("inf"), "1200-2000 K"),
]


def _bin(value: Optional[float], bins: Sequence[Tuple[float, float, str]]) -> str:
    if value is None or not np.isfinite(value):
        return "unknown"
    for lo, hi, label in bins:
        if lo <= value < hi:
            return label
    return "unknown"


def _load_frames(path: str, max_frames: int) -> List[ase.Atoms]:
    frames = ase.io.read(str(path), index=":", format="extxyz")
    if isinstance(frames, ase.Atoms):
        frames = [frames]
    if max_frames and max_frames < len(frames):
        # Deterministic spread over the whole file, not the first N states.
        idx = np.unique(np.linspace(0, len(frames) - 1, max_frames).round().astype(int))
        frames = [frames[i] for i in idx]
    return frames


class Accumulator:
    __slots__ = ("n_frames", "n_atoms", "e_abs_sum", "f_abs_sum",
                 "n_force_components", "e_per_atom", "f_per_atom")

    def __init__(self) -> None:
        self.n_frames = 0
        self.n_atoms = 0
        self.e_abs_sum = 0.0
        self.f_abs_sum = 0.0
        self.n_force_components = 0
        self.e_per_atom: List[float] = []
        self.f_per_atom: List[float] = []

    def add(self, e_abs: float, f_abs: float, n_atoms: int) -> None:
        self.n_frames += 1
        self.n_atoms += n_atoms
        self.e_abs_sum += e_abs
        self.f_abs_sum += f_abs
        self.n_force_components += 3 * n_atoms
        self.e_per_atom.append(e_abs / max(n_atoms, 1))
        self.f_per_atom.append(f_abs / max(3 * n_atoms, 1))

    def metrics(self) -> Dict[str, Any]:
        if not self.n_frames:
            return {"n_frames": 0, "n_atoms": 0,
                    "mae_e_meV_per_atom": None, "mae_f_meV_per_A": None,
                    "median_e_meV_per_atom": None, "median_f_meV_per_A": None,
                    "p90_e_meV_per_atom": None, "p90_f_meV_per_A": None,
                    "n_frames_e_gt_100meV_per_atom": 0,
                    "n_frames_f_gt_1000meV_per_A": 0}
        e = np.asarray(self.e_per_atom, dtype=np.float64) * 1000.0
        f = np.asarray(self.f_per_atom, dtype=np.float64) * 1000.0
        return {
            "n_frames": self.n_frames,
            "n_atoms": self.n_atoms,
            "mae_e_meV_per_atom": 1000.0 * self.e_abs_sum / max(self.n_atoms, 1),
            "mae_f_meV_per_A": 1000.0 * self.f_abs_sum / max(self.n_force_components, 1),
            "median_e_meV_per_atom": float(np.median(e)),
            "median_f_meV_per_A": float(np.median(f)),
            "p90_e_meV_per_atom": float(np.percentile(e, 90)),
            "p90_f_meV_per_A": float(np.percentile(f, 90)),
            "n_frames_e_gt_100meV_per_atom": int(np.sum(e > 100.0)),
            "n_frames_f_gt_1000meV_per_A": int(np.sum(f > 1000.0)),
        }


def _add(accs: Dict[str, Accumulator], key: str, e_abs: float, f_abs: float, n_atoms: int) -> None:
    accs.setdefault(key, Accumulator()).add(e_abs, f_abs, n_atoms)


def _fmt(v: Optional[float], nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def print_table(title: str, rows: Sequence[Tuple[str, Accumulator]]) -> None:
    print(f"\n{title}")
    print(f"{'stratum':34s} {'frames':>7s} {'atoms':>8s} {'MAE E':>12s} {'MAE F':>11s} "
          f"{'med E':>10s} {'med F':>10s} {'>100meV':>8s}")
    print("-" * 108)
    for label, acc in rows:
        m = acc.metrics()
        print(f"{label:34s} {m['n_frames']:7d} {m['n_atoms']:8d} "
              f"{_fmt(m['mae_e_meV_per_atom']):>12s} {_fmt(m['mae_f_meV_per_A']):>11s} "
              f"{_fmt(m['median_e_meV_per_atom']):>10s} "
              f"{_fmt(m['median_f_meV_per_A']):>10s} "
              f"{m['n_frames_e_gt_100meV_per_atom']:8d}")
    print("  E in meV/atom, F in meV/A; >100meV counts frames with E error > 100 meV/atom")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--valid", default=str(REPO / "data/valid.xyz"))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--max-frames", type=int, default=0,
                    help="evaluate a deterministic spread of at most N frames (0 = all)")
    ap.add_argument("--only-dataset", default=None, choices=["bulk", "gas"],
                    help="restrict evaluation to one dataset before subsampling")
    ap.add_argument("--speed", action="store_true",
                    help="measure calculator wall time and report us/atom/step")
    ap.add_argument("--out", default=None, help="write the results JSON here")
    ap.add_argument("--md", default=None, help="write a markdown summary here")
    args = ap.parse_args()

    if GRAPH_LONG_RANGE not in sys.path:
        sys.path.insert(0, GRAPH_LONG_RANGE)
    import torch
    from mace.calculators import mace_polar

    frames = _load_frames(args.valid, 0)
    if args.only_dataset:
        frames = [
            a for a in frames
            if str(a.info.get("dataset") or ("gas" if a.info.get("source") else "bulk"))
            == args.only_dataset
        ]
    if args.max_frames and args.max_frames < len(frames):
        idx = np.unique(np.linspace(0, len(frames) - 1, args.max_frames).round().astype(int))
        frames = [frames[i] for i in idx]
    if not frames:
        print(f"[eval] ERROR: no frames in {args.valid} after filtering", file=sys.stderr)
        return 2

    print(f"[eval] loading {args.model} on {args.device} ({args.dtype})")
    t_load = time.time()
    calc = mace_polar(model=args.model, device=args.device, default_dtype=args.dtype)
    model_load_s = time.time() - t_load
    print(f"[eval] model loaded in {model_load_s:.2f} s; evaluating {len(frames)} frames")

    record: Dict[str, Any] = {
        "model": str(Path(args.model).resolve()),
        "valid": str(Path(args.valid).resolve()),
        "device": args.device,
        "dtype": args.dtype,
        "n_frames_evaluated": len(frames),
        "max_frames": args.max_frames,
        "only_dataset": args.only_dataset,
        "model_load_s": model_load_s,
        "speed_requested": bool(args.speed),
    }

    overall = Accumulator()
    by_dataset: Dict[str, Accumulator] = {}
    by_species: Dict[str, Accumulator] = {}
    by_density: Dict[str, Accumulator] = {}
    by_temperature: Dict[str, Accumulator] = {}
    by_species_density: Dict[str, Accumulator] = {}
    by_species_temperature: Dict[str, Accumulator] = {}
    per_state: Dict[str, Dict[str, Any]] = collections.defaultdict(
        lambda: {"n_frames": 0, "e_abs_sum": 0.0, "n_atoms": 0,
                 "f_abs_sum": 0.0, "n_force_components": 0,
                 "dataset": None, "species": None,
                 "density": None, "temperature": None}
    )

    speed_atoms = 0
    speed_frames = 0
    t_predict = 0.0
    skipped = collections.Counter()

    for i, atoms in enumerate(frames, 1):
        info = atoms.info
        dataset = str(info.get("dataset") or ("gas" if info.get("source") else "bulk"))
        species = str(info.get("species") or atoms.get_chemical_formula())
        density = info.get("density_mol_A3")
        density = None if density is None else float(density)
        temperature = info.get("temperature_K")
        temperature = None if temperature is None else float(temperature)

        ref_e = info.get("REF_energy", info.get("energy"))
        ref_f = atoms.arrays.get("REF_forces", atoms.arrays.get("forces"))
        if ref_e is None or ref_f is None:
            skipped["missing_reference"] += 1
            continue
        ref_e = float(ref_e)
        ref_f = np.asarray(ref_f, dtype=np.float64)

        atoms.calc = calc
        t0 = time.perf_counter()
        try:
            e = float(atoms.get_potential_energy())
            f = np.asarray(atoms.get_forces(), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            skipped[f"eval_error:{type(exc).__name__}"] += 1
            continue
        if args.speed and i > 1:  # first frame is a warm-up, not timed
            if args.device == "cuda":
                torch.cuda.synchronize()
            t_predict += time.perf_counter() - t0
            speed_atoms += len(atoms)
            speed_frames += 1

        if not np.isfinite(e) or not np.isfinite(f).all():
            skipped["non_finite_prediction"] += 1
            continue

        n_atoms = len(atoms)
        e_abs = abs(e - ref_e)
        f_abs = float(np.abs(f - ref_f).sum())

        overall.add(e_abs, f_abs, n_atoms)
        _add(by_dataset, dataset, e_abs, f_abs, n_atoms)
        _add(by_species, f"{dataset}/{species}", e_abs, f_abs, n_atoms)
        if dataset == "bulk":
            db = _bin(density, DENSITY_BINS)
            tb = _bin(temperature, TEMPERATURE_BINS)
            _add(by_density, db, e_abs, f_abs, n_atoms)
            _add(by_temperature, tb, e_abs, f_abs, n_atoms)
            _add(by_species_density, f"{species} | {db}", e_abs, f_abs, n_atoms)
            _add(by_species_temperature, f"{species} | {tb}", e_abs, f_abs, n_atoms)

        state_id = str(info.get("state_id") or
                       (f"{species}|{density}|{temperature}" if dataset == "bulk"
                        else f"{species}|{info.get('source', '')}"))
        st = per_state[state_id]
        st["n_frames"] += 1
        st["e_abs_sum"] += e_abs
        st["n_atoms"] += n_atoms
        st["f_abs_sum"] += f_abs
        st["n_force_components"] += 3 * n_atoms
        st["dataset"] = dataset
        st["species"] = species
        st["density"] = density
        st["temperature"] = temperature

        if i % 200 == 0:
            print(f"[eval] {i}/{len(frames)} frames", flush=True)

    if not overall.n_frames:
        print("[eval] ERROR: no frames were evaluated", file=sys.stderr)
        return 2

    print_table("=== overall ===", [("all", overall)] +
                [(f"dataset={k}", v) for k, v in sorted(by_dataset.items())])
    print_table("=== per species (dataset/species) ===",
                [(k, v) for k, v in sorted(by_species.items())])
    print_table("=== bulk density bins ===",
                [(k, v) for k, v in sorted(by_density.items())])
    print_table("=== bulk temperature bins ===",
                [(k, v) for k, v in sorted(by_temperature.items())])
    print_table("=== bulk species x density bin ===",
                [(k, v) for k, v in sorted(by_species_density.items())])
    print_table("=== bulk species x temperature bin ===",
                [(k, v) for k, v in sorted(by_species_temperature.items())])

    # Explicit gas-only reference row (empty -> null, never fabricated).
    gas_acc = by_dataset.get("gas")
    gas_row = gas_acc.metrics() if gas_acc else {
        "n_frames": 0, "n_atoms": 0,
        "mae_e_meV_per_atom": None, "mae_f_meV_per_A": None,
    }
    print("\n=== gas-only reference ===")
    print(f"frames={gas_row['n_frames']} MAE E={_fmt(gas_row['mae_e_meV_per_atom'])} meV/atom "
          f"MAE F={_fmt(gas_row['mae_f_meV_per_A'])} meV/A")

    record["metrics"] = {
        "overall": overall.metrics(),
        "by_dataset": {k: v.metrics() for k, v in sorted(by_dataset.items())},
        "by_species": {k: v.metrics() for k, v in sorted(by_species.items())},
        "by_density_bin": {k: v.metrics() for k, v in sorted(by_density.items())},
        "by_temperature_bin": {k: v.metrics() for k, v in sorted(by_temperature.items())},
        "by_species_density_bin": {k: v.metrics() for k, v in sorted(by_species_density.items())},
        "by_species_temperature_bin": {
            k: v.metrics() for k, v in sorted(by_species_temperature.items())
        },
        "gas_only_reference": gas_row,
    }
    record["by_state"] = {
        k: {
            "dataset": v["dataset"], "species": v["species"],
            "density_mol_A3": v["density"], "temperature_K": v["temperature"],
            "n_frames": v["n_frames"],
            "mae_e_meV_per_atom": 1000.0 * v["e_abs_sum"] / max(v["n_atoms"], 1),
            "mae_f_meV_per_A": 1000.0 * v["f_abs_sum"] / max(v["n_force_components"], 1),
        }
        for k, v in sorted(per_state.items())
    }
    record["skipped"] = dict(skipped)

    if args.speed:
        speed = {
            "n_frames": speed_frames,
            "n_atoms": speed_atoms,
            "wall_s": t_predict,
            "us_per_atom_step": (
                1e6 * t_predict / speed_atoms if speed_atoms else None
            ),
            "note": (
                "PyTorch MACECalculator wall time per single-point evaluation "
                "(one MD step), model load and file I/O excluded; includes "
                "Python/calculator overhead and is not an engine benchmark."
            ),
        }
        record["speed"] = speed
        print(f"\n=== speed (measured, {speed_frames} frames) ===")
        print(f"{speed['us_per_atom_step']:.2f} us/atom/step "
              f"({speed_atoms} atoms, {t_predict:.3f} s total; calculator, not engine)")
    else:
        record["speed"] = None
        print("\n=== speed ===\nnot measured (pass --speed to measure)")

    if skipped:
        print(f"\n[eval] skipped frames: {dict(skipped)}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1))
        print(f"[eval] wrote {out}")

    if args.md:
        md = Path(args.md)
        md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Student bulk evaluation -- {Path(args.model).name}",
            "",
            f"- valid: `{record['valid']}`",
            f"- frames: {record['n_frames_evaluated']}",
            f"- device/dtype: {args.device}/{args.dtype}",
            "",
            "| stratum | frames | MAE E (meV/atom) | MAE F (meV/A) |",
            "|---|---:|---:|---:|",
        ]
        for label, acc in ([("all", overall)] +
                           [(f"dataset={k}", v) for k, v in sorted(by_dataset.items())] +
                           [(k, v) for k, v in sorted(by_species.items())] +
                           [(f"rho {k}", v) for k, v in sorted(by_density.items())] +
                           [(f"T {k}", v) for k, v in sorted(by_temperature.items())]):
            m = acc.metrics()
            lines.append(f"| {label} | {m['n_frames']} | "
                         f"{_fmt(m['mae_e_meV_per_atom'])} | {_fmt(m['mae_f_meV_per_A'])} |")
        if record.get("speed"):
            lines += ["", f"Speed (measured): **{record['speed']['us_per_atom_step']:.2f} "
                          f"us/atom/step** (PyTorch calculator, not engine)."]
        md.write_text("\n".join(lines) + "\n")
        print(f"[eval] wrote {md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
