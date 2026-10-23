#!/usr/bin/env python3
"""Merge gas + bulk teacher extxyz, validate, split at the trajectory-state level.

A *state* is one physical trajectory:

  bulk: (species, density_mol_A3, temperature_K[, seed/traj_id])
        Frames written by more than one run at the same thermodynamic point
        are merged into one state unless an explicit seed/traj_id is present,
        so a state can never leak across train/valid.
  gas : (species, md<temperature>K) for the 800-step Langevin trajectories
        (the ``source`` field is only a frame index, not a trajectory id);
        each rattle / equilibrium config is its own single-frame state.

The output manifest reports raw frame counts, the state split, the
species x density x temperature coverage, and an integrated-autocorrelation
effective-sample-size estimate.  Short states (fewer than --min-frames-for-acf
frames) cannot support an ACF estimate; their N_eff is reported as the raw
frame count and is an *upper bound*, not evidence of independence.

Usage (the MACE environment is required):
  /opt/matos-tools/conda/envs/matos-compute/bin/python \
      scripts/prepare_bulk_dataset.py
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ase
import ase.io

REPO = Path(__file__).resolve().parents[1]

DEFAULT_BULK_GLOBS = [
    str(REPO / "bench/path_b_bulk_q4_2026-10-23/attempt-01/data/*.xyz"),
    str(REPO / "bench/path_b_bulk2_a_2026-10-23/attempt-01/data/*.xyz"),
    str(REPO / "bench/path_b_bulk2_b_2026-10-23/attempt-01/data/*.xyz"),
    str(REPO / "bench/path_b_bulk2_c_2026-10-23/attempt-01/data/*.xyz"),
]
DEFAULT_GAS_TRAIN = str(
    REPO / "bench/path_b_data2_q4_2026-10-22/attempt-01/data/train.xyz"
)
DEFAULT_GAS_VALID = str(
    REPO / "bench/path_b_data2_q4_2026-10-22/attempt-01/data/valid.xyz"
)

TRAJ_ID_KEYS = ("traj_id", "trajectory_id", "seed", "step_block")
MD_SOURCE_RE = re.compile(r"^md(?P<temp>[-+0-9.eE]+)K(?:_(?P<idx>\d+))?$")
LOG_TEMPS = (150.0, 250.0, 400.0, 800.0, 1200.0, 2000.0)
LOG_DENSITIES = (1e-4, 5e-4, 2e-3, 1e-2, 3.5e-2, 1e-1)


# --------------------------------------------------------------------------- #
# frames and state keys
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    atoms: ase.Atoms
    dataset: str            # "gas" or "bulk"
    species: str
    density: Optional[float]
    temperature: Optional[float]
    step: Optional[float]
    source: str
    origin: str             # run/task directory that produced the file
    state_key: Tuple[Any, ...]
    traj_key: Tuple[Any, ...]
    group_key: Tuple[Any, ...]
    acf_order: float


def _origin_tag(path: Path) -> str:
    """Run/task directory name, e.g. path_b_bulk2_a_2026-10-23."""
    parts = path.resolve().parts
    if "bench" in parts:
        i = parts.index("bench")
        if i + 1 < len(parts):
            return parts[i + 1]
    return path.parent.name


def _species_of(atoms: ase.Atoms, info: Dict[str, Any], path: Path) -> str:
    species = info.get("species")
    if species is not None:
        return str(species)
    config_type = str(info.get("config_type", ""))
    if "_" in config_type:
        return config_type.split("_")[0]
    formula = atoms.get_chemical_formula()
    print(f"[prepare] WARNING: no species metadata in {path}; using {formula}")
    return formula


def _explicit_traj_token(info: Dict[str, Any]) -> str:
    for key in TRAJ_ID_KEYS:
        if key in info and info[key] not in (None, ""):
            return f"{key}={info[key]}"
    return ""


def _gas_kind_and_traj(source: str) -> Tuple[str, str]:
    m = MD_SOURCE_RE.match(source)
    if m:
        return "md", f"md{m.group('temp')}K"
    if source.startswith("md"):
        # e.g. md200K_seed3_07: keep everything but the frame index
        head = re.sub(r"_\d+$", "", source)
        return "md", head
    if source.startswith("rattle"):
        return "rattle", source
    if source == "equilibrium":
        return "static", source
    return "static", source or "unlabelled"


def make_frame(atoms: ase.Atoms, dataset: str, path: Path) -> Frame:
    info = dict(atoms.info)
    species = _species_of(atoms, info, path)
    origin = _origin_tag(path)

    if dataset == "bulk":
        density = info.get("density_mol_A3")
        temperature = info.get("temperature_K")
        density = None if density is None else float(density)
        temperature = None if temperature is None else float(temperature)
        step = info.get("step")
        step = None if step is None else float(step)
        token = _explicit_traj_token(info)
        state_key = (
            "bulk",
            species,
            round(density, 10) if density is not None else None,
            round(temperature, 6) if temperature is not None else None,
            token,
        )
        # Independent replicas of the same thermodynamic point are merged for
        # the split (no seed in the current data), but kept apart for ACF.
        traj_key = state_key + (origin,)
        group_key = ("bulk", species, "bulk")
        acf_order = step if step is not None else 0.0
        source = str(info.get("config_type", ""))
    else:
        density = None
        temperature = info.get("temperature_K")
        if temperature is None and "md_temperature_K" in info:
            temperature = info["md_temperature_K"]
        temperature = None if temperature is None else float(temperature)
        source = str(info.get("source", info.get("config_type", "")))
        kind, traj = _gas_kind_and_traj(source)
        token = _explicit_traj_token(info)
        state_key = ("gas", species, kind, traj, token)
        traj_key = state_key
        group_key = ("gas", species, kind)
        m = MD_SOURCE_RE.match(source)
        if m and m.group("idx") is not None:
            acf_order = float(m.group("idx"))
        else:
            step = info.get("step")
            acf_order = float(step) if step is not None else 0.0
        step = info.get("step")
        step = None if step is None else float(step)

    return Frame(
        atoms=atoms,
        dataset=dataset,
        species=species,
        density=density,
        temperature=temperature,
        step=step,
        source=source,
        origin=origin,
        state_key=state_key,
        traj_key=traj_key,
        group_key=group_key,
        acf_order=acf_order,
    )


def classify(atoms: ase.Atoms, path: Path) -> str:
    info = atoms.info
    if "density_mol_A3" in info or "temperature_K" in info:
        return "bulk"
    if "source" in info or str(info.get("config_type", "")).startswith(
        ("md", "rattle", "equilibrium")
    ):
        return "gas"
    # Fallback: a 16 A single-molecule box is the gas dataset convention.
    cell = np.asarray(atoms.cell.array, dtype=float)
    if len(atoms) <= 10 and np.allclose(np.diag(cell), 16.0, atol=1e-6):
        return "gas"
    raise ValueError(
        f"cannot classify {path}: no density_mol_A3/source metadata and cell "
        f"is not the 16 A gas box"
    )


# --------------------------------------------------------------------------- #
# validation / de-duplication
# --------------------------------------------------------------------------- #
def frame_hash(atoms: ase.Atoms, decimals: int = 6) -> str:
    species = ",".join(atoms.get_chemical_symbols())
    pos = np.round(np.asarray(atoms.positions, dtype=np.float64), decimals)
    cell = np.round(np.asarray(atoms.cell.array, dtype=np.float64), decimals)
    h = hashlib.sha256()
    h.update(species.encode())
    h.update(pos.tobytes())
    h.update(cell.tobytes())
    return h.hexdigest()


def validate_frame(atoms: ase.Atoms, min_dist: float) -> Tuple[bool, str, Dict[str, Any]]:
    """Return (ok, reason, diagnostics).  Checks labels, cell and min distance."""
    info = atoms.info
    energy = info.get("REF_energy", info.get("energy"))
    if energy is None:
        return False, "missing_energy", {}
    try:
        energy = float(energy)
    except (TypeError, ValueError):
        return False, "non_numeric_energy", {}
    if not np.isfinite(energy):
        return False, "non_finite_energy", {}

    forces = atoms.arrays.get("REF_forces", atoms.arrays.get("forces"))
    if forces is None:
        return False, "missing_forces", {}
    forces = np.asarray(forces, dtype=np.float64)
    if forces.shape != (len(atoms), 3):
        return False, "bad_forces_shape", {"forces_shape": list(forces.shape)}
    if not np.isfinite(forces).all():
        return False, "non_finite_forces", {}

    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    volume = float(abs(np.linalg.det(cell)))
    if not np.isfinite(cell).all() or volume < 1e-6:
        return False, "bad_cell", {"cell_volume": volume}

    try:
        d = atoms.get_all_distances(mic=bool(np.any(atoms.pbc)))
    except Exception as exc:  # noqa: BLE001
        return False, "distance_error", {"error": str(exc)}
    np.fill_diagonal(d, np.inf)
    rmin = float(d.min()) if d.size else float("inf")
    if not np.isfinite(rmin):
        return False, "distance_error", {}
    if rmin < min_dist:
        return False, "min_distance", {"min_distance_A": rmin}

    return True, "", {
        "min_distance_A": rmin,
        "cell_volume_A3": volume,
        "energy_eV": energy,
        "fmax_eV_A": float(np.abs(forces).max()),
    }


# --------------------------------------------------------------------------- #
# autocorrelation effective sample size
# --------------------------------------------------------------------------- #
def acf_tau_neff(series: Sequence[float], min_frames: int) -> Optional[Dict[str, Any]]:
    """Integrated autocorrelation time and N_eff for a scalar time series.

    Uses the standard biased ACF and the initial-positive-sequence truncation
    with a Sokal window (stop once lag > 5*tau).  Returns None when the series
    is too short or constant.
    """
    x = np.asarray(series, dtype=np.float64)
    n = x.size
    if n < max(2, min_frames):
        return None
    x = x - x.mean()
    var = float(np.dot(x, x) / n)
    if not np.isfinite(var) or var <= 1e-24:
        return None
    f = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(f * np.conj(f), n=2 * n)[:n].real / n
    acf = acov / acov[0]
    tau = 1.0
    max_lag = n - 1
    for k in range(1, max_lag + 1):
        rho = float(acf[k])
        if rho <= 0.0:
            break
        tau += 2.0 * rho
        if k >= max(4.0, 5.0 * tau):
            break
    tau = max(1.0, tau)
    neff = max(1.0, n / tau)
    return {
        "n_frames": int(n),
        "tau_int": float(tau),
        "neff": float(neff),
        "acf_lag1": float(acf[1]) if n > 1 else None,
    }


def _state_series(frames: Sequence[Frame]) -> Tuple[List[float], List[float]]:
    energy_per_atom: List[float] = []
    force_rms: List[float] = []
    for f in frames:
        n = len(f.atoms)
        e = f.atoms.info.get("REF_energy", f.atoms.info.get("energy"))
        forces = f.atoms.arrays.get("REF_forces", f.atoms.arrays.get("forces"))
        energy_per_atom.append(float(e) / max(n, 1))
        force_rms.append(float(np.sqrt(np.mean(np.asarray(forces, dtype=np.float64) ** 2))))
    return energy_per_atom, force_rms


def autocorrelation_report(
    frames: Sequence[Frame], min_frames: int
) -> Dict[str, Any]:
    by_state: Dict[Tuple[Any, ...], Dict[Tuple[Any, ...], List[Frame]]] = (
        collections.defaultdict(lambda: collections.defaultdict(list))
    )
    for f in frames:
        by_state[f.state_key][f.traj_key].append(f)

    states_out: Dict[str, Any] = {}
    for state_key in sorted(by_state, key=lambda k: repr(k)):
        traj_entries = []
        state_frames = 0
        state_eff_upper = 0.0
        state_est_frames = 0
        state_eff_est = 0.0
        for traj_key in sorted(by_state[state_key], key=lambda k: repr(k)):
            tr = sorted(by_state[state_key][traj_key], key=lambda f: (f.acf_order, f.source))
            e_series, f_series = _state_series(tr)
            e_acf = acf_tau_neff(e_series, min_frames)
            f_acf = acf_tau_neff(f_series, min_frames)
            n = len(tr)
            if e_acf is None and f_acf is None:
                neff = float(n)
                tau_e = tau_f = None
                estimable = False
            else:
                tau_e = e_acf["tau_int"] if e_acf else None
                tau_f = f_acf["tau_int"] if f_acf else None
                # conservative: use the slower-decaying observable
                tau = max([t for t in (tau_e, tau_f) if t is not None])
                neff = max(1.0, n / tau)
                estimable = True
            state_frames += n
            state_eff_upper += neff
            if estimable:
                state_est_frames += n
                state_eff_est += neff
            traj_entries.append(
                {
                    "traj_key": [str(x) for x in traj_key],
                    "origin": tr[0].origin if tr else "",
                    "n_frames": n,
                    "step_min": min((f.step for f in tr if f.step is not None), default=None),
                    "step_max": max((f.step for f in tr if f.step is not None), default=None),
                    "acf_estimable": estimable,
                    "tau_int_energy": tau_e,
                    "tau_int_force": tau_f,
                    "neff": neff,
                }
            )
        states_out[repr(state_key)] = {
            "state_key": [str(x) for x in state_key],
            "dataset": state_key[0],
            "species": state_key[1],
            "kind": state_key[2] if state_key[0] == "gas" else "bulk",
            "n_frames": state_frames,
            "n_eff_upper": state_eff_upper,
            "n_estimable_frames": state_est_frames,
            "n_eff_estimable": state_eff_est,
            "trajectories": traj_entries,
        }

    def _aggregate(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        n_frames = sum(e["n_frames"] for e in entries)
        n_eff_upper = sum(e["n_eff_upper"] for e in entries)
        n_est_frames = sum(e["n_estimable_frames"] for e in entries)
        n_eff_est = sum(e["n_eff_estimable"] for e in entries)
        tau_e_values: List[float] = []
        tau_f_values: List[float] = []
        n_traj = 0
        n_est_traj = 0
        neff_values: List[float] = []
        for e in entries:
            for tr in e["trajectories"]:
                n_traj += 1
                if tr["acf_estimable"]:
                    n_est_traj += 1
                    neff_values.append(tr["neff"])
                    if tr["tau_int_energy"] is not None:
                        tau_e_values.append(tr["tau_int_energy"])
                    if tr["tau_int_force"] is not None:
                        tau_f_values.append(tr["tau_int_force"])
        return {
            "n_frames": n_frames,
            "n_eff_upper": n_eff_upper,
            "effective_fraction_upper": n_eff_upper / n_frames if n_frames else None,
            "n_estimable_trajectories": n_est_traj,
            "n_trajectories": n_traj,
            "n_estimable_frames": n_est_frames,
            "n_eff_estimable": n_eff_est,
            "effective_fraction_estimable": (
                n_eff_est / n_est_frames if n_est_frames else None
            ),
            "median_tau_int_energy": (
                float(np.median(tau_e_values)) if tau_e_values else None
            ),
            "median_tau_int_force": (
                float(np.median(tau_f_values)) if tau_f_values else None
            ),
            "estimable_neff_min": min(neff_values) if neff_values else None,
            "estimable_neff_median": (
                float(np.median(neff_values)) if neff_values else None
            ),
            "estimable_neff_max": max(neff_values) if neff_values else None,
        }

    entries = list(states_out.values())
    by_dataset = {
        ds: _aggregate([e for e in entries if e["dataset"] == ds])
        for ds in sorted({e["dataset"] for e in entries})
    }
    groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for e in entries:
        groups[f"{e['dataset']}/{e['species']}/{e['kind']}"].append(e)
    by_group = {k: _aggregate(v) for k, v in sorted(groups.items())}
    return {
        "aggregate": _aggregate(entries),
        "by_dataset": by_dataset,
        "by_group": by_group,
        "states": states_out,
    }


# --------------------------------------------------------------------------- #
# coverage / split
# --------------------------------------------------------------------------- #
def state_sort_key(state_key: Tuple[Any, ...]) -> Tuple[Any, ...]:
    if state_key[0] == "bulk":
        return (
            state_key[1],
            state_key[2] if state_key[2] is not None else -1.0,
            state_key[3] if state_key[3] is not None else -1.0,
            str(state_key[4]),
        )
    kind = state_key[2]
    kind_order = {"static": 0, "md": 1, "rattle": 2}.get(kind, 3)
    return (state_key[1], kind_order, str(state_key[3]), str(state_key[4]))


def select_valid_states(
    frames: Sequence[Frame], valid_fraction: float, seed: int
) -> Tuple[set, Dict[str, List[Any]]]:
    """Stratified whole-state holdout per (dataset, species, kind)."""
    groups: Dict[Tuple[Any, ...], set] = collections.defaultdict(set)
    for f in frames:
        groups[f.group_key].add(f.state_key)

    k = max(2, int(round(1.0 / max(valid_fraction, 1e-6))))
    valid_states: set = set()
    detail: Dict[str, List[Any]] = {}
    for group_key in sorted(groups, key=lambda g: repr(g)):
        states = sorted(groups[group_key], key=state_sort_key)
        n = len(states)
        chosen: List[Tuple[Any, ...]] = []
        if n >= 2:
            offset = seed % k
            chosen = [s for i, s in enumerate(states) if i % k == offset]
            if not chosen:
                chosen = [states[seed % n]]
        valid_states.update(chosen)
        detail[repr(group_key)] = [
            {"state_key": [str(x) for x in s],
             "n_frames": sum(1 for f in frames if f.state_key == s),
             "split": "valid" if s in valid_states else "train"}
            for s in states
        ]
    return valid_states, detail


def coverage_report(frames: Sequence[Frame], valid_states: set) -> Dict[str, Any]:
    cov: Dict[str, Any] = {}
    for dataset, species in sorted({(f.dataset, f.species) for f in frames}):
        subset = [f for f in frames if f.dataset == dataset and f.species == species]
        by_state: Dict[Tuple[Any, ...], List[Frame]] = collections.defaultdict(list)
        for f in subset:
            by_state[f.state_key].append(f)
        entry: Dict[str, Any] = {
            "n_states": len(by_state),
            "n_states_train": sum(1 for s in by_state if s not in valid_states),
            "n_states_valid": sum(1 for s in by_state if s in valid_states),
            "n_frames": len(subset),
            "n_frames_train": sum(len(v) for s, v in by_state.items() if s not in valid_states),
            "n_frames_valid": sum(len(v) for s, v in by_state.items() if s in valid_states),
        }
        if dataset == "bulk":
            densities = sorted({f.density for f in subset if f.density is not None})
            temps = sorted({f.temperature for f in subset if f.temperature is not None})
            entry["densities_mol_A3"] = densities
            entry["temperatures_K"] = temps
            grid: Dict[str, Any] = {}
            for rho in densities:
                for temp in temps:
                    match = [s for s in by_state
                             if s[2] is not None and s[3] is not None
                             and abs(float(s[2]) - rho) < 1e-12
                             and abs(float(s[3]) - temp) < 1e-9]
                    if not match:
                        continue
                    key = f"{rho:.6g}|{temp:.6g}"
                    grid[key] = {
                        "n_states": len(match),
                        "n_states_train": sum(1 for s in match if s not in valid_states),
                        "n_states_valid": sum(1 for s in match if s in valid_states),
                        "n_frames": sum(len(by_state[s]) for s in match),
                    }
            entry["density_temperature_grid"] = grid
            entry["n_frames_by_density"] = {
                f"{rho:.6g}": sum(len(by_state[s]) for s in by_state
                                  if s[2] is not None and abs(float(s[2]) - rho) < 1e-12)
                for rho in densities
            }
            entry["n_frames_by_temperature"] = {
                f"{temp:.6g}": sum(len(by_state[s]) for s in by_state
                                   if s[3] is not None and abs(float(s[3]) - temp) < 1e-9)
                for temp in temps
            }
        else:
            kinds = collections.Counter(s[2] for s in by_state)
            entry["state_kinds"] = dict(kinds)
            entry["n_frames_by_kind"] = {
                kind: sum(len(v) for s, v in by_state.items() if s[2] == kind)
                for kind in kinds
            }
        cov[f"{dataset}/{species}"] = entry
    return cov


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def write_extxyz(path: Path, frames: Sequence[Frame], keep_momenta: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    n_written = 0
    for f in frames:
        atoms = f.atoms
        atoms.info["dataset"] = f.dataset
        atoms.info["state_id"] = repr(f.state_key)
        atoms.info["traj_id"] = repr(f.traj_key)
        atoms.info.setdefault("species", f.species)
        if not keep_momenta and "momenta" in atoms.arrays:
            del atoms.arrays["momenta"]
        ase.io.write(str(path), atoms, format="extxyz", append=True)
        n_written += 1
        if n_written % 5000 == 0:
            print(f"[prepare] wrote {n_written} frames to {path}", flush=True)
    print(f"[prepare] wrote {n_written} frames to {path}", flush=True)


def _fmt(value: Optional[float], nd: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{nd}g}"


def print_coverage_table(frames: Sequence[Frame], valid_states: set, cov: Dict[str, Any]) -> None:
    print("\n=== coverage (states are whole trajectories) ===")
    header = (
        f"{'dataset/species':18s} {'states':>7s} {'tr':>5s} {'va':>5s} "
        f"{'frames':>8s} {'tr':>8s} {'va':>8s}  densities / temperatures"
    )
    print(header)
    print("-" * len(header))
    for key, e in cov.items():
        if key.startswith("bulk/"):
            rho_txt = (f"[{_fmt(e['densities_mol_A3'][0])}.."
                       f"{_fmt(e['densities_mol_A3'][-1])}]"
                       if e["densities_mol_A3"] else "[]")
            t_txt = (f"[{_fmt(e['temperatures_K'][0])}.."
                     f"{_fmt(e['temperatures_K'][-1])}]"
                     if e["temperatures_K"] else "[]")
            extra = (f"rho={len(e['densities_mol_A3'])} {rho_txt} "
                     f"T={len(e['temperatures_K'])} {t_txt}")
        else:
            extra = "kinds=" + ",".join(f"{k}:{v}" for k, v in sorted(e["state_kinds"].items()))
        print(
            f"{key:18s} {e['n_states']:7d} {e['n_states_train']:5d} {e['n_states_valid']:5d} "
            f"{e['n_frames']:8d} {e['n_frames_train']:8d} {e['n_frames_valid']:8d}  {extra}"
        )

    bulk_keys = [k for k in cov if k.startswith("bulk/")]
    if bulk_keys:
        densities = sorted({d for k in bulk_keys for d in cov[k]["densities_mol_A3"]})
        temps = sorted({t for k in bulk_keys for t in cov[k]["temperatures_K"]})
        print("\nbulk state counts, all species pooled: rows=density, cols=temperature")
        print("  rho\\T      " + " ".join(f"{t:>10.0f}" for t in temps))
        for rho in densities:
            cells = []
            for temp in temps:
                ns = ntr = nva = 0
                for k in bulk_keys:
                    g = cov[k]["density_temperature_grid"].get(f"{rho:.6g}|{temp:.6g}")
                    if g:
                        ns += g["n_states"]
                        ntr += g["n_states_train"]
                        nva += g["n_states_valid"]
                cells.append(f"{ns:2d}({ntr:2d}/{nva:2d})" if ns else f"{'-':>10s}")
            print(f"  {rho:<10.4g} " + " ".join(f"{c:>10s}" for c in cells))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bulk", nargs="*", default=None,
                    help="bulk extxyz glob(s); default: pilot + shards A/B/C")
    ap.add_argument("--gas-train", default=DEFAULT_GAS_TRAIN, help="gas extxyz (merged; re-split by state)")
    ap.add_argument("--gas-valid", default=DEFAULT_GAS_VALID, help="gas extxyz (merged; re-split by state)")
    ap.add_argument("--no-gas", action="store_true", help="do not include the gas data")
    ap.add_argument("--out-dir", default=str(REPO / "data"))
    ap.add_argument("--valid-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--min-dist", type=float, default=0.5,
                    help="reject frames with a minimum-image distance below this (A)")
    ap.add_argument("--min-frames-for-acf", type=int, default=5)
    ap.add_argument("--keep-momenta", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="debug: stop after N kept frames (0 = all)")
    args = ap.parse_args()

    t0 = time.time()
    bulk_globs = args.bulk if args.bulk is not None else DEFAULT_BULK_GLOBS
    files: List[Tuple[str, Path]] = []
    for pattern in bulk_globs:
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"[prepare] WARNING: no files match bulk glob {pattern}")
        files.extend(("bulk", Path(p)) for p in matches)
    if not args.no_gas:
        for p in (args.gas_train, args.gas_valid):
            if p and Path(p).exists():
                files.append(("gas", Path(p)))
            elif p:
                print(f"[prepare] WARNING: gas file missing: {p}")
    if not files:
        print("[prepare] ERROR: no input files found", file=sys.stderr)
        return 2

    frames_read = 0
    dup_removed = 0
    invalid: Dict[str, int] = collections.Counter()
    seen_hashes: Dict[str, str] = {}
    kept: List[Frame] = []
    rmin_values: List[float] = []
    files_meta = []

    for dataset, path in files:
        print(f"[prepare] reading {dataset}: {path}", flush=True)
        try:
            atoms_list = ase.io.read(str(path), index=":", format="extxyz")
        except Exception as exc:  # noqa: BLE001
            print(f"[prepare] WARNING: cannot read {path}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue
        if isinstance(atoms_list, ase.Atoms):
            atoms_list = [atoms_list]
        files_meta.append({
            "path": str(path),
            "dataset": dataset,
            "n_frames": len(atoms_list),
            "sha256": _file_sha256(path),
        })
        for atoms in atoms_list:
            frames_read += 1
            h = frame_hash(atoms)
            if h in seen_hashes:
                dup_removed += 1
                continue
            ok, reason, diag = validate_frame(atoms, args.min_dist)
            if not ok:
                invalid[reason] += 1
                continue
            seen_hashes[h] = str(path)
            f = make_frame(atoms, dataset, path)
            f.atoms.info["_source_file"] = str(path)
            kept.append(f)
            rmin_values.append(diag["min_distance_A"])
            if args.max_frames and len(kept) >= args.max_frames:
                break
        if args.max_frames and len(kept) >= args.max_frames:
            print(f"[prepare] debug --max-frames {args.max_frames} reached")
            break

    if not kept:
        print("[prepare] ERROR: no valid frames", file=sys.stderr)
        return 2

    valid_states, split_detail = select_valid_states(kept, args.valid_fraction, args.seed)
    train = [f for f in kept if f.state_key not in valid_states]
    valid = [f for f in kept if f.state_key in valid_states]
    if not valid:
        print("[prepare] WARNING: valid split is empty; check --valid-fraction")
    if not train:
        print("[prepare] ERROR: train split is empty", file=sys.stderr)
        return 2

    print(f"[prepare] frames read={frames_read} duplicates={dup_removed} "
          f"invalid={sum(invalid.values())} kept={len(kept)}")
    print(f"[prepare] states: train={len({f.state_key for f in train})} "
          f"valid={len(valid_states)} | frames: train={len(train)} valid={len(valid)}")

    acf = autocorrelation_report(kept, args.min_frames_for_acf)
    cov = coverage_report(kept, valid_states)
    print_coverage_table(kept, valid_states, cov)

    a = acf["aggregate"]
    print("\n=== independent-sample estimate (integrated autocorrelation) ===")
    print(f"states with >= {args.min_frames_for_acf} frames: "
          f"{a['n_estimable_trajectories']}/{a['n_trajectories']} trajectories")
    print(f"median tau_int: E={_fmt(a['median_tau_int_energy'])} frames, "
          f"F={_fmt(a['median_tau_int_force'])} frames")
    print(f"all frames: N_eff_upper={a['n_eff_upper']:.1f}/{a['n_frames']} "
          f"({100.0 * (a['effective_fraction_upper'] or 0.0):.1f}%) "
          f"[short states counted as independent -> upper bound]")
    print(f"estimable frames only: N_eff={a['n_eff_estimable']:.1f}/"
          f"{a['n_estimable_frames']} "
          f"({100.0 * (a['effective_fraction_estimable'] or 0.0):.1f}%)")
    for ds, agg in acf["by_dataset"].items():
        print(f"  {ds}: N_eff_upper={agg['n_eff_upper']:.1f}/{agg['n_frames']} "
              f"({100.0 * (agg['effective_fraction_upper'] or 0.0):.1f}%), "
              f"estimable {agg['n_eff_estimable']:.1f}/{agg['n_estimable_frames']} "
              f"({100.0 * (agg['effective_fraction_estimable'] or 0.0):.1f}%)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.xyz"
    valid_path = out_dir / "valid.xyz"
    write_extxyz(train_path, train, args.keep_momenta)
    write_extxyz(valid_path, valid, args.keep_momenta)

    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "repo": str(REPO),
        "params": {
            "bulk_globs": bulk_globs,
            "gas_train": None if args.no_gas else args.gas_train,
            "gas_valid": None if args.no_gas else args.gas_valid,
            "valid_fraction": args.valid_fraction,
            "seed": args.seed,
            "min_dist_A": args.min_dist,
            "min_frames_for_acf": args.min_frames_for_acf,
            "keep_momenta": args.keep_momenta,
        },
        "inputs": files_meta,
        "counts": {
            "frames_read": frames_read,
            "duplicates_removed": dup_removed,
            "invalid_total": int(sum(invalid.values())),
            "invalid_by_reason": dict(invalid),
            "frames_kept": len(kept),
            "atoms_total": int(sum(len(f.atoms) for f in kept)),
            "min_distance_A_min": float(min(rmin_values)) if rmin_values else None,
            "min_distance_A_p01": float(np.percentile(rmin_values, 1)) if rmin_values else None,
        },
        "split": {
            "train": {
                "n_frames": len(train),
                "n_states": len({f.state_key for f in train}),
                "n_atoms": int(sum(len(f.atoms) for f in train)),
            },
            "valid": {
                "n_frames": len(valid),
                "n_states": len(valid_states),
                "n_atoms": int(sum(len(f.atoms) for f in valid)),
            },
            "method": (
                "whole-state holdout, stratified by (dataset, species, kind); "
                "every k-th state in density-temperature order with k=round(1/valid_fraction); "
                "at least one state per stratum when >=2 states exist"
            ),
            "valid_states": [list(s) for s in sorted(valid_states, key=state_sort_key)],
            "groups": split_detail,
        },
        "coverage": cov,
        "autocorrelation": acf,
        "outputs": {
            "train_xyz": str(train_path),
            "valid_xyz": str(valid_path),
            "train_sha256": _file_sha256(train_path),
            "valid_sha256": _file_sha256(valid_path),
        },
    }
    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"\n[prepare] wrote {manifest_path}")
    print(f"[prepare] train={train_path} ({len(train)} frames), "
          f"valid={valid_path} ({len(valid)} frames)")
    print(f"[prepare] done in {time.time() - t0:.1f} s")
    return 0


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
