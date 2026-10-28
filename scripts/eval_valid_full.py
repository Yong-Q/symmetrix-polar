#!/usr/bin/env python3
"""One pass over the held-out valid set: per-species E/F plus the dense-CO2
state breakdown (the 23.62 / 11.5 meV/A calibers reported in the README).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import ase.io
import numpy as np
from mace.calculators import MACECalculator

REPO = Path("/workspace/group/shared/bza/symmetrix-polar")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--valid", default=str(REPO / "data_20261025/valid.xyz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    calc = MACECalculator(model_paths=args.model, device=args.device,
                          default_dtype="float64")
    frames = ase.io.read(args.valid, index=":")
    print(f"[valid] {len(frames)} frames", flush=True)

    acc = defaultdict(lambda: {"n_frames": 0, "n_atoms": 0,
                               "se": 0.0, "sf": 0.0, "nf": 0})
    co2 = []
    for i, atoms in enumerate(frames):
        atoms.calc = calc
        e_pred = float(atoms.get_potential_energy())
        f_pred = atoms.get_forces()
        e_ref = float(atoms.info["REF_energy"])
        f_ref = np.asarray(atoms.arrays["REF_forces"])
        de = abs(e_pred - e_ref) / len(atoms) * 1e3
        d = np.abs(f_pred - f_ref)
        df = float(d.mean()) * 1e3
        species = str(atoms.info.get("config_type", "")).split("_")[0] or "?"
        dataset = str(atoms.info.get("dataset", "")) or "?"
        for key in ("ALL", f"species:{species}", f"dataset:{dataset}",
                    f"cross:{species}|{dataset}"):
            a = acc[key]
            a["n_frames"] += 1
            a["n_atoms"] += len(atoms)
            a["se"] += de * len(atoms)
            a["sf"] += df * len(atoms)
            a["nf"] += len(atoms)
        if species == "CO2" and dataset == "bulk":
            co2.append({
                "state_id": str(atoms.info.get("state_id", "")),
                "density": float(atoms.info.get("density_mol_A3", np.nan)),
                "temperature": float(atoms.info.get("temperature_K", np.nan)),
                "n_atoms": len(atoms),
                "e_err": (e_pred - e_ref) / len(atoms) * 1e3,
                "f_mae": df,
                "f_ref_mean": float(np.abs(f_ref).mean()) * 1e3,
            })
        if (i + 1) % 500 == 0:
            print(f"[valid] {i+1}/{len(frames)}", flush=True)

    res = {k: {"n_frames": a["n_frames"], "n_atoms": a["n_atoms"],
               "mae_e_meV_per_atom": a["se"] / a["n_atoms"],
               "mae_f_meV_per_A": a["sf"] / a["nf"]}
           for k, a in acc.items()}

    by_state = defaultdict(list)
    for r in co2:
        by_state[r["state_id"]].append(r)
    states = []
    for sid, rs in by_state.items():
        w = np.array([r["n_atoms"] for r in rs], dtype=float)
        states.append({
            "state_id": sid, "density": rs[0]["density"],
            "temperature": rs[0]["temperature"],
            "n_frames": len(rs), "n_atoms": int(w.sum()),
            "e_abs_mean": float(np.mean([abs(r["e_err"]) for r in rs])),
            "f_mae": float(np.average([r["f_mae"] for r in rs], weights=w)),
            "f_ref_mean": float(np.average([r["f_ref_mean"] for r in rs],
                                           weights=w)),
        })
    states.sort(key=lambda s: (s["density"], s["temperature"]))
    w_all = np.array([r["n_atoms"] for r in co2], dtype=float)
    f_all = float(np.average([r["f_mae"] for r in co2], weights=w_all))
    keep = [r for r in co2 if not (abs(r["density"] - 0.02) < 1e-9
                                   and abs(r["temperature"] - 150.0) < 1e-6)]
    w_keep = np.array([r["n_atoms"] for r in keep], dtype=float)
    f_keep = float(np.average([r["f_mae"] for r in keep], weights=w_keep))

    res["co2_dense"] = {
        "n_frames": len(co2),
        "f_mae_all": f_all,
        "f_mae_excl_002_150K": f_keep,
        "n_frames_excl": len(keep),
        "states": states,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if k != "co2_dense" else
                          {kk: vv for kk, vv in v.items() if kk != "states"})
                      for k, v in res.items()
                      if k.startswith(("ALL", "species:", "dataset:", "co2"))},
                     indent=1), flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
