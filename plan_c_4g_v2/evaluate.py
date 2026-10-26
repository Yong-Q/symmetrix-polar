"""Evaluate a Plan C 4G student checkpoint on extxyz data.

Reports energy MAE per atom and force MAE (meV/A), overall and grouped by
species / density / temperature.  Batches are formed only from frames with an
identical cell, so the PME grid is shared.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plan_c_4g_v2.data import collate, load_frames  # noqa: E402
from plan_c_4g_v2.descriptors import DescriptorConfig  # noqa: E402
from plan_c_4g_v2.model import FourGStudent, ModelConfig  # noqa: E402

HA_TO_MEV = 1000.0


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dtype = ck["model"]["trunk.0.weight"].dtype
    torch.set_default_dtype(dtype)
    dcfg = DescriptorConfig(**ck["descriptor_config"])
    mcfg = ModelConfig(**ck["model_config"])
    model = FourGStudent(dcfg, mcfg).to(device=device, dtype=dtype)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def cell_key(f):
    return tuple(round(float(x), 5) for x in f.cell.reshape(-1).tolist())


def group_indices(frames):
    groups = defaultdict(list)
    for i, f in enumerate(frames):
        groups[cell_key(f)].append(i)
    return groups


def evaluate(model, frames, batch_size, device, max_frames=None):
    idxs = list(range(len(frames)))
    if max_frames:
        idxs = idxs[:max_frames]
    groups = group_indices([frames[i] for i in idxs])
    # map local -> global
    globals_ = []
    for key, loc in groups.items():
        globals_.append((key, [idxs[j] for j in loc]))
    e_abs = 0.0
    e_sq = 0.0
    f_abs = 0.0
    f_sq = 0.0
    n_atoms = 0
    n_frames = 0
    per = defaultdict(lambda: [0.0, 0.0, 0, 0.0, 0.0, 0, 0.0])
    with torch.enable_grad():
        for key, gidx in globals_:
            for start in range(0, len(gidx), batch_size):
                chunk = gidx[start:start + batch_size]
                b = collate(frames, chunk, device)
                out = model.predict(b, compute_forces=True, create_graph=False)
                e_err = (out["E_graph"] - b["energy"]) / b["sizes"]
                f_err = out["F"] - b["forces"]
                e_abs += float(e_err.abs().sum())
                e_sq += float((e_err * e_err).sum())
                f_abs += float(f_err.abs().sum())
                f_sq += float((f_err * f_err).sum())
                n_atoms += int(b["sizes"].sum())
                n_frames += len(chunk)
                for j, gi in enumerate(chunk):
                    f = frames[gi]
                    info = f.info
                    sp = str(info.get("species", f.name.split("_")[0]))
                    dens = info.get("density_mol_A3", None)
                    temp = info.get("temperature_K", None)
                    keys = [("species", sp)]
                    if dens is not None:
                        keys.append(("density", round(float(dens), 8)))
                    if temp is not None:
                        keys.append(("temperature", round(float(temp), 3)))
                    mask = b["batch"] == j
                    for k in keys:
                        d = per[k]
                        d[0] += float(e_err[j].abs()); d[1] += 1
                        d[3] += float(f_err[mask].abs().sum()); d[4] += int(mask.sum()) * 3
                        d[6] += float(f.forces.pow(2).mean().sqrt())
    metrics = {
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "E_mae_eV_per_atom": e_abs / max(n_frames, 1),
        "E_rmse_eV_per_atom": (e_sq / max(n_frames, 1)) ** 0.5,
        "F_mae_eV_per_A": f_abs / max(n_atoms * 3, 1),
        "F_rmse_eV_per_A": (f_sq / max(n_atoms * 3, 1)) ** 0.5,
        "groups": {},
    }
    for k, d in sorted(per.items(), key=lambda kv: str(kv[0])):
        metrics["groups"]["/".join(str(x) for x in k)] = {
            "n_frames": d[1],
            "E_mae_meV_per_atom": d[0] / max(d[1], 1) * HA_TO_MEV,
            "F_mae_meV_per_A": d[3] / max(d[4], 1) * HA_TO_MEV,
            "teacher_F_rms_eV_per_A": d[6] / max(d[1], 1),
        }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.ckpt, device)
    print(f"checkpoint step={ck.get('step')} valid={ck.get('valid')}")
    results = {}
    for path in args.files:
        for p in sorted(glob.glob(path)):
            print(f"evaluating {p} ...", flush=True)
            cfg = model.dcfg
            frames = load_frames(p, cfg, max_frames=args.max_frames or None,
                                 dtype=next(model.parameters()).dtype)
            m = evaluate(model, frames, args.batch_size, device)
            results[p] = m
            print(f"  {m['n_frames']} frames / {m['n_atoms']} atoms  "
                  f"E {m['E_mae_eV_per_atom']*1000:.3f} meV/atom  "
                  f"F {m['F_mae_eV_per_A']*1000:.3f} meV/A")
            for k, v in m["groups"].items():
                print(f"    {k:34s} E {v['E_mae_meV_per_atom']:9.3f} meV/atom  "
                      f"F {v['F_mae_meV_per_A']:9.3f} meV/A  "
                      f"(teacher F rms {v['teacher_F_rms_eV_per_A']:.3f})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print("json:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
