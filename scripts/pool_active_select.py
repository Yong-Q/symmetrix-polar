#!/usr/bin/env python3
"""Pool-based committee active selection for the bulk corpus.

The MD-proposal path in plan_c_4g.active is correct but its per-step CPU
neighbour rebuild makes a production round very slow.  This script performs
the selection half on an existing unlabeled/labeled pool:

  1. score every pool frame by committee force variance;
  2. keep a deterministic ACF-like spacing per trajectory;
  3. FPS on force signatures for diversity;
  4. write the selected extxyz + a JSON audit.

The selected frames already carry REF_energy / REF_forces (the pool is the
teacher-labelled bulk corpus), so they can be merged into a Plan D training
file directly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plan_c_4g.active import fps_diversity, _write_xyz  # noqa: E402
from plan_c_4g.data import load_frames  # noqa: E402
from plan_c_4g.descriptors import DescriptorConfig  # noqa: E402
from plan_c_4g.model import FourGStudent, ModelConfig  # noqa: E402


def load_member(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    kw = dict(ck["model_config"])
    if "use_vdw" not in kw:
        kw["use_vdw"] = False
    if "include_self_energy" not in kw:
        kw["include_self_energy"] = False
    m = FourGStudent(DescriptorConfig(**ck["descriptor_config"]),
                     ModelConfig(**kw))
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"[pool] {path}: missing {list(missing)} "
              f"unexpected {list(unexpected)}", flush=True)
    return m.to(device).eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--pool-stride", type=int, default=8)
    ap.add_argument("--pool-limit", type=int, default=3000)
    ap.add_argument("--students", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-select", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tau-mult", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import ase.io
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Materialise a strided pool so load_frames sees only the selected slice.
    raw = ase.io.read(args.pool, index=":")
    raw = raw[::max(args.pool_stride, 1)][:args.pool_limit]
    pool_path = out / "pool_slice.xyz"
    _write_xyz(pool_path, raw)
    print(f"[pool] {len(raw)} frames -> {pool_path}", flush=True)

    cfg = DescriptorConfig()
    frames = load_frames(str(pool_path), cfg, verbose=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float64)
    members = [load_member(p, device) for p in args.students]
    print(f"[pool] {len(members)} committee members", flush=True)

    groups = defaultdict(list)
    for i, f in enumerate(frames):
        key = tuple(round(float(x), 5) for x in f.cell.reshape(-1).tolist())
        groups[key].append(i)

    scores = np.zeros(len(frames), dtype=float)
    e_std = np.zeros(len(frames), dtype=float)
    emb = np.zeros((len(frames), 10), dtype=float)   # 5 percentiles + 4 elements + log V
    with torch.enable_grad():
        for gi, gidx in enumerate(groups.values()):
            for start in range(0, len(gidx), args.batch_size):
                chunk = gidx[start:start + args.batch_size]
                from plan_c_4g.data import collate
                b = collate(frames, chunk, device)
                fs, es = [], []
                for m in members:
                    o = m.predict(b, compute_forces=True, create_graph=False)
                    fs.append(o["F"].detach().cpu().numpy())
                    es.append(o["E_graph"].detach().cpu().numpy())
                fs = np.stack(fs)                      # [M, N, 3]
                es = np.stack(es)                      # [M, G]
                sizes = b["sizes"].detach().cpu().numpy()
                off = 0
                for j, i in enumerate(chunk):
                    n = int(sizes[j])
                    f = fs[:, off:off + n, :]
                    scores[i] = float(f.var(axis=0).mean())
                    e_std[i] = float(es[:, j].std())
                    mag = np.linalg.norm(f[0], axis=1)
                    qs = np.percentile(mag, [5, 25, 50, 75, 95])
                    z = frames[i].z.cpu().numpy()
                    per_el = [float(mag[z == e].mean()) if np.any(z == e) else 0.0
                              for e in (1, 6, 7, 8)]
                    emb[i] = np.concatenate([qs, per_el,
                                             [np.log(max(float(frames[i].cell.detach().abs().det().item()), 1e-6))]])
                    off += n
            if (gi + 1) % 20 == 0:
                print(f"[pool] scored {gi+1}/{len(groups)} cell groups", flush=True)

    # Deterministic refractory spacing inside each state/trajectory.
    order = np.argsort(-scores)
    used = defaultdict(list)
    kept = []
    for i in order:
        f = frames[i]
        traj = str(f.info.get("traj_id") or f.info.get("state_id") or f.name)
        step = int(f.info.get("step", 0))
        win = max(1.0, args.tau_mult * 15.0)       # median bulk tau_E ~15 frames
        if any(abs(step - s) < win for s in used[traj]):
            continue
        kept.append(int(i))
        used[traj].append(step)
        if len(kept) >= max(args.max_select * 6, args.max_select):
            break

    pick_local = fps_diversity(emb[kept], scores[kept],
                               min(args.max_select, len(kept)))
    picked = [kept[j] for j in pick_local]
    # ``frames`` are plan_c_4g.data.Frame wrappers; write the original ASE
    # Atoms objects (same order, same slice) so the extxyz keeps all arrays.
    selected = [raw[i] for i in picked]
    _write_xyz(out / "selected.xyz", selected)

    by_species = defaultdict(int)
    by_density = defaultdict(int)
    for i in picked:
        by_species[frames[i].name.split("_")[0]] += 1
        by_density[f"{frames[i].info.get('density_mol_A3', 0.0):.4g}"] += 1
    summary = {
        "pool": args.pool,
        "pool_stride": args.pool_stride,
        "pool_frames": len(frames),
        "students": args.students,
        "n_kept": len(kept),
        "n_selected": len(picked),
        "score_quantiles": {
            "p50": float(np.percentile(scores, 50)),
            "p90": float(np.percentile(scores, 90)),
            "p99": float(np.percentile(scores, 99)),
            "max": float(scores.max()),
        },
        "energy_std_quantiles": {
            "p50": float(np.percentile(e_std, 50)),
            "p90": float(np.percentile(e_std, 90)),
            "max": float(e_std.max()),
        },
        "selected_species": dict(sorted(by_species.items())),
        "selected_density": dict(sorted(by_density.items())),
        "selected_indices": picked,
    }
    (out / "select.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
