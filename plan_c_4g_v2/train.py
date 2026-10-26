"""Train the Plan C 4G-HDNNP student on PolarMACE teacher labels.

Loss (per batch):
    L = mean(((E_graph - E_ref)/N_atoms)^2) + wF * mean((F - F_ref)^2)

The energy term uses the straight-through ``E_graph_train`` whose gradient is
the exact QEq-consistent derivative (envelope theorem), while its value is the
physical energy.  Forces come from the force trick.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plan_c_4g_v2.data import collate, load_frames, per_element_reference  # noqa: E402
from plan_c_4g_v2.descriptors import DescriptorConfig, ELEMENT_INDEX  # noqa: E402
from plan_c_4g_v2.model import FourGStudent, ModelConfig  # noqa: E402


def evaluate(model, frames, indices, batch_size, device, max_frames=None):
    model.eval()
    idxs = list(indices)
    if max_frames is not None:
        idxs = idxs[:max_frames]
    # group by identical cell (bulk frames span many densities)
    from collections import defaultdict as _dd
    _groups = _dd(list)
    for i in idxs:
        key = tuple(round(float(x), 5) for x in frames[i].cell.reshape(-1).tolist())
        _groups[key].append(i)
    e_abs, e_cnt, f_abs, f_cnt = 0.0, 0, 0.0, 0
    per_species = {}
    with torch.enable_grad():
        for gidx in _groups.values():
            for start in range(0, len(gidx), batch_size):
                chunk = gidx[start:start + batch_size]
                b = collate(frames, chunk, device)
                out = model.predict(b, compute_forces=True, create_graph=False)
                e_err = (out["E_graph"] - b["energy"]) / b["sizes"]
                f_err = (out["F"] - b["forces"])
                e_abs += float(e_err.abs().sum())
                e_cnt += e_err.numel()
                f_abs += float(f_err.abs().sum())
                f_cnt += f_err.numel()
                # per-species (frame names carry the species prefix)
                for j, i in enumerate(chunk):
                    sp = frames[i].name.split("_")[0]
                    d = per_species.setdefault(sp, [0.0, 0, 0.0, 0])
                    d[0] += float(e_err[j].abs())
                    d[1] += 1
                    mask = b["batch"] == j
                    d[2] += float(f_err[mask].abs().sum())
                    d[3] += int(mask.sum()) * 3
    model.train()
    res = {"E_mae_eV_per_atom": e_abs / max(e_cnt, 1),
           "F_mae_eV_per_A": f_abs / max(f_cnt, 1),
           "n_frames": e_cnt}
    res["per_species"] = {
        k: {"E_mae": v[0] / max(v[1], 1), "F_mae": v[2] / max(v[3], 1)}
        for k, v in sorted(per_species.items())
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str,
                    default="/workspace/group/shared/bza/symmetrix-polar/bench/"
                            "path_b_data2_q4_2026-10-22/attempt-01/data/train.xyz")
    ap.add_argument("--valid", type=str,
                    default="/workspace/group/shared/bza/symmetrix-polar/bench/"
                            "path_b_data2_q4_2026-10-22/attempt-01/data/valid.xyz")
    ap.add_argument("--extra-train", type=str, default="",
                    help="optional second extxyz file appended to train")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--force-weight", type=float, default=10.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-rbf", type=int, default=12)
    ap.add_argument("--ang-preset", type=int, choices=[8, 12, 16], default=8)
    ap.add_argument("--cg-iters", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--valid-every", type=int, default=10)
    ap.add_argument("--valid-frames", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--max-train-frames", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--tag", type=str, default="gas")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", choices=["fp64", "fp32"], default="fp64")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tdt = torch.float64 if args.dtype == "fp64" else torch.float32
    torch.set_default_dtype(tdt)
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    if args.ang_preset == 12:
        basis = tuple((e, z, l) for e in (0.3, 1.5, 6.0) for z in (2.0, 4.0)
                      for l in (1.0, -1.0))
    elif args.ang_preset == 16:
        basis = tuple((e, z, l) for e in (0.1, 0.5, 2.0, 6.0) for z in (2.0, 4.0)
                      for l in (1.0, -1.0))
    else:
        basis = DescriptorConfig().angular_basis
    cfg = DescriptorConfig(n_rbf=args.n_rbf, angular_basis=basis)
    print("descriptor dim", cfg.cont_dim, "angular basis", cfg.n_ang, flush=True)
    print("loading train ...", flush=True)
    train_frames = load_frames(args.train, cfg, verbose=True, dtype=tdt)
    if args.extra_train:
        print("loading extra train ...", flush=True)
        train_frames += load_frames(args.extra_train, cfg, verbose=True, dtype=tdt)
    print("loading valid ...", flush=True)
    valid_frames = load_frames(args.valid, cfg, verbose=True, dtype=tdt)
    if args.max_train_frames:
        train_frames = train_frames[:args.max_train_frames]
    train_idx = list(range(len(train_frames)))
    valid_idx = list(range(len(valid_frames)))
    print(f"train {len(train_idx)}  valid {len(valid_idx)}", flush=True)

    model = FourGStudent(cfg, ModelConfig(hidden=args.hidden,
                                          trunk_hidden=int(args.hidden * 0.75),
                                          cg_iters=args.cg_iters,
                                          cg_tol=1e-8)).to(device)
    eps = per_element_reference(train_frames, train_idx).to(device)
    model.set_element_reference(eps)
    model.fit_scaler(train_frames, train_idx, max_frames=600)
    print("eps", eps.tolist(), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_idx) // args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(total_steps, 1), eta_min=args.min_lr)

    best = {"E_mae_eV_per_atom": 1e9, "step": -1}
    history = []
    step = 0
    t0 = time.time()
    model.train()
    log_path = os.path.join(args.outdir, f"train_{args.tag}.jsonl")
    logf = open(log_path, "w")
    for epoch in range(args.epochs):
        random.shuffle(train_idx)
        for start in range(0, len(train_idx), args.batch_size):
            if args.max_steps and step >= args.max_steps:
                break
            chunk = train_idx[start:start + args.batch_size]
            if len(chunk) < 2:
                continue
            b = collate(train_frames, chunk, device)
            out = model.predict(b, compute_forces=True, create_graph=True)
            e_err = (out["E_graph_train"] - b["energy"]) / b["sizes"]
            f_err = out["F"] - b["forces"]
            loss_e = (e_err * e_err).mean()
            loss_f = (f_err * f_err).mean()
            loss = loss_e + args.force_weight * loss_f
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            if step % 10 == 0 or step == 1:
                rec = {"step": step, "epoch": epoch,
                       "loss": float(loss), "loss_e": float(loss_e),
                       "loss_f": float(loss_f),
                       "E_mae_batch": float(e_err.abs().mean()),
                       "F_mae_batch": float(f_err.abs().mean()),
                       "lr": float(sched.get_last_lr()[0]),
                       "wall_s": time.time() - t0}
                print(json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
            if step % (args.valid_every * steps_per_epoch) == 0 or step == total_steps:
                v = evaluate(model, valid_frames, valid_idx, args.batch_size,
                             device, max_frames=args.valid_frames or None)
                rec = {"step": step, "valid": v, "wall_s": time.time() - t0}
                print("VALID " + json.dumps(rec), flush=True)
                logf.write("VALID " + json.dumps(rec) + "\n")
                logf.flush()
                history.append(rec)
                if v["E_mae_eV_per_atom"] < best["E_mae_eV_per_atom"]:
                    best = {"E_mae_eV_per_atom": v["E_mae_eV_per_atom"],
                            "F_mae_eV_per_A": v["F_mae_eV_per_A"],
                            "step": step}
                    ckpt = os.path.join(args.outdir, f"plan_c_4g_v2_{args.tag}.pt")
                    torch.save({"model": model.state_dict(),
                                "descriptor_config": cfg.__dict__,
                                "model_config": model.mcfg.__dict__,
                                "eps": model.eps.detach().cpu(),
                                "valid": v, "step": step}, ckpt)
                print("BEST " + json.dumps(best), flush=True)
        if args.max_steps and step >= args.max_steps:
            break
    logf.close()
    summary = {"best": best, "steps": step, "wall_s": time.time() - t0,
               "history": history,
               "args": vars(args)}
    with open(os.path.join(args.outdir, f"train_{args.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("DONE " + json.dumps(summary["best"]), flush=True)


if __name__ == "__main__":
    main()
