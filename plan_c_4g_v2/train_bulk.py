"""Bulk-density fine-tuning for the Plan C 4G student.

Batches are formed only inside a cell group (the PME grid is shared), and the
bulk holdout is split at the (species, density, temperature) state level so
frames from the same trajectory never leak across the split.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plan_c_4g_v2.data import collate, load_frames, per_element_reference  # noqa: E402
from plan_c_4g_v2.descriptors import DescriptorConfig, angular_basis_preset  # noqa: E402
from plan_c_4g_v2.model import FourGStudent, ModelConfig  # noqa: E402
from plan_c_4g_v2.train import evaluate  # noqa: E402


def cell_key(f):
    return tuple(round(float(x), 5) for x in f.cell.reshape(-1).tolist())


def state_key(f):
    i = f.info
    return (str(i.get("species", f.name.split("_")[0])),
            round(float(i.get("density_mol_A3", 0.0)), 8),
            round(float(i.get("temperature_K", 0.0)), 3))


def split_states(frames, valid_fraction, seed):
    states = sorted({state_key(f) for f in frames})
    rng = random.Random(seed)
    rng.shuffle(states)
    n_valid = max(1, int(round(len(states) * valid_fraction)))
    valid_states = set(states[:n_valid])
    train_idx, valid_idx = [], []
    for i, f in enumerate(frames):
        (valid_idx if state_key(f) in valid_states else train_idx).append(i)
    return train_idx, valid_idx, sorted(valid_states)


def make_batches(frames, indices, batch_size, rng=None):
    groups = defaultdict(list)
    for i in indices:
        groups[cell_key(frames[i])].append(i)
    batches = []
    for g in groups.values():
        g = list(g)
        if rng is not None:
            rng.shuffle(g)
        for s in range(0, len(g), batch_size):
            batches.append(g[s:s + batch_size])
    if rng is not None:
        rng.shuffle(batches)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gas-train", default="")
    ap.add_argument("--gas-valid", default="")
    ap.add_argument("--bulk", nargs="+", default=[])
    ap.add_argument("--bulk-train-file", default="",
                    help="pre-split bulk train extxyz; skips the internal state split")
    ap.add_argument("--bulk-valid-file", default="",
                    help="pre-split bulk valid extxyz; required with --bulk-train-file")
    ap.add_argument("--init", default="")
    ap.add_argument("--valid-fraction", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--force-weight", type=float, default=20.0)
    ap.add_argument("--cg-iters", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=160)
    ap.add_argument("--trunk-layers", type=int, default=1,
                    help="hidden layers in the shared trunk (1 = original)")
    ap.add_argument("--property-layers", type=int, default=1,
                    help="layers in each chi/J head (1 = original)")
    ap.add_argument("--head-hidden", type=int, default=64)
    ap.add_argument("--property-hidden", type=int, default=32)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--valid-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--tag", default="bulk")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--refit-scaler", action="store_true")
    ap.add_argument("--gas-every", type=int, default=4)
    ap.add_argument("--gas-batch", type=int, default=16)
    ap.add_argument("--select", choices=["gas", "bulk", "mixed"], default="bulk")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ang-preset", type=int, choices=[8, 16, 24], default=8)
    ap.add_argument("--n-rbf", type=int, default=12)
    ap.add_argument("--r-cut-ang", type=float, default=3.5)
    ap.add_argument("--use-ee4g", action="store_true",
                    help="append the QEq electrostatic potential as a local feature")
    ap.add_argument("--no-vdw", action="store_true")
    # RuNNer2 4G fixes / two-stage training.
    ap.add_argument("--physical-init", action="store_true",
                    help="initialise element chi/J from Pearson values (RuNNer2 4G)")
    ap.add_argument("--no-self-energy", action="store_true",
                    help="ablation: omit the Gaussian self term from QEq/energy")
    ap.add_argument("--charge-stage-epochs", type=int, default=0,
                    help="stage-1 epochs on teacher charges (0 = no charge stage)")
    ap.add_argument("--charge-weight", type=float, default=0.0,
                    help="stage-2 charge-loss weight (0 = energy/force only)")
    ap.add_argument("--charge-lr", type=float, default=1e-3)
    ap.add_argument("--charge-batch-size", type=int, default=8)
    ap.add_argument("--charge-frames", type=int, default=0,
                    help="limit stage-1 frames (0 = all frames with REF_charges)")
    ap.add_argument("--charge-train-file", default="",
                    help="stage-1 extxyz with REF_charges (bulk)")
    ap.add_argument("--charge-gas-file", default="",
                    help="stage-1 extxyz with REF_charges (gas)")
    ap.add_argument("--charge-scale", type=float, default=1.0,
                    help="multiplier on REF_charges; add_teacher_charges.py "
                         "already applies the 4*pi coefficient->e conversion")
    ap.add_argument("--charge-loss-scale", type=float, default=1e4,
                    help="stage-1 loss multiplier (keeps gradients above Adam eps)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)
    cfg = DescriptorConfig(n_rbf=args.n_rbf,
                           r_cut_ang=args.r_cut_ang,
                           angular_basis=angular_basis_preset(args.ang_preset))

    bulk_frames = []
    if args.bulk_train_file:
        if not args.bulk_valid_file:
            raise SystemExit("--bulk-train-file requires --bulk-valid-file")
        print("loading pre-split bulk train", args.bulk_train_file, flush=True)
        bulk_frames = load_frames(args.bulk_train_file, cfg, verbose=True)
        print("loading pre-split bulk valid", args.bulk_valid_file, flush=True)
        bulk_valid_frames = load_frames(args.bulk_valid_file, cfg, verbose=True)
        b_train = list(range(len(bulk_frames)))
        b_valid = list(range(len(bulk_valid_frames)))
        valid_states = sorted({state_key(f) for f in bulk_valid_frames})
        print(f"pre-split bulk frames {len(bulk_frames)} train {len(b_train)} "
              f"valid {len(b_valid)}")
        print("valid states:", valid_states)
    else:
        for pat in args.bulk:
            for p in sorted(glob.glob(pat)):
                print("loading", p, flush=True)
                bulk_frames += load_frames(p, cfg, verbose=True)
        if not bulk_frames:
            raise SystemExit("no bulk data found")
        b_train, b_valid, valid_states = split_states(bulk_frames,
                                                      args.valid_fraction, args.seed)
        bulk_valid_frames = bulk_frames
        print(f"bulk frames {len(bulk_frames)} train {len(b_train)} valid {len(b_valid)}")
        print("valid states:", valid_states)

    gas_train, gas_valid = [], []
    if args.gas_train:
        gas_train = load_frames(args.gas_train, cfg, verbose=True)
    if args.gas_valid:
        gas_valid = load_frames(args.gas_valid, cfg, verbose=True)

    if args.init:
        if (args.ang_preset != 8 or args.n_rbf != 12
                or abs(args.r_cut_ang - 3.5) > 1e-9 or args.use_ee4g):
            raise SystemExit(
                "--init cannot change the descriptor or enable ee4G; train the "
                "new architecture from scratch or provide a matching checkpoint")
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        mcfg_kwargs = dict(ck["model_config"])
        # Old checkpoints predate the vdW ablation and were trained without it;
        # do not silently create randomly initialised v_heads on load.
        if "use_vdw" not in mcfg_kwargs:
            mcfg_kwargs["use_vdw"] = False
        mcfg = ModelConfig(**mcfg_kwargs)
        mcfg.include_self_energy = not args.no_self_energy
        mcfg.physical_init = bool(args.physical_init)
        mcfg.use_ee4g = bool(args.use_ee4g)
        if args.no_vdw:
            mcfg.use_vdw = False
        model = FourGStudent(DescriptorConfig(**ck["descriptor_config"]), mcfg)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        model = model.to(device)
        print("initialised from", args.init, "step", ck.get("step"))
        if missing or unexpected:
            print("checkpoint load: missing", list(missing),
                  "unexpected", list(unexpected))
    else:
        model = FourGStudent(cfg, ModelConfig(hidden=args.hidden,
                                              trunk_hidden=int(args.hidden * 0.75),
                                              head_hidden=args.head_hidden,
                                              property_hidden=args.property_hidden,
                                              n_trunk_layers=args.trunk_layers,
                                              n_property_layers=args.property_layers,
                                              cg_iters=args.cg_iters,
                                              cg_tol=1e-8,
                                              include_self_energy=not args.no_self_energy,
                                              physical_init=bool(args.physical_init),
                                              use_ee4g=bool(args.use_ee4g),
                                              use_vdw=not args.no_vdw)).to(device)
        all_ref = list(gas_train) + list(bulk_frames)
        eps = per_element_reference(all_ref, list(range(len(all_ref))))
        model.set_element_reference(eps)
        # combined scaler: stratified sample of gas + bulk (all densities)
        gas_sample = gas_train[:min(len(gas_train), 300)]
        bulk_sample = bulk_frames[:min(len(bulk_frames), 500)]
        sample = list(gas_sample) + list(bulk_sample)
        model.fit_scaler(sample, list(range(len(sample))))
        print("fitted combined scaler on", len(sample), "frames")
    if args.refit_scaler:
        refit = bulk_frames[:min(len(bulk_frames), 400)]
        n = len(refit) + min(len(gas_train), 200)
        model.fit_scaler(refit + gas_train[:200], list(range(n)))
        print("refit scaler on", n, "frames")
    else:
        print("keeping the scaler from the initial checkpoint")

    rng = random.Random(args.seed)

    # ---- stage 1: RuNNer2 4G charge training ---------------------------
    # Train chi/J (and the shared trunk) on teacher reference charges.  This
    # is the missing "two-step" half of the 4G recipe: without it, chi/J are
    # only an indirect by-product of the energy/force loss.
    if args.charge_stage_epochs > 0:
        if args.charge_train_file:
            qframes = load_frames(args.charge_train_file, cfg, verbose=True)
            if args.charge_gas_file:
                qframes += load_frames(args.charge_gas_file, cfg, verbose=True)
        else:
            pool = [bulk_frames[i] for i in b_train] + list(gas_train)
            qframes = [f for f in pool if f.charges is not None]
        qframes = [f for f in qframes if f.charges is not None]
        if not qframes:
            raise SystemExit("--charge-stage-epochs requires REF_charges in the data")
        if args.charge_frames:
            qframes = qframes[:args.charge_frames]
        print(f"stage 1: {len(qframes)} frames with REF_charges, "
              f"{args.charge_stage_epochs} epochs", flush=True)
        # Keep the shared trunk frozen: it already carries the c1 local
        # features, and the charge stage only needs to learn chi/J.  Training
        # the trunk on a charge-only loss was the destabilising path.
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(
                ("chi_head", "j_head", "chi_elem", "j_elem"))
        opt_q = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.charge_lr, weight_decay=1e-6)
        logq = open(os.path.join(args.outdir, f"train_{args.tag}_charge.jsonl"), "w")
        tq0 = time.time()
        for ep in range(args.charge_stage_epochs):
            model.train()
            qb = make_batches(qframes, list(range(len(qframes))),
                              args.charge_batch_size, rng)
            q_abs, q_cnt = 0.0, 0
            for chunk in qb:
                b = collate(qframes, chunk, device)
                if b["charges"] is None:
                    continue
                out = model.predict(b, compute_forces=False)
                q_ref = b["charges"] * args.charge_scale
                q_surr = model.charge_surrogate(out, q_ref)
                loss_q_raw = ((q_surr - q_ref) ** 2).mean()
                loss_q = loss_q_raw * args.charge_loss_scale
                opt_q.zero_grad(set_to_none=True)
                loss_q.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.grad_clip)
                opt_q.step()
                q_abs += float((q_surr - q_ref).abs().sum())
                q_cnt += q_ref.numel()
            rec = {"stage": "charge", "epoch": ep,
                   "loss_q": float(loss_q) if q_cnt else None,
                   "loss_q_raw": float(loss_q_raw) if q_cnt else None,
                   "q_mae": q_abs / max(q_cnt, 1),
                   "wall_s": time.time() - tq0}
            print("CHARGE " + json.dumps(rec), flush=True)
            logq.write(json.dumps(rec) + "\n"); logq.flush()
        logq.close()
        for p in model.parameters():
            p.requires_grad = True
        print("stage 1 complete; unfroze all parameters", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    batches_per_epoch = len(make_batches(bulk_frames, b_train, args.batch_size))
    total_steps = args.epochs * batches_per_epoch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1),
                                                       eta_min=args.min_lr)
    try:
        import tqdm  # noqa: F401
        have_tqdm = True
    except Exception:
        have_tqdm = False

    best = {"E": 1e9, "step": -1}
    step = 0
    t0 = time.time()
    logf = open(os.path.join(args.outdir, f"train_{args.tag}.jsonl"), "w")
    history = []
    for epoch in range(args.epochs):
        model.train()
        batches = make_batches(bulk_frames, b_train, args.batch_size, rng)
        # gas batches (single 16 A cell group) mixed in at 25% of the bulk rate
        gas_batches = (make_batches(gas_train, list(range(len(gas_train))),
                                    max(args.batch_size, args.gas_batch), rng)
                       if gas_train else [])
        for bi, chunk in enumerate(batches):
            if args.max_steps and step >= args.max_steps:
                break
            use_gas = bool(gas_batches) and (bi % max(args.gas_every, 1) == 0)
            frames = gas_train if use_gas else bulk_frames
            idxs = rng.choice(gas_batches) if use_gas else chunk
            if len(idxs) < 2:
                continue
            b = collate(frames, idxs, device)
            out = model.predict(b, compute_forces=True, create_graph=True)
            e_err = (out["E_graph_train"] - b["energy"]) / b["sizes"]
            f_err = out["F"] - b["forces"]
            loss_e = (e_err * e_err).mean()
            loss_f = (f_err * f_err).mean()
            loss = loss_e + args.force_weight * loss_f
            loss_q = None
            if args.charge_weight > 0.0 and b["charges"] is not None:
                q_ref = b["charges"] * args.charge_scale
                q_surr = model.charge_surrogate(out, q_ref)
                loss_q = ((q_surr - q_ref) ** 2).mean()
                loss = loss + args.charge_weight * loss_q
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            if step % 10 == 0 or step == 1:
                rec = {"step": step, "epoch": epoch, "gas": bool(use_gas),
                       "loss": float(loss), "loss_e": float(loss_e),
                       "loss_f": float(loss_f),
                       "loss_q": None if loss_q is None else float(loss_q),
                       "E_mae_batch": float(e_err.abs().mean()),
                       "F_mae_batch": float(f_err.abs().mean()),
                       "lr": float(sched.get_last_lr()[0]),
                       "wall_s": time.time() - t0}
                print(json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + "\n"); logf.flush()
        if (epoch + 1) % args.valid_every == 0 or epoch == args.epochs - 1:
            vb = evaluate(model, bulk_valid_frames, b_valid, args.batch_size, device)
            vg = (evaluate(model, gas_valid, list(range(len(gas_valid))),
                           args.batch_size, device) if gas_valid else None)
            rec = {"step": step, "epoch": epoch, "valid_bulk": vb,
                   "valid_gas": vg, "wall_s": time.time() - t0}
            print("VALID " + json.dumps(rec), flush=True)
            logf.write("VALID " + json.dumps(rec) + "\n"); logf.flush()
            history.append(rec)
            if args.select == "gas" and vg is not None:
                score = vg["E_mae_eV_per_atom"]
            elif args.select == "mixed" and vg is not None:
                score = vb["E_mae_eV_per_atom"] + vg["E_mae_eV_per_atom"]
            else:
                score = vb["E_mae_eV_per_atom"]
            if score < best["E"]:
                best = {"E": score, "F": vb["F_mae_eV_per_A"],
                        "bulk_E": vb["E_mae_eV_per_atom"],
                        "bulk_F": vb["F_mae_eV_per_A"],
                        "gas_E": None if vg is None else vg["E_mae_eV_per_atom"],
                        "gas_F": None if vg is None else vg["F_mae_eV_per_A"],
                        "step": step, "epoch": epoch}
                torch.save({"model": model.state_dict(),
                            "descriptor_config": model.dcfg.__dict__,
                            "model_config": model.mcfg.__dict__,
                            "eps": model.eps.detach().cpu(),
                            "valid_bulk": vb, "valid_gas": vg,
                            "step": step}, os.path.join(args.outdir,
                                                        f"plan_c_4g_v2_{args.tag}.pt"))
            print("BEST " + json.dumps(best), flush=True)
        if args.max_steps and step >= args.max_steps:
            break
    torch.save({"model": model.state_dict(),
                "descriptor_config": model.dcfg.__dict__,
                "model_config": model.mcfg.__dict__,
                "eps": model.eps.detach().cpu(),
                "step": step}, os.path.join(args.outdir,
                                            f"plan_c_4g_v2_{args.tag}_final.pt"))
    logf.close()
    summary = {"best": best, "steps": step, "wall_s": time.time() - t0,
               "history": history, "args": vars(args)}
    with open(os.path.join(args.outdir, f"train_{args.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("DONE " + json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
