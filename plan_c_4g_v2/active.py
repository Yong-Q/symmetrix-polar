#!/usr/bin/env python3
"""ACF-aware committee active learning for the 4G-HDNNP distillate.

What the previous round did wrong (bench/path_b_active_q4_2026-10-22):
  * "diversity" was `spacing=3` on candidate frame indices. The measured bulk
    integrated autocorrelation time is tau_int(E) ~ 15 frames (median), so
    frames 3 apart are still ~5x correlated -- the selection did not produce
    independent samples;
  * candidates came only from single molecules in a 16 A box. No bulk density
    was ever proposed, so the loop could not fix the dense-state failure that
    actually mattered, and random bulk expansion beat it.

This script fixes both:
  * candidates are generated per (species, density, temperature) state of the
    deployment grid, using the current committee, so the proposal distribution
    matches where the student will be used;
  * selection is greedy committee-disagreement with a hard refractory window of
    `--tau-mult * tau_int` frames inside each trajectory, so retained frames
    are ~independent by the same ACF metric used to audit the dataset;
  * an FPS diversity pass over a force-descriptor embedding prevents the
    ranking from collapsing onto one local environment;
  * labels come from the FP64 PolarMACE teacher and are written to a new
    round directory; nothing is silently merged into the training set.

Stages:
  propose : generate -> score -> select -> teacher-label
  merge   : append a round's labeled frames to a base train file (explicit)

Example:
  python -m plan_c_4g_v2.active --stage propose --round 1 \
      --students results_al/plan_c_4g_v2_4g_c1.pt \
      --outdir bench/plan_c_4g_v2_2026-10-23/active/r1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Deployment grid the distillate must cover (matches the bulk campaign design).
TARGET_DENSITIES = (1e-4, 5e-4, 2e-3, 5e-3, 1e-2, 2e-2, 3.5e-2)
TARGET_TEMPERATURES = (150.0, 250.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 2000.0)
SPECIES = ("H2O", "NH3", "C2H4", "CH4", "C2H6", "CO2")


def _write_xyz(path, frames):
    """Write variable-size Atoms frames without stacking arrays."""
    import ase.io
    from io import StringIO
    with open(path, "w") as fh:
        for atoms in frames:
            buf = StringIO()
            ase.io.write(buf, atoms, format="extxyz")
            fh.write(buf.getvalue())


def _random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def build_state_boxes(species, densities, temperatures, n_mol, seed):
    """One deterministic starting box per (species, density, temperature)."""
    import ase
    from ase.build import molecule

    rng = np.random.default_rng(seed)
    states = []
    for sp in species:
        base = molecule(sp)
        base.center(vacuum=0.0)
        sym = list(base.get_chemical_symbols())
        ref = base.get_positions()
        for rho in densities:
            box = (n_mol / rho) ** (1.0 / 3.0)
            for temp in temperatures:
                if rho >= 2e-2 and temp < 250.0:
                    continue          # glassy / non-ergodic corner, skipped
                chunks = []
                for _ in range(n_mol):
                    rot = _random_rotation(rng)
                    off = rng.random(3) * box
                    chunks.append(np.mod(ref @ rot.T + off, box))
                atoms = ase.Atoms(symbols=sym * n_mol, positions=np.vstack(chunks),
                                  cell=[box] * 3, pbc=True)
                atoms.info.update({
                    "species": sp, "density_mol_A3": float(rho),
                    "temperature_K": float(temp),
                    "traj": f"{sp}|{rho:.6g}|{temp:.0f}",
                })
                states.append(atoms)
    return states


class _FourGCalculator:
    """Minimal ASE-style calculator for a single 4G committee member.

    Deliberately not an ase.calculators.Calculator subclass: the MD loop only
    needs get_forces()/get_potential_energy() on the *same* Atoms object, and
    avoiding ASE's results cache makes the reused-geometry contract explicit.
    """

    def __init__(self, model, cfg, device, torch):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.torch = torch

    def _eval(self, atoms):
        import numpy as np
        from plan_c_4g_v2.descriptors import species_to_index
        from plan_c_4g_v2.neighbors import build_neighbor_list

        t = self.torch
        pos = t.tensor(atoms.get_positions(), dtype=t.float64, device=self.device)
        cell = t.tensor(atoms.get_cell().array, dtype=t.float64, device=self.device)
        zi = species_to_index(t.tensor(atoms.get_atomic_numbers(),
                                       dtype=t.long, device=self.device))
        nl_r = build_neighbor_list(pos, cell, self.cfg.r_cut_rad)
        nl_a = build_neighbor_list(pos, cell, self.cfg.r_cut_ang)
        batch = t.zeros(pos.shape[0], dtype=t.long, device=self.device)
        pos = pos.requires_grad_(True)
        out = self.model.forward(pos, cell.unsqueeze(0), zi, nl_r, nl_a, batch, 1,
                                 compute_forces=True, create_graph=False)
        return (float(out["E"]),
                out["F"].detach().cpu().numpy().astype(np.float64))

    def get_potential_energy(self, atoms=None):
        return self._eval(atoms)[0]

    def get_forces(self, atoms=None):
        return self._eval(atoms)[1]

    def get_stress(self, atoms=None):
        import numpy as np
        return np.zeros(6)


def integrated_tau(series, dump_every):
    """Initial-positive-sequence integrated autocorrelation time, in frames."""
    arr = np.asarray(series, dtype=float)
    if len(arr) < 6 or np.allclose(arr, arr[0]):
        return 1.0
    arr = arr - arr.mean()
    ac = np.correlate(arr, arr, "full")[len(arr) - 1:]
    ac = ac / ac[0]
    tau = 1.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        tau += 2.0 * ac[k]
    return max(1.0, tau) * dump_every


def fps_diversity(embeddings, prior_scores, n_pick, weights=None):
    """Max-min FPS on normalized embeddings, seeded by highest disagreement."""
    emb = np.asarray(embeddings, dtype=np.float64)
    emb = emb - emb.mean(axis=0, keepdims=True)
    scale = emb.std(axis=0, keepdims=True)
    emb = emb / np.where(scale > 0, scale, 1.0)
    n = len(emb)
    d = np.full(n, np.inf)
    picked = [int(np.argmax(prior_scores))]
    d = np.minimum(d, np.linalg.norm(emb - emb[picked[0]], axis=1))
    while len(picked) < min(n_pick, n):
        cand = np.where(np.isfinite(d), d, -np.inf)
        # break exact ties with the disagreement score
        best = np.max(cand)
        ties = np.flatnonzero(cand >= best - 1e-12)
        nxt = int(ties[np.argmax(np.asarray(prior_scores)[ties])])
        picked.append(nxt)
        d = np.minimum(d, np.linalg.norm(emb - emb[nxt], axis=1))
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["propose", "merge"], required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--students", nargs="+", default=[])
    ap.add_argument("--teacher", default="/workspace/group/shared/bza/"
                    "task3_sampling_database/mlip/models/mace-polar/MACE-POLAR-1-M.model")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--base-train", default="")
    ap.add_argument("--n-mol", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--dump-every", type=int, default=10)
    ap.add_argument("--tau-mult", type=float, default=1.0)
    ap.add_argument("--max-select", type=int, default=240)
    ap.add_argument("--species", nargs="+", default=list(SPECIES))
    ap.add_argument("--densities", nargs="+", type=float, default=list(TARGET_DENSITIES))
    ap.add_argument("--temperatures", nargs="+", type=float,
                    default=list(TARGET_TEMPERATURES))
    ap.add_argument("--charge-scale", type=float, default=4.0 * np.pi,
                    help="teacher l=0 coefficient -> elementary charge scale")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "merge":
        return _merge(args, out)
    if len(args.students) < 2:
        raise SystemExit("propose needs >=2 committee checkpoints for disagreement")
    return _propose(args, out)


def _propose(args, out: Path) -> int:
    import torch
    import ase.units as units
    import ase.io
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    from plan_c_4g_v2.descriptors import DescriptorConfig, species_to_index
    from plan_c_4g_v2.model import FourGStudent, ModelConfig
    from plan_c_4g_v2.neighbors import build_neighbor_list

    torch.set_default_dtype(torch.float64)
    device = args.device if torch.cuda.is_available() else "cpu"
    cfg = DescriptorConfig()

    members = []
    for ckpt in args.students:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        mcfg_kwargs = dict(ck["model_config"])
        if "use_vdw" not in mcfg_kwargs:
            mcfg_kwargs["use_vdw"] = False
        if "include_self_energy" not in mcfg_kwargs:
            # Old checkpoints were trained without the self term; preserve
            # their original QEq operator when they are used as committee
            # members rather than silently changing their forces.
            mcfg_kwargs["include_self_energy"] = False
        m = FourGStudent(DescriptorConfig(**ck["descriptor_config"]),
                         ModelConfig(**mcfg_kwargs))
        missing, unexpected = m.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            print(f"[active] {ckpt}: missing {list(missing)} "
                  f"unexpected {list(unexpected)}", flush=True)
        members.append(m.to(device).eval())
        print(f"[active] loaded {ckpt} (step {ck.get('step')})", flush=True)

    def score_atoms(atoms):
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=device)
        cell = torch.tensor(atoms.get_cell().array, dtype=torch.float64, device=device)
        zi = species_to_index(torch.tensor(atoms.get_atomic_numbers(),
                                           dtype=torch.long, device=device))
        nl_r = build_neighbor_list(pos, cell, cfg.r_cut_rad)
        nl_a = build_neighbor_list(pos, cell, cfg.r_cut_ang)
        batch = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
        pos = pos.requires_grad_(True)
        energies, forces = [], []
        for m in members:
            o = m.forward(pos, cell.unsqueeze(0), zi, nl_r, nl_a, batch, 1,
                          compute_forces=True, create_graph=False)
            energies.append(float(o["E"]))
            forces.append(o["F"].detach().cpu().numpy())
        es = np.array(energies)
        fs = np.stack(forces)
        return es, fs

    states = build_state_boxes(args.species, args.densities, args.temperatures,
                               args.n_mol, seed=20261025 + args.round)
    print(f"[active] {len(states)} (species, rho, T) states", flush=True)

    candidates, scores, taus = [], [], {}
    for si, state in enumerate(states):
        atoms = state.copy()
        T = state.info["temperature_K"]
        MaxwellBoltzmannDistribution(atoms, temperature_K=T)
        atoms.calc = _FourGCalculator(members[0], cfg, device, torch)
        ase_units = units
        dyn = Langevin(atoms, timestep=0.5 * ase_units.fs, temperature_K=T,
                       friction=0.01, fixcm=False)
        traj = state.info["traj"]
        e_hist = []
        for step in range(args.n_steps + 1):
            if step % args.dump_every == 0:
                es, fs = score_atoms(atoms)
                e_hist.append(es.mean())
                frame = atoms.copy()
                frame.info.update(state.info)
                frame.info["active_round"] = args.round
                frame.info["frame_index"] = step
                candidates.append(frame)
                scores.append(float(fs.var(axis=0).mean()) if len(fs) > 1 else 0.0)
            if step < args.n_steps:
                dyn.run(1)
        taus[traj] = integrated_tau(e_hist, args.dump_every)
        if (si + 1) % 10 == 0:
            print(f"[active] {si+1}/{len(states)} states, "
                  f"{len(candidates)} candidates, tau~{taus[traj]:.1f} frames",
                  flush=True)

    scores = np.asarray(scores)

    # --- stage 1: ACF refractory filter -----------------------------------
    order = np.argsort(-scores)
    kept, used = [], defaultdict(list)
    for i in order:
        if not np.isfinite(scores[i]):
            continue
        c = candidates[i]
        win = max(1.0, args.tau_mult * taus.get(c.info["traj"], 1.0))
        if any(abs(c.info["frame_index"] - j) < win for j in used[c.info["traj"]]):
            continue
        kept.append(int(i))
        used[c.info["traj"]].append(c.info["frame_index"])
    print(f"[active] ACF filter: {len(candidates)} -> {len(kept)}", flush=True)

    # --- stage 2: FPS diversity on per-atom force signatures --------------
    emb = []
    for i in kept:
        es, fs = score_atoms(candidates[i])
        # rotation/translation-invariant-ish signature: sorted force-magnitude
        # percentiles plus per-element means
        mag = np.linalg.norm(fs[0], axis=1)
        qs = np.percentile(mag, [5, 25, 50, 75, 95])
        z = candidates[i].get_atomic_numbers()
        per_el = [float(mag[z == e].mean()) if np.any(z == e) else 0.0
                  for e in (1, 6, 7, 8)]
        emb.append(np.concatenate([qs, per_el, [np.log(candidates[i].get_volume())]]))
    pick_local = fps_diversity(np.asarray(emb), scores[kept],
                              min(args.max_select, len(kept)))
    picked = [kept[j] for j in pick_local]
    print(f"[active] FPS: {len(kept)} -> {len(picked)}", flush=True)

    selected = [candidates[i] for i in picked]
    _write_xyz(out / "selected_unlabeled.xyz", selected)

    # --- teacher labels (FP64, GPU) ---------------------------------------
    sys.path.insert(0, "/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0")
    from mace.calculators import mace_polar
    teacher = mace_polar(model=args.teacher, device=args.device,
                         default_dtype="float64")
    labeled = []
    for j, atoms in enumerate(selected):
        atoms.calc = teacher
        atoms.info["REF_energy"] = float(atoms.get_potential_energy())
        atoms.arrays["REF_forces"] = atoms.get_forces()
        q = np.asarray(teacher.results["charges"]).reshape(-1)
        if q.size != len(atoms):
            if q.size < len(atoms):
                raise RuntimeError(
                    f"teacher returned {q.size} charges for {len(atoms)} atoms")
            q = q[:len(atoms)]          # strip ASE calculator padding
        atoms.arrays["REF_charges"] = q * args.charge_scale
        atoms.calc = None
        atoms.info["dataset"] = "bulk"
        atoms.info["source"] = f"active_r{args.round}"
        labeled.append(atoms)
        if (j + 1) % 25 == 0:
            print(f"[active] labelled {j+1}/{len(selected)}", flush=True)
    _write_xyz(out / "selected_labeled.xyz", labeled)

    summary = {
        "round": args.round,
        "students": args.students,
        "teacher": args.teacher,
        "n_states": len(states),
        "n_candidates": len(candidates),
        "n_after_acf": len(kept),
        "n_selected": len(picked),
        "tau_mult": args.tau_mult,
        "tau_frames_median": float(np.median(list(taus.values()))),
        "densities": list(TARGET_DENSITIES),
        "temperatures": list(TARGET_TEMPERATURES),
        "species": list(SPECIES),
        "n_mol": args.n_mol,
        "charge_scale": args.charge_scale,
    }
    (out / "propose.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


def _merge(args, out: Path) -> int:
    import ase.io

    if not args.base_train:
        raise SystemExit("--base-train is required for merge")
    labeled = ase.io.read(out / "selected_labeled.xyz", index=":")
    base = ase.io.read(args.base_train, index=":")
    merged = list(base) + list(labeled)
    target = out / "train_merged.xyz"
    _write_xyz(target, merged)
    info = {"base": args.base_train, "base_frames": len(base),
            "added": len(labeled), "merged_frames": len(merged),
            "output": str(target)}
    (out / "merge.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
