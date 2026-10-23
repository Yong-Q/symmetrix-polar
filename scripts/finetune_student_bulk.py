#!/usr/bin/env python3
"""Thin, auditable launcher for the 128L1 bulk fine-tune.

It stages a run directory containing:

  student128L1_bulk_config.yaml  resolved MACE config (absolute paths)
  run.sh                         reproducible entry point
  command.txt                    plain run_train command + cell-grouped command
  seed.txt                       random seed
  manifest.json                  hashes/inputs/resolved settings

and, only when ``launch`` is used, runs MACE ``run_train`` in-process.

Cell-grouped batching
---------------------
MACE 0.3.16 builds a globally shuffled DataLoader, so configs with different
cells are mixed in a batch.  The bulk boxes span L = 6.8 .. 68.4 A plus the
16 A gas box.  We install a small BatchSampler (only for shuffled training
loaders) that keeps all configs of one cell in the same batch: cells are
grouped, indices are shuffled inside a group, chunked by batch size, and the
resulting batches are shuffled.  This changes batching order only, not the
model or loss.  ``--no-cell-grouped`` falls back to stock MACE.

The production run is NOT started by ``stage``.  The real launch is:

  PYTHONPATH=<shared>/external-repos/graph_electrostatics-0.4.0 \
    <mace-python> scripts/finetune_student_bulk.py launch \
    --run-dir bench/path_b_bulk_ft_2026-10-23/attempt-01
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INIT = str(
    REPO
    / "bench/path_b_128L1_data2_q4_2026-10-22/attempt-01/student_run/models/"
    "path_b_student128L1_data2.model"
)
DEFAULT_TRAIN = str(REPO / "data/train.xyz")
DEFAULT_VALID = str(REPO / "data/valid.xyz")
GRAPH_LONG_RANGE = str(
    REPO.parent / "external-repos/graph_electrostatics-0.4.0"
)

E0S = "{1: -15.99000358581543, 6: -1037.1678466796875, 7: -1490.378173828125, 8: -2047.592529296875}"
AVG_NUM_NEIGHBORS = 4.4816765785217285  # from the 128L1 gas student checkpoint


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _check_inputs(paths: Dict[str, Optional[str]]) -> Dict[str, Any]:
    out = {}
    for label, p in paths.items():
        if not p:
            continue
        path = Path(p)
        out[label] = {
            "path": str(path.resolve()),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": _file_sha256(path) if path.exists() else None,
        }
    return out


def _extract_init_config(init_model: str) -> Dict[str, Any]:
    """Best-effort read of the student checkpoint metadata."""
    info: Dict[str, Any] = {"path": str(Path(init_model).resolve())}
    try:
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        if GRAPH_LONG_RANGE not in sys.path:
            sys.path.insert(0, GRAPH_LONG_RANGE)
        import torch  # noqa: F401
        from mace.tools.scripts_utils import extract_config_mace_model

        model = torch.load(init_model, map_location="cpu", weights_only=False)
        cfg = extract_config_mace_model(model)
        info["class"] = type(model).__name__
        info["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
        info["avg_num_neighbors"] = float(cfg.get("avg_num_neighbors", AVG_NUM_NEIGHBORS))
        info["hidden_irreps"] = str(cfg.get("hidden_irreps"))
        info["r_max"] = float(cfg.get("r_max", 6.0))
        info["num_interactions"] = int(cfg.get("num_interactions", 2))
    except Exception as exc:  # noqa: BLE001
        info["extract_error"] = f"{type(exc).__name__}: {exc}"
        info["avg_num_neighbors"] = AVG_NUM_NEIGHBORS
    return info


def build_config(args: argparse.Namespace, run_dir: Path, init_info: Dict[str, Any]) -> Dict[str, Any]:
    avg_nn = float(args.avg_num_neighbors or init_info.get("avg_num_neighbors") or AVG_NUM_NEIGHBORS)
    cfg: Dict[str, Any] = {
        # --- identity ---
        "name": args.name,
        "seed": int(args.seed),
        "model": "PolarMACE",
        # --- backbone: identical to path_b_student128L1_data2 ---
        "r_max": 6.0,
        "num_radial_basis": 8,
        "num_cutoff_basis": 5,
        "max_ell": 3,
        "num_channels": 128,
        "max_L": 1,
        "num_interactions": 2,
        "correlation": 3,
        "interaction": "RealAgnosticResidualNonLinearInteractionBlock",
        "interaction_first": "RealAgnosticResidualNonLinearInteractionBlock",
        "use_agnostic_product": True,
        "gate": "silu",
        "MLP_irreps": "32x0e",
        "radial_MLP": "[64, 64, 64]",
        "radial_type": "bessel",
        "scaling": "rms_forces_scaling",
        "avg_num_neighbors": avg_nn,
        "compute_avg_num_neighbors": False,
        # --- PolarMACE electrostatics: match the teacher / gas student ---
        "kspace_cutoff_factor": 1.5,
        "atomic_multipoles_max_l": 1,
        "atomic_multipoles_smearing_width": 1.5,
        "field_feature_max_l": 1,
        "field_feature_widths": "[1.5, 3.0]",
        "field_feature_norms": "[20.0, 20.0, 0.5, 0.5]",
        "num_recursion_steps": 2,
        "field_si": False,
        "include_electrostatic_self_interaction": True,
        "add_local_electron_energy": True,
        "quadrupole_feature_corrections": False,
        "field_norm_factor": 1.0,
        # --- outputs ---
        "compute_forces": True,
        "compute_stress": False,
        "compute_polarizability": False,
        "compute_atomic_dipole": False,
        # --- data / labels ---
        "train_file": str(Path(args.train).resolve()),
        "valid_file": str(Path(args.valid).resolve()),
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "E0s": E0S,
        # --- initialisation from the existing engine-compatible student ---
        "foundation_model": str(Path(args.init_model).resolve()),
        # Also transfer the readout heads (MACE otherwise re-initialises them).
        "foundation_filter_elements": True,
        # --- loss / optimisation ---
        "loss": "weighted",
        "error_table": "PerAtomMAE",
        "energy_weight": float(args.energy_weight),
        "forces_weight": float(args.forces_weight),
        "optimizer": "adam",
        "lr": float(args.lr),
        "weight_decay": 5.0e-7,
        "clip_grad": 10.0,
        "scheduler": "ReduceLROnPlateau",
        "lr_factor": 0.8,
        "scheduler_patience": 15,
        "batch_size": int(args.batch_size),
        "valid_batch_size": int(args.batch_size),
        "max_num_epochs": int(args.epochs),
        "patience": int(args.patience),
        "eval_interval": 1,
        "swa": False,
        "ema": False,
        # --- runtime ---
        "device": args.device,
        "default_dtype": args.dtype,
        "num_workers": 0,
        "plot": False,
        "work_dir": str(run_dir / "student_run"),
        "model_dir": str(run_dir / "student_run/models"),
        "checkpoints_dir": str(run_dir / "student_run/checkpoints"),
        "results_dir": str(run_dir / "student_run/results"),
        "log_dir": str(run_dir / "student_run/logs"),
    }
    return cfg


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        # Always quote: YAML would otherwise turn "[64, 64, 64]" into a list and
        # "{1: -15.99,...}" into a dict, which MACE's argparse rejects.
        return json.dumps(value)
    return str(value)


def write_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    lines = ["# Resolved by scripts/finetune_student_bulk.py -- do not hand-edit."]
    for key, value in cfg.items():
        lines.append(f"{key}: {_yaml_scalar(value)}")
    path.write_text("\n".join(lines) + "\n")


def stage(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "student128L1_bulk_config.yaml"
    if cfg_path.exists() and not args.overwrite:
        print(f"[finetune] {cfg_path} exists; keeping it (use --overwrite to regenerate)")
        return run_dir

    for sub in ("student_run/models", "student_run/checkpoints",
                "student_run/logs", "student_run/results"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    init_info = _extract_init_config(args.init_model)
    cfg = build_config(args, run_dir, init_info)
    cfg_path = run_dir / "student128L1_bulk_config.yaml"
    write_yaml(cfg_path, cfg)

    python = args.python or sys.executable
    env_prefix = (
        f"PYTHONPATH={GRAPH_LONG_RANGE}:{REPO}:${{PYTHONPATH:-}} "
        f"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
    )
    plain_cmd = (
        f"{env_prefix} {python} -m mace.cli.run_train "
        f"--config {cfg_path} --name {args.name} --seed {args.seed}"
    )
    group_cmd = (
        f"{env_prefix} {python} {Path(__file__).resolve()} launch "
        f"--run-dir {run_dir} --no-stage"
    )
    (run_dir / "command.txt").write_text(
        "# Plain MACE run_train (global shuffle, no cell grouping):\n"
        f"{plain_cmd}\n\n"
        "# Cell-grouped batching used by this pipeline (recommended):\n"
        f"{group_cmd}\n"
    )
    (run_dir / "seed.txt").write_text(f"{args.seed}\n")

    run_sh = run_dir / "run.sh"
    run_sh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cd "$(dirname "$0")"\n'
        f'export PYTHONPATH="{GRAPH_LONG_RANGE}:${{REPO:-{REPO}}}:${{PYTHONPATH:-}}"\n'
        "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1\n"
        f'exec "${{PYTHON:-{python}}}" "{Path(__file__).resolve()}" launch '
        f'--run-dir "$PWD" --no-stage "$@"\n'
    )
    run_sh.chmod(0o755)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "run_dir": str(run_dir),
        "name": args.name,
        "seed": int(args.seed),
        "python": python,
        "cell_grouped_batching": not args.no_cell_grouped,
        "init_model": init_info,
        "data": _check_inputs({"train": args.train, "valid": args.valid}),
        "config": cfg,
        "commands": {"plain": plain_cmd, "cell_grouped": group_cmd},
        "note": (
            "stage only writes the run directory; it does not start training. "
            "The model is initialised from the existing 128L1 .model via MACE "
            "foundation-model loading; foundation_filter_elements=true also "
            "transfers the readout heads. default_dtype=float64, 2 SCF steps."
        ),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"[finetune] staged {run_dir}")
    print(f"[finetune] config: {cfg_path}")
    print(f"[finetune] command: {plain_cmd}")
    print(f"[finetune] cell-grouped: {group_cmd}")
    if "extract_error" in init_info:
        print(f"[finetune] WARNING: could not read init model metadata: "
              f"{init_info['extract_error']}")
    return run_dir


# --------------------------------------------------------------------------- #
# cell-grouped batch sampler
# --------------------------------------------------------------------------- #
def _cell_key(dataset, index: int) -> str:
    item = dataset[index]
    cell = getattr(item, "cell", None)
    if cell is None:
        return f"__no_cell__{index}"
    arr = cell.detach().cpu().numpy() if hasattr(cell, "detach") else np.asarray(cell)
    return ",".join(f"{x:.4f}" for x in np.asarray(arr).ravel())


class CellGroupBatchSampler:
    """Yield batches containing configs with one identical cell."""

    def __init__(self, dataset, batch_size: int, seed: int):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        groups: Dict[str, List[int]] = collections.defaultdict(list)
        for i in range(len(dataset)):
            groups[_cell_key(dataset, i)].append(i)
        self.groups = [np.asarray(v, dtype=np.int64) for v in groups.values()]
        self._len = sum(
            (len(g) + self.batch_size - 1) // self.batch_size for g in self.groups
        )
        print(
            f"[finetune] cell-grouped sampler: {len(self.groups)} distinct cells, "
            f"group sizes {sorted(len(g) for g in self.groups)}, "
            f"{self._len} batches/epoch (batch_size={self.batch_size})",
            flush=True,
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        batches: List[List[int]] = []
        for g in self.groups:
            idx = g.copy()
            rng.shuffle(idx)
            for start in range(0, len(idx), self.batch_size):
                batches.append(idx[start:start + self.batch_size].tolist())
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return self._len


def install_cell_grouped_loader(seed: int) -> None:
    """Monkey-patch MACE's vendored DataLoader for shuffled training loaders."""
    import torch.utils.data
    from mace.tools.torch_geometric import dataloader as tg_loader

    orig = tg_loader.DataLoader
    Collater = tg_loader.Collater

    class CellGroupedDataLoader(torch.utils.data.DataLoader):
        def __init__(self, dataset, batch_size=1, shuffle=False,
                     follow_batch=(None,), exclude_keys=(None,), **kwargs):
            if "collate_fn" in kwargs:
                del kwargs["collate_fn"]
            self.follow_batch = follow_batch
            self.exclude_keys = exclude_keys
            collate = Collater(follow_batch, exclude_keys)
            if shuffle and isinstance(batch_size, int) and batch_size > 0:
                kwargs.pop("drop_last", None)
                kwargs.pop("generator", None)
                sampler = CellGroupBatchSampler(dataset, batch_size, seed)
                torch.utils.data.DataLoader.__init__(
                    self, dataset, batch_sampler=sampler, collate_fn=collate, **kwargs
                )
            else:
                torch.utils.data.DataLoader.__init__(
                    self, dataset, batch_size=batch_size, shuffle=shuffle,
                    collate_fn=collate, **kwargs
                )

    CellGroupedDataLoader.__name__ = "CellGroupedDataLoader"
    tg_loader.DataLoader = CellGroupedDataLoader
    print("[finetune] installed cell-grouped BatchSampler (training loaders only)")


def launch(args: argparse.Namespace) -> int:
    if not args.no_stage:
        stage(args)
    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "student128L1_bulk_config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing config {cfg_path}; run stage first")

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if GRAPH_LONG_RANGE not in sys.path:
        sys.path.insert(0, GRAPH_LONG_RANGE)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    import mace.cli.run_train as run_train

    if not args.no_cell_grouped:
        install_cell_grouped_loader(seed=args.seed)

    sys.argv = [
        "run_train",
        "--config", str(cfg_path),
        "--name", args.name,
        "--seed", str(args.seed),
    ] + list(getattr(args, "run_train_arg", []) or [])
    print(f"[finetune] launching: {' '.join(shlex.quote(a) for a in sys.argv)}")
    run_train.main()
    return 0


def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-dir", required=True)
    common.add_argument("--train", default=DEFAULT_TRAIN)
    common.add_argument("--valid", default=DEFAULT_VALID)
    common.add_argument("--init-model", default=DEFAULT_INIT)
    common.add_argument("--name", default="path_b_student128L1_bulk")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--epochs", type=int, default=60)
    common.add_argument("--patience", type=int, default=30)
    common.add_argument("--batch-size", type=int, default=16)
    common.add_argument("--lr", type=float, default=1.0e-3)
    common.add_argument("--energy-weight", type=float, default=1.0)
    common.add_argument("--forces-weight", type=float, default=50.0)
    common.add_argument("--avg-num-neighbors", type=float, default=None)
    common.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    common.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    common.add_argument("--python", default=None, help="interpreter recorded in run.sh")
    common.add_argument("--overwrite", action="store_true")
    common.add_argument("--no-cell-grouped", action="store_true",
                        help="use stock MACE global shuffling instead of cell-grouped batches")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage", parents=[common], help="write the run directory only")
    p_launch = sub.add_parser("launch", parents=[common], help="stage (unless --no-stage) and train")
    p_launch.add_argument("--no-stage", action="store_true")
    p_launch.add_argument("--run-train-arg", action="append", default=[],
                          help="extra argument passed to mace.cli.run_train (e.g. --dry_run)")
    args = ap.parse_args()

    if args.cmd == "stage":
        stage(args)
        return 0
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
