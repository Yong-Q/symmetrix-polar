#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PolarMACE SCF 链中间量钩子夹具(P3 门 ①-⑤, 2026-10-15)。

通过 nn.Module forward-hook + 实例方法包装,捕获:
  - lr_source_maps[0/1] 输出(→ 初值密度的逐层贡献)
  - fukui_source_map 输出(→ Fukui 源)
  - 每个 SCF 步: potential_features(势特征)→ field_dependent_charges_maps[i] 输出
    (density 增量 + Fukui 源)
落盘到 bench/polar_fixtures/scf/{mol}-f{k}.npz + index.json(张量清单+范数摘要)。

用法: CUDA_VISIBLE_DEVICES=<gpu> <polar venv python> polar_scf_hooks_dump.py
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
import mace_md as mm  # noqa: E402

POLAR = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"
OUTDIR = f"{ROOT}/bench/polar_fixtures/scf"
MOLS = [("NH3", 770600), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]


def main():
    from mace.calculators import mace_polar
    os.makedirs(OUTDIR, exist_ok=True)
    calc = mace_polar(model=POLAR, device="cuda", default_dtype="float64")
    model = calc.models[0]

    caps = {}
    handles = []

    def hook(name):
        def fn(module, inp, out):
            arr = out.detach().cpu().numpy() if isinstance(out, torch.Tensor) else out
            caps.setdefault(name, []).append(np.asarray(arr))
        return fn

    for i, m in enumerate(model.lr_source_maps):
        handles.append(m.register_forward_hook(hook(f"lr_src_{i}")))
    handles.append(model.fukui_source_map.register_forward_hook(hook("fukui_src")))

    orig_fd = model.electric_potential_descriptor.forward_dynamic
    def fd_wrapped(*a, **kw):
        out = orig_fd(*a, **kw)
        caps.setdefault("potential_feats_step", []).append(
            out.detach().cpu().numpy())
        return out
    model.electric_potential_descriptor.forward_dynamic = fd_wrapped

    for i, m in enumerate(model.field_dependent_charges_maps):
        handles.append(m.register_forward_hook(hook(f"charges_step_{i}")))

    index = {}
    for mol, seed in MOLS:
        base, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
        atoms = base.copy()
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        atoms.calc = calc
        caps.clear()
        e = atoms.get_potential_energy()
        r = calc.results
        name = f"{mol.lower()}-f0"
        payload = {
            "positions": atoms.get_positions(),
            "symbols": np.asarray(atoms.get_chemical_symbols(), dtype="U2"),
            "final_spin_charge_density": np.asarray(r["spin_charge_density"]),
            "energy": np.float64(e),
        }
        summary = {"energy": float(e)}
        for k, v in caps.items():
            for j, arr in enumerate(v):
                key = k if len(v) == 1 else f"{k}_{j}"
                payload[key] = arr
                summary[key] = {"shape": list(arr.shape),
                                "absmax": float(np.abs(arr).max()),
                                "sum": float(np.sum(arr))}
        np.savez(f"{OUTDIR}/{name}.npz", **payload)
        index[name] = summary
        print(f"  {name}: keys={len(payload)}  E={e:.4f}", flush=True)
        for k in sorted(summary):
            if k != "energy":
                print(f"      {k:26s} {summary[k]['shape']}  absmax={summary[k]['absmax']:.5f}",
                      flush=True)

    with open(f"{OUTDIR}/index.json", "w") as fh:
        json.dump(index, fh, indent=1)
    print(f"SCF-HOOKS-DONE -> {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
