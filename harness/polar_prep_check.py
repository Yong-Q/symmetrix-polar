#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MACE-POLAR 批处理替换件准备核验（2026-10-14）。

A) COM 修复后等价性（OFF23，16×NH3-32，nh3-500 生产场文件）：
   A1) site 模式：重构版 scripts/torchsim_field_model.py vs 波次冻存版 → 应一致；
   A2) com 模式：重构版（live-state 映射，无需 system_idx）vs 冻存版（显式 system_idx）
       → 应一致（≤ float32 噪声）；并给出冻存版生产同款调用（不传）的偏差量级（事故签名）。
B) PolarMACE × 官方 torch_sim 集成（MaceTorchSimModel 系包装）：
   16 流 minigate（5 步 Langevin，T=500K）→ 有限性 / 温度 / 能量尺度。
C) 同机同盒吞吐：POLAR vs OFF23（各 2 预热 + 6 计时步）→ 聚合/单流步速与波次 ETA。

用法：CUDA_VISIBLE_DEVICES=0 <polar venv python> bench/polar_prep_check.py
"""
import glob
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np
import torch

ROOT = "/workspace/group/shared/bza/task3_sampling_database"
W = f"{ROOT}/mlip/remote/wave-batch/nh3-500-01/attempt-01"
sys.path.insert(0, f"{ROOT}/mlip/scripts")

import mace_md as mm  # noqa: E402
import torch_sim as ts  # noqa: E402
from torch_sim import integrators as ti  # noqa: E402
from torch_sim import io as tio  # noqa: E402
from torchsim_field_model import (  # noqa: E402
    PerSystemSiteFieldMaceModel,
    PerSystemSiteFieldMaceTorchSimModel,
)

OFF23 = f"{ROOT}/mlip/models/MACE-OFF23_medium.model"
POLAR = f"{ROOT}/mlip/models/mace-polar/MACE-POLAR-1-M.model"

N, NMOL, RHO, T, SEED = 16, 32, 0.01, 500.0, 770600
DT, GAMMA = 0.5e-3, 10.0
KB = 8.617333262e-5
RESULTS = {}


def numkey(p):
    m = re.search(r"(\d+)", os.path.basename(p))
    return (int(m.group(1)) if m else 10**9, p)


def load_frozen():
    spec = importlib.util.spec_from_file_location("frozen_wave_tfm",
                                                  f"{W}/torchsim_field_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["frozen_wave_tfm"] = mod
    spec.loader.exec_module(mod)
    return mod


def emax(x):
    return float(np.abs(x).max())


def main():
    dev = "cuda"
    fields = sorted(glob.glob(f"{W}/fields/field_s*.json"), key=numkey)[:N]
    assert len(fields) == N

    boxes, masses = [], []
    for k in range(N):
        a, _, _ = mm.build_gas("NH3", NMOL, RHO, SEED + k, 2.0)
        boxes.append(a)
        masses.append(np.asarray(a.get_masses(), dtype=np.float64))
    n_atoms_per = [len(a) for a in boxes]
    # runner 约定：gmol = per-ATOM 全局分子号；frac = per-atom 质量分数
    gmol_list, frac_list = [], []
    for k, m in enumerate(masses):
        mloc = np.repeat(np.arange(NMOL), n_atoms_per[k] // NMOL)
        M_mol = np.bincount(mloc, weights=m, minlength=NMOL)
        frac_list.append(m / M_mol[mloc])
        gmol_list.append(k * NMOL + mloc)
    gmol = np.concatenate(gmol_list)
    frac = np.concatenate(frac_list)
    sys_of_atom = np.repeat(np.arange(N), n_atoms_per)

    state = tio.atoms_to_state(boxes, device=dev, dtype=torch.float32)
    frozen = load_frozen()

    # ---------------- A) equivalence ----------------
    print("== A1) site 模式：重构版 vs 冻存版 ==", flush=True)
    wr_site_ref = frozen.PerSystemSiteFieldMaceModel(
        model=OFF23, field_configs=fields, mode="site", device=dev, dtype=torch.float32)
    wr_site_new = PerSystemSiteFieldMaceModel(
        model=OFF23, field_configs=fields, mode="site", device=dev, dtype=torch.float32)
    oR, oN = wr_site_ref(state), wr_site_new(state)
    dE_site = emax((oR["energy"] - oN["energy"]).detach().cpu().numpy())
    dF_site = emax((oR["forces"] - oN["forces"]).detach().cpu().numpy())
    print(f"   dE(sys)max={dE_site:.3e} eV  dFmax={dF_site:.3e} eV/Å (期望 float32 噪声级)", flush=True)
    RESULTS["A1_site"] = {"dE_sys_max": dE_site, "dF_max": dF_site}

    print("== A2) com 模式：新(no arg) vs 冻存(带 system_idx) ==", flush=True)
    wr_com_ref = frozen.PerSystemSiteFieldMaceModel(
        model=OFF23, field_configs=fields, mode="com",
        molecule_of_atom=gmol, mass_fraction=frac,
        system_idx=torch.as_tensor(sys_of_atom, dtype=torch.long, device=dev),
        device=dev, dtype=torch.float32)
    wr_com_new = PerSystemSiteFieldMaceModel(
        model=OFF23, field_configs=fields, mode="com",
        molecule_of_atom=gmol, mass_fraction=frac, device=dev, dtype=torch.float32)
    oCR, oCN = wr_com_ref(state), wr_com_new(state)
    feR = oCR["field_energy"].detach().cpu().numpy()
    feN = oCN["field_energy"].detach().cpu().numpy()
    dE_com = emax((oCR["energy"] - oCN["energy"]).detach().cpu().numpy())
    dF_com = emax((oCR["forces"] - oCN["forces"]).detach().cpu().numpy())
    dfeld = emax(feR - feN)
    print(f"   dE(sys)max={dE_com:.3e} eV  dFmax={dF_com:.3e} eV/Å  d(field_energy)max={dfeld:.3e} eV",
          flush=True)
    print(f"   field_energy(new): {np.round(feN, 4).tolist()}", flush=True)

    print("== A2b) 事故签名（冻存版生产同款：不传 system_idx） ==", flush=True)
    wr_com_bug = frozen.PerSystemSiteFieldMaceModel(
        model=OFF23, field_configs=fields, mode="com",
        molecule_of_atom=gmol, mass_fraction=frac, device=dev, dtype=torch.float32)
    oCB = wr_com_bug(state)
    feB = oCB["field_energy"].detach().cpu().numpy()
    dE_bug = emax((oCB["energy"] - oCN["energy"]).detach().cpu().numpy())
    dF_bug = emax((oCB["forces"] - oCN["forces"]).detach().cpu().numpy())
    print(f"   bug fe[0]={feB[0]:.4f} eV (vs correct {feN[0]:.4f}) | "
          f"bug fe[1:]max={emax(feB[1:]):.2e} | dE(sys)max={dE_bug:.4f} eV  dFmax={dF_bug:.4f} eV/Å",
          flush=True)
    RESULTS["A2_com"] = {"dE_sys_max": dE_com, "dF_max": dF_com,
                         "d_field_energy_max": dfeld,
                         "bug_dE_sys_max": dE_bug, "bug_dF_max": dF_bug,
                         "bug_fe0": float(feB[0]), "new_fe0": float(feN[0])}

    # ---------------- B) POLAR minigate ----------------
    print("== B) PolarMACE × PerSystemSiteFieldMaceTorchSimModel minigate ==", flush=True)
    # 用全新状态（A 节的 MaceModel 系包装器会经 mace 的原地 requires_grad_
    # 污染共享 state.positions；生产 POLAR 波次不会经过该路径，这里保持同构）
    state = tio.atoms_to_state(boxes, device=dev, dtype=torch.float32)
    state = ts.SimState.from_state(
        state,
        charge=torch.zeros(N, dtype=torch.float32, device=state.device),
        spin=torch.ones(N, dtype=torch.float32, device=state.device),
    )
    print(f"   extras: in _system_extras={'charge' in state._system_extras} "
          f"readback charge={float(state.charge[0])} spin={float(state.spin[0])}",
          flush=True)
    pm = PerSystemSiteFieldMaceTorchSimModel(
        model=POLAR, field_configs=fields, mode="com",
        molecule_of_atom=gmol, mass_fraction=frac, device=dev, dtype=torch.float32)
    kT = KB * T
    md = ti.nvt_langevin_init(state, pm, kT=kT)
    q = getattr(md, "charge", None)
    print(f"   md extras: charge={'ok' if q is not None else 'MISSING'} "
          f"(spin={'ok' if getattr(md, 'spin', None) is not None else 'MISSING'})", flush=True)
    if q is None:
        md._system_extras["charge"] = state._system_extras["charge"]
        md._system_extras["spin"] = state._system_extras["spin"]
        print("   extras re-attached into md._system_extras", flush=True)
    t0 = time.time()
    for _ in range(5):
        md = ti.nvt_langevin_step(md, pm, dt=DT, kT=kT, gamma=GAMMA)
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(md.positions.detach()).all().item())
    temps = md.calc_temperature().detach().cpu().numpy()
    out_p = pm(md)
    pes = out_p["energy"].detach().cpu().numpy()
    print(f"   minigate finite={finite}  5步 wall={time.time()-t0:.1f}s  "
          f"T均值={temps.mean():.1f}K  PE[0:3]={np.round(pes[:3], 2).tolist()} eV", flush=True)
    RESULTS["B_polar_minigate"] = {"finite": finite, "T_mean": float(temps.mean()),
                                   "PE_first3": pes[:3].tolist()}

    # ---------------- C) timing ----------------
    def timed(md_, model, n=6, warmup=2):
        for _ in range(warmup):
            md_ = ti.nvt_langevin_step(md_, model, dt=DT, kT=kT, gamma=GAMMA)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            md_ = ti.nvt_langevin_step(md_, model, dt=DT, kT=kT, gamma=GAMMA)
        torch.cuda.synchronize()
        return (time.time() - t0) / n

    print("== C) 吞吐（同机同盒 16 流） ==", flush=True)
    s_polar = timed(md, pm)
    md2 = ti.nvt_langevin_init(state, wr_com_new, kT=kT)
    s_off = timed(md2, wr_com_new)
    print(f"   POLAR : {s_polar:.2f} s/步(16流) → {1/s_polar:.2f} agg 步/s → {1/s_polar/N:.3f} 步/s·流"
          f" | 2.1M 步/流 ETA {2.1e6/(1/s_polar/N)/3600:.0f} h", flush=True)
    print(f"   OFF23 : {s_off:.2f} s/步(16流) → {1/s_off:.2f} agg 步/s → {1/s_off/N:.3f} 步/s·流"
          f" | 2.1M 步/流 ETA {2.1e6/(1/s_off/N)/3600:.0f} h", flush=True)
    print(f"   ratio POLAR/OFF23 单步耗时 = {s_polar/s_off:.2f}×", flush=True)
    RESULTS["C_timing"] = {"s_per_step_16streams_polar": s_polar,
                           "s_per_step_16streams_off23": s_off,
                           "ratio": s_polar / s_off}

    with open(f"{ROOT}/mlip/bench/polar_prep_check.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1)
    print("PREP-CHECK-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
