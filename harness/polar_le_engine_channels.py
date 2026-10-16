#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""⑦ le 反向引擎侧逐通道隔离 (2026-10-16): POLAR_DBG_SKIP 有线差分。

对帧: (a) F_full(skip=7)=极性关?(不 — skip=7 只关 le 的三通道投递 ⇒ le 力全无);
  (b) skip∈{0,1,2,4,7} 各取一次 F_full(基位置单次求值);
  通道有线力: d_wired=F(0)−F(1), P_wired=F(0)−F(2), nf_wired=F(0)−F(4);
  总量: ΔF_le=F(0)−F_leoff(leoff=无 le 块);
  FD_le(参考, 中央差分 h): −d(E_full−E_leoff)/dr。
输出 bench/polar_fixtures/le_engine_channels_<frame>.npz + 与 python 真值
  (le_channel_ref_<frame>.npz) 的逐通道 max|Δ|/rel 表。
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/bench")

FIX = f"{ROOT}/bench/polar_fixtures"
J_FULL = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
J_LEOFF = f"{ROOT}/models/mace-polar/polar_symmetrix_probe_leoff.json"


def load_atoms(name):
    from ase import Atoms
    d = np.load(f"{FIX}/{name}.npz", allow_pickle=False)
    return Atoms(symbols=[str(s) for s in d["symbols"]],
                 positions=np.asarray(d["positions"], dtype=np.float64),
                 cell=np.asarray(d["cell"], dtype=np.float64), pbc=True)


def _uncache(engine):
    # ASE check_state 位置比对: 同位置复用同一 calculator 会命中缓存 ⇒ 强制失效。
    try:
        engine.atoms = None
        engine.results = {}
    except Exception:
        pass


def forces_of(engine, atoms):
    _uncache(engine)
    a = atoms.copy()
    a.info["charge"] = 0
    a.info["spin"] = 1
    a.calc = engine
    _ = float(a.get_potential_energy())
    return np.asarray(a.get_forces(), dtype=np.float64).reshape(-1, 3)


def energy_of(engine, atoms):
    _uncache(engine)
    a = atoms.copy()
    a.info["charge"] = 0
    a.info["spin"] = 1
    a.calc = engine
    return float(a.get_potential_energy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="nh3-f0,co2-f0")
    ap.add_argument("--h", type=float, default=2e-3)
    ap.add_argument("--fd-atoms", type=int, default=2)
    args = ap.parse_args()

    from symmetrix.calculator import Symmetrix

    engine_full = Symmetrix(J_FULL, dtype="float64")
    engine_leoff = Symmetrix(J_LEOFF, dtype="float64")

    for fname in args.frames.split(","):
        atoms = load_atoms(fname)
        n = len(atoms)
        ids = list(range(0, n, max(1, n // args.fd_atoms)))[:args.fd_atoms]
        f_by_skip = {}
        for skip in (0, 1, 2, 4, 7):
            os.environ["POLAR_DBG_SKIP"] = str(skip)
            f_by_skip[skip] = forces_of(engine_full, atoms)
        os.environ["POLAR_DBG_SKIP"] = "0"
        f_leoff = forces_of(engine_leoff, atoms)

        d_wired = f_by_skip[0] - f_by_skip[1]
        p_wired = f_by_skip[0] - f_by_skip[2]
        nf_wired = f_by_skip[0] - f_by_skip[4]
        df_le = f_by_skip[0] - f_leoff
        consistency = np.abs(df_le - (d_wired + p_wired + nf_wired)).max()

        # FD_le 参考
        fd_rows = []
        for ai in ids:
            for comp in range(3):
                vals = []
                for sign in (+1.0, -1.0):
                    p2 = atoms.copy()
                    p = p2.get_positions()
                    p[ai, comp] += sign * args.h
                    p2.set_positions(p)
                    vals.append(energy_of(engine_full, p2)
                                - energy_of(engine_leoff, p2))
                fd_rows.append((ai, comp, -(vals[0] - vals[1]) / (2 * args.h)))

        np.savez(f"{FIX}/le_engine_channels_{fname}.npz",
                 d_wired=d_wired, p_wired=p_wired, nf_wired=nf_wired,
                 df_le=df_le)
        print(f"== {fname} ==", flush=True)
        print(f"  线性自洽 |ΔF_le−(d+P+nf)|max = {consistency:.3e}", flush=True)
        ref = np.load(f"{FIX}/le_channel_ref_{fname}.npz")
        for label, wired, truth in (("d", d_wired, ref["A_d"]),
                                    ("P", p_wired, ref["A_P"]),
                                    ("nf", nf_wired, ref["A_nf"]),
                                    ("total", df_le, ref["F_le"])):
            w = np.asarray(wired, dtype=np.float64)
            t = np.asarray(truth, dtype=np.float64)
            d = np.abs(w - t)
            idx = np.unravel_index(np.argmax(d), d.shape)
            denom = max(float(np.abs(t).max()), 1e-30)
            print(f"  ch {label:5s}: |wired|max={np.abs(w).max():.4e} "
                  f"|truth|max={np.abs(t).max():.4e} max|Δ|={d.max():.4e} "
                  f"rel={d.max()/denom:.3e} @a{idx[0]}c{idx[1]}", flush=True)
        for ai, comp, fd in fd_rows:
            w = df_le[ai, comp]
            print(f"  FD_le a{ai}c{comp}={fd:+.6e}  ΔF_le={w:+.6e}  "
                  f"|Δ|={abs(fd - w):.3e}", flush=True)
    os.environ["POLAR_DBG_SKIP"] = "0"
    print("\nENGINE-CHANNELS-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
