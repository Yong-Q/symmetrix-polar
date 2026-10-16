#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""极性反向链逐段 FD 自洽剖分 (2026-10-16)。

三变体(除 JSON 外引擎/输入全同):
  full     = 全模型(⑥+⑦ 全装配)
  leoff    = 无 local_electron_energy 块(⑦ 关)
  polaroff = 无 polar_blocks(极地层全关 ⇒ P2 骨架基线)
差分对(引擎内生, 同符号约定, 与 python 无关):
  ΔE_le=E_full−E_leoff,  ΔF_le=F_full−F_leoff   (⑦ le 段, 含其经链影响)
  ΔE_c =E_leoff−E_polaroff, ΔF_c=F_leoff−F_polaroff (⑥ coulomb 段+链路)
  ΔE_p =E_full−E_polaroff,  ΔF_p=F_full−F_polaroff (极地层合计)
判据: FD(ΔE_x) ≈ ΔF_x(中央差分 h; 引擎力已证与物理/python 同号)。
另打印各变体 ΣF(平移不变性)与 ΔF 平衡。
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
MODELS = {
    "full": f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json",
    "leoff": f"{ROOT}/models/mace-polar/polar_symmetrix_probe_leoff.json",
    "polaroff": f"{ROOT}/models/mace-polar/polar_symmetrix_probe_polaroff.json",
}


def build_atoms_from_npz(d):
    from ase import Atoms
    return Atoms(symbols=[str(s) for s in d["symbols"]],
                 positions=np.asarray(d["positions"], dtype=np.float64),
                 cell=np.asarray(d["cell"], dtype=np.float64), pbc=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="nh3-f0,co2-f0")
    ap.add_argument("--h", type=float, default=2e-3)
    ap.add_argument("--fd-atoms", type=int, default=2)
    args = ap.parse_args()

    from symmetrix.calculator import Symmetrix

    engines = {}
    for name, path in MODELS.items():
        engines[name] = Symmetrix(path, dtype="float64")
    print("engines loaded:", list(engines), flush=True)

    def eval_variant(name, atoms):
        a = atoms.copy()
        a.info["charge"] = 0
        a.info["spin"] = 1
        a.calc = engines[name]
        return float(a.get_potential_energy()), \
            np.asarray(a.get_forces(), dtype=np.float64).reshape(-1, 3)

    for fname in args.frames.split(","):
        d0 = np.load(f"{FIX}/{fname}.npz", allow_pickle=False)
        atoms = build_atoms_from_npz(d0)
        n = len(atoms)
        ids = list(range(0, n, max(1, n // args.fd_atoms)))[:args.fd_atoms]

        values = {}
        for name in MODELS:
            e, f = eval_variant(name, atoms)
            values[name] = {"e": e, "f": f}
        e_full = values["full"]["e"]
        e_leoff = values["leoff"]["e"]
        e_polaroff = values["polaroff"]["e"]
        f_full, f_leoff, f_polaroff = (values["full"]["f"],
                                       values["leoff"]["f"],
                                       values["polaroff"]["f"])
        b_full = np.abs(f_full.sum(axis=0)).max()
        b_leoff = np.abs(f_leoff.sum(axis=0)).max()
        b_pol = np.abs(f_polaroff.sum(axis=0)).max()
        dfl = f_full - f_leoff
        dfc = f_leoff - f_polaroff
        dfp = f_full - f_polaroff
        print(f"\n== {fname} (n={n}) 抽样原子={ids} ==", flush=True)
        print(f"  ΔE_le={e_full - e_leoff:+.6e}  ΔE_c={e_leoff - e_polaroff:+.6e}"
              f"  ΔE_p={e_full - e_polaroff:+.6e} eV", flush=True)
        print(f"  ΣF: full={b_full:.3e} leoff={b_leoff:.3e} poloff={b_pol:.3e}"
              f"  |ΣΔF_le|={np.abs(dfl.sum(axis=0)).max():.3e}"
              f"  |ΣΔF_c|={np.abs(dfc.sum(axis=0)).max():.3e}", flush=True)

        rows = []
        for ai in ids:
            for comp in range(3):
                ev = {}
                for name in MODELS:
                    for sign in (+1.0, -1.0):
                        x = atoms.copy()
                        p = x.get_positions()
                        p[ai, comp] += sign * args.h
                        x.set_positions(p)
                        ev[(name, sign)] = eval_variant(name, x)[0]
                def fd_of(a, b):  # −d( Ea−Eb )/dr
                    return -((ev[(a, +1.0)] - ev[(b, +1.0)])
                             - (ev[(a, -1.0)] - ev[(b, -1.0)])) / (2 * args.h)
                fd_le = fd_of("full", "leoff")
                fd_c = fd_of("leoff", "polaroff")
                fd_p = fd_of("full", "polaroff")
                rows.append((ai, comp, fd_le, fd_c, fd_p,
                             dfl[ai, comp], dfc[ai, comp], dfp[ai, comp]))
        for ai, comp, fd_le, fd_c, fd_p, an_le, an_c, an_p in rows:
            print(f"  a{ai:3d} c{comp}:  le FD={fd_le:+.6e} ΔF={an_le:+.6e}"
                  f" |d|={abs(fd_le - an_le):.3e} || c FD={fd_c:+.6e}"
                  f" ΔF={an_c:+.6e} |d|={abs(fd_c - an_c):.3e} || all |d|={abs(fd_p - an_p):.3e}",
                  flush=True)
        mle = max(abs(fd - an) for _, _, fd, _, _, an, _, _ in rows)
        mc = max(abs(fd - an) for _, _, _, fd, _, _, an, _ in rows)
        mp = max(abs(fd - an) for _, _, _, _, fd, _, _, an in rows)
        print(f"  MAX: le={mle:.3e}  coulomb={mc:.3e}  total={mp:.3e}", flush=True)
    print("\nCHAIN-BISECT-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
