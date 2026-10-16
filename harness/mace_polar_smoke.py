#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MACE-POLAR-1-M 冒烟 + 计时（与 MACE-OFF23 同盒子配对对照）。

体系 = 生产同款：NH3/CH4/N2/CO2 × 32 分子（ρ=0.01，seed 同生产）。
协议 = 缓存破除（每步微扰）、预热 3、计时 10，取中位。
"""
import os, sys, time, statistics
import numpy as np
import torch

sys.path.insert(0, "/workspace/group/shared/bza/task3_sampling_database/mlip/scripts")
import mace_md as mm  # noqa: E402

POLAR = "/workspace/group/shared/bza/task3_sampling_database/mlip/models/mace-polar/MACE-POLAR-1-M.model"
MACEOFF = "/workspace/group/shared/bza/task3_sampling_database/mlip/models/MACE-OFF23_medium.model"


def bench(calc, atoms, n=10, warmup=3):
    atoms = atoms.copy()
    atoms.calc = calc
    for w in range(warmup):
        atoms.rattle(stdev=1e-4, seed=5000 + w)
        atoms.get_potential_energy(); atoms.get_forces()
    ts = []
    for i in range(n):
        atoms.rattle(stdev=1e-4, seed=9000 + i)
        t0 = time.perf_counter()
        e = atoms.get_potential_energy(); f = atoms.get_forces()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), e, float(np.abs(f).max())


def install_compat_shim():
    """上游快照错位修补：
    1) precompute_geometry 不接受 force_pbc_evaluator → pop 掉（同外部源模块的处理）；
    2) checkpoint 反序列化对象可能缺 _precompute_geometry_impl/_forward_dynamic_impl
       （pickle 不执行 __init__）→ 惰性自愈；pbc_handling 缺失时回退 'pbc'。"""
    import graph_longrange.features as F
    patched = 0
    targets = [getattr(F, "GTOElectrostaticFeatures", None)]
    try:
        import graph_longrange.external_source_features as E
        for n in dir(E):
            o = getattr(E, n)
            if isinstance(o, type) and "ElectrostaticFeatures" in n:
                targets.append(o)
    except Exception:
        pass
    for cls in [c for c in set(targets) if c is not None]:
        if getattr(getattr(cls, "precompute_geometry", None), "_shim", False):
            continue
        orig_pg = cls.precompute_geometry
        orig_fw = cls.forward

        def _ensure(self):
            if not hasattr(self, "pbc_handling"):
                self.pbc_handling = "pbc"
                print("  [shim] pbc_handling -> 'pbc'", flush=True)
            if not hasattr(self, "_precompute_geometry_impl"):
                self._precompute_geometry_impl = self._select_precompute_geometry_impl()
                print("  [shim] lazy _precompute_geometry_impl installed", flush=True)
            if not hasattr(self, "_forward_dynamic_impl"):
                self._forward_dynamic_impl = self._select_forward_dynamic_impl()

        def _pg(self, *args, **kwargs):
            kwargs.pop("force_pbc_evaluator", None)
            _ensure(self)
            return orig_pg(self, *args, **kwargs)
        def _fw(self, *args, **kwargs):
            _ensure(self)
            return orig_fw(self, *args, **kwargs)
        _pg._shim = True
        cls.precompute_geometry = _pg
        cls.forward = _fw
        patched += 1
    print(f"compat shim installed on {patched} class(es)", flush=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} torch={torch.__version__}", flush=True)

    if os.environ.get("MACE_POLAR_SHIM") == "1":
        install_compat_shim()
    else:
        print("shim disabled (clean official-combo run)", flush=True)
    from mace.calculators import mace_polar, MACECalculator
    import inspect
    try:
        print("mace_polar sig:", inspect.signature(mace_polar), flush=True)
    except Exception:
        pass

    def make_polar():
        return mace_polar(model=POLAR, device=dev, default_dtype="float32")

    def make_maceoff():
        return MACECalculator(model_paths=MACEOFF, device=dev, default_dtype="float32")

    specs = [("NH3", 770500), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]
    for mol, seed in specs:
        atoms, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        n = len(atoms)
        print(f"{mol}-32: {n} atoms", flush=True)
        try:
            tA, eA, fA = bench(make_maceoff(), atoms)
            print(f"  MACE-OFF23 : {tA*1e6/n:9.2f} us/atom  {tA*1e3:7.1f} ms/call", flush=True)
        except Exception as e:
            tA = eA = fA = None
            print(f"  MACE-OFF23 FAIL: {type(e).__name__} {str(e)[:150]}", flush=True)
        try:
            tB, eB, fB = bench(make_polar(), atoms)
            print(f"  MACE-POLAR : {tB*1e6/n:9.2f} us/atom  {tB*1e3:7.1f} ms/call", flush=True)
            if tA is not None:
                print(f"  -> POLAR/OFF23 用时比 {tB/tA:.2f}x | dE={abs(eA-eB):.4f} eV | 各自 fmax {fA:.3f} / {fB:.3f}", flush=True)
        except Exception as e:
            print(f"  MACE-POLAR FAIL: {type(e).__name__} {str(e)[:200]}", flush=True)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
