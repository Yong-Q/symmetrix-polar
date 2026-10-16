#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Symmetrix-XL vs MACE-Torch 生产形态基准（task3 MLIP 线；本地 A100）。

体系 = 生产波次同款：NH3/CH4/N2/CO2 × 32 分子（ρ=0.01, seed≈生产 7705xx）。
指标：energy+forces(+stress) 完整调用 → µs/atom、ms/call、隐含单流步/s；
对比我们现行：ASE 单进程（~12.5 步/s·流 远程）与 torch_sim 16 流批处理（~8.4 步/s·流）。

用法：CUDA_VISIBLE_DEVICES=3 python symmetrix_vs_mace.py <model.model> <model.json>
"""
import sys, time, statistics
import numpy as np
import torch

sys.path.insert(0, "/workspace/group/shared/bza/task3_sampling_database/mlip/scripts")
import mace_md as mm  # noqa: E402


def bench(calc, atoms, n=20, warmup=10):
    """稳健协议：每次调用前微扰构型（破除 ASE 结果缓存），预热后取中位数。"""
    atoms = atoms.copy()
    atoms.calc = calc
    sflag = True
    for w in range(warmup):
        atoms.rattle(stdev=1e-4, seed=5000 + w)
        atoms.get_potential_energy(); atoms.get_forces()
        if sflag:
            try: atoms.get_stress()
            except Exception: sflag = False
    ts, es, fmax = [], [], []
    for i in range(n):
        atoms.rattle(stdev=1e-4, seed=9000 + i)
        t0 = time.perf_counter()
        e = atoms.get_potential_energy(); f = atoms.get_forces()
        if sflag:
            try: atoms.get_stress()
            except Exception: sflag = False
        ts.append(time.perf_counter() - t0)
        es.append(e); fmax.append(float(np.abs(f).max()))
    return statistics.median(ts), statistics.median(es), max(fmax)


def main():
    model, json_model = sys.argv[1], sys.argv[2]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} torch={torch.__version__}", flush=True)

    from mace.calculators import MACECalculator
    import symmetrix.calculator as sc
    import inspect
    print("Symmetrix signature:", inspect.signature(sc.Symmetrix), flush=True)

    calcA = MACECalculator(model_paths=model, device=dev, default_dtype="float32")
    try:
        calcB = sc.Symmetrix(json_model)
        b_desc = "default"
    except TypeError as e:
        print("default ctor failed:", e, flush=True)
        raise
    print(f"symmetrix calculator: {b_desc}", flush=True)

    specs = [("NH3", 770500), ("CH4", 770700), ("N2", 770800), ("CO2", 770500)]
    for mol, seed in specs:
        atoms, _, _ = mm.build_gas(mol, 32, 0.01, seed, 2.0)
        n = len(atoms)
        print(f"{mol}-32: {n} atoms")
        try:
            tA, eA, fA = bench(calcA, atoms)
            print(f"  MACE-Torch : {tA*1e6/n:8.3f} us/atom  {tA*1e3:7.2f} ms/call  -> {1/tA:6.1f} 步/s·流(单进程)")
        except Exception as e:
            tA = eA = fA = None
            print(f"  MACE-Torch FAIL: {type(e).__name__} {str(e)[:160]}")
        try:
            tB, eB, fB = bench(calcB, atoms)
            print(f"  Symmetrix  : {tB*1e6/n:8.3f} us/atom  {tB*1e3:7.2f} ms/call  -> {1/tB:6.1f} 步/s·流")
            if tA is not None:
                print(f"  -> 提速 {tA/tB:.2f}x | dE={abs(eA-eB):.4f} eV ({(abs(eA-eB)/max(1e-9,abs(eA))*1e6):.1f} ppm) | dmaxF={abs(fA-fB):.4f} eV/A")
        except Exception as e:
            print(f"  Symmetrix FAIL: {type(e).__name__} {str(e)[:200]}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
