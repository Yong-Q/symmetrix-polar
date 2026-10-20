# Testing Module — Numerical Gate Catalog

**中文摘要：** 每个数值门都列清楚“保护什么、用什么输入、指标怎么定义、
容差多少、跑什么命令、结果是多少”。优化只有在最终干净构建上通过全套门才算
验收；gate 是硬标准，不是“参考”。

Every optimization in this project is admitted only after the complete gate
suite passes on a **clean build of the final tree**. This document lists each
gate with its purpose, input, metric definition, tolerance, command, and the
accepted round-2 result. The gate is the contract; a speedup without a gate is
not a result.

## 0. Common setup

```bash
export MLIP=/workspace/group/shared/bza/task3_sampling_database/mlip
export CUDA_VISIBLE_DEVICES=0
export SYMMETRIX_BACKEND=cuda12-sm80
export SYMMETRIX_BACKEND_DESCRIPTOR_PATH=$MLIP/models/mace-polar/backend_cuda12_sm80.json
export LD_LIBRARY_PATH=/opt/matos-tools/cuda/12.9/lib64:/opt/matos-tools/conda/envs/matos-compute/lib:$LD_LIBRARY_PATH
PY=/workspace/caches/conda/envs/polar/bin/python
LMP=/workspace/group/shared/bza/external-repos/symmetrix-xl/symmetrix/build-lammps-aacd036df381800a/lmp
```

- Model: `polar_symmetrix_probe.json` (`MACE_Nonlinear` + polar, 316 MB,
  83 elements, \(r_c=6\) Å).
- Reference: frozen PyTorch fixtures plus a live recomputation at gate time.
- All gates run under
  `flock -w 10800 /workspace/group/shared/bza/.tmp/gpu.lock`.
- Python snapshots: `bench/kernel_round2_2026-10-19/pyfinal_r2` (tested build),
  `bench/e3lin_opt_2026-10-19/pybase` (baseline).

## 1. Gate summary (round 2 final build; round-3 row added)

| Gate | What it protects | Metric | Tolerance | Round-2 result | Verdict |
|---|---|---|---|---|---|
| module 5 | density / per-stage fixtures | dens abs, dens rel, stage rel | abs ≤5e-6, rel ≤1e-6 | 6.689e-8, 4.218e-8, 7.127e-8 | **PASS** |
| module 6 | Coulomb energy / SCF final | dE rel, dE abs, ρ̂ rel | rel ≤1e-6, abs ≤5e-6 eV | 1.095e-15, 5.551e-16 eV, 1.4e-15 | **PASS** |
| module 7 | local-electron stage | stage rel, out abs | rel ≤1e-6, abs ≤5e-6 (or rel) | 1.174e-7, 4.546e-7 | **PASS** |
| module 8 | forces (FD + PyTorch parity) | FD max, parity max | both ≤1e-3 | 5.783e-4, 7.191e-7 | **PASS** |
| end-to-end | full engine vs PyTorch | 32 frames, dE rel/abs, dF | rel ≤1e-6, abs ≤5e-6 eV, dF ≤1e-4 eV/Å | 32/32; 1.084e-11; 1.034e-6 eV; 1.624e-7 | **PASS** |
| LAMMPS parity 1 rank | pair style vs engine | dE, max dF | ≤5e-5 eV, ≤1e-4 eV/Å | 2.910e-11 eV, 2.579e-14 eV/Å | **PASS** |
| **LAMMPS parity 1 rank, round-3 build 8** | pair style vs round-2 Python reference | dE, max dF | ≤5e-5 eV, ≤1e-4 eV/Å | **2.910e-11 eV, 2.586e-14 eV/Å** | **PASS** |
| LAMMPS parity, exact 2-hop 2 ranks | MPI correctness | dE, max dF | ≤5e-5 eV, ≤1e-4 eV/Å | 2.910e-11 eV, 1.687e-6 eV/Å (e3lin round) | **PASS** |
| NVT MD 50 steps | dynamics stability | completion, lost atoms, finite | 50/50, none, finite | 50/50, none, Etot span 0.213 eV | **PASS** |
| RBE force gate | approximation quality | RMS_F | ≤1e-4 eV/Å | 4.1e-2 … 1.0e-2 eV/Å (p=50…800); final-build p=800 check **6.1e-3 eV/Å**, max 3.0e-2, energy noise ±0.05 eV | **REJECTED** |
| RBE P4 convergence/unbiasedness | implementation/statistics | slope, RMS drop, 3·SEM | slope ∈ [−0.8,−0.2], drop ≥2× | slope −0.499, unbiased within 3·SEM, same-seed bitwise | **PASS** |

The two RBE rows are deliberately separate: the P4 implementation/statistics
gate passes, while the production force gate is rejected. The p-scan and the
final-build p=800 check are in
`docs/results/KERNEL_ROUND3_FINAL.md` §6. Reaching 1e-4 eV/Å would need
p≈8.5e6 (≈17 TB of cos/sin storage), so exact full-k Ewald remains production.
| 1-hop ghost MPI | approximation quality | dE, max dF | ≤5e-5 eV, ≤1e-4 eV/Å | 7.94e-5 eV, 4.116e-3 eV/Å | **REJECTED** |
| H1 2-rank parity | H1 exactness | dE, max dF | ≤5e-5 eV, ≤1e-4 eV/Å | 2.910e-11 eV, 1.413e-15 eV/Å | **PASS** (pre-fix binary) |

## 2. Gate details and commands

### 2.1 module 5 — density and stage fixtures

- **Purpose:** the initial spin-charge density, the per-SCF-step density, and
  the mixer/update-block stage tensors must match the frozen PyTorch fixtures.
- **Input:** 4 frozen configurations + live recomputation; 12 configs total.
- **Metric:** max absolute / relative difference of the density tensor and of
  each named stage.
- **Command:**
  ```bash
  flock -w 10800 /workspace/group/shared/bza/.tmp/gpu.lock -c \
    "env -u PYTHONPATH PYTHONPATH=<run>/pyfinal_r2 $PY $MLIP/bench/polar_module5_gate.py \
     --out <run>/gates/polar_module5_gate"
  ```
- **Result:** dens abs 6.689e-8, rel 4.218e-8, stage rel 7.127e-8 → PASS.

### 2.2 module 6 — Coulomb / SCF final

- **Purpose:** the full-k Ewald Coulomb energy and the converged density.
- **Input:** 8 configurations (4 fixtures + 4 live), mode `scf-final`.
- **Metric:** \(\Delta E_{\rm rel}\), \(\Delta E_{\rm abs}\), \(\hat\rho\) relative.
- **Command:**
  ```bash
  env -u PYTHONPATH PYTHONPATH=<run>/pyfinal_r2 $PY $MLIP/bench/polar_module6_gate.py \
    --out <run>/gates/polar_module6_gate
  ```
- **Result:** dE rel 1.095e-15, abs 5.551e-16 eV, ρ̂ rel 1.4e-15 → PASS.

### 2.3 module 7 — local-electron

- **Purpose:** the local-electron stage (`Le`, SCF final) and its output.
- **Input:** 16 configurations; live recomputation.
- **Metric:** stage relative difference, output absolute difference.
- **Criterion:** stage rel ≤1e-6 (or abs ≤5e-6), output ≤1e-6.
- **Command:**
  ```bash
  env -u PYTHONPATH PYTHONPATH=<run>/pyfinal_r2 $PY $MLIP/bench/polar_module7_gate.py \
    --live 2 --out <run>/gates/polar_module7_gate
  ```
- **Result:** stage rel 1.174e-7, out 4.546e-7 → PASS. (The log prints a
  worst absolute stage value 2.396e-5 for a large-magnitude quantity; the
  acceptance criterion is abs ≤5e-6 **or** rel ≤1e-6, and the relative value is
  1.17e-7.)

### 2.4 module 8 — forces

- **Purpose:** verify the analytic force against finite differences and against
  the PyTorch reference.
- **Input:** 12 configurations; `--fd-atoms 4 --h 1e-3 --live 1 --stage auto`.
- **Metric:** max FD error (eV/Å), max parity error (eV/Å), \(\sum_i \mathbf F_i\).
- **Command:**
  ```bash
  env -u PYTHONPATH PYTHONPATH=<run>/pyfinal_r2 $PY $MLIP/bench/polar_module8_gate.py \
    --fd-atoms 4 --h 1e-3 --live 1 --stage auto --out <run>/gates/polar_module8_gate
  ```
- **Result:** FD 5.783e-4, parity 7.191e-7, ΣF 6.7e-15 → PASS.

### 2.5 End-to-end — 32 frames

- **Purpose:** full engine energy+forces against PyTorch on 32 independent
  frames (N2, CO2, CH4, NH3, ...).
- **Metric:** per-frame \(\Delta E_{\rm rel}\), \(\Delta E_{\rm abs}\),
  \(\Delta F_{\max}\), position drift.
- **Criterion:** rel ≤1e-6, abs ≤5e-6 eV, dF ≤1e-4 eV/Å.
- **Command:**
  ```bash
  env -u PYTHONPATH PYTHONPATH=<run>/pyfinal_r2 $PY $MLIP/bench/polar_endtoend_parity.py \
    --frames 8 --out <run>/gates/polar_endtoend_parity
  ```
- **Result:** 32/32 PASS; dE rel max 1.084e-11, abs max 1.034e-6 eV, dF max
  1.624e-7 eV/Å, position drift 0.0.

### 2.6 LAMMPS pair-style parity (1 rank)

- **Purpose:** the LAMMPS pair style must reproduce the engine's energy and
  forces on the same structure.
- **Input:** `polar_parity.data` (128 atoms), model, `pair_style
  symmetrix/mace/kk`, `run 0` + `write_dump`.
- **Metric:** dE, max dF, position drift.
- **Command:** `run_lammps_parity_r2.py` (wraps the same comparison as the
  e3lin round) under the GPU lock.
- **Result:** dE 2.910e-11 eV, max dF 2.579e-14 eV/Å → PASS.

### 2.7 LAMMPS parity 2 ranks (H1)

- **Purpose:** the H1 exchange must be exact on 2 ranks.
- **Input:** the same 128-atom NH3 box, `pair_style ... polar_mpi_message_passing`,
  `mpiexec -np 2`.
- **Metric:** dE, max dF vs the 1-rank engine reference.
- **Command:** `run_h1_parity.py` (see
  `mlip/bench/kernel_gate_2026-10-19/run_h1_parity.py`).
- **Result (pre-fix binary):** dE 2.910e-11 eV, max dF 1.413e-15 eV/Å → PASS.
  The current H1 rework is **not accepted** until this gate passes again.

### 2.8 NVT MD

- **Purpose:** a 50-step NVT run must complete with no lost atoms and stable
  energy/temperature.
- **Input:** 128-atom NH3, 300 K target, 0.5 fs.
- **Metric:** completion, lost atoms, finite values, Etot span, T range.
- **Result:** 50/50 steps, no lost atoms, Etot span 0.21295 eV, T mean 276.8 K
  (188.96–335.95) → PASS.

### 2.9 Rejected approximations

- **RBE:** unbiased and bitwise seed-reproducible, but RMS_F 4.1e-2 to
  1.0e-2 eV/Å (100–400× the gate) → rejected. A final-build p=800 check on
  128-atom NH3 gives RMS dF **6.1e-3 eV/Å**, max component **3.0e-2 eV/Å**,
  energy noise **±0.05 eV**, and **70.4 ms/step** — slower than exact Ewald
  (~53–62 ms). The observed 1/√p slope implies p≈8.5e6 for 1e-4 eV/Å, with
  ≈17 TB of cos/sin storage; the full-support vs truncated-reference mismatch
  adds a systematic 1e-4–1e-3 floor. Exact full-k Ewald remains production.
- **1-hop ghost MPI:** 2.86× faster than exact 2-hop, but dE 7.94e-5 eV and
  max dF 4.116e-3 eV/Å → rejected.
- These are documented as negative results; they are not silently enabled.

## 3. Teardown-abort note

Every gate process exits with code 134 **after** printing `GATE PASS` and
writing its JSON:

```
cudaErrorCudartUnloading: driver shutting down
```

This is a pre-existing Kokkos/CUDA teardown abort at interpreter shutdown; the
computed metrics and the PASS markers are already complete. The gate is
accepted only when the PASS marker and JSON exist. Do not treat exit 134 alone
as a failure, and do not ignore a gate that lacks the PASS marker.

## 4. Adding a new gate

1. Define the metric and the tolerance in the table above.
2. Freeze the reference (fixture dump or live recomputation) and hash the input.
3. Make the comparison independent of the engine under test where possible.
4. Run it on the pre-change and post-change builds.
5. Record the JSON and log under the round directory; update the table.
6. Only then accept the optimization.
