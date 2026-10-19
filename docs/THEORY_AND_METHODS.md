# Theory, Performance Model, and Test Methodology

This is the **master index** for the scientific and engineering documentation of
Symmetrix-Polar. It is organized as independent modules so that a reader can go
straight to the part they need. Every quantitative claim in the result modules
is traceable to a JSON/log artifact under `docs/examples/data/` or a path named
in the module.

## How to read this documentation

| Reader | Recommended path |
|---|---|
| New to the project | §1 notation → `theory/ENERGY_MODEL.md` → `theory/MPI_H1.md` → `results/KERNEL_ROUND2.md` |
| Kernel / performance engineer | `theory/PERFORMANCE_MODEL.md` → `results/KERNEL_ROUND2.md` → `testing/BENCHMARKS.md` |
| MPI / distributed engineer | `theory/MPI_H1.md` → `results/MPI_H1.md` → `testing/GATES.md` |
| Reviewer / gate owner | `testing/GATES.md` → `testing/BENCHMARKS.md` → `results/KERNEL_ROUND2.md` |
| Reproducing a number | `testing/BENCHMARKS.md` → `testing/GATES.md` → the artifact path named in the result table |

## 1. Notation

| Symbol | Meaning | Unit |
|---|---|---|
| \(N\) | number of atoms in the local simulation | — |
| \(N_l\), \(N_g\) | local atoms, ghost atoms (per rank) | — |
| \(E\) | total potential energy | eV |
| \(\mathbf F_i\) | force on atom \(i\), \(\mathbf F_i=-\partial E/\partial \mathbf R_i\) | eV/Å |
| \(C\) | hidden channel count (512 for the probe model) | — |
| \(L\) | angular-momentum channels (`l_max = 3`) | — |
| \(r_c\) | model cutoff (6.0 Å for the probe model) | Å |
| \(V\) | simulation-cell volume | Å\(^3\) |
| \(T\) | wall time per energy+force evaluation | ms |
| \(S\) | speedup \(S=T_{\rm base}/T_{\rm opt}\) | — |
| \(N_{\rm SM}\) | SM count of one A100 (108) | — |
| \(G_i\), \(B_i\) | grid blocks, resident blocks per SM for kernel \(i\) | — |
| \(O_i\) | static SM coverage of kernel \(i\) | — |
| \(T_{\rm busy}\) | \(\sum_i t_i\), sum of kernel durations | ms |
| \(T_{\rm SM}\) | \(\sum_i t_i O_i\), SM-weighted busy time | ms |

**FP64 everywhere.** All production numbers use double precision. FP32/TF32 is
not used anywhere in the accepted path because it would violate the force gate.

## 2. Module map

| Module | What it answers |
|---|---|
| [`theory/ENERGY_MODEL.md`](theory/ENERGY_MODEL.md) | What energy is being computed: short-range MACE, GTO long-range Ewald, SCF fixed point, RBE estimator and why it was rejected |
| [`theory/TENSOR_ALGEBRA.md`](theory/TENSOR_ALGEBRA.md) | e3nn linear, tensor product, UVU factorization, fused gate normalization |
| [`theory/PERFORMANCE_MODEL.md`](theory/PERFORMANCE_MODEL.md) | Speedup, Amdahl, occupancy, SM-weighted time, roofline, launch overhead |
| [`theory/MPI_H1.md`](theory/MPI_H1.md) | Domain decomposition, ghost atoms, H1 feature/adjoint exchange, communication volume, one-GPU vs multi-GPU |
| [`testing/GATES.md`](testing/GATES.md) | Every numerical gate: purpose, input, metric, tolerance, command, result |
| [`testing/BENCHMARKS.md`](testing/BENCHMARKS.md) | Speed protocol, GPU locking, profiler, static-occupancy caveat, reproducibility |
| [`results/KERNEL_ROUND2.md`](results/KERNEL_ROUND2.md) | Round-2 results, waterfall, per-kernel table, gates, stale-binary lesson |
| [`results/MPI_H1.md`](results/MPI_H1.md) | H1 status, the 1042-ghost diagnosis, current fix state, multi-GPU plan |

## 3. Results at a glance (2026-10-19)

| Configuration | Before | After round 2 | Speedup |
|---|---:|---:|---:|
| NH3 128 atoms, full PolarMACE | 122.46 ms/eval | **56.50 ms/eval** | **2.17×** |
| NH3 128 atoms, polar-off | 40.58 ms/eval | **27.66 ms/eval** | 1.47× |
| N2 432 atoms, full PolarMACE | 294.39 ms/step | **126.59 ms/step** | 2.33× |
| GPU busy (CUPTI, 128 atoms) | 116.66 ms | **49.97 ms** | 2.33× |

Cumulative, same model and hardware (one A100-80GB, f64, energy+forces):

| Baseline | Time | Cumulative speedup |
|---|---:|---:|
| v28 original | 6966.79 ms/eval | — |
| v29 (P5 wiring + UVU rewrite) | 183.78 ms/eval | 37.9× vs v28 |
| kernel round 1 (e3 linear packed GEMM) | 122.21 ms/eval | 1.50× vs v29 |
| **kernel round 2 (this document)** | **56.50 ms/eval** | **3.25× vs v29, 123× vs v28** |

The PyTorch `mace_polar` reference for the same 128-atom frame is 94.0 ms/eval;
the engine is now **1.66× faster** than that reference.

All round-2 gates pass: module 5–8, 32/32 end-to-end frames, and LAMMPS
1-rank parity. The gate specifics are in [`testing/GATES.md`](testing/GATES.md).

## 4. What is *not* claimed

- The 2-rank MPI number is **not** a production speedup on one GPU. The H1
  diagnosis and the planned multi-GPU test are in
  [`results/MPI_H1.md`](results/MPI_H1.md).
- RBE and 1-hop ghost MPI are rejected by the force gate; they are documented
  as rejected alternatives, not production options.
- The SM-coverage numbers are **static estimates** from grid/block/register
  data. Hardware counters (`ncu`) are blocked in this container. See
  [`testing/BENCHMARKS.md`](testing/BENCHMARKS.md).
- The 432-atom "second 10-step loop" is pathologically slower in both binaries
  and is not used as a headline metric. The first loop is the reported metric,
  matching the prior 293.34 ms/step report.
