# Reading Guide — Symmetrix-Polar Report

This is the English reading guide for the consolidated comparison report. The
Chinese guide is [`阅读说明.md`](阅读说明.md); the theory and test modules are in
[`THEORY_AND_METHODS.md`](THEORY_AND_METHODS.md).

## 1. What to read first

| Goal | Read |
|---|---|
| Get the accepted numbers in two minutes | this page, §2; PDF page 2 “How to read” |
| Understand the energy model / Ewald / SCF | [`theory/ENERGY_MODEL.md`](theory/ENERGY_MODEL.md) |
| Understand e3nn linear, tensor product, UVU | [`theory/TENSOR_ALGEBRA.md`](theory/TENSOR_ALGEBRA.md) |
| Understand speedup, occupancy, Amdahl | [`theory/PERFORMANCE_MODEL.md`](theory/PERFORMANCE_MODEL.md) |
| Understand MPI H1, ghosts, communication | [`theory/MPI_H1.md`](theory/MPI_H1.md) |
| Run or audit a gate | [`testing/GATES.md`](testing/GATES.md) |
| Reproduce a speed number | [`testing/BENCHMARKS.md`](testing/BENCHMARKS.md) |
| See the round-2 result and stale-binary lesson | [`results/KERNEL_ROUND2.md`](results/KERNEL_ROUND2.md) |
| See the MPI H1 diagnosis and pending fix | [`results/MPI_H1.md`](results/MPI_H1.md) |
| See the round-3 implementation plan and 20–30 ms target | [`design/KERNEL_ROUND3_PLAN.md`](design/KERNEL_ROUND3_PLAN.md), [`design/KERNEL_ROUND3_TARGET.md`](design/KERNEL_ROUND3_TARGET.md) |

## 2. Key results at a glance

| Configuration | Round 1 | Round 2 | Speedup |
|---|---:|---:|---:|
| NH3 128 atoms, full PolarMACE | 122.46 ms/eval | **56.50 ms/eval** | **2.17x** |
| NH3 128 atoms, polar-off | 40.58 ms/eval | **27.66 ms/eval** | 1.47x |
| N2 432 atoms, full PolarMACE | 294.39 ms/step | **126.59 ms/step** | 2.33x |
| GPU busy (CUPTI, 128 atoms) | 116.66 ms | **49.97 ms** | 2.33x |

Cumulative: **3.25x vs v29**, **123x vs v28**. The PyTorch reference for the
same 128-atom frame is 94.0 ms/eval; the engine is now **1.66x faster**.

All accepted gates pass: module 5–8, 32/32 end-to-end frames, and LAMMPS
1-rank parity. See [`testing/GATES.md`](testing/GATES.md) for the exact
input/metric/tolerance/command/result of each gate.

**MPI:** the H1 one-rank defect is fixed. H1 1-rank is **418.64 -> 50.20 ms**
(default 50.84; **0.99x**); the same-binary A/B is **3.24x**, and the 1-rank
parity/NVT gates pass. The two-rank path is **blocked by engine workspace
sizing**: six 2.0-5.4 GiB workspaces per rank (~21 GiB/rank, peak
80,966 MiB) make exact 2-hop OOM and H1 SIGSEGV. A dedicated fix is in
progress; no 2-rank speed claim until it passes parity. See
[`results/MPI_H1.md`](results/MPI_H1.md).

## 3. Report map

| Section | Content |
|---|---|
| 0 | Summary and scope |
| 1 | Model and what is compared (linear vs nonlinear, MH-1 status) |
| 2 | Verification and gate status |
| 3 | Speed: v28/v29, round 1, round 2, MPI, homogeneous MD |
| 4 | Findings: cost structure, MH-1 block, RBE/1-hop rejection, one-GPU MPI |
| 5 | Kernel optimization rounds, stale-object lesson, MPI H1 diagnosis |
| 6 | Reproduction commands |
| 7 | Artifact index |

## 4. Figure index (PDF gallery)

| No. | Figure | What to look at |
|---:|---|---|
| 1 | Cost structure at 128 and 432 atoms | Which family dominates; polar is not just k-space |
| 2 | v28 → v29 speedup | The 37.9x jump that started the campaign |
| 3 | Homogeneous MD cost and 1 ns extrapolation | Production cost for five fluids |
| 4 | Linear MACE vs nonlinear backbone vs full PolarMACE | The 12x gap is model size/architecture |
| 5 | Kernel round 1 speed | e3 linear 12.07x kernel gain → 1.50x model gain |
| 6 | Kernel round 2 speed | 122.46 → 56.50 ms/eval; 432 atoms 294.4 → 126.6 ms/step |
| 7 | Kernel round 2 waterfall | 65.96 ms saved; 38.6 ms from stale-object activation |
| 8 | Kernel-family roll-up | Polar 73.9 → 20.9 ms; tensor products now 10.8 ms |
| 9 | RBE speed–accuracy tradeoff | RBE is 100–400x over the force gate |
| 10 | RBE force error vs batch size | Every batch size fails the hard gate |
| 11 | Gate margins | All accepted gates pass with large margin |
| 12 | LAMMPS parity | dE 2.9e-11 eV, max dF 2.6e-14 eV/Å |
| 13 | NVT MD stability | 50 steps, no lost atoms, Etot span 0.21 eV |
| 14 | MPI status | Correctness achieved; no one-GPU speedup |
| 15 | MPI H1 pre-fix diagnosis | Historical: 1042 ghosts, 4.87x; fixed by the one-rank fold |
| 16 | H1 exchange schematic | Forward H1, reverse adjoints |
| 17 | MPI H1 final + 2-rank blocker | H1 1-rank 0.99x default; 2-rank blocked by ~21 GiB/rank workspaces |
| 18 | Top kernels (pre-round-2) | Starting point: polar + e3 scalar linear |
| 19 | Stage spans vs GPU busy | Host stages vs measured kernel time |
| 20 | Polar-specific kernel delta | What the polar layer adds |
| 21 | Remaining kernels + SM coverage | Red <50%; tensor products 100% covered but instruction/store-bound |

Equations in the PDF gallery:

1. GTO-Ewald reciprocal energy
2. SCF fixed point
3. e3nn linear
4. e3nn tensor product
5. Static SM coverage / SM-weighted time
6. Amdahl speedup
7. H1 communication volume / multi-rank model

## 5. Reproduction and conventions

- Speed protocol: warmup 1 + 12 measured evals, 2 interleaved rounds, median of
  medians, GPU `flock`; see [`testing/BENCHMARKS.md`](testing/BENCHMARKS.md).
- `ms/eval` = one energy+force evaluation; `ms/step` = one MD step.
- SM coverage is a **static estimate** from grid/block/register data; hardware
  counters are unavailable in this container.
- A binary hash alone is not proof of the source state. Round 2 found stale
  objects; every round records the source state, CUPTI kernel names, and the
  final binary hash.
- FP64 is mandatory. FP32/TF32, dropped terms, RBE, and 1-hop ghost MPI are not
  accepted production paths.
