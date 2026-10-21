# Plan B-core — host-gap elimination and safe work removal

**Status:** gated result on the isolated B tree. **Not the accepted headline.**
**Date:** 2026-10-21
**System:** 128-atom NH3 (32 molecules), full PolarMACE, FP64, A100.

## 0. Headline

| Configuration | ms/eval | Δ |
|---|---:|---:|
| Accepted Plan B-lite final (`18d5fc94…`, default) | 40.95 | — |
| Same binary, same-session A/B baseline | 41.32 | — |
| **B-core final (B2 host-gap + B4 + fused MDRange MH-1)** | **38.73** | **−2.59 same-session** |
| Projection vs the accepted 40.95 headline (same delta) | ≈38.4 | ≈−2.6 |

Interleaved A/B, 2 rounds × 12, median of per-round medians, per-run GPU
`flock`. The same-session baseline (41.32) is 0.9 % above the accepted
40.95; the accepted-relative projection is therefore stated separately and
is not claimed as a measurement. The gated measurement of this build is
**38.73 ms/eval**.

Final binary `lammps-b/build-b/lmp` sha256
**`ed331d239d85c091d3bcaf14b383aa9955e7f6c20deb2269dfea6d4097a86af4`**
(198,512,528 B). Python module sha256
**`0fd372e245422fd5d6e69fe83b58fdacb342d60087711d290ba1a289b6d7e28f`**
(Python 3.12, same source tree).

## 1. Source integrity (important)

The shared `symmetrix-xl` / `symmetrix-xl-b` trees were being **modified and
rebuilt concurrently** during this session. Two independent problems were
found and handled:

1. The engine source present in the B tree at session start was not a
   consistent copy of the source that built the 40.95 binary
   (`mace_nonlinear_kokkos_impl.tpp` / `e3nn_kokkos.cpp` had already been
   edited, and were edited again at 11:01–11:15). Rebuilding that state
   reproduced a **cross-evaluation state bug**: a second evaluation in the
   same process (module 5's N-shrink sequence, or LAMMPS `run 0` + `run 1`)
   produced a 0.1–0.9 % wrong energy.
2. The main tree's new **FP64 MH-1 team kernels** have the same
   cross-evaluation bug. With the two team dispatches restored to the gated
   B-lite condition (`Precision == float`), the same sources are exact again
   (NH3→N2 density 2.88e-8; LAMMPS repeat runs bitwise equal; 17-digit force
   dump vs the accepted binary max |dF| = **7.5e-16 eV/Å**).

The final build uses this frozen source set (`src_final/`, hashes in
`snapshot/source_hashes_final_core.txt`). The shared tree was edited again
after the final build (`e3nn_kokkos.cpp` `bf227f6c…` vs the built
`fc112659…`); the gated binary and `src_final/` are authoritative for this
result.

| File | Base | Change |
|---|---|---|
| `mace_polar_impl.tpp` | B-lite final `9442f1…` | all B2 slices (patch `b2_polar_impl.patch`) |
| `mace_nonlinear_kokkos.hpp` | B-lite final `75f263…` | B2 members + 1-rank global-sum skip |
| `mace_nonlinear_kokkos_impl.tpp` | main tree B4 `11e546…` | unchanged |
| `e3nn_kokkos.cpp` | main tree B4 `91a680…` | FP64 team dispatch reverted to float-only (`b2_e3nn_guard.patch`) |
| `e3nn_kokkos.hpp` | main tree `7cff64…` | unchanged |
| `pair_symmetrix_mace_kokkos.cpp` | B-lite final `2c9cd7…` | `set_polar_global_size(comm->nprocs)` (`b2_pair_gsum.patch`) |

## 2. B2 slices

### 2.1 What was removed

| Site | Before | After |
|---|---|---|
| UVU/LE path metadata (dot/tp/LE, per SCF step + reverse) | 24 blocking D2H copies/eval | cached host vectors at construction |
| Stage-2, per-SCF-step, LE and Fukui-reverse sums | 13 full-array D2H mirrors + host serial sums | 7 single-thread device sums + 4/2/1-scalar D2H copies (same FP64 node order, bitwise on 1 rank) |
| `polar_global_sum_view` (16 calls/eval) | D2H + MPI_Allreduce + H2D on **every** rank, including 1 rank | skipped when `comm->nprocs == 1` (identity); multi-rank path unchanged |
| k-space geometry | k enumeration, 3 H2D copies, GTO-basis kernel and k-weight kernel every eval | cached by a bitwise cell key; cos/sin (position-dependent) still rebuilt; opt-out `MATOS_KGEOM_CACHE=0` |

### 2.2 Measured attribution

| Slice | A/B | Saving | Decision |
|---|---|---:|---|
| B2a metadata cache + device sums | 40.97 → 40.51 (2 × 12, intermediate build) | **0.45** | accepted, default on |
| B2b k-geometry cell cache | 39.39 → 38.68 (2 × 8, `MATOS_KGEOM_CACHE=0/1`) | **0.71** | accepted, default on |
| B2b 1-rank global-sum skip + remainder | total − above | ≈1.4 (derived) | accepted, default on |
| **B2 total** | **41.32 → 38.73 (2 × 12)** | **2.59** | **accepted** |
| UVU reverse sweep = 1 | 39.51 vs 38.68 (2 × 8) | −0.83 | **rejected**; sweep 4 stays default |

The k-geometry cache removes the `polar gto basis` kernel (0.32 ms GPU) from
every evaluation after the first, plus the host enumeration/copies.
The global-sum skip is exact on one rank; 2-rank uses the original
`MPI_Allreduce` path (verified below).

## 3. Gate suite (final binary, defaults)

All gates rerun on the final build and module. `rc=134` is the known
Kokkos/CUDA teardown abort **after** `GATE PASS`.

| Gate | Metric | Result | Verdict |
|---|---|---|---|
| module 5 | density abs/rel, stage rel | 12/12; stage rel max 7.13e-8 | **PASS** |
| module 6 | ρ̂ rel, ΔE rel, ΔE abs | 1.27e-15, 1.02e-15, 7.772e-16 eV | **PASS** |
| module 7 | 16/16, stage, output | stage max 2.396e-5 (criterion abs≤5e-6 or rel≤1e-6), out 4.546e-7 | **PASS** |
| module 8 | FD max, parity max | 5.783e-4, 7.191e-7 | **PASS** |
| end-to-end | 32 frames | 32/32; dE abs max 1.034e-6 eV; dF max 1.624e-7 eV/Å | **PASS** |
| LAMMPS parity, 1 rank | dE, max dF | 2.910e-11 eV, 2.584e-14 eV/Å | **PASS** |
| LAMMPS parity, 2 ranks exact 2-hop | dE, max dF | 2.910e-11 eV, 1.687475e-6 eV/Å | **PASS** |

These metrics reproduce the B-lite build-8 values to round-off. The 17-digit
LAMMPS probe against the accepted binary (`probe/`) gives max |dF| =
7.49e-16 and 6.40e-16 eV/Å on the two dumped frames (cross-process noise).

## 4. Host-gap decomposition

Fenced `MATOS_POLAR_PROFILE` trace, one warm eval per binary (absolute
values are inflated by the profiler's fences; the before/after comparison
uses the same protocol). Per-stage `host_ms` is the fenced span and
`gap = host − GPU busy`.

| Phase (interval ending at mark) | prior host / busy / gap | final host / busy / gap |
|---|---:|---:|
| forward backbone | 14.65 / 9.37 / 5.28 | 13.84 / 8.71 / 5.13 |
| polar Fukui | 0.33 / 0.04 / 0.29 | 0.32 / 0.07 / 0.25 |
| **polar k-geometry** | 0.66 / 0.07 / **0.59** | 0.06 / 0.01 / **0.05** |
| polar potential features | 0.62 / 0.27 / 0.35 | 0.47 / 0.29 / 0.18 |
| **SCF, 2 steps** | 7.68 / 3.83 / **3.84** | 6.91 / 3.85 / **3.06** |
| Coulomb energy | 0.23 / 0.06 / 0.16 | 0.13 / 0.07 / 0.07 |
| **local electron** | 2.16 / 1.22 / **0.95** | 1.94 / 1.17 / **0.77** |
| **polar reverse** | 12.69 / 7.59 / **5.10** | 12.06 / 7.25 / **4.81** |
| backbone reverse | 15.25 / 9.59 / 5.65 | 13.84 / 9.28 / 4.56 |
| **total eval** | 60.42 / 35.58 / **24.84** | 55.14 / 34.26 / **20.88** |

Clean-wall host gap (`wall − GPU busy`, same-trace busy): prior
**41.32 − 35.58 = 5.74 ms**, final **38.73 − 34.26 = 4.47 ms** (−1.27 ms).
Launches: 1077 → 1082 (+5 from the device sums; the k-geometry and
global-sum kernels/copies removed are not in the launch count).
GPU busy also fell 35.58 → 34.26 ms, mostly the removed GTO-basis kernel.

## 5. B9 and other work removal

- **B9 (SCF/k-space fusion): no safe design found; not merged.** The two SCF
  steps are sequential and fixed. The `assemble fourier → project features`
  pair is separated by a global node reduction (`rho(k)` must be complete
  before any node's projection), so fusing them requires a global barrier or
  recomputing the reduction per node; neither is a safe FP64-preserving
  transformation with the existing launch geometry. The remaining
  reducible launches are micro-fusions (correction+snapshot ≈0.15–0.25 ms,
  normalize+project, concat+dense0), none ≥1 ms, and B3/B5 already showed
  this class is neutral or negative on this workload.
- **B7 (MH-1 work removal): not chased.** Code inspection confirms the path
  weight is applied once per instruction; the audit estimate was not
  supported (as noted in the B-lite report).
- **CUDA graph: not attempted/claimed.** The SCF still needs host-visible
  Fukui scalar factors for the multi-rank `MPI_Allreduce` and the factor
  computation, and the Python/LAMMPS entry paths retain host decisions;
  capturing only the backbone would not cover the measured gap and would
  need a bitwise proof. No graph is claimed.
- **UVU reverse work removal:** x1+x2 together are ≈2× the forward
  bilinear arithmetic (both gradients), which is the irreducible cost of
  this contraction; no ≥1 ms safe removal was found.

## 6. Remaining gap to 30 ms

**8.73 ms** (38.73 → 30; 1.29×). The final GPU busy is **34.26 ms**, i.e.
even a zero host gap cannot reach 30 ms: ≥4.3 ms of GPU kernel work would
have to be removed. Blocking kernels (final trace):

| Kernel | ms | Note |
|---|---:|---|
| polar reverse UVU mode0 x2 | 3.57 | 2 gradients of the bilinear product |
| cuBLAS/external (272 launches) | 3.56 | e3-linear GEMMs; merged unpack rejected |
| MH-1 forward (fused MDRange) | 3.19 | FP64; B7 not supported |
| polar reverse UVU mode0 x1 | 2.76 | as above |
| MH-1 reverse channels / harmonics | 1.81 / 1.69 | 73×8 / 73×4 grids |
| reverse convolution cutoff | 1.53 | SM coverage 3 % |
| polar forward UVU mode0 | 1.10 | |
| compiled product reverse | 0.99 | |
| affine linear epilogue | 0.94 | |
| e3 linear pack/unpack | ~1.8 (combined) | 204 launches |

**20–25 ms remains not credible** under the current FP64 exact-physics
contract: it would require removing most of the polar/UVU and MH-1 work,
not launch or host-gap engineering.

## 7. Cross-checks

| Configuration | prior `18d5fc94` | final | Δ |
|---|---:|---:|---:|
| polar-off, 128 atoms (1 round × 12) | 19.65 | 19.49 | −0.16 (noise) |
| N2 432 atoms, first 10-step loop (ms/step) | 83.94 | 81.37 | −2.57 (−3.1 %) |

## 8. Artifacts

| Artifact | Path |
|---|---|
| Final binary | `bin/lmp_core` (`ed331d23…`) |
| Final Python module | `pyb/` (`0fd372e2…`) |
| Final source set + hashes | `src_final/`, `snapshot/source_hashes_final_core.txt` |
| B2 patches | `b2_polar_impl.patch`, `b2_e3nn_guard.patch`, `b2_pair_gsum.patch` |
| A/B JSON | `ab/final_full.json`, `ab/b2a_full.json`, `ab/screen_full.json` |
| Gates | `gates/final/`, `logs/gate_final_*` |
| Traces + stage tables | `trace/{base,final}/`, `analysis/*_stages_core.{csv,json}`, `analysis_exact/` |
| Exactness probe | `probe/` (17-digit dumps) |
| Summary / figure | `summary.json`, `plan_b_core_speed.png` |
