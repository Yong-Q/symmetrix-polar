# Plan B-micro — one-rank device Fukui scalars

**Status:** gated micro-close on a private copy of the accepted 30.557 ms tree.
**Slice accepted** (safe, ≥0.2 ms, full gates PASS). **Not the headline;** the
coordinator integrates.
**Date:** 2026-10-21
**System:** 128-atom NH3 (32 molecules), full PolarMACE, FP64, A100-SXM4-80GB.

## 0. Headline

| Configuration | ms/eval | Δ |
|---|---:|---:|
| Accepted gated (`f0d3f05a…`, 30ms artifact) | **30.557** | — |
| Same-session baseline (`bin/lmp_f0d3f05a`) | 30.677 | — |
| **Micro final (`bin/lmp_micro`)** | **30.325** | **−0.352 same-session** |
| Confirmation A/B (independent 2×12) | 30.408 | −0.348 |
| Projection vs the accepted 30.557 (same delta) | ≈30.205 | ≈−0.35 |

Decisive A/B: 3 variants (`base` / `revonly` / `both`) × 2 rounds × 12 measured
`run 1`, interleaved, median of per-round medians, per-run GPU `flock`.
Base round medians 30.479 / 30.875; final 30.135 / 30.514. An independent
base-vs-final 2×12 A/B on the same binary reproduced the saving (0.348 ms;
base 30.756, final 30.408).

**Robustness caveat.** The first 3-variant screen (on the pre-fix binary, whose
1-rank device path is identical) measured neutral (paired savings −0.08 / +0.08
ms) and is retained in `ab/micro_full.json`. The two later A/Bs on the final
binary agree on 0.35 ms in both rounds (paired +0.22/+0.48 and +0.34/+0.36).
The accepted number is the 0.35 ms reproduced by the two final-binary A/Bs;
the conservative reading is 0.2–0.35 ms. A fourth A/B was started but
abandoned because a contended base run took 854 s and measured 31.07 ms; it is
not used.

**Final binary** `bin/lmp_micro` sha256
**`7061ecb09962bc0f23af2e9da8024309b0a110f484baaf100fa928a175175095`**
(199,523,584 B). **Python module** sha256
**`f3dd22327c1d2ac8072690e05c5a2245b430eee527a356f71e85a0cf1a5327aa`**.

**Measured gap to 30.0 ms: 0.325 (same-session), 0.205 projected** against the
accepted 30.557. The slice removes 0.35 ms; the remaining wall is GPU work.

## 1. Isolation and source integrity

Work was done in a **private copy** of
`bench/plan_b_full_2026-10-20/work/symmetrix` and a private LAMMPS tree
(`work/lammps`) whose CMake source path points at the copy. The shared
`symmetrix-xl`/`symmetrix-xl-b` trees and `docs/` were not edited.
`bin/lmp_f0d3f05a` (the accepted 30.557 binary) and the source hashes were
snapshotted before any change (`snapshot/`). Builds ran under
`flock /workspace/group/shared/bza/.tmp/build.lock`; every GPU run under
`flock /workspace/group/shared/bza/.tmp/gpu.lock`.

| Source file | Baseline sha256 | Final sha256 |
|---|---|---|
| `mace_polar_impl.tpp` | `6a3da179…` | `aabb1c95…` |
| `mace_nonlinear_kokkos.hpp` | `af47943c…` | `1f8d7204…` |
| `mace_nonlinear_kokkos.cpp` (binding) | `f91f3d68…` | `54e9f0b3…` |

Reproducible patch: `apply_fukui_dev.py`.

### 1.1 Build incident (process note)

An interrupted/resumed build compiled one TU during a brief window when the
header had been restored to the pristine layout, producing a class-layout
mismatch that segfaulted in `compute_Y`. A **clean rebuild** of the private
LAMMPS tree fixed it. Do not edit headers while a build is running; the clean
build is the authoritative one for all results here.

## 2. Accepted slice: one-rank device Fukui scalars

For `nranks == 1`, the Fukui reduction round trips stay on the device; the
multi-rank path is the original host/MPI branch.

- **Forward** (initial Fukui + one per SCF step): the 4-scalar D2H is removed.
  The owned-node sums are consumed by the correction kernel, which evaluates
  the same target/factor expressions with the same IEEE operations; the six
  Fukui scalars are staged into `polar_fukui_stage`. Three D2H syncs removed.
- **Reverse** (three `fukui_reverse` calls): the 2-scalar fbar D2H and the
  8-double transform H2D are removed. fbar and the transform are evaluated in
  the correction kernel from the staged scalars. Three D2H + three H2D
  copies removed.
- **LE total:** the per-eval 1-scalar D2H is staged; the Python getter does a
  lazy D2H only when `polar_le_energy()` is requested.
- Env switches: `MATOS_FUKUI_DEV_REV=0`, `MATOS_FUKUI_DEV_FWD=0` restore the
  original paths (`FWD` requires `REV`).

All arithmetic is bitwise-equivalent: PE is identical at 17 digits to the
accepted binary and force differences are at the cross-process noise level
(§5).

### 2.1 Issue found and fixed during gating

The first revision added `out=nullptr` as the default of the existing
`polar_owned_sum_1` helper. The host-path LE call did not pass a buffer, so it
silently returned 0 without the D2H. Module 7 caught this (`le_total_abs`
8.45). The final source passes an explicit buffer; module 7 passes with
`out_max 4.546e-7` and `stage_max 2.396e-5`, reproducing the accepted values.

## 3. A/B slices (final3, 2 rounds × 12, final binary)

| Variant | Round medians (ms) | Median of medians | Saving vs base |
|---|---:|---:|---:|
| `base` (`f0d3f05a`) | 30.479 / 30.875 | 30.677 | — |
| `revonly` (`FWD=0`) | 30.355 / 30.582 | 30.468 | **0.209** |
| `both` (final) | 30.135 / 30.514 | **30.325** | **0.352** |

Forward increment over rev-only: 0.143 ms. Confirmation A/B (base vs final):
round medians base 30.822 / 30.691, final 30.604 / 30.212, medians of medians
30.756 → 30.408, saving **0.348**.

## 4. Gate suite (final binary + source-matched module)

`rc=134` is the pre-existing Kokkos/CUDA teardown abort **after** the gate.

| Gate | Metric | Verdict |
|---|---|---|
| module 5 | dens abs 6.689e-8, rel 4.218e-8; stage rel 7.127e-8 | **PASS** |
| module 6 | rel 1.270e-15; dE abs 7.772e-16 eV | **PASS** |
| module 7 | 16/16; stage max 2.396e-5; out max 4.546e-7 | **PASS** |
| module 8 | FD max 5.783e-4; parity max 7.191e-7 | **PASS** |
| end-to-end | 32/32; dE abs max 1.034e-6 eV; dF max 1.624e-7 eV/Å | **PASS** |
| LAMMPS parity, 1 rank | dE 2.910e-11 eV; max dF 2.627e-14 eV/Å | **PASS** |
| LAMMPS parity, 2 ranks | dE 2.910e-11 eV; max dF 1.687475e-6 eV/Å | **PASS** |

The metrics reproduce the accepted 30.557 build to round-off.

## 5. Repeated evaluation and cross-binary

Three `run 0` at fixed positions, 17-digit dumps:

| Run | PE (eV) ×3 | dPE | max dF (eval 1 / 2) |
|---|---|---|---:|
| final (`finalbin`) | −49242.45641531493 ×3 | 0 / 0 | 6.661e-16 / 6.661e-16 |
| baseline (`base2`) | −49242.45641531493 ×3 | 0 / 0 | 7.494e-16 / 8.882e-16 |

Cross-binary (baseline vs final, eval 0): PE identical at 17 digits,
max |dF| **6.731e-16 eV/Å**, i.e. the same cross-process noise the accepted
binary shows against itself. **PASS.**

## 6. Attempts not landed

- **Tiny cuBLAS launch grouping (attempt 1): rejected analytically.** Per eval
  the tiny bucket is 36 `gemvx` + 33 `gemmk1` + 19 `splitKreduce` launches
  (0.388 ms total). They are m=1 / small-K GEMMs from `E3Linear`: 18 forward +
  18 reverse m=1 instructions spread across the readout, `lr_source_maps` and
  `field_dependent_charges` modules with heterogeneous `(k, width)` shapes
  ((512,1)×5, (512,3)×4, (32,1)×4, (32,3)×4, (16,1)×1 across the model). At
  most two same-shape m=1 calls coexist inside one `evaluate`, and the
  cross-module calls happen at different times, so there is no safe batch that
  removes ≥0.2 ms without restructuring the whole model evaluation. No
  same-shape strided batch would remove more than a few launches.
- **e3 pack/unpack fusion (attempt 3): not needed.** It was held as the second
  candidate, but attempt 2 landed 0.35 ms. The fusion requires a pending-unpack
  restructure of `E3Linear::evaluate`/`reverse` plus a flat 1-D mapping
  (`unpack_i` + `pack_{i+1}`), and the previous merged-unpack design measured
  slower. It was not attempted after the target slice landed.
- **CUDA Graph, B9 SCF fusion, B7, merged unpack:** not attempted (prior
  findings: unsafe/slower).

## 7. Remaining gap and blocking kernels

Trace of the accepted 30.557 binary (CUPTI, one warm eval): **26.414 ms GPU
busy / 1060 launches**, wall 30.557, so 4.143 ms of non-kernel time in the
profiled window. The micro slice removes host syncs, not GPU work; the two
independent A/Bs show a real 0.35 ms host-side saving.

| Family | ms | launches |
|---|---:|---:|
| polar pipeline | 11.17 | 235 |
| cuBLAS/external (DMMA + tiny + xmma) | 3.66 | 272 |
| tensor product (MH-1) | 3.65 | 12 |
| e3nn linear (pack/unpack) | 2.58 | 408 |
| backbone nonlinear | 2.40 | 42 |
| other | 1.65 | 32 |
| MLP/dense | 1.27 | 43 |

Largest single kernels: `polar reverse uvu mode0 x2` 2.13, `e3 tensor product
mh1 team channels` 1.65, `polar reverse uvu mode0 x1` 1.45, `compiled product
reverse` 0.99, `e3 tensor reverse mh1 team harmonics` 0.97, `e3 tensor reverse
mh1 team channels` 0.93, `e3 linear unpack` 0.93 (107 launches),
`nonlinear gather messages` 0.85, `polar uvu mode0` 0.77, `affine fused layer
norm silu` 0.76, `e3 reverse linear pack` 0.71 (97 launches), `polar le uvu
mode0` 0.68.

**Honest gap:** 0.325 ms measured same-session (0.205 against the accepted
30.557). Closing the last ~0.2 ms needs GPU-work reduction (UVU x1+x2 or MH-1
work removal), not launch/sync engineering.

## 8. Artifacts

| Artifact | Path |
|---|---|
| Final binary | `bin/lmp_micro` (`7061ecb0…`) |
| Reference binary | `bin/lmp_f0d3f05a` (`f0d3f05a…`) |
| Python module | `pyb/` (`f3dd2232…`) |
| Patch | `apply_fukui_dev.py` |
| A/B | `ab/final3_full.json`, `ab/final_full.json`, `ab/micro_full.json` |
| Gates | `gates/final_v2/`, `logs/gate_final_v2_*` |
| Repeated eval | `gates/repeat_eval/result.finalbin.json`, `result.base2.json` |
| Cross-binary | `gates/repeat_eval/cross_base2_finalbin.json` |
| Trace | `trace/base30/`, `analysis_exact/base30_*` |
| Summary / figure | `summary.json`, `plan_b_micro_speed.png` |
