# Plan B-30ms — UVU forward/LE d1, affine epilogue, reverse LN reuse

**Status:** gated result on the private B-full work tree (baseline snapshotted
into this artifact directory). **Not the accepted headline;** the coordinator
integrates.
**Date:** 2026-10-21
**System:** 128-atom NH3 (32 molecules), full PolarMACE, FP64, A100-SXM4-80GB.

## 0. Headline

| Configuration | ms/eval | Δ |
|---|---:|---:|
| B-full gated headline (`bcecbf90…`) | 31.33 | — |
| Same-session A/B baseline (`bin/lmp_baseline`, `bcecbf90…`) | 31.80 | — |
| **B-30ms final** | **30.56** | **−1.24 same-session** |
| Projection vs the accepted 31.33 (same delta) | ≈30.10 | ≈−1.24 |

Interleaved A/B: 5 variants in one session, 2 rounds × 12 measured `run 1`,
median of per-round medians, GPU `flock` held for the whole run. Baseline round
medians 31.458 / 32.133; final 30.643 / 30.471. The same-session baseline ran
1.5 % above the gated 31.33, so the accepted-relative projection is stated
separately and is **not** claimed as a measurement. The gated measurement of
this build is **30.557 ms/eval**.

**Final binary** `bin/lmp_final` sha256
**`f0d3f05aabecb7021dca847c27434fff706a541cf5f62f545f953eadf9daf103`**
(199,011,656 B). **Python module** sha256
**`d01e0eb7e4bb719c82819e7c3f95402004005921bc7056d2711f5a0adc527c62`**.

**Measured gap to 30.0 ms: 0.557 ms** (projected 0.096 ms against the accepted
31.33). Three safe slices landed (sum of marginals 1.109 ms; end-to-end
1.238 ms). No further safe ≥0.2 ms slice was found; see §5.

## 1. Isolation and source integrity

Work was done directly in `bench/plan_b_full_2026-10-20/work/symmetrix` (the
session's private tree) **after snapshotting** the accepted state into this
directory: `snapshot/src_baseline/` (source), `bin/lmp_baseline`
(`bcecbf90…`), `pyb_baseline/`. The shared `symmetrix-xl`/`symmetrix-xl-b`
trees and `docs/` were not edited. Builds ran under
`flock /workspace/group/shared/bza/.tmp/build.lock`; all GPU runs under
`flock /workspace/group/shared/bza/.tmp/gpu.lock`.

| Source file | Baseline sha256 | Final sha256 |
|---|---|---|
| `mace_polar_impl.tpp` | `085e2b45…` | `6a3da179…` |
| `affine_mlp_kokkos.cpp` | `922ace1e…` (src_frozen) | `5fcc7bd4…` |
| `affine_mlp_kokkos.hpp` | `9a873ee7…` (src_frozen) | `8aeb657a…` |
| `e3nn_kokkos.cpp` | `d0a2b3b6…` | unchanged |
| `mace_nonlinear_kokkos_impl.tpp` | `2b0d1293…` | unchanged |
| `mace_nonlinear_kokkos.hpp` | `af47943c…` | unchanged |
| `e3nn_kokkos.hpp` | `7cff6474…` | unchanged |
| `pair_symmetrix_mace_kokkos.cpp` | `df3b3f0c…` | unchanged |

Full hashes: `snapshot/final_source_hashes.txt`; reproducible patches:
`apply_uvu_fwd_specialize_v2.py`, `apply_affine_epi_fuse.py`,
`apply_affine_ln_reuse.py`.

## 2. Accepted slices (5-variant interleaved A/B, 2 × 12)

Each "off" variant disables exactly one slice on the final binary; all other
slices stay on. Round medians (ms) are per configuration.

| Slice | Off round medians | On round medians | Off → On | Saving | Verdict |
|---|---|---|---:|---:|---|
| **UVU forward/LE d1 split** | 30.956 / 30.980 | 30.643 / 30.471 | 30.968 → **30.557** | **0.411** | accepted, default on |
| **Affine epilogue** (fuse + init) | 30.787 / 30.755 | 30.643 / 30.471 | 30.771 → **30.557** | **0.214** | accepted, default on |
| **Affine reverse LN reuse** | 31.277 / 30.806 | 30.643 / 30.471 | 31.041 → **30.557** | **0.484** | accepted, default on |

Sum of marginals **1.109 ms**; end-to-end same-session **1.238 ms** (the
marginals are measured with the other slices on, so they are not additive).
The final build with all three slices on passes all gates and the repeated-evaluation check; each slice is isolated by its env switch in the A/B above.

### 2.1 UVU forward/LE d1 split (`apply_uvu_fwd_specialize_v2.py`)

D1-specialized (`d1=1`, `d1=3`) forward mode0 accumulation applied to the
chunked (`v_begin..v_end`), non-chunked (`0..mul2`) and local-electron mode0
loops. The per-`v` pair sum keeps ascending `c` and `mixed` keeps ascending
`v`, so the FP64 accumulation order is unchanged. `MATOS_UVU_FWD_D1SPEC=0`
restores the runtime-`d1` loop. This is the slice previously rejected only for
being below the 1 ms bar; it is bitwise-correct and now measured at 0.411 ms.

### 2.2 Affine epilogue (`apply_affine_epi_fuse.py`)

Two changes, both default on:

- **EPI-FUSE** (`MATOS_AFFINE_EPI_FUSE`): when a linear output is consumed by
  the fused LayerNorm+SiLU layer, the bias/contribution epilogue is deferred
  into that kernel's first (mean) pass. The tape value is materialised as
  `source + (bias + contribution)` — the same expression order as the old
  epilogue — so all downstream values are unchanged; one launch and one
  full-tensor read are removed.
- **EPI-INIT** (`MATOS_AFFINE_EPI_INIT`): for the remaining GEMM linears, C is
  prefilled with `bias + contribution` and the GEMM runs with `beta=1`
  instead of `beta=0` plus a read-modify-write epilogue. This removes the read
  of C.

Combined saving 0.214 ms. The fusion part alone measured neutral in a
single-round screen; the init part carries the saving.

### 2.3 Affine reverse LN reuse (`apply_affine_ln_reuse.py`)

The forward fused LayerNorm+SiLU kernel already computes `mean` and
`1/sqrt(var+eps)` per row. They are stored per (LN layer, row) and reused by
the reverse fused LayerNorm+SiLU kernel, which therefore drops two full-tensor
reads and two deterministic team reductions. Forward and reverse use the same
`TeamThreadRange` reduction order and scalar expressions; the reverse result is
unchanged at the round-off level (cross-binary PE identical).
`MATOS_AFFINE_LN_REUSE=0` restores recompute.

## 3. Gate suite (final binary + source-matched module)

All gates were rerun on `bin/lmp_final` and `pyb_final`. `rc=134` is the
pre-existing Kokkos/CUDA teardown abort **after** `GATE PASS`.

| Gate | Metric | Verdict |
|---|---|---|
| module 5 | dens abs 6.689e-8, rel 4.218e-8; stage rel max 7.127e-8 | **PASS** |
| module 6 | rel max 1.270e-15; dE abs 7.772e-16 eV | **PASS** |
| module 7 | 16/16; stage max 2.396e-5; out 4.546e-7 | **PASS** |
| module 8 | FD max 5.783e-4; parity max 7.191e-7 | **PASS** |
| end-to-end | 32/32; dE abs max 1.034e-6 eV; dF max 1.624e-7 eV/Å | **PASS** |
| LAMMPS parity, 1 rank | dE 2.910e-11 eV; max dF 2.611e-14 eV/Å | **PASS** |
| LAMMPS parity, 2 ranks | dE 2.910e-11 eV; max dF 1.687475e-6 eV/Å | **PASS** |

The gate metrics reproduce the B-full 31.33 values to round-off.

## 4. Repeated-evaluation check (critical)

Three `run 0` at fixed positions with 17-digit force dumps, run three times:

| Run | PE (eV) × 3 | dPE | max dF (eval 1 / 2) |
|---|---|---|---:|
| `final` | −49242.45641531493 ×3 | 0 / 0 | 8.88e-16 / **1.055e-15** |
| `final_rerun` | −49242.45641531493 ×3 | 0 / 0 | **7.77e-16 / 7.49e-16** |
| `final_rerun2` | −49242.45641531493 ×3 | 0 / 0 | **6.11e-16 / 8.33e-16** |

**PASS.** PE is bitwise identical in all three runs and does not drift; the 1.055e-15
in the first run is the same cross-process FP64 round-off level measured on the
accepted 31.33 build (1.1e-15), and the two reruns are ≤8.4e-16. Cross-binary check
against the 31.33 binary at eval 0: PE identical (17 digits), max |dF| =
**6.661e-16 eV/Å** (cross-process noise). No state accumulation.

## 5. Remaining gap and blockers

**Measured 0.557 ms above 30.0** (projected 0.096 ms against the accepted
31.33). The wall is GPU work, not launch/host overhead:

| Blocking kernel family | ms (B-full final trace) |
|---|---:|
| polar reverse UVU mode0 x1 + x2 | 3.91 |
| cuBLAS/external (DMMA 2.12 + tiny 0.39 + xmma/other) | 3.91 |
| MH-1 team forward + reverse | 3.81 |
| e3 linear pack/unpack (408 launches) | 2.86 |
| affine MLP (B-full trace, before these slices) | 2.71 |
| polar forward UVU (mode0 + mode1 + LE) | 2.88 |
| compiled product reverse | 1.06 |
| nonlinear gather messages | 0.93 |

A final-build CUPTI trace was not re-taken; the labels above are from the
B-full final trace and the affine labels are the only ones these slices moved.

Not attempted or rejected: merged e3 unpack (already measured slower; not
redone); CUDA Graph (host-visible Fukui/fbar sums remain, capture not safe);
SCF/k-space fusion (global reduction barrier); UVU x1+x2 single-pass fusion
(rejected design: no tile sharing, fewer/longer blocks regress); B7 MH-1 work
removal (no supporting evidence). The remaining candidates (tiny cuBLAS
grouping, e3 pack/unpack regrouping) each require a workspace/layout redesign
and were not shown to clear the 0.2 ms bar with a safe design in this session.

**20–25 ms remains not credible** under the FP64, exact-physics, full-gate
contract.

## 6. Artifacts

| Artifact | Path |
|---|---|
| Final binary | `bin/lmp_final` (`f0d3f05a…`, 199,011,656 B) |
| Python module | `pyb_final/` (`d01e0eb7…`) |
| Baseline snapshot | `snapshot/src_baseline/`, `bin/lmp_baseline`, `pyb_baseline/` |
| Final source snapshot + hashes | `snapshot/src_final/`, `snapshot/final_source_hashes.txt` |
| Patches | `apply_uvu_fwd_specialize_v2.py`, `apply_affine_epi_fuse.py`, `apply_affine_ln_reuse.py` |
| Final A/B | `ab/final5_full.json` (5 variants × 2 rounds × 12) |
| Earlier A/B screens | `ab/screen_fwd_full.json`, `ab/screen_epi_full.json`, `ab/ab_epi_2x8_full.json` |
| Gates | `gates/final/`, `logs/gate_final_*`, `logs/gate_suite_final.log` |
| Repeated eval | `gates/repeat_eval/result.final*.json`, `logs/repeat_final*` |
| Summary / figure | `summary.json`, `plan_b_30ms_speed.png` |
