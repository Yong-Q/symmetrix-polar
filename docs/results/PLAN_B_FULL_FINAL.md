# Plan B-full — MH-1 FP64 team, UVU reverse d1 split, cutoff team

**Status:** gated result on a private copy of the frozen B-core source. **Not
the accepted headline;** the coordinator integrates.
**Date:** 2026-10-21
**System:** 128-atom NH3 (32 molecules), full PolarMACE, FP64, A100-SXM4-80GB.

## 0. Headline

| Configuration | ms/eval | Δ |
|---|---:|---:|
| B-core gated build (`ed331d23…`) | 38.73 | — |
| Same-session A/B baseline (`bin/lmp_core`) | 38.51 | — |
| **B-full final** | **31.33** | **−7.18 same-session (−18.6 %)** |
| Projection vs the B-core gated headline (same delta) | ≈31.55 | ≈−7.2 |

Interleaved A/B, 2 rounds × 12 measured `run 1`, median of per-round medians,
per-run GPU `flock`. Baseline round medians 38.61 / 38.41; final 31.27 / 31.40.
The same-session baseline reproduces the B-core gated 38.73 to 0.6 %.

**Final binary** `bin/lmp_full_final` sha256
**`bcecbf9051cf8ab830e6b3af303fa44e6f0911f95768cfe57a652f7ebaef14ec`**
(198,780,712 B). **Python module** sha256
**`3a8d55d886a520629dd309324dbd72e10d4c216f0eb0aa1fec53c570889b1495`**.

**Remaining gap to 30 ms: 1.33 ms** (measured). The remaining host/launch gap
is ~4.5 ms and was not attacked (no safe graph capture).

## 1. Isolation and source integrity

All work was done in a **private copy** of the frozen B-core source
(`work/symmetrix`, overlaid from `plan_b_core_2026-10-20/src_final/` and
verified against `snapshot/source_hashes_final_core.txt`) and a private
LAMMPS tree (`work/lammps`) whose CMake source path points at that copy. The
shared `symmetrix-xl` / `symmetrix-xl-b` trees were never edited or built.

The frozen baseline was rebuilt in the private tree (binary
`1c1db48f…`) and reproduced the gated binary numerically: same PE
(−49242.45641531493, 17 digits), max |dF| **4.4e-16 eV/Å** at eval 0 and
repeated evaluations stable to 1e-15. Builds ran under
`flock /workspace/group/shared/bza/.tmp/build.lock`; every GPU run under
`flock /workspace/group/shared/bza/.tmp/gpu.lock`.

### Accepted source changes (vs `src_final`)

| File | Change |
|---|---|
| `e3nn_kokkos.cpp` | FP64 MH-1 team dispatch enabled (float unchanged); the forward team kernel branches on `write_through` (`=` when write-through, `+=` after zero-fill otherwise). `MATOS_MH1_TEAM_FP64=0` restores the fused MDRange path. This is the write-through fix for the main-tree cross-evaluation bug; the frozen float-only guard is replaced only for FP64 and the numerical contract is preserved. |
| `mace_polar_impl.tpp` | UVU reverse mode0 x1/x2: `d1=1`/`d1=3` specializations with four-`v` prefetch, same ascending-`v` FP64 accumulation order (bitwise-identical). `MATOS_UVU_REV_D1SPEC=0` restores the runtime-`d1` loop. |
| `mace_nonlinear_kokkos_impl.tpp` | `reverse convolution cutoff` team-per-edge single-stage reduction is the default; `MATOS_CONV_CUTOFF_BACKEND=scalar` restores the serial path. |
| `mace_nonlinear_kokkos.hpp`, `e3nn_kokkos.hpp`, `pair_symmetrix_mace_kokkos.cpp` | unchanged (`af47943c…`, `7cff6474…`, `df3b3f0c…`) |

## 2. Accepted slices (interleaved A/B, same binary per A/B)

| Slice | Off (ms) | On (ms) | Saving | Verdict |
|---|---:|---:|---:|---|
| **MH-1 FP64 team** (write-through fix) | 35.34* | **32.24** | **3.10** | accepted, default on |
| **UVU reverse d1 split + v-prefetch** | 36.19 | **32.36** | **3.83** | accepted, default on |
| **conv cutoff team** | 39.52 | **37.48** | **2.03** (clean-round 1.84) | accepted, default on |
| **UVU forward/LE d1 split** | 32.58 | 31.84 | 0.74 | **REJECTED** (<1 ms), reverted |

\* The MH-1 fused round 0 was externally contended (median 193.75, min 44.2 vs
the team arm 31.3–35.7); the saving is quoted against the clean fused round 1
(35.34) and the team median-of-medians (32.24); team rounds 32.08 / 32.41.
The other two A/Bs were stable in both rounds. The cutoff team slice was
measured on the gated B-core binary (its env switch is independent of the
other two), mom delta 2.03 and clean-round delta 1.84.

The three accepted slices sum to 8.96 ms of same-configuration A/B saving;
the end-to-end measurement is **7.18 ms**, i.e. the slices are sub-additive
by ~1.8 ms (the cutoff gain was measured on the slower baseline and is
partially overlapped once the polar/MH-1 work is shorter). The end-to-end
A/B is the authoritative number.

## 3. CUPTI attribution

One profiled warm eval per configuration; absolute kernel totals vary ~5 %
between traces, so the wall A/B is authoritative. Label totals (ms):

| Label | fused/scalar/generic | team/spec | Δ |
|---|---:|---:|---:|
| MH-1 forward | 3.205 | 1.739 | −1.47 |
| MH-1 reverse channels | 1.806 | 1.016 | −0.79 |
| MH-1 reverse harmonics | 1.697 | 1.058 | −0.64 |
| UVU reverse mode0 x1 | 2.773 | 1.47–1.66 | −1.1…−1.3 |
| UVU reverse mode0 x2 | 3.534 | 2.13–2.25 | −1.3…−1.4 |
| reverse convolution cutoff | 1.528 | 0.234 | −1.29 |

The MH-1 kernels moved from 37-block / 3-block / 2-block MDRange launches to
73 / 73 / 292 team blocks respectively, which is the source of the gain at
128 atoms. The UVU specialization removes the runtime-`d1` inner-loop branch
and address arithmetic while keeping the exact accumulation order.

## 4. Gate suite (final binary + source-matched module)

All gates were rerun on the final build and the module built from the same
private source. `rc=134` is the pre-existing Kokkos/CUDA teardown abort
**after** `GATE PASS`.

| Gate | Metric | Verdict |
|---|---|---|
| module 5 | dens abs 6.566e-8, rel 3.30e-8; stage rel max 7.13e-8 | **PASS** |
| module 6 | ρ̂ rel 1.27e-15, ΔE rel 1.02e-15, abs 7.772e-16 eV | **PASS** |
| module 7 | 16/16; stage max 2.396e-5 (abs≤5e-6 or rel≤1e-6); out 4.546e-7 | **PASS** |
| module 8 | FD max 5.783e-4, parity max 7.191e-7 | **PASS** |
| end-to-end | 32/32; dE abs max 1.034e-6 eV; dF max 1.624e-7 eV/Å | **PASS** |
| LAMMPS parity, 1 rank | dE 2.910e-11 eV, max dF 2.608e-14 eV/Å | **PASS** |
| LAMMPS parity, 2 ranks | dE 2.910e-11 eV, max dF 1.687475e-6 eV/Å | **PASS** |

These reproduce the B-core final gate metrics to round-off; the 2-rank path
is unchanged but was rerun for completeness.

## 5. Repeated-evaluation check (critical)

Three `run 0` at fixed positions with 17-digit force dumps:

| Configuration | PE (eV) × 3 | dPE | max dF (eval 1 / 2) |
|---|---|---:|---:|
| final default | −49242.45641531493 ×3 | 0 / 0 | 8.9e-16 / 1.1e-15 |
| MH-1 team on | same | 0 / 0 | 1.0e-15 / 8.6e-16 |
| UVU rev d1 spec on | same | 0 / 0 | 6.9e-16 / 1.0e-15 |
| conv cutoff team | same | 0 / 0 | 1.1e-15 / 8.0e-16 |

Cross-binary eval-0 forces (private frozen baseline vs final):
max |dF| **7.2e-16 eV/Å** — cross-process noise. The write-through bug that
previously accumulated into the previous result does **not** reproduce with
the fix. **PASS.**

## 6. Not landed

- **UVU fused single-pass x1+x2:** rejected at design review. x1 and x2 need
  transposed weight tiles (no row/column sharing), and each thread would do
  twice the dependent work with half the blocks; the sweep=1 result already
  showed that fewer, longer blocks regress on this workload.
- **UVU forward/LE d1 split:** implemented and bitwise-correct, measured
  **0.74 ms** (32.58 → 31.84), below the 1 ms acceptance bar. Reverted; the
  final source has no forward/LE specialization.
- **cuBLAS/external (3.56–3.91 ms):** no safe ≥1 ms design. The medium FP64
  DMMA GEMMs are 69 + 72 launches of 32×32×16 tiles; the merged-unpack path was
  already measured slower and was not repeated. A fused epilogue was not
  attempted within this session.
- **CUDA Graph / host gap (~4.5 ms):** not attempted. The reverse still has
  host-visible `polar_global_sum_host`/`create_mirror_view_and_copy` sites
  and host-side SCF/Fukui decisions, so full-eval capture is not safe; a
  forward-only graph would not cover the measured gap and would need a
  bitwise proof. Kokkos Graph support exists in this Kokkos, but capture was
  not provably safe without removing those syncs.
- **B9 SCF fusion:** not attempted (global reduction barrier per the audit).
- **B7/other MH-1 work removal:** not chased; the audit's path-weight estimate
  is still unsupported. The team dispatch change is the evidence-backed
  MH-1 slice.

## 7. Remaining gap to 30 ms

**1.33 ms** (31.33 → 30; 1.044×). The wall is now dominated by GPU work that
is close to its measured floor for this contract plus the ~4.5 ms host/launch
gap. The largest remaining kernels are cuBLAS/external (~3.6–3.9),
`polar reverse uvu mode0 x2` (~2.2), MH-1 forward (~1.7), UVU x1 (~1.6),
`e3 linear pack/unpack` (~1.4 combined), `compiled product reverse` (~1.0),
and `affine linear epilogue` (~1.0). Reaching 30 ms would require either the
0.74 ms forward/LE UVU slice to be accepted despite the 1 ms bar, plus
~0.6 ms from cuBLAS/epilogue work, or a safe host-gap graph; neither is a
single ≥1 ms gated slice in hand.

**20–25 ms remains not credible under the current FP64, exact-physics,
full-gate contract.**

## 8. Artifacts

| Artifact | Path |
|---|---|
| Final binary | `bin/lmp_full_final` (`bcecbf90…`) |
| Python module | `pyb/` (`3a8d55d8…`) |
| Private source + final hashes | `work/symmetrix/`, `snapshot/final_hashes.txt` |
| Patches | `apply_mh1_team_fp64.py`, `apply_uvu_specialize.py` (forward variant `apply_uvu_fwd_specialize.py`, not applied) |
| A/B JSON | `ab/final2_ab_full.json`, `ab/screen_mh1_full.json`, `ab/screen_uvu_full.json`, `ab/cutoff_core2_full.json`, `ab/screen_uvufwd_full.json` |
| Gates | `gates/final_full/`, `logs/gate_final_full_*` |
| Repeated eval | `gates/repeat_eval/`, `logs/`, script `repeat_eval_full.py` |
| CUPTI traces | `trace/{final_default,mh1_fused,uvu_generic,cut_scalar}/`, `analysis_exact/` |
| Summary / figure | `summary.json`, `plan_b_full_speed.png` |
