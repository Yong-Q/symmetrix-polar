# Progress toward 80 µs/atom/step

**Date:** 2026-10-21 (container clock).
**System:** 1,296 H₂O, full nonlinear PolarMACE, FP64, exact Ewald,
one A100-SXM4-80GB, one MPI rank.
**Target:** **80 µs/atom/step** = **103.68 ms/eval** at 1,296 atoms.
**Convention:** `µs/atom = 1000 × ms/eval / 1296`.

![Progress toward 80 µs/atom](figures/PROGRESS_80US.png)

## 1. Where we are

| Quantity | Value | Status |
|---|---:|---|
| Best measured (S1.1 streaming + A1 UVU tile, generic) | **253.84 ms/eval = 195.86 µs/atom** | [M] `S1_1_A1_INTEGRATION.md` |
| A1 UVU tile on `main` (materialized anchor) | 266.21 ms = 205.4 µs/atom | [M] `UVU_TILE_MAIN_1296.md` |
| Target | 103.68 ms = 80.0 µs/atom | — |
| Remaining gap | **2.45×**, −115.9 µs/atom | — |
| Current peak VRAM (generic path) | 8.55 GiB | [M] |
| 5,184 best measured control | 986 ms = 190.2 µs/atom | [M] `S1_1_A1_INTEGRATION.md` §6 |

The measured optimization ladder is 462.6 → 311.3 → 269.3 → 257.1 → 247.7 →
230.5 → 224.1 → 205.4 → **195.9 µs/atom**. The last row is the S1.1
node-contiguous streaming merge with the A1 UVU tile.

## 2. Where the remaining 195.9 µs/atom goes

Fresh CUPTI decomposition of the merged generic binary (`6582bc6`), GPU busy
239.75 ms + 14.1 ms non-kernel:

| Component | ms/eval | µs/atom | Share |
|---|---:|---:|---:|
| polar pipeline | 87.95 | 67.9 | 34.6 % |
| tensor product (MH-1) | 66.09 | 51.0 | 26.0 % |
| backbone nonlinear | 28.85 | 22.3 | 11.4 % |
| cuBLAS / external | 22.85 | 17.6 | 9.0 % |
| MLP / dense | 20.90 | 16.1 | 8.2 % |
| e3 linear | 13.30 | 10.3 | 5.2 % |
| non-kernel (host) | 14.12 | 10.9 | 5.6 % |
| **total** | **253.84** | **195.9** | 100 % |

Source: `PERFORMANCE_DECOMPOSITION_1296.md` and
`bench/perf_decomp_1296_2026-10-21/analysis/1296_generic_summary.json`.

## 3. Work in flight

1. **N1 grouped FP64 path-GEMM for the polar UVU** (highest leverage).
   Implementation is in the isolated worktree
   `external-repos/symmetrix-xl-polar-uvu-gemm` on top of `6582bc6`
   (`mace_polar_impl.tpp` +507/−24, `mace_nonlinear_kokkos.hpp` +41).
   The microbench projects **87.95 → 22.03 ms** for the polar pipeline and
   **253.84 → 188.37 ms/eval = 145.3 µs/atom** (−25.8 %), based on
   4.845 ms forward+reverse for the 20 real checkpoint GEMMs plus
   1.885/5.632 ms forward packaging at 1,296/5,184.
   A full clean build is running; parity (128/1,296, materialized and generic)
   and an interleaved 1,296 A/B are queued after it.
   **Status: build in progress (~89 % at 14:26); no measured engine number yet.**
   `run_gemm_validation.sh` is queued: it waits for the build lock, verifies
   the binary hash, then runs 128/1,296 parity and the interleaved 1,296 A/B
   automatically.
   Checkpoint metadata check: the polar pipeline has exactly six UVU tables
   (`field_dependent_charges_maps[0,1]` dot/tp, `local_electron_energy`
   q/v), all `mul1=mul2=512`; plane counts are 4+2 per SCF step and 4+4 for
   LE, i.e. **20 path-GEMM planes per evaluation**, matching the microbench
   and the `S*scf_planes+8 = 20` Z-cache allocation. All six tables satisfy
   the new `supported()` predicate, so the GEMM path is eligible for the full
   polar pipeline at 1,296 and 5,184 atoms.
   The CPU formula check `verify_gemm_formula.py` reproduces the legacy
   per-path kernel with the grouped Z/epilogue formula on all six real tables
   (`max|diff| = 1.7e-14`, random FP64 inputs), so the remaining risk is
   engine integration and parity, not the algebra.
2. **P1 k-space chunk (K1–K3 remap).** A first 1,296 A/B of the
   `MATOS_POLAR_KSPACE_CHUNK` slice gave 266.31 ms (baseline) vs 267.40 ms
   (candidate) in the clean round — **no measurable gain**. The second
   candidate round was contaminated by an external 3× GPU slowdown
   (798.6 ms), so the slice is not promoted. This is a **negative result**,
   not a regression claim.
3. **S1.1+A1 clean revalidation.** A source-clean parity rerun
   (128-atom A1/materialized/generic plus 1,296) is executing to rule out
   the 1-ULP first-evaluation drift. It does not change the speed baseline.

## 4. Honest projection to 80 µs/atom

All forward-looking numbers are projections, tagged [P].

| Scenario | 1,296 ms/eval | µs/atom | Source |
|---|---:|---:|---|
| Measured current | 253.84 | 195.9 | [M] |
| N1 grouped GEMM | 188.37 | 145.3 | [P] `POLAR_UVU_GEMM_MICROBENCH.md` |
| V2 full, 1 GPU | 168–195 | 129.6–150.5 | [P] `ARCHITECTURE_V2_PLAN.md` |
| V2 stretch, 1 GPU | 159–169 | 122.7–130.4 | [P] |
| V2 on 4 GPUs, 1,296 (1.5–2.0×, low confidence) | 84–130 | 64.8–100.3 | [P] |
| **Target** | **103.68** | **80.0** | — |

The single-GPU FP64 exact-Ewald floor supported by the measured record is
**~123–130 µs/atom** (V2 stretch), i.e. still **1.5–1.6× above 80**.
Reaching 80 µs/atom on one GPU would require every remaining group
(tensor 51, polar 17, backbone 22, MLP 16, e3 10, host 11 µs/atom) to roughly
halve at once; no measured group has shown 2×. The routes that can reach 80
are therefore: (a) the optimistic end of 4-GPU V2 at 1,296, or (b) a
precision/approximation change (FP32/TF32 UVU, PPPM/approximate Ewald, or
1 SCF step), each of which is currently rejected by the accepted physics
contract. The 4.5 µs/atom paper-normalized target is 43× away and explicitly
out of scope for FP64 PolarMACE.

## 5. Next actions

1. Finish the N1 build, run `parity_gemm.py` (128 then 1,296; both
   materialized and generic) and `ab_gemm_1296.py`; accept only if the
   dPE/dF gates and the repeated-eval check pass. **Expected: ~145 µs/atom.**
2. If N1 passes, re-decompose the GEMM binary and attack the new top item
   (tensor product, ~51 µs/atom) with the same output-owned/path-grouped
   approach; target the 130 µs/atom class.
3. Keep the 80 µs/atom query open as a **multi-GPU or precision-policy**
   decision, not as a single-GPU optimization target.

## 6. Benchmark caveat

During the P1 A/B, the same binary produced 267 ms in one round and 799 ms
in the next (3.0×), with identical env and input. This is external GPU
contention or a clock/power state, not a code-path change. Future A/B runs
must reject rounds whose median deviates by more than ~15 % from the
same-arm previous round and re-run them; the 799 ms samples are excluded from
all conclusions above.
