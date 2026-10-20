# Round-3 post-final follow-up — nonlinear message kernels

**Status:** implemented and bitwise-verified; clean interleaved A/B pending
because another GPU job is currently holding the shared A100.

## What changed (build 9)

- `nonlinear gather messages`: four contiguous message features per thread,
  with four independent accumulators. The per-feature edge order is unchanged,
  so the reduction is bitwise-identical to build 8.
- `gather edge message adjoint`: four contiguous features per thread, giving
  the compiler a contiguous 32-byte load/store and 4× fewer threads.
- Revert switches: `MATOS_FWD_MSG_VEC=1`, `MATOS_REV_GATHER_VEC=1`.

## Verification

- Build 9 vs build 8 on the 128-atom input: PE identical, **max |dF| = 0.0**
  over all 384 force components (bitwise).
- Clean single A/B round (before contention):
  **49.25 → 48.63 ms** (≈0.6 ms, ≈1.3%).
- Official interleaved A/B (2 rounds × 12) was attempted, but all rounds were
  contaminated by another GPU job (base 137.7 ms, candidate 147.9 ms, i.e.
  ~3× the normal 49 ms); the raw file is kept as
  `speed_ab_r3gather2_INVALID_contended.json` and is **not** used. A clean
  interleaved A/B will be rerun when the A100 is idle. The clean single-round
  result above is the current evidence for this slice.

## How to reproduce

```bash
# build 9 is the current build-kernel-r3/lmp
export MATOS_FWD_MSG_VEC=4
export MATOS_REV_GATHER_VEC=4
```

The two switches default to 4; setting either to 1 restores the build-8
kernel. No model, precision, or gate tolerance changed.

## UVU reverse x2 shared-memory staging (build 10)

- Each x2 team now stages a `(64 nodes x 32 u)` `mixed_bar` tile and a
  `(4 v x 32 u)` weight tile in shared memory (≈17 KB). Four v threads share
  one node's `mixed_bar` row, so those global loads drop 4×.
- The ascending-`u` accumulation order is unchanged. **Verified:** build 10
  vs build 8 on the 128-atom input gives PE identical and
  **max |dF| = 0.0** over all 384 force components (bitwise).
- Revert switch: `MATOS_UVU_X2_MB_SMEM=0` restores the build-9 x2 kernel.
- Build 10 compiled successfully; bitwise verification **PASS**. The clean
  interleaved speed A/B is queued behind another GPU job.

## Next nonlinear slices

1. UVU reverse x2: stage `mixed_bar` in shared memory (target 0.3–0.5 ms).
2. Fuse the reverse message gather into the MH-1 reverse consumer (target
   0.5–1.0 ms).
3. Remaining small polar kernels (`rev replay dense0`, readout gates).
