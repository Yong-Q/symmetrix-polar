# Plan B-micro: device-side Fukui scalars — measured null result

**Date:** 2026-10-21
**System:** 128 NH3 atoms, full PolarMACE, FP64, one A100-80GB.
**Binary:** `bench/plan_b_micro_2026-10-21/bin/lmp_micro`
sha256 `19c372c6…` (clean rebuild 2026-10-21 08:26).
**Baseline:** `bin/lmp_f0d3f05a` (Plan B-30ms accepted, 30.56 ms/eval).

## 1. What was tested

The patch removes host↔device scalar round trips in the Fukui correction:

- forward: 3 × (4-scalar D2H) → 0;
- reverse: 3 × (2-scalar D2H + 8-double H2D transform) → 0;
- LE scalar: lazy getter;
- switches `MATOS_FUKUI_DEV_REV` / `MATOS_FUKUI_DEV_FWD`; multi-rank unchanged.

## 2. Interleaved A/B (2 rounds × 12 measured `run 1`)

Per-round medians (ms/eval):

| Variant | round 0 | round 1 | median of medians |
|---|---:|---:|---:|
| base `f0d3f05a` | 30.6073 | 30.5959 | **30.6016** |
| `revonly` (`FWD=0`) | 30.5337 | 30.6183 | **30.5760** |
| `both` (`REV=1,FWD=1`) | 30.6913 | 30.5153 | **30.6033** |

All three variants are within **0.1 %** of each other.  The micro slice has
**no measurable speed benefit** at 128 atoms.

## 3. Correctness

`both` repeat-eval PASS:

```text
PE = -49242.45641531493  ×3
dPE = 0
max dF = 6.4e-16 / 8.9e-16
```

`revonly` single-point (`run 0`) can fail with `cudaErrorIllegalAddress`;
root cause: the reverse-only `else` branch dereferences the host
`scalars_host` inside a `KOKKOS_LAMBDA` (see
`bench/plan_b_micro_2026-10-21/debug/ROOT_CAUSE_REV.md`).  It passed in the
A/B (`run 1`) but remains unsafe as a diagnostic path.

## 4. Decision

- **Do not merge / do not make this the default.**  Keep the patch opt-in or
  drop it.
- The host/launch gap is not the critical path at 128 atoms for this slice;
  removing these D2H/H2D calls does not move the wall time.
- Move the effort to the measured UVU tile win (same-binary `32,32` vs
  `64,4`: 330.90 → 305.78 ms at 1,296, **−7.6 %**) and to MH-1 reverse.
- Source: `bench/plan_b_micro_2026-10-21/apply_fukui_dev.py`,
  A/B under `bench/plan_b_micro_2026-10-21/ab/`.

## 5. Caveat

The A/B was run on the shared A100 with the accepted Plan B-30ms binary as
baseline; the 30.60 ms baseline is within 0.13 % of the accepted 30.56 ms.
No gate-passed branch should be created from this slice.
