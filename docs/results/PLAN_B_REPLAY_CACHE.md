# Plan B1: forward SCF replay cache

**Status:** experimental, isolated tree only (`symmetrix-xl-b` /
`lammps-b`). **Not in the accepted headline.** The accepted result remains
build 8: **48.74 ms/eval, full gate suite PASS**.

## What changed

`MATOS_REPLAY_CACHE=1` caches the forward SCF per-step state in the polar
pipeline and feeds it to the reverse, instead of replaying steps ③④⑤:

- views `polar_rev_cache_{pe,nl_h,nl_a,nl_out,ro1,ro2,out}`, shape
  `(steps, nodes, width)`, 128 atoms / 2 steps / FP64: **6.39 MiB** total;
- forward captures after each step (`mace_polar_impl.tpp` 1248–1358,
  1485–1502), sets `polar_rev_cache_valid`;
- reverse `load_step_cache(step)` replaces `replay_step_forward(step)` when
  the switch is on (3470–3512, 3820–3845);
- default `MATOS_REPLAY_CACHE=0` keeps the legacy replay path unchanged; the
  reverse throws if the cache is enabled but not populated.

## Numerical result

| quantity | cache=0 (replay) | cache=1 | gate |
|---|---:|---:|---|
| PE (128 atoms) | −49242.4564153149 | identical | PASS |
| max abs force difference (384 components, 17-digit dump) | — | **7.77e-16** | 1e-4 |
| same-mode repeat control | 7.29e-16 | 6.66e-16 | — |

The residual is **pre-existing cross-process noise**, not a cache effect:
two fresh processes of the same mode differ by the same ~7e-16.
`MATOS_POLAR_PROFILE=1` confirms `rev5a:cache-load` (4 marks) and zero
`rev5a:replay` marks with cache=1.

## Timing (A100, 128 atoms, full PolarMACE)

Same binary, only the env switch changes; interleaved one round × 8 measured
`run 1` samples on an idle A100:

| binary | cache=0 | cache=1 | saving |
|---|---:|---:|---:|
| isolated `lammps-b` | 49.69 ms (388.2 µs/atom) | 45.02 ms (351.7 µs/atom) | **4.67 ms (9.4%)** |
| main tree `build-kernel-r3` | **48.86 ms (381.7 µs/atom)** | **45.12 ms (352.5 µs/atom)** | **3.74 ms (7.6%)** |

The main-tree numbers are the authoritative port result; the isolated-tree
run is the first measurement.

This is larger than the 3.5 ms replay slice in the CUPTI trace. The B tree
shows a **+4.67 ms** improvement on this protocol; it has not yet been
re-measured against the accepted build 8 binary and has not passed the full
gate suite.

## Gate status

**LAMMPS 1-rank parity with cache=1: PASS** on both the isolated `lammps-b`
build and the main-tree `build-kernel-r3` (2026-10-20).

| binary | energy difference | max force difference | tolerance | result |
|---|---:|---:|---:|---|
| isolated `lammps-b` | 2.91e-11 | 2.59e-14 | 5e-5 / 1e-4 | PASS |
| main tree | 2.91e-11 | 2.56e-14 | 5e-5 / 1e-4 | PASS |

The gate ran with `MATOS_REPLAY_CACHE=1` and split Python/engine backend
descriptors (the `symmetrix` package and the LAMMPS pair style use different
`SYMMETRIX_BACKEND` selectors). The trailing `cudaErrorCudartUnloading`
message is process-exit driver shutdown after `GATE PASS`, not a gate
failure.

Module 5–8 and the 32/32 end-to-end gates are engine-independent Python
reference gates and remain the build-8 PASS; the LAMMPS parity gate is the
engine-side gate for this change. The headline is still build 8 until the
clean full-suite decision; 45.02 ms / 351.7 µs/atom/step is a gate-passed
experimental candidate.

## Artifacts

- `bench/plan_b_2026-10-20/ab/result.json` — timing.
- `bench/plan_b_2026-10-20/bitwise/` — 17-digit force dumps and controls.
- `bench/plan_b_2026-10-20` — implementation notes and scripts.
- Isolated sources: `external-repos/symmetrix-xl-b`, `external-repos/lammps-b`.
