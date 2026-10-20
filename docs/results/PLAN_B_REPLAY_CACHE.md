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

| mode | median ms/eval |
|---|---:|
| `MATOS_REPLAY_CACHE=0` | **49.69** |
| `MATOS_REPLAY_CACHE=1` | **45.02** |
| saving | **4.67 ms (9.4%)** |

This is larger than the 3.5 ms replay slice in the CUPTI trace. The B tree
shows a **+4.67 ms** improvement on this protocol; it has not yet been
re-measured against the accepted build 8 binary and has not passed the full
gate suite.

## Gate status

- LAMMPS 1-rank parity with cache=1: the direct invocation of
  `run_lammps_parity_b_cache1.py` hit a **Python-reference environment
  conflict** (the `symmetrix` package and the LAMMPS pair style both read
  `SYMMETRIX_BACKEND` but expect different descriptors), so it has not
  completed. This is a harness issue, not a numerical failure.
- The 17-digit differential against cache=0 is established: PE identical,
  max force difference 7.77e-16, equal to same-mode cross-process noise.
  This is strong numerical evidence but **not** a substitute for the full
  parity gate.
- Module 5–8 and 32/32 end-to-end gates are engine-independent (Python
  reference) and remain the build-8 PASS; they do not exercise the cache.
- Do not replace the headline or quote 45.02 ms until the parity gate passes
  on a clean build.

## Artifacts

- `bench/plan_b_2026-10-20/ab/result.json` — timing.
- `bench/plan_b_2026-10-20/bitwise/` — 17-digit force dumps and controls.
- `bench/plan_b_2026-10-20` — implementation notes and scripts.
- Isolated sources: `external-repos/symmetrix-xl-b`, `external-repos/lammps-b`.
