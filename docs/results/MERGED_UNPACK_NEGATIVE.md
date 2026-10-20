# Merged e3-linear unpack: numerical pass, performance negative

**Status:** experimental, default off. **Not accepted, not enabled in the
headline.** The accepted headline remains build 8: **48.74 ms/eval, full gate
suite PASS**.

## What it does

The e3-linear block-sparse sum launches one unpack kernel per instruction
(488 e3-linear launches per evaluation). The merged path writes every
instruction's GEMM output into one flat buffer and replaces the per-instruction
unpacks with a single kernel driven by an `(output_offset, multiplicity, width,
columns, first_contribution, count)` plan.

## Numerical result

After two indexing fixes the merged output is numerically correct:

| system | PE | max abs force difference | gate (1e-8) |
|---|---:|---:|---|
| 128 atoms (32 NH3) | identical | **1.01e-15** over 384 components | PASS |
| 1,024 atoms (256 NH3) | identical | **1.03e-15** over 3,072 components | PASS |

Fixes required:
1. rewind `merged_offset` before the GEMM pass (plan construction had advanced
   it to the end of the flat buffer);
2. group contributions by output block with a **stable sort by
   `output_offset`** — the instruction order is weight-offset order, so writers
   of one block can be interleaved (widths `1,3,1,3`), and the old contiguous
   `first/count` range read the wrong block's contribution.

## Performance result (A100, 128 atoms, full PolarMACE)

Interleaved A/B, 2 rounds x 12 measured `run 1` samples, same binary:

| variant | median ms/eval | µs/atom/step (128 atoms) |
|---|---:|---:|
| baseline (merged off) | **49.00** | 382.8 |
| merged unpack | **54.69** | 427.3 |

The merged path is **5.69 ms slower (11.6%)**, not the 1.5–2.2 ms faster that
the Plan B audit projected.

Attribution:
- per-call plan construction dominated the first version (56.1 ms);
- caching the plan per layer and per direction (with per-object plan views and
  the shared-workspace invalidation) recovered only ~1.4 ms (54.7 ms);
- the remaining overhead is the per-pass plan rebuild for layers called once
  per pass, the per-call sorted instruction copy, and the flat-buffer GEMM
  path; the saved unpack launches are too few at 128 atoms to compensate.

## Decision

- Keep `MATOS_E3_LINEAR_MERGED_UNPACK=0` (default). Do not enable in build 8 or
  any accepted binary.
- Do not quote the merged path as a speed result. It is a **negative result**
  for the current e3-linear size mix.
- If revisited, the minimum viable change is separate persistent forward and
  reverse plan views per layer (no rebuild per pass) plus caching the sorted
  instruction order; measure again before enabling.

## Reproduce

```bash
# numerical
TEST_CASE=merged_unpack_probe python bench/kernel_round3_2026-10-19/test_merged_unpack.py
TEST_CASE=merged_unpack_probe_1024 TEST_REPLICATE="2 2 2" TEST_STEPS=0 \
  python bench/kernel_round3_2026-10-19/test_merged_unpack.py

# A/B (same binary, env switch)
KERNEL_GATE_LOCK_HELD=1 python bench/kernel_round3_2026-10-19/speed_ab_r3.py \
  --binary base=.../build-kernel-r3/lmp \
  --binary merged=bench/kernel_round3_2026-10-19/lmp_merged.sh \
  --rounds 2 --measured 12 --prefix merged_unpack_cachefix
```

Artifacts: `bench/kernel_round3_2026-10-19/speed_ab_merged_unpack_cachefix.json`,
`merged_unpack_probe_cachefix.log`, `merged_unpack_probe_1024.log`.
