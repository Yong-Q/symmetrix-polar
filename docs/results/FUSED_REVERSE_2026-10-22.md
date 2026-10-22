# Fused reverse MH-1 (fused2) — 2026-10-22

**Status:** implemented, exact, **not promoted**. The two-kernel reverse remains
the default. Opt-in `MATOS_MH1_REV_FUSED2=1`.

## What was built

The reverse MH-1 previously ran two kernels per edge:

1. a per-edge team over channels for `d input_1` and `d weights`;
2. a per-edge, per-harmonic-component team for `d input_2`.

The fused2 kernel runs one team per edge. Each thread owns a contiguous
channel set, keeps the 16 `d input_2` components in registers with the
harmonic component as an unrolled compile-time index, and performs one
shared-memory reduction over threads. The sparse terms are regrouped by
`(instruction, global input_2 component)` at model load; the regrouped plan is
a re-association of the same terms, not a different operator.

## Parity

1,296 water, one `run 0`, same binary, `fused2` on/off:

| Quantity | Result |
|---|---|
| Potential energy | identical (−895939.24763167975 eV) |
| max position difference | 0.0 Å |
| max force difference | **4.97e-14 eV/Å** |
| gate | 1e-8 eV/Å → PASS |

## Speed

The local A100 was shared with external processes (67–100 % utilization,
1.6–6.5 GiB held by other containers), so wall-clock A/B is dominated by
contention. In one interleaved two-round A/B the on/off ratio was **1.048**
and **0.993** (median **1.02**); the minimum-step samples were also 2–3 %
slower for fused2. In the same window the absolute values swung between
**299 and 452 µs/atom/step**, so this is a “no measured gain” verdict, not a
clean speed claim.

The likely cause is register pressure: 16 live `d input_2` accumulators plus
the per-channel `d input_1`/`d weights` scalars reduce occupancy relative to
the two simple kernels. The two-kernel path is already sparse and memory-
bound, so the fusion saves redundant reads but loses more to occupancy.

## Decision

- Keep the two-kernel reverse as the default.
- Keep fused2 opt-in for a future occupancy-tuned variant (smaller channel
  block, warp-level reduction, or a 32-thread team).
- Do not count it toward the 100 µs/atom/step plan.

## Artifacts

- Parity dumps: `bench/arch_dataflow_2026-10-22/parity_rev_off/`,
  `parity_rev_f2/`.
- A/B: `bench/arch_dataflow_2026-10-22/ab_1296_rev_fused2.json`,
  `ab_rev_fused2_1296.py`.
- Clean q4 A/B (baseline/node/rev/both): task `t-20261022180247-qqdqw`,
  `bench/q4_arch_ab_2026-10-22/attempt-01/`.
