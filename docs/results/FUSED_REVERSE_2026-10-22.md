# Fused reverse MH-1 (fused2) — 2026-10-22

**Status:** implemented, exact, **promoted to default** after the clean q4 A/B.
`MATOS_MH1_REV_FUSED2=0` opts out.

## Clean q4 result (decisive)

Task `t-20261022180640-dslc9`, 1×A100, 2 rounds × 4 arms, 1,296 water. All
speeds in **µs/atom/step**.

| Arm | q4 median | Saving |
|---|---:|---:|
| Baseline (N1 + R1) | 138.91 | — |
| Fused2 reverse | **136.08** | **2.83 (2.0 %)** |
| Node-owned forward + fused2 | **133.04** | **5.87 (4.2 %)** |

The fused2-only arm changes the potential energy by 1.1e-10 eV (inside the
1e-9 eV gate); the default combination has max force difference 4.62e-14 eV/Å
against the baseline. The earlier local shared-GPU verdict (“no gain”) was
noise and is superseded.

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

The local shared-GPU A/B (on/off ratios 1.048 and 0.993) was noise; the
dedicated q4 run resolves it. Fused2 saves **2.83 µs/atom/step (2.0 %)** alone
and **5.87 (4.2 %)** together with the node-owned forward. The 16 live
`d input_2` accumulators do cost occupancy, but the removed second kernel and
its redundant reads more than compensate on a dedicated A100.

## Decision

- **Promote fused2 to the default reverse** (`MATOS_MH1_REV_FUSED2=0` opts out).
- Keep the two-kernel path as the fallback for non-16-component models.
- Bank 2.83 µs/atom/step toward the plan; 100 still needs the persistent
  SCF/host component.

## Artifacts

- Parity dumps: `bench/arch_dataflow_2026-10-22/parity_rev_off/`,
  `parity_rev_f2/`.
- Local A/B: `bench/arch_dataflow_2026-10-22/ab_1296_rev_fused2.json`,
  `ab_rev_fused2_1296.py`.
- Clean q4 A/B: task `t-20261022180640-dslc9`,
  `bench/q4_arch_ab_2026-10-22/attempt-01/ab_summary.txt`.
