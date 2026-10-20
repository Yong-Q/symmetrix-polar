# Data index — final vs historical

**Final (round 3, 2026-10-20):**

- `kernel_round3_summary.json` — 48.74 ms at 128 atoms (32 NH3),
  312.1 ms at 1024 atoms (256 NH3), speedup ladder, RBE p=800 check,
  parity numbers, final binary hash.
- `engine_compare_round3.json` — same-box engine comparison with explicit
  gate status: round-3 full PolarMACE 48.74 ms (LAMMPS parity PASS), round-2
  56.50 ms (full suite PASS), round-2 backbone 27.7 ms, and the v30 linear
  MACE-OFF23 reference rows (different model).

**Historical evidence:**

- `kernel_round2_summary.json` — round-2 accepted 56.50 ms.
- `kernel_round2_baseline_families.csv`, `kernel_round2_families.csv`,
  `kernel_round2_labels.csv` — CUPTI trace roll-ups used for the round-2
  attribution; trace-level, not headline numbers.
- `kernel_round_gate_summary.json`, `kernel_round_speed_ablation.csv/.json` —
  round-1 evidence.
- `mpi_h1_*`, `polar_*`, `rbe_*`, `engine_compare.json`,
  `homog_timing.json`, `streamed_compare.json`, `linear_mpi.json` — MPI,
  RBE, engine-comparison, and scaling evidence from earlier stages.

Headline numbers should be taken from
[`../../results/KERNEL_ROUND3_FINAL.md`](../../results/KERNEL_ROUND3_FINAL.md)
and `kernel_round3_summary.json`, not from the process files above.
