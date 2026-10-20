# Results index — what is final

**Final (authoritative):**

- [`../technical/SYMMETRIX_POLAR_TECHNICAL_REPORT.pdf`](../technical/SYMMETRIX_POLAR_TECHNICAL_REPORT.pdf)
  — extendable technical report (theory, implementation, results, gaps).
- [`KERNEL_ROUND3_FINAL.md`](KERNEL_ROUND3_FINAL.md) — **48.74 ms/eval**
  on 128 atoms (**32 NH3 molecules**), full nonlinear PolarMACE, FP64;
  1024 atoms (**256 NH3 molecules**) at **312 ms/eval**; RBE decision;
  numerical validation; remaining gap to 30 ms. This is the **latest
  full-gate-suite-passing version**: module 5–8, 32/32 end-to-end, and
  LAMMPS 1-rank parity all PASS on the round-3 snapshot.
- [`../examples/data/kernel_round3_summary.json`](../examples/data/kernel_round3_summary.json)
  — machine-readable final numbers.
- [`../../results/comparison_summary.csv`](../../results/comparison_summary.csv)
  — CSV; the round-2 rows are historical, the round-3 row is final.

**Paper comparability (separate):**

- [`PAPER_COMPARABILITY.md`](PAPER_COMPARABILITY.md) — why the 48.74 ms
  PolarMACE number must not be compared with the Symmetrix-XL paper's
  3.1–5.0× / 4.50 µs/atom (different model, precision, system size, and
  execution path), plus the matched linear MACE-OFF23 scaling measurement
  (12.26 µs/atom/step at 6,144 atoms, materialized float64).

**Plan B audit (separate):**

- [`../design/PLAN_B_AUDIT.md`](../design/PLAN_B_AUDIT.md) — what a 20–25 ms
  rewrite would require, with expected savings, effort, risk, host-sync
  inventory, and kill criteria. Conclusion: 36–40 ms is credible; 20–25 ms is
  not credible under the current FP64/exact-physics/full-gate contract.

**Post-final experiments (not in the accepted headline):**

- [`PAPER_COMPARABILITY.md`](PAPER_COMPARABILITY.md) — paper-vs-ours
  alignment; direct/JIT float32 path now measures **3.72 µs/atom/step**
  (median of medians) at 6,144 atoms, in the paper's 4.50 range.
- [`PLAN_B_REPLAY_CACHE.md`](PLAN_B_REPLAY_CACHE.md) — forward SCF replay
  cache; PE identical, max force difference 7.77e-16 (cross-process noise);
  main-tree A/B **48.86 → 45.12 ms (3.74 ms, 7.6%; 381.7 → 352.5
  µs/atom)**, LAMMPS parity gate **PASS**. The isolated-tree first run was
  49.69 → 45.02 ms. Default off; not yet the headline.
- [`MERGED_UNPACK_NEGATIVE.md`](MERGED_UNPACK_NEGATIVE.md) — merged e3-linear
  unpack: numerically correct (1e-15), but **54.69 vs 49.00 ms**, a
  performance negative; default off.

**Round-3 follow-up (pending clean A/B):**

- [`KERNEL_ROUND3_FOLLOWUP.md`](KERNEL_ROUND3_FOLLOWUP.md) — vectorized
  `nonlinear gather messages` / `gather edge message adjoint`; bitwise vs
  build 8, clean single round 49.25 → 48.63 ms. The official interleaved A/B
  is queued because another GPU job currently holds the A100.

**Historical evidence (numbers superseded by the final report):**

- [`KERNEL_ROUND2.md`](KERNEL_ROUND2.md) — round-2 accepted 56.50 ms.
- [`MPI_H1.md`](MPI_H1.md) — MPI H1 diagnosis; the H1 one-rank fix was later
  accepted by the MPI workstream, but the two-rank engine blocker remains.
- `docs/COMPARISON_REPORT_2026-10-19.md` / `.pdf` / `docs/index.html` —
  pre-round-3 consolidated comparison (round-2 headline).
- `docs/REPORT_2026-10-16.md`, `docs/REPORT_2026-10-19.md`.
- `bench/` round-1/2/3 logs and traces (workspace-only; `bench/` is
  gitignored).

**One-line baseline ladder (128 atoms = 32 NH3, full PolarMACE, FP64):**

| Implementation | ms/eval | Final 48.74 ms vs it |
|---|---:|---:|
| v28 initial polar | 6951.7 | 142.6× |
| v29 Symmetrix-XL Kokkos engine | 183.8 | 3.77× |
| round 1 | 122.2 | 2.51× |
| round 2 accepted | 56.50 | 1.159× |
| PyTorch `mace_polar` | 94.0 | 1.93× |
| **round 3 (build 8)** | **48.74** | 1× |
