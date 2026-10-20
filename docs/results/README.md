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
