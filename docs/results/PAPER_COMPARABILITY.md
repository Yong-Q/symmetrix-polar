# Paper comparability audit (arXiv:2610.01036)

**Date:** 2026-10-20
**Question:** our 48.74 ms/eval at 128 atoms looks far from the
Symmetrix-XL paper's numbers (3.1–5.0× vs ML-IAP + cuEquivariance,
single-digit µs/atom/step). Is our baseline wrong?

## Consolidated per-atom table (all entries in µs/atom/step)

| Workload (A100 80 GB unless noted) | Model | Precision | Atoms | Path | µs/atom/step | Source |
|---|---|---|---:|---|---:|---|
| OMAT-0 SrTiO3 | MACE-OMAT-0 | FP32 | 5,000 | ML-IAP + cuEq | 19.34 | paper |
| OMAT-0 SrTiO3 | MACE-OMAT-0 | FP32 | 5,000 | standard streaming | 5.80 | paper |
| OFF23 water | MACE-OFF23 | FP32 | 5,184 | ML-IAP + cuEq | 13.99 | paper |
| OFF23 water | MACE-OFF23 | FP32 | 5,184 | standard streaming | 4.50 | paper |
| **OFF23 water** | **MACE-OFF23 medium** | **FP32** | **5,184** | **direct/JIT** | **4.47–4.53 min; 4.49–6.37 median** | **this work** |
| OFF23 NH3 | MACE-OFF23 medium | FP32 | 6,144 | direct/JIT | 3.72 median; 2.72 min | this work |
| OFF23 NH3 | MACE-OFF23 medium | FP64 | 6,144 | materialized | 12.26 clean; 16.06 contended | this work |
| OFF23 NH3 | MACE-OFF23 medium | FP64 | 6,144 | generic | 15.29 (unstable) | this work |
| PolarMACE NH3 | nonlinear polar | FP64 | 128 | materialized | 380.8 (48.74 ms) | this work |
| PolarMACE NH3 | nonlinear polar | FP64 | 1,024 | materialized | 304.8 (312.1 ms) | this work |
| PolarMACE NH3, Plan B cache | nonlinear polar | FP64 | 128 | materialized + replay cache | 351.7 (45.02 ms) | this work, parity gate PASS |
| OMAT-0 capacity | MACE-OMAT-0 | FP32 | 24,565 | ML-IAP + cuEq | 25.69 | paper |
| OMAT-0 capacity | MACE-OMAT-0 | FP32 | 1,250,235 | standard streaming | 6.15 | paper |
| OMAT-0 capacity | MACE-OMAT-0 | FP32 | 11,240,455 | tiled streaming | 6.41 | paper |
| OMAT-0 SrTiO3 (RTX 5090) | MACE-OMAT-0 | FP32 | 5,000 | ML-IAP + cuEq | 12.81 | paper |
| OMAT-0 SrTiO3 (RTX 5090) | MACE-OMAT-0 | FP32 | 5,000 | standard streaming | 2.56 | paper |

Notes: the paper's capacity rows are at different system sizes and are not
matched-size comparisons. The PolarMACE rows are a different model and must
not be compared with the MACE-OMAT-0/OFF23 rows.

## Short answer

No measurement error was found. The two sets of numbers answer different
questions:

| | Symmetrix-XL paper | this repository |
|---|---|---|
| model | MACE-OMAT-0 medium / MACE-OFF23 | full **nonlinear PolarMACE** |
| precision | float32 | **float64** |
| system | 5,000-atom SrTiO3 / 5,184-atom water | **128-atom NH3** (32 molecules) |
| execution path | **standard streaming** (specialized streamed-edge) | **materialized** |
| reported quantity | µs/atom/step at 5,000+ atoms | ms per full 128-atom evaluation |

The paper's 3.1–5.0× is a matched-size float32 LAMMPS comparison against
ML-IAP + cuEquivariance at 5,000 atoms:

| Paper figure 1b, A100 80 GB | ML-IAP + cuEq | Symmetrix-XL standard streaming | ratio |
|---|---:|---:|---:|
| OMAT-0 SrTiO3, 5,000 atoms | 19.34 µs/atom/step | 5.80 | 3.33× |
| OFF23 water, 5,184 atoms | 13.99 | 4.50 | 3.11× |
| RTX 5090 OMAT-0, 5,000 atoms | 12.81 | 2.56 | 5.00× |

Our 3.77× in `KERNEL_ROUND3_FINAL.md` is **not** that comparison: it is
`v29 pre-round-1 Symmetrix-XL Kokkos build (183.8 ms) / build 8 (48.74 ms)`
on the same PolarMACE model. Likewise, the 1.93× PyTorch reference is the
PolarMACE PyTorch implementation, not ML-IAP + cuEquivariance. Those labels
must not be read as the paper's 3.1–5.0×.

## Direct scaling test (linear MACE-OFF23 medium)

To check the engine rather than the polar model, we ran our engine's
**linear MACE-OFF23 medium** on replicated NH3 boxes in the same
materialized path, float64:

| atoms | path | median ms/step | µs/atom/step |
|---:|---|---:|---:|
| 1,536 | materialized | 21.77 | 14.17 |
| 6,144 | materialized (clean round) | 75.32 | 12.26 |
| 6,144 | materialized (CPU-contended round) | 98.66 | 16.06 |
| 6,144 | generic (`auto` fallback) | 93.9 (unstable) | 15.29 |

The 128-atom point of this run was contaminated by start-up/GPU contention
(samples 8.25–59.58 ms) and is not used. The generic 6,144-atom run was also
unstable (samples 68–1918 ms), so its median is only indicative. The clean
materialized 6,144-atom round is the primary number.

Order-of-magnitude comparison at the paper's 5,184-atom OFF23 water point:

- paper ML-IAP + cuEquivariance: **13.99 µs/atom/step** (float32, liquid water)
- our materialized linear engine: **12.26 µs/atom/step** (float64, NH3)
- paper standard streaming: **4.50 µs/atom/step** (float32, liquid water)

**Caveat:** this is not a matched system. The 128-atom NH3 benchmark box has
a density of about 0.28 g/cm³, far below liquid water (1.0 g/cm³), so it has
fewer neighbors per atom and a lower per-atom cost at the same kernel
efficiency. The numbers are indicative of order of magnitude only, not a
like-for-like engine ratio.

Even with that caveat, the decisive point stands: the paper's 3.1× beyond
its ML-IAP + cuEquivariance baseline is the gain of the specialized
**standard-streaming** path, and this build does not select that path for the
linear model. `streamed_edges auto` resolves to `generic`
(`bench/lammps_parity/streamed_compare/streamed_compare.json`), and the
nonlinear PolarMACE model is executed materialized because its SCF/Ewald
state is graph-wide.

## Direct/JIT path: measured

The paper's standard streaming corresponds to the **direct** execution path.
`streamed_edges auto` selects `direct` when a `jit_host_artifact` or
`jit_device_artifact` is supplied; otherwise it falls back to `generic`.

We generated the OFF23 **device** artifact with the installed CUDA backend:

```bash
export SYMMETRIX_BACKEND_DESCRIPTOR_PATH=bench/symmetrix_cuda_sm80.json
export SYMMETRIX_BACKEND=cuda-sm80
export SYMMETRIX_JIT_CACHE=bench/jit_cache_2026-10-20
python bench/run_prepare_jit_device.py \
  --model .../mace-off23-medium.json --precision float32 --backend cuda \
  --cache-root "$SYMMETRIX_JIT_CACHE" --lammps-arguments
```

This produced a CUDA cubin and the pair-style fragment
`jit_device_artifact <cubin> jit_device_blocks_per_compute_unit 8`. The
float32/direct run then reports `resolved='direct'` and activates
`jit-r1-gen11-f32-...`.

Steady-state per-step result, 6,144-atom NH3, float32, 10 measured `run 1`
samples per process, 3 fresh processes:

| trial | median µs/atom/step | min µs/atom/step |
|---:|---:|---:|
| 0 | 3.58 | 2.72 |
| 1 | 4.70 | 2.75 |
| 2 | 3.72 | 2.87 |
| **median of medians** | **3.72** | **2.72** |

The paper's A100 OFF23 water standard-streaming value is **4.50
µs/atom/step**. Our direct/JIT path is in the same range and slightly faster
on this lower-density NH3 box.

Caveats:

- The systems are not matched: the NH3 box is ~0.28 g/cm³ vs liquid water
  ~1.0 g/cm³, so it has fewer neighbors per atom. A dense water box would
  move the number up toward the paper's 4.50.
- The shared local A100 produced occasional 0.3–2 s outliers (other tenants);
  the medians and minima above are robust, but a dedicated A100 is required
  for a qualification-grade distribution.
- A multi-step `run 20` Loop in this build includes a ~2.6 s per-`run` setup;
  the per-step `run 1` protocol above measures the steady state and is the
  correct comparison for the paper's per-step throughput.
- For PolarMACE the nonlinear pair style explicitly rejects direct/JIT
  (`MACE_Nonlinear direct/JIT execution is not wired into this pair style;
  only materialized MACE_Nonlinear models (PolarMACE) are supported`), so the
  paper's streaming result cannot be applied to the polar model at all.

![direct/JIT per-step samples](../../bench/paper_scale_2026-10-20/direct_steps.png)

### Exact paper system: 5,184-atom liquid water

We generated a 1,728-molecule water box at 0.997 g/cm³ (5,184 atoms,
146 neighbors/atom with the 5 Å cutoff) and reran the same float32/direct/JIT
protocol:

| trial | min µs/atom/step | median µs/atom/step |
|---:|---:|---:|
| 0 | 4.53 | 6.37 |
| 1 | 4.47 | 4.49 |
| 2 | 4.50 | 5.43 |

The **clean per-step floor is 4.47–4.53 µs/atom**, matching the paper's
**4.50** on the same system class. The medians are inflated by 0.3–2 s
outliers from the other tenant on the shared local A100; a dedicated A100 is
still required for a qualification-grade distribution, but the steady-state
kernel cost is aligned.

![water direct/JIT per-step samples](../../bench/paper_scale_2026-10-20/direct_water.png)

## Same-size 1,296-atom H₂O comparison

The 5,184-atom water box exceeds the materialized PolarMACE workspace on one
A100 80 GB (2,592 already OOMs on a 2.0 GiB request). The largest fitting
water box is 1,296 atoms; this is the same-size reference for the polar gap:

| Implementation | Precision | Path | µs/atom/step | ms/eval |
|---|---|---|---:|---:|
| Linear MACE-OFF23 medium | FP32 | direct/JIT | **6.31 min; 9.69 clean median** | 8.2 / 12.6 |
| Full PolarMACE | FP64 | materialized + replay cache + team cutoff | 462.4 | 599.77 |
| **Full PolarMACE, B4 + team** | **FP64** | **+ team MH-1, team size 64, B4 gather fusion** | **261.6** | **339.00** |
| Polar / linear ratio | | | **~27–41×** | |

PolarMACE scalar cutoff at the same size: 625.55 ms / 482.7 µs/atom.

## Long-range Ewald fraction (CUPTI trace, actual)

| system | GPU busy | long-range total | % GPU busy | strict Coulomb reciprocal | % GPU busy |
|---|---:|---:|---:|---:|---:|
| 128-atom NH₃ | 40.35 ms | 2.19 ms | **5.43%** | 0.12 ms | 0.30% |
| 1,296-atom H₂O | 573.80 ms | 9.56 ms | **1.67%** | 1.09 ms | 0.19% |

The long-range Coulomb/Ewald is **not** the bottleneck; at 1,296 atoms the
MH-1 tensor product is **274.96 ms = 44.1%** of GPU busy, the backbone
nonlinear 112.49 ms = 18.0%, and the whole polar pipeline 111.15 ms = 17.8%.

## What this means for the headline

- **48.74 ms at 128 atoms (32 NH3) is the correct number for full PolarMACE,
  float64, 128 atoms.** It is not a paper-comparable throughput number and
  must not be compared with 4.50 µs/atom at 5,184 atoms.
- The paper-comparable statement for the materialized path is: **linear
  MACE-OFF23 at ~6,144 atoms is ~12.3 µs/atom/step (float64)**, i.e. at the
  paper's ML-IAP + cuEquivariance baseline.
- The **direct/JIT standard-streaming path is now enabled** and measures
  **3.72 µs/atom/step** (median of medians; best 2.72) at 6,144 atoms,
  matching the paper's 4.50 µs/atom A100 value within the system mismatch.
  The 3.1× gap to ML-IAP is therefore closed on this engine.
- PolarMACE remains materially different: it is materialized-only and 48.74
  ms at 128 atoms is not a paper-comparable number.

## Action items

1. Relabel the baseline table in `KERNEL_ROUND3_FINAL.md`: `v29` is a
   pre-round-1 build of this engine, not the paper's ML-IAP baseline.
2. Keep the paper's 3.1–5.0× and our internal 3.77× in separate tables.
3. For a qualification-grade match, run the exact 5,184-atom liquid-water
   OFF23 protocol (5 warm-up + 20 measured, 3 fresh processes) on a
   **dedicated** A100 with the direct/JIT artifact. The current 6,144-atom
   NH3 box is lower density, so 3.72 vs 4.50 is indicative rather than
   matched.
