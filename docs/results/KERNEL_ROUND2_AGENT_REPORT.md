# Symmetrix-Polar kernel round 2 — final report (2026-10-19)

## Headline

| metric | before (accepted lmp) | after (round 2) | speedup |
|---|---|---|---|
| 128-atom NH3, full polar, wall ms/eval | **122.46** | **56.50** | **2.167x** (−65.96 ms) |
| 128-atom NH3, polar off, wall ms/eval | 40.58 | 27.66 | 1.467x |
| 432-atom N2, full polar, ms/step (first 10-step loop) | 294.39 | 126.59 | 2.325x |
| GPU busy (CUPTI, one eval) | 116.66 ms | 49.97 ms | 2.33x |

Wall protocol: one warmup `run 1` + 12 measured `run 1`, two interleaved rounds
(baseline, candidate, baseline, candidate), median of the two per-round medians.
Polar-off: same protocol. 432-atom: existing N2 NVT input, first of two `run 10`
loops, 2 rounds (this is the metric used by the prior 293.34 ms/step report).
Note: the *second* `run 10` loop is pathologically slower in both binaries
(baseline 1.42–1.53 s/step, final 0.44–0.48 s/step); that is pre-existing
behaviour, not caused by these changes.

## Most important finding: the accepted lmp was stale

The accepted binary `7e9c6bec…` was linked from stale objects. At runtime it
still executed the **old per-node `polar reverse dense`** (26.00 ms/eval, grid
1×1×1, 128 threads) and the **unfused reverse gate normalization** (13.03 ms/eval),
even though the working tree already contained (a) the 2-D `MDRangePolicy`
`dense_reverse` and (b) the fused `mh1 fused reverse gate normalization` path.
The object files were newer than the source but contained the older code
(most likely copied in from a cached build). A forced rebuild of
`mace_polar_impl.tpp` (content unchanged: only `touch`) activated both:

* `polar reverse dense`: 26.00 → 0.42 ms/eval (2-D MDRange over node×input-channel,
  same `o=0..out_w-1` accumulation order).
* reverse gate normalization: fallback 13.03 ms → fused path 0.17 ms/eval.

Those two alone account for ≈39 ms/eval. No source edit was needed for them.

## Changes made (this round)

All edits are in files owned by the kernel agent. No pair_symmetrix file was touched.

1. `libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp:5996-6046` — task (a):
   parallel two-pass fallback for reverse nonlinear normalization. Pass 1:
   MDRange over `(node, k)` writes `linear_adj` (single writer, identical
   expression). Pass 2: one thread per node folds `density_adj` over k in the
   original ascending order (bitwise-identical FP64 reduction). Revert switch:
   `MATOS_RN_BACKEND=scalar`. For this model the branch is dormant (the fused
   path is used), but it was validated bitwise with `MATOS_FUSED_GATE_REVERSE=0`.
2. `libsymmetrix/source/mace_polar_impl.tpp:680` (`polar project features`),
   `:1002` (`polar scf project features`), `:2478` (`polar rev replay project`) —
   task (c). Each output column j uses only `feat[permutation(j)]`, so the
   kernel is now MDRange over `(node, j)`; each thread runs the identical
   ascending-k accumulation. 6.14 → 1.00 ms/eval.
3. `libsymmetrix/source/mace_polar_impl.tpp:1974-2066` — task (b). Reverse UVU
   mode-0 x1/x2 adjoints rewritten with `TeamPolicy` + a shared-memory weight
   tile (4×`mul` weights, padded stride; ~16 KB/team) and `mixed_bar` hoisted
   from the x2 inner loop into `polar_rev_uvu_mixed_bar`. Each thread keeps the
   original ascending v (x1) / u (x2) accumulation. 16.00 → 7.47 ms/eval
   (x1 5.45 → 3.21, x2 10.90 → 3.55). The `d1>8` generic MDRange fallback is kept.
4. `libsymmetrix/source/mace_polar_impl.tpp:873/924` (forward `polar uvu mode0`),
   `:1486/1536` (LE), `:2593/2633` (replay) — same smem weight-tile scheme for the
   forward/replay/LE mode-0 UVU products (with the same generic fallback).
   17.19 → 3.31 ms/eval.

## Per-kernel CUPTI time (ms per eval, 128-atom NH3 full model)

| kernel / region | before | after |
|---|---:|---:|
| polar reverse dense | 26.00 | 0.41 |
| reverse nonlinear normalization | 13.03 | 0.00 (fused path 0.17) |
| polar reverse uvu mode0 x2 | 10.90 | 3.55 |
| polar reverse uvu mode0 x1 | 5.45 | 3.21 |
| polar uvu mode0 (forward) | 5.76 | 1.20 |
| polar rev replay uvu mode0 | 5.77 | 1.20 |
| polar le uvu mode0 | 4.59 | 0.91 |
| polar scf project features | 2.39 | 0.39 |
| polar rev replay project | 2.38 | 0.39 |
| polar project features | 1.22 | 0.21 |

Family roll-up (ms/eval): polar pipeline 73.89 → 20.87; backbone nonlinear
19.37 → 6.38; tensor product 11.09 → 10.81; cuBLAS/external 3.84 → 3.80;
MLP/dense 3.96 → 3.78; e3nn linear 2.89 → 2.68; other 1.53 → 1.60.
Remaining top kernels: e3 tensor reverse mh1 fused channels 4.29, cuBLAS
(312 small GEMMs) 3.80, reverse UVU x2 3.55, e3 tensor product 3.34,
e3 tensor reverse harmonics 3.10, gather edge message adjoint 2.01.

Trace labels/files: `analysis_exact/{base_full,r2_full,r3_full,r4_full,r5_full}_*.{csv,json}`.
Each trace is one warm eval selected between the `eval:enter` and
`backbone-reverse:done` profile markers; Kokkos pf_begin events are paired to
CUPTI kernels by monotone timestamp (873/882 paired; the rest are unlabelled).

## Numerical validation

Incremental binaries were compared at the LAMMPS level on the 128-atom input
(PE and all 128×3 force components):

* r2 → r3 (project kernels): PE identical, max |dF| = 0.0
* r3 → r4 (reverse UVU smem): PE identical, max |dF| = 0.0
* r4 → r5 (forward/replay/LE UVU smem): PE identical, max |dF| = 0.0
* RN fallback 2-D vs `MATOS_RN_BACKEND=scalar` (with `MATOS_FUSED_GATE_REVERSE=0`):
  PE identical, max |dF| = 0.0

So every kernel change is bitwise-neutral on this model; the only non-bitwise
change relative to the accepted binary is the activation of the already-written
fused gate-reverse path (which is *more* accurate: module5 dens abs max dropped
from the e3lin-round level to 6.7e-8).

## Final gates (all PASS)

| gate | verdict | metric | tolerance |
|---|---|---|---|
| module5 | PASS | dens abs 6.69e-8, rel 4.22e-8, stage rel 7.13e-8 | 5e-6 / 1e-6 |
| module6 | PASS | dE abs 5.55e-16 eV, rel 1.09e-15 | 5e-6 eV / 1e-6 |
| module7 | PASS | stage rel 1.17e-7, out 4.55e-7 | 1e-6 / 1e-6 |
| module8 | PASS | FD max 5.78e-4, parity 7.19e-7 | 1e-3 / 1e-3 |
| endtoend | PASS | 32/32 frames, dE rel 1.08e-11, dE abs 1.03e-6 eV, dF 1.62e-7 eV/Å | 1e-6 / 5e-6 eV / 1e-4 |
| LAMMPS 1-rank parity | PASS | dE 2.91e-11 eV, max dF 2.58e-14 eV/Å | 5e-5 eV / 1e-4 |

Gate logs: `r2_gate_module{5,6,7,8}.log`, `r2_gate_endtoend.log`,
`r2_gate_lammps_parity.log`; JSON in `gates/`.
Every gate process exits with code 134 **after** printing GATE PASS and writing
its JSON: `cudaErrorCudartUnloading: driver shutting down` during interpreter
teardown. This teardown abort is pre-existing — the same error appears in all
e3lin-round gate logs — and does not affect the computed metrics.

## Rejected / not pursued

* UVU mode-1 kernels (0.61–0.67 ms each) were left as MDRange; a smem rewrite
  would save well under 1 ms.
* The RN 2-D fallback is dormant for this model (fused path used). It is kept
  because it is the requested safety net for models where the fused gate
  reverse is unavailable, and it was proven bitwise with the env revert.
* The `polar reverse dense` 2-D source was already present and correct; no edit
  was made. Only the stale object was replaced by rebuilding.
* Further e3 tensor-product work was not attempted: those kernels already run at
  100% SM coverage / 62–100% occupancy.

## Binaries and environment notes

* Shared build-path `lmp` at final verification (11:33 UTC): sha256
  `820953cd983cd0ecd6107942bcbec0b39837ffa0a44bbac0cddd69dc3b59355c`.
  The MPI agent relinked this path twice during my round (`7e9c6bec` was the
  accepted baseline; `2c063450` at 11:23; `820953cd` at 11:31). Every relink
  compiled the same kernel sources and measured 55.97–57.14 ms/eval, i.e.
  identical within noise to my snapshot. The hash may change again if the MPI
  agent rebuilds; `make -n lmp` confirms no kernel TU is out of date.
* My snapshot of the final build: `lmp_r5`, sha256
  `54d8fe3978aa07f559224cea048489b1e433981804c45566a52a94bc1baef855`
  (56.12 ms/eval).
* Python package snapshot used for gates: `pyfinal_r2`, module sha256
  `cb7098a08cfbe17fed31c17246b66669a67c80cd2fc97e607159b1a8ab4381b5`.
* Kernel env modes: default = parallel/smem paths. Reverts:
  `MATOS_RN_BACKEND=scalar` (reverse-normalization fallback only),
  `MATOS_E3_LINEAR_CUDA_BACKEND=scalar` (pre-existing e3 linear revert).
* The MPI agent must know: `mace_polar_impl.tpp` and
  `mace_nonlinear_kokkos_impl.tpp` were edited; `pair_symmetrix/*` was not.
  The shared `lmp` now contains the new UVU/project kernels.

## Artifacts (all under this directory)

* `FINAL_REPORT.md`, `summary.json`, `kernel_before_after.csv`
* `round2_speedup_summary.png` (figsize 16×6.6, dpi 160)
* `speed_ab_r5ab.json` (3-way A/B), `speed_ab_r4ab.json`, `speed_ab_r2ab.json`,
  `speed_polaroff.json`, `speed_n2_432.json`, `speed_ab_sharedcheck.json`
* `analysis_exact/` — per-tag label/family/kernel CSVs and JSON summaries
* `trace_runs/{base_full,r2_full,r3_full,r4_full,r5_full}/` — raw CUPTI + Kokkos traces
* `gates/` — gate JSON/PNG; `bitwise/` — PE/force dumps and comparisons
* Build logs: `build_r2*_symmetrix.log`, `build_r2*_lmp.log`, `build_pymodule_r2*.log`
