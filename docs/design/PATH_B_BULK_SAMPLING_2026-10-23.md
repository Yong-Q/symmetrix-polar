# Path B bulk-density sampling and FP64 4G speed target — 2026-10-23

## Why the old data was insufficient

The 457/107 and 2,166-config datasets placed **one molecule in a 16 Å box**
(H2O atom density 0.00073 atoms/Å³, ~0.7 % of liquid water). They covered
gas-phase conformations only. The PRL reference (Dijkman et al.,
PRL 134, 056103 (2025)) samples homogeneous Lennard-Jones bulk at reduced
density ρα³ ∈ [0.02, 0.67] with GCMC (μβ ∈ [−4.0, 0.5], L=10σ, 10⁵ moves) and
validates on 150 external potentials (μβ ∈ [−4, 6]). Our single-molecule box
is below that range.

## Bulk dataset (q4 `t-20261023095902-fwmjq`)

- 32 molecules per periodic box for each of H2O, NH3, C2H4, CH4, C2H6, CO2.
- Molecular number density: **1e-4, 5e-4, 2e-3, 5e-3, 1e-2, 2e-2, 3.5e-2,
  5e-2, 7e-2, 1e-1 molecules/Å³** — from extreme dilute vapor through liquid
  to above-liquid/high-pressure, covering and exceeding the PRL range.
- Temperature: **150, 200, 250, 300, 400, 600, 800, 1000, 1200, 1500,
  2000 K**.
- 150 Langevin steps per state, 3 frames/state, FP64 PolarMACE teacher labels
  (`REF_energy` / `REF_forces`); unstable extreme states are dropped and
  counted in `summary.json`.
- Up to 330 frames/species (1,980 total) before filtering.

## FP64 4G speed prototype

A PyTorch FP64 speed prototype at 5,184 atoms — random local descriptors,
K=30 neighbor aggregation, MLP 64→64→64→1, and a dense Coulomb matrix with
20 CG iterations for QEq — measured **3.75 ms/eval = 0.72 µs/atom/step**.
This excludes the real descriptor/force-trick work, but it shows that FP64
is not the blocker for a 4G-style model: **30 µs/atom/step is comfortable,
4 µs is plausible with an optimized kernel**.

## Data budget: 2,166 is a prototype, not a bulk dataset

The 2,166 gas-phase configs and the ~2k-frame bulk pilot are enough only to
validate the pipeline and the gas-phase gate. For condensed-phase density and
temperature coverage, the budget should be:

| Model | Target teacher frames | Why |
|---|---:|---|
| PolarMACE 128L1 fine-tune | 10k–20k | narrow six-molecule domain, active selection |
| 4G / linear student | 50k–100k | lower-capacity local model needs broader coverage |
| DFT validation subset | 100–500 | detect teacher bias, not training |

The 10 densities × 11 temperatures × 6 species grid gives 660 states; at
100–500 frames/state that is 66k–330k candidates before selection. Use
longer NVT/NPT trajectories (2–10 ps, sample every 10–20 steps), 4 q4 GPUs
in parallel, and an active/random hybrid to keep the labeled set near the
target. Teacher labeling at 0.1–0.5 s/frame is 1.4–7 h of GPU time per 50k
frames, so the scale-up is dominated by MD generation and teacher wall time,
not by the student.

## Next

1. Train a real 4G-HDNNP-style student (local MLP + QEq/FFT, FP64) on the
   bulk dataset and measure accuracy against PolarMACE.
2. Distill the 128L1 PolarMACE student further on the bulk data to test
   whether the <100 µs result survives condensed-phase densities.
3. A linear FP32 MACE path (4.47–4.53 µs measured) remains the surest 4 µs
   route if the 4G accuracy target is not met.
