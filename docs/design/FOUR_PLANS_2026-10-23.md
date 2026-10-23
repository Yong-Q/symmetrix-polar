# Four-plan program — 2026-10-23

All four plans run in parallel. Speed is always **µs/atom/step**; accuracy is
held-out energy/force MAE. "Exact" means the 512-channel PolarMACE checkpoint
with FP64, 2-step SCF and exact Ewald; the students are new models.

| Plan | What | Current | Target | Gate |
|---|---|---|---|---|
| **A** | Exact PolarMACE engine optimization | 133.04 / 128.99 (1,296 / 5,184) | 100 exact | E/F parity, then speed |
| **B** | PolarMACE student distillation | 128L1: 91.64 / 95.11; 3.8 / 6.1 on 432 gas-phase | keep <100 on bulk | 5 meV/atom, 30 meV/Å on bulk |
| **C** | Custom 4G-HDNNP-like student (local MLP + QEq/FFT) | FP64 speed prototype 0.72 (fake) | **~20** | 5 / 30 on six-molecule bulk |
| **D** | Linear student distilled from PolarMACE, long-range correction if needed | linear MACE FP32 engine: 4.47–4.53 | **~4** | accuracy vs PolarMACE on bulk |

## Plan A next levers

1. Persistent polar SCF/host execution (28.1 → 8.5–12.5 projected).
2. Tensor reverse dataflow beyond fused2 (tensor 45 → 27–35).
3. Grouped small FP64 GEMM/MLP/e3 kernels (67 → 57.8).
The exact 100 target requires all three; realistic exact landing is 113–125.

## Plan B next steps

- The 43k-frame bulk density/temperature campaign is running
  (`t-20261023100114-9pdj8`, `t-20261023100130-kvvrg`,
  `t-20261023100147-8bq5c`, pilot `t-20261023095902-fwmjq`).
- Fine-tune 128L1 on the bulk set; evaluate on density/temperature-stratified
  held-out trajectories; keep the engine speed <100.

## Plan C next steps

- Build a real speed prototype: radial/angular local descriptors, small MLP,
  QEq with FFT/PME and the force trick, FP64 or mixed.
- Train/distill on the 2,166 gas data first, then the 43k bulk set.
- Target ~20 µs; report a measured number, not a projection.

## Plan D next steps

- Inspect the existing linear MACE FP32 engine path and model export.
- Distill PolarMACE into a linear MACE student on the gas data, then bulk.
- If accuracy fails, add a cheap long-range correction (QEq/mean-field) and
  re-gate; target 4–6 µs.

## Shared constraints

- Do not modify the exact engine in Plan B/C/D; Plan A owns engine changes.
- Do not overwrite the 128L1/192L1 models or the 2,166/43k data.
- Active-learning result (9.2 / 9.8 on 697) is rejected; random expansion is
  the current best.
- Every speed claim needs a q4 or local measured number with task id.
