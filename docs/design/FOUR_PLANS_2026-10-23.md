# Four-plan program — 2026-10-23

All four plans run in parallel. Speed is always **µs/atom/step**; accuracy is
held-out energy/force MAE. "Exact" means the 512-channel PolarMACE checkpoint
with FP64, 2-step SCF and exact Ewald; the students are new models.

| Plan | What | Current | Target | Gate |
|---|---|---|---|---|
| **A** | Exact PolarMACE engine optimization | 133.04 / 128.99 (1,296 / 5,184) | 100 exact | E/F parity, then speed |
| **B** | PolarMACE student distillation | 128L1: 91.64 / 95.11; 3.8 / 6.1 on 432 gas-phase | keep <100 on bulk | 5 meV/atom, 30 meV/Å on bulk |
| **C** | Custom 4G-HDNNP-like student (local MLP + QEq/FFT) | real FP64 prototype: **19.26 converged / 13.45 at 60 CG** at 5,184, 72k params; accuracy **not gated** (gas 38.2/40.3; bulk 73.2/95.0) | **~20** | 5 / 30 on six-molecule bulk |
| **D** | Linear student distilled from PolarMACE, FP64 only | gas gate PASS in FP32 student (**0.924 / 3.984**); FP64 inference re-eval pending; FP32 reference 4.473, FP64 generic probe 284.8; bulk **BLOCKED** | **~9–10 FP64** | 5 / 30 on gas and bulk |

## Plan A next levers

1. Persistent polar SCF/host execution (28.1 → 8.5–12.5 projected).
2. Tensor reverse dataflow beyond fused2 (tensor 45 → 27–35).
3. Grouped small FP64 GEMM/MLP/e3 kernels (67 → 57.8).
The exact 100 target requires all three; realistic exact landing is 113–125.

## Plan B status and next steps

- **Bulk campaign (snapshot 03:32Z):** pilot complete (975 frames); shard
  C2H4 3,300, NH3 4,400; 15,641 frames merged with gas, 0 duplicates, all
  labels finite and stable.
- **Split:** whole-state holdout, 1,098 train / 176 valid states; zero state
  or frame overlap. ACF says N_eff **4,165 / 15,641 (26.6%)**; 43k raw frames
  are not 43k independent samples.
- **Pipeline:** `scripts/prepare_bulk_dataset.py`,
  `finetune_student_bulk.py`, `eval_student_bulk.py` validated end-to-end.
  Fine-tune staged but **not launched** (shards still running).
- **Smoke:** gas-only 128L1 is fine on gas (4.60 / 4.93 meV) but diverges on
  dense bulk (one CH4 frame ΔE 1.3e7 eV/atom); labels are clean, so this is
  the expected OOD failure. Bulk fine-tuning is required.
- **Next:** rerun prepare after all shards land; launch the staged 128L1
  fine-tune on the q4 slot freed by the pilot; evaluate per species / density
  / temperature; gate 5 meV/atom, 30 meV/Å; keep <100 µs/atom/step.

## Plan C status and next steps

- **Architecture:** 4G-HDNNP: element-resolved radial G2 and angular G4
  descriptors (48 + 80 features), a 72k-parameter local MLP, global QEq with
  FFT/PME Coulomb, charges fed back into the atomic energy, and a force
  trick (one adjoint solve + envelope theorem). This is our PyTorch
  reimplementation; the separate H4 owner is aligning it to RuNNer2 defaults.
- **Speed (measured):** FP64, 5,184 atoms, full E+F **19.26 µs/atom/step**
  converged (88 CG, residual 6.6e-10) and **13.45** at 60 CG
  (residual 8.8e-7); energy-only 8.48; neighbor 2.36; descriptors 0.44.
  FP32 is 15.10, not faster. The earlier 0.72 was a kernel-only fake
  prototype and is superseded.
- **Accuracy (measured, failing):** gas 38.15 meV/atom / 40.26 meV/Å (combined
  gas+bulk model); bulk pilot 73.21 / 94.99; bulk2 C2H4 94.92 / 116.15.
  Best gas-only model is 23.18 / 32.61 (force 1.09x gate, energy 4.6x).
  C2H4/C2H6 energy bias and CO2 forces dominate.
- **Failure fixed:** the gas-only feature scaler amplified absent O-O channels
  by up to 1e5, giving 22 eV/atom on bulk; an RMS floor plus a combined
  gas+bulk scaler/training cut the bulk error to 73 meV/atom.
- **Next:** train on the full 43k bulk corpus with cell-grouped batches and a
  gas+bulk schedule; then re-gate. Port the forward to LAMMPS only after the
  accuracy gate.

## Plan D status and next steps

- **Done:** 128-channel linear student, 848,528 parameters, initialized from
  MACE-OFF23 medium (H/C/N/O), 123 epochs on 1,734 gas frames. Held-out gas:
  **0.924 meV/atom / 3.984 meV/Å**; speed **4.473–4.499 µs/atom/step** at
  5,184 atoms (FP32). Report: `docs/design/PLAN_D_LINEAR_2026-10-23.md`.
- **Blocked:** dense-water stress probes fail (2.27 eV/atom, 18.2 eV/Å)
  because the student has seen gas only. A fixed-charge Gaussian-Ewald
  correction is designed and fitted, but it does not fix the short-range
  distribution shift (<0.1 % change).
- **Next:** retrain on bulk frames once the 43k campaign lands, re-gate on
  density/temperature-stratified held-out trajectories, then decide whether
  the mean-field correction is still needed. Do not promote the gas-only
  model as a bulk potential.

## Shared constraints

- Do not modify the exact engine in Plan B/C/D; Plan A owns engine changes.
- Do not overwrite the 128L1/192L1 models or the 2,166/43k data.
- Active-learning result (9.2 / 9.8 on 697) is rejected; random expansion is
  the current best.
- Every speed claim needs a q4 or local measured number with task id.
