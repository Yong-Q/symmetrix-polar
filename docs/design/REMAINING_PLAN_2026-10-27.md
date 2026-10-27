# Remaining plan — charge/multipole student (2026-10-27)

Status snapshot after the 2026-10-27 charge/l=1 work. The main line is the
linear MACE student plus q/μ heads and a direct 4-channel PME. The unfinished
items are listed here so they survive context switches.

## Done

- Plan D 50 ps reweighting: ESS 3.37 %, σ_β 10.83 (marginal; 7× better than OFF).
- Route decision: linear distillation is the E/F main line; 4G accuracy line
  closed (v2 baseline 97/122, far from the 20/50 kill criterion).
- Fixed-state probe: 0.5–0.6 meV/atom at 96 atoms; 0.20–0.27 relative on the
  1,152-atom box; 0.1 is below the local-model floor.
- Charge head: MLP on MACE 0e features, q MAE **0.00945 e**, R² 0.9962.
- 4G χ/J charge stage: fails (0.34–0.46 e and diverges); closed.
- Direct 4-channel q+μ PME: **4.3–9.3 µs/atom** at 1,152 atoms; with the
  linear backbone the total is **17–22 µs** (40–50 steps/s at 1,152 atoms).
- State-split oracle multipole comparison (600 frames, 226 states):
  H2O l=1 1.30→1.05 meV/atom; monopole wins for C2H6/CH4/CO2; C2H4/NH3 do not
  benefit; a single global weight does not transfer.
- Dataset audit: 23,376 train frames / 1,100 states; stride-4 covers only
  496 states; ~4,450 effective independent samples in the full 26,841-frame
  corpus. State supplement built: 965 frames to bring every state to ≥5.
- Git: commits through `e40b981` pushed.

## In progress

1. **Full q/μ labels**
   - `data_20261025_charges/train_stride4_multipole.xyz` — 5,844 frames, done.
   - `data_20261025_charges/valid_stride4_multipole.xyz` — 867 frames, done.
   - `data_20261025_charges/train_supplement_multipole.xyz` — 965 frames,
     labeling now (~6 min).
2. **Dense homogeneous tranche** (0.035 molecules/Å³, 64-molecule boxes,
   L=12.23 Å, T=300/400/500 K, six species) — collection script ready
   (`bench/plan_d_bulk_2026-10-25/collect_dense_slabs.py`), not started.
3. **Inhomogeneous validation tranche** (liquid-vapour slabs: H2O 300/400/500 K,
   CO2 300 K, NH3 300 K; 125 molecules, cell elongated 3× in z) — same script,
   not started.

## Pending

1. Merge stride-4 + supplement into a **6,809-frame / 1,100-state** train set;
   keep `valid_stride4_multipole` (867) for state-split evaluation.
2. Extract MACE 0e/1o features for the merged train/valid and train the q/μ
   heads (freeze backbone). Target q ≤0.01 e, μ relative ≤2–5 %.
3. Add an environment-dependent gate mixing monopole and l=1; loss =
   E + λ_F F + λ_q q + λ_μ μ; state-split six-species evaluation.
4. Teacher-label the dense and slab frames with MACE-POLAR E/F/q/μ.
   - Dense frames go into training (dense homogeneous coverage).
   - Slab frames go into a **validation** set; compute density profiles and
     compare to the PRL 800-density validation bins (validation-level only).
5. Measure the hybrid at 1,152 atoms on a dedicated A100 (local or Q4):
   backbone + q/μ heads + PME, target 17–22 µs; 30 fps ceiling 1,600–1,900 atoms.
6. Engine export: the Symmetrix standard MACE path does not support charge/
   dipole outputs; a prototype PyTorch path is fine for the paper, but a
   production number needs an engine interface (new pair style or PolarMACE
   path extension).
7. Paper claims: homogeneous bulk/PNAS is supported; PRL inhomogeneous is a
   validation-level addition, not a training requirement.

## Priority under GPU constraints

1. Finish supplement q/μ labels (prerequisite for training).
2. **Joint training** (main deliverable): merged train → features → q/μ heads →
   gating → state-split eval.
3. Dense homogeneous + slab collection and teacher labeling (coverage).
4. Dedicated speed measurement and engine export.

## Data paths

- Labels: `data_20261025_charges/*_multipole.xyz`
- New collection: `data_20261025_new/dense_0035_64mol.xyz`,
  `data_20261025_new/slabs_validation.xyz`
- Scripts: `scripts/add_teacher_multipoles.py`, `scripts/charge_mlp_head.py`,
  `scripts/dipole_pme_bench.py`, `scripts/longrange_probe_l1.py`,
  `scripts/collect_dense_slabs.py` (to be copied from bench).
- Figures/docs: `docs/results/CHARGE_MULTIPOLE_PROGRESS_2026-10-27.md`.

## Status update 2026-10-27 03:45 UTC

- q/μ labels complete: 5,844 train + 867 valid + 965 supplement.
- Merged train: `data_20261025_charges/train_6809_multipole.xyz`
  (6,809 frames / 1,100 states).
- MACE 0e/1o features extracted: `features_train.npz` (822,940 atoms),
  `features_valid.npz` (111,670 atoms).
- Dense 0.035 and slab sampling launched in parallel (`--mode dense`,
  `--mode slabs`). Current 3 ps slab run is a smoke; the production slab run
  should use 10 ps equilibration + 20 ps production (20 frames/state) or
  20 + 30 ps (30 frames/state).
- Q4 still 0/64 free; local A100 is shared but has memory headroom.
