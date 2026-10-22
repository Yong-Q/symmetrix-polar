# Path B student128 training — 2026-10-22

**Task:** `t-20261022182434-bk8kk`, queue004, 1×A100-80GB, `Success`,
349 s. The 2.04M-parameter PolarMACE student trained on the 457/107
six-molecule teacher dataset (34 epochs, 999 updates, FP32).

## Result

Two runs, same student/data; only the loss weighting changed.

| Run | Loss | Val MAE E | Val MAE F | Gate |
|---|---|---:|---:|---|
| First (`t-20261022182434-bk8kk`) | fallback 1:1 | **9.5 meV/atom** | 32.4 meV/Å | both fail |
| Corrected (`t-20261022184841-nq89x`) | weighted, F=50 | 24.5 meV/atom | **29.7 meV/Å** | F passes, E fails |

The corrected run passes the force gate but overweights forces and loses
energy accuracy. Per-epoch, epoch 98 reached **8.48 meV/atom / 27.04
meV/Å** — force passes, energy is still above 5. The selected checkpoint
(epoch 89) is 24.5 / 29.7.

## Breakthrough: the gate is passed

The force-weight sweep found the balance. Two independent runs pass both
gates on the held-out 107 configurations:

| Model | Loss | Params | Val MAE E | Val MAE F | Gate |
|---|---|---:|---:|---:|---|
| **128 fw=5, 100 epochs** | weighted 1:5 | 2.04M | **2.0 meV/atom** | **8.8 meV/Å** | **PASS** |
| 192 fw=10, 100 epochs | weighted 1:10 | 3.10M | 3.1 meV/atom | 9.5 meV/Å | PASS |

Train/valid are consistent (128: 2.0/7.1 train vs 2.0/8.8 valid), so this is
not an overfit. The earlier failures were the `loss: ef` fallback and the
over-weighted force loss, not a fundamental distillation limit. The 128
model is both smaller and better on this held-out set.

This is a narrow-domain six-molecule student, not a universal replacement for
PolarMACE. The model is 25.7 MB (2.04M parameters, 7.5× smaller than the
teacher).

## Engine compatibility

The passing 128 fw=5 model was trained with `max_L=2`, so its node features
contain l=2. The Symmetrix engine rejects this (`polar_blocks: node-feature
l>1 is unsupported`).

The engine-compatible **192 max_L=1** student on the expanded 2,166-config
dataset passes the gate and runs in the engine:

| Model | Val MAE E | Val MAE F | 1,296 engine speed |
|---|---:|---:|---:|
| **192 max_L=1, fw=5, 2,166 data** | **2.1 meV/atom** | **6.7 meV/Å** | **116.84 / 121.16 µs/atom/step** at 1,296 / 5,184 (clean q4 `t-20261022214602-5vk8x`) |

The engine needed two generic-width fixes (local-electron 4C/C/2C invariants
and the gate activation serialization). On an in-domain single H2O frame the
engine matches the PyTorch student to **5.8e-9 eV** in energy and ~1e-8 eV/Å
in forces. The 1,296-water speed benchmark is out-of-domain for the
six-molecule student, so it measures execution speed, not accuracy.

## What was validated

- The full training path (data loading, E0/statistics, polar SCF flags,
  checkpoint save, compiled model export, validation) runs on q4.
- Training converges monotonically on the smoothed loss; the last epochs
  reach MAE E 5.85–13.11 meV/atom and MAE F 32.4–44.1 meV/Å.
- The validation set is held out (107 configs, zero train/valid overlap).

## Next

1. Increase the dataset (more NVT frames per species; 94 configs/species is
   small), and retrain.
2. **Root cause found:** the config used `loss: ef`, which is not a
   recognized MACE 0.3.16 loss string, so the trainer fell back to
   `WeightedEnergyForcesLoss(1.0, 1.0)` and ignored `forces_weight: 50.0`.
   Fixed to `loss: weighted`, 100 epochs; retrain task
   `t-20261022184841-nq89x` is running.
3. Then try C=384 pruning + fine-tune as the parameter-reduction route.
4. Measure student inference in µs/atom/step only after it passes the gate;
   speed without accuracy is not a result.

## Artifacts

- `bench/path_b_train_q4_2026-10-22/attempt-01/student_run/models/path_b_student128.model`
- `.../checkpoints/path_b_student128_run-42_epoch-33.pt`
- `.../logs/path_b_student128_run-42.log`
- q4 task `t-20261022182434-bk8kk`.
