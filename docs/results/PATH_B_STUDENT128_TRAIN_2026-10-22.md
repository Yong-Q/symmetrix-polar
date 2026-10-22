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
(epoch 89) is 24.5 / 29.7. The energy accuracy is the binding constraint.

This is a prototype, not a replacement for PolarMACE. The model is 25.7 MB
(2.04M parameters, 7.5× smaller than the teacher).

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
