# plan_c_4g_v2

Clean fork of the 4G student. The old `plan_c_4g` is frozen as a fast-surrogate
prototype; do not patch it further for the 5/30 gate.

## Baseline (this package)

- monopole QEq with the Gaussian self term in the operator and energy
- element-resolved Pearson chi/J initialisation
- optional ee4G electrostatic-potential feature
- 16/24-function element-resolved G4 angular presets, 4.5 A angular cutoff
- two-stage training: charge loss on REF_charges, then energy/force
- `q_tot` support is still fixed to the neutral subspace

Baseline command:

```bash
python -m plan_c_4g_v2.train_bulk \
  --bulk-train-file data_20261025_subset/bulk_train.xyz \
  --bulk-valid-file data_20261025_subset/bulk_valid.xyz \
  --gas-train data_20261025_subset/gas_train.xyz \
  --gas-valid data_20261025_subset/gas_valid.xyz \
  --ang-preset 24 --n-rbf 16 --r-cut-ang 4.5 --use-ee4g --no-vdw \
  --physical-init --charge-stage-epochs 3 \
  --charge-train-file data_20261025_subset/bulk_train_charges.xyz \
  --charge-gas-file data_20261025_subset/gas_train_charges.xyz \
  --epochs 5 --batch-size 8 --lr 3e-4 --outdir results_v2 --tag v2_baseline
```

## Next milestone

Add explicit l=1 atomic multipoles (or a multipole Ewald) and a proper charge
label/training protocol. Target a realistic 20/60 before attempting 5/30.
