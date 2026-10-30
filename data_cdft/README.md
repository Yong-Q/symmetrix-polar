# cDFT training-data export (PRL + PNAS schemas)

Built by `test/sampling/cdft_dataset.py` from the site-field MD runs.  Every
entry is one simulation: a 1D density profile along the field direction `x`,
the analytic external field, and the supervised direct-correlation targets.

## Units

| quantity | unit |
|---|---|
| length, `x`, `r` | Å |
| density `rho` | Å⁻³ (molecular COM by default, `--density atomic` for atoms) |
| `Vext`, `mu`, `c1` | kT (dimensionless); `Vext` is the four-sine field divided by kT |
| energy (`c2`) | dimensionless (kT⁻¹) |

The PRL dataset stores the same quantities in reduced hard-sphere units;
rescaling is a choice of the trainer.

## PRL F1 (learn c1 from inhomogeneous profiles)

`f1/` per simulation id:

| file | content |
|---|---|
| `rho_<id>.npy` | density profile ρ(x), shape (bins,) |
| `Vext_<id>.npy` | external potential V_ext(x)/kT, shape (bins,) |
| `mu_<id>.npy` | scalar chemical potential (kT) |
| `c1_<id>.npy` | supervised c1(x) = ln ρ + V_ext/kT − μ (NaN where ρ≈0) |
| `x_<id>.npy` | bin centres (Å) |
| `c1_<id>.csv` | the same table for eyeballing / pandas |

The PRL loader builds the target itself as
`dF_drho = mu - Vext - ln(rho)` for `rho > cutoff`, which equals `-c1(x)`.

`mu` is **not** known for a fixed-N run.  Default convention:
`mu = <ln rho + Vext>` over the profile (zero-mean target).  Pass `--mu` to
override with a Widom/TI value; the convention is recorded per run in
`manifest.json`.

## PRL F2 (learn c2 from bulk pair correlations)

`f2/` per bulk simulation:

| file | content |
|---|---|
| `rhob_<id>.npy` | bulk number density of the pair's first species (Å⁻³) |
| `r_values_<id>.npy` | r grid (Å) |
| `c2_<id>.npy` | c2(r) from g(r) through Ornstein–Zernike |
| `mu_<id>.npy`, `dF_drho_<id>.npy` | bulk scalars (convention: 0) |
| `c2k_<id>.npz` | r, g(r), k, c2(k), rho — full diagnostic |

The OZ inversion is the validated `sampling.direct_correlation.c2_from_g`
(analytic Gaussian and PY hard-sphere tests pass to ~1e-13).  For molecular
fluids this is the scalar O–O (or C–C) c2, not the full position-orientation
c2 — see `test/docs/AUDIT_2026-10-23.md`.

## PNAS sim_dict

`pnas_sim_dict.npy` is a dict `{sim_id: {"rho": (bins,), "Vext": (bins,)}}`,
same arrays as F1.  The PNAS `c1_mu` models take `(rho, Vext)` as input and
predict `c1` and `mu`.

## Splits and index

- `train_files.txt` / `val_files.txt`: whole simulations, no frame leakage.
- `index.csv`: one row per run with species, T, density, μ, weights source,
  field amplitude, c2 pair, split.
- `manifest.json`: full provenance (source SHA-256, field SHA-256, units,
  frames, conventions).

## POLAR target ensemble (reweighting)

The field runs above were sampled with MACE-OFF23.  To estimate **POLAR**
profiles, run the validated reweighting first and pass its per-frame weights:

```bash
off-polar-reweight --traj <run>/traj.xyz --polar-model MACE-POLAR-1-M.model \
  --outdir <run>/reweight
python -m sampling.cdft_dataset --runs '<run>' --weights-in-run --outdir out/
```

Weights are applied to the density histogram and to g(r).  Gate on
`ESS >= 10 %` and `sigma_beta <= 3` before treating the weighted profile as a
POLAR estimate; otherwise sample POLAR directly (the q4 `polar_md_q4.py`
runs already do).

## Rebuild

```bash
cd /workspace/group/shared/bza/test
python -m sampling.cdft_dataset \
  --runs '/workspace/group/shared/bza/symmetrix-polar/bench/plan_d_inhomog_2026-10-27/site-*' \
  --outdir /workspace/group/shared/bza/symmetrix-polar/data_cdft/<date> \
  --bins 2000 --c2-bins 80 --stride 1 --bulk-stride 5 \
  --bulk-traj .../traj_500ps_1fs.lammpstrj \
  --bulk-elements O H --bulk-species H2O --bulk-T 300 --bulk-density 0.02
```
