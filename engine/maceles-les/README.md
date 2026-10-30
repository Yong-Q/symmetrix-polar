# MACELES / LES integration for Symmetrix-XL

Source snapshot of the `maceles-les` branch (commit `ccdde6b`) of the
Symmetrix-XL engine, based on `bonan-group/symmetrix-xl` `main` at `6c5f73b`.
It adds the complete Latent Ewald Summation (LES) path used by MACELES:

- periodic triclinic GTO-Ewald with analytic forces and `dE/dq`
  (`libsymmetrix/source/les_ewald.hpp`, with host reference and
  finite-difference self-test);
- LES charge readouts (linear H1 + MLP H2), charge finalization, and the
  model-provided `les_norm_factor`;
- `apply_les_charge_adjoint()`: backpropagates `dE_LR/dq` through both LES
  readouts into `H1_adj/H2_adj` (the charge-response force term);
- `pair_symmetrix_mace_kokkos`: LAMMPS box/positions plumbing, Ewald force
  merge, optional charge debug dump;
- `symmetrix/source/symmetrix/extract_maceles.py`: checkpoint export with
  the compact-engine fusion basis, e3nn path normalization, and the model
  norm factor.

`0001-maceles-complete-LES-integration-with-charge-respons.patch` is the
exact `git format-patch` of that commit.

## Apply

```bash
git clone git@github.com:bonan-group/symmetrix-xl.git
cd symmetrix-xl
git checkout -b maceles-les 6c5f73b
git am /path/to/0001-maceles-complete-LES-integration-with-charge-respons.patch
```

or copy the files in this directory over the engine tree (same relative
paths).

## Build

```bash
export PATH=/opt/matos-tools/cuda/12.9/bin:/opt/matos-tools/conda/envs/matos-compute/bin:$PATH
export LIBRARY_PATH=/opt/matos-tools/conda/envs/matos-compute/lib:$LIBRARY_PATH
cmake --build symmetrix/build-lammps-99ca96e1225f5a30 -j 64
```

The final link needs `-lblas`; `LIBRARY_PATH` above supplies the OpenBLAS
`libblas.so` from the shared Conda environment.

## Validate

- `docs/results/MACELES_FORCE_FIX_2026-10-30.md` — force/energy validation
  against PyTorch for the official MACELES-OFF checkpoint (1,002 / 1,500 /
  six gas molecules / six 3,000-atom systems).
- `docs/results/MACELES_Q4_VERIFY_2026-10-30.md` — clean q4 speed.
- `/workspace/group/shared/bza/test/sampling/les_c1c2.py` and
  `les_batch.py` — MD -> c1/c2 pipeline with trajectory retention.
