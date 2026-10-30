# MACELES in the Symmetrix legacy engine — validation and clean q4 speed

Date: 2026-10-30. Engine: `MACEKokkos` legacy path + `les_ewald.hpp`
(periodic triclinic GTO-Ewald), `streamed_edges direct` with a model-specific
JIT cubin, FP32, single A100 on queue004.

## Model identity and numerical accuracy

- q4 model JSON: `model_type = "MACE_LES"`, 15 `les_*` keys, the MACELES
  readouts/Ewald parameters. The JIT artifact was generated from the same
  backbone weights.
- LES is active in the q4 runs: step-0 PE at 1,002 atoms is -174.69754 eV,
  identical to the local LES engine, and 6.79 eV below the backbone-only
  -167.90609 eV. At 5,001 atoms: -956.65211 (LES) vs -900.26159 (backbone).
- Accuracy on the exact q4 1,002-atom geometry vs PyTorch MACELES (FP64):
  energy -174.69754 vs -174.69670 eV -> **-0.00084 meV/atom**;
  force MAE 0.0033 eV/A, corr 0.9989 (the box is unminimised, |F|max 0.27 eV/A).
- On the 1,152-atom water box: E -217.8345 vs -217.8336 eV
  (-0.0008 meV/atom), force MAE 0.00082 eV/A, corr 0.99995, charges match to
  1e-7. Both `generic` and `direct/JIT` paths were validated.

## Clean q4 A100 speed (3 x 20 steps per size)

| atoms | MACELES (LES) | backbone only | LES delta |
|---:|---:|---:|---:|
| 1,002 | 6.50 / 12.27 us/atom (min/median) | 4.40 / 5.31 | +2.10 (+48 %) |
| 5,001 | **4.23 / 4.24** | 2.86 / 2.87 | **+1.37 (+48 %)** |

MACELES is ~48 % slower than the same backbone, as expected for the latent
Ewald. The 4.47 us/atom "linear MACE-OFF23" figure in the project notes is a
different model; the standard compact MACE backbone measures 2.86 us/atom at
5,001 atoms on this clean A100.

Earlier local (shared-GPU) measurements of 40-70 us/atom were inflated by
GPU contention and must not be used as absolute numbers.

## Tasks / logs

- MACELES: `t-20261030161606-kf4hv` (attempt-02), 57 s, Success
- Backbone: `t-20261030162027-vdwhn` (attempt-03), 41 s, Success
- Logs: `bench/q4_maceles_speed/attempt-0{2,3}/log.{1000,5000}`
