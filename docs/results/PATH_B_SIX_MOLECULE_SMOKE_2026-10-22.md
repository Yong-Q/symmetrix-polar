# Path B six-molecule teacher smoke — 2026-10-22

**Task:** `t-20261022145019-xkd46`, queue004, 1×A100-80GB, `Success`.
**Scope:** H2O, NH3, C2H4, CH4, C2H6, CO2; 8 configurations × 12 molecules per
species; FP64 PolarMACE teacher; no training.

The goal is to verify that the teacher covers all six chemistries and that the
extxyz label pipeline works before scaling the distillation dataset.

| Species | Atoms / box | E mean (eV) | E std (eV) | F rms (eV/Å) | F max (eV/Å) |
|---|---:|---:|---:|---:|---:|
| H2O | 36 | −24955.965 | 1.457 | 2.093 | 14.80 |
| NH3 | 48 | −18463.130 | 1.635 | 1.620 | 12.95 |
| C2H4 | 72 | −25649.939 | 3.333 | 1.890 | 15.84 |
| CH4 | 60 | −13218.635 | 1.942 | 1.395 | 8.10 |
| C2H6 | 96 | −26049.688 | 2.722 | 1.496 | 11.33 |
| CO2 | 36 | −61588.207 | 2.371 | 3.816 | **37.47** |

## What passed

- The teacher loads and evaluates all six species in FP64 on one A100.
- The extxyz writer produces one file per species with energies and forces.
- The summary JSON and the script are reproducible from the committed copy.

## What needs fixing before training

- The current boxes use random positions and rattle amplitudes 0.02–0.06 Å.
  This creates close contacts and high-force configurations, especially CO2
  (max 37.5 eV/Å). Training on this distribution would overweight repulsive
  walls.
- For the real distillation set, use:
  - equilibrium and gently rattled configurations (rattle ≤0.02 Å);
  - short NVT snapshots at 300–800 K;
  - dimers/clusters and a few compressed/stretched single points;
  - 2,000–5,000 configurations per species, with a 10–20% held-out split.
- Add a small DFT single-point subset (100–500 configurations) to validate
  that the student does not inherit teacher bias; teacher-only validation is
  not enough.
- Validate E/F/stress, 10 ps NVT stability, and (for the 4G contract) charge
  conservation / dipole response where available.

## Artifacts

- Task log: `t-20261022145019-xkd46` (queue004).
- Summary: `docs/results/data/path_b_six_molecule_smoke_2026-10-22.json`.
- Script: `docs/results/path_b/six_molecule_teacher_smoke.py`.
- Extxyz (local bench, not committed): `bench/path_b_six_molecule_smoke_2026-10-22/attempt-01/path_b_six_molecule_smoke/`.

## Next

1. Generate the gentler six-molecule dataset and a DFT validation subset.
2. Train a small student only after the data/validation protocol is fixed.
3. Keep the current PolarMACE model unchanged until the student passes the
   parity gate (E/F + MD), per the Path A/B decision matrix.
