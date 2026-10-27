# Charge/dipole student — final effect (2026-10-27)

| component | training data | validation | result |
|---|---|---|---|
| E/F backbone (Plan D bulk) | 23,376 frames | 3,465 | **1.2 meV/atom / 7.8 meV/Å**; gas 1.17 / 1.00 |
| q head (MLP) | 6,809 frames (1,100 states) | 867 | **0.0117 e** (4π; 0.00093 raw) |
| μ head (full linear ridge 384→3) | 6,809 frames | 867 | **0.0475** (4π; 0.00378 raw) |
| Ewald diagnostic | 6,809-trained heads | 867 | **0.311 meV/atom** |

Per-species Ewald (867 valid): C2H4 0.035, C2H6 0.055, CH4 0.013, CO2 0.628,
H2O 0.516, NH3 0.375 meV/atom.

Per-species E/F on the 3,465 held-out frames (FP64, MACE component-MAE
convention; overall 1.124 meV/atom / 7.753 meV/Å):

| species | gas E | gas F | dense E | dense F |
|---|---:|---:|---:|---:|
| C2H4 | 0.94 | 1.32 | 0.64 | 4.25 |
| C2H6 | 0.65 | 1.14 | 0.82 | 2.46 |
| CH4 | 1.35 | 0.38 | 1.10 | 2.05 |
| CO2 | 1.44 | 0.71 | 1.47 | 23.62 |
| H2O | 2.05 | 0.73 | 1.51 | 11.12 |
| NH3 | 1.46 | 1.46 | 1.40 | 8.33 |

Units: E meV/atom, F meV/Å. Dense CO2/H2O/NH3 dominate the force residual;
the other three species stay below 4.3 meV/Å. Raw table:
`docs/results/data/valid_by_species.json`.

Stability: force rotation 4.0e-5; NVE 0.245 meV/atom/ps over 1 ps;
canonical-frame rotation 4.6e-14 meV/atom (vs 0.825 without).

Speed: backbone ~4.5 µs/atom (engine FP32), heads <0.1, Ewald 7.9 (PyTorch);
projected total ~20.2 µs, cutoff 1.2× → ≤20. Engine hybrid pair style is
still required for an end-to-end measurement.

Caveats: q/μ trained on 6,809 frames, not 23K; CO2/H2O/NH3 are
representation-limited (0.4–0.6 meV/atom); 20 µs is projected.

Reweighting: student→POLAR one-step reweighting at 1,152 atoms gives
σ_β=8.02 and ESS=0.24 % (top frame 91.5 % of the weight), so it is not
usable at production size; 96 atoms gives ESS≈44 % and gas 98 %. Use direct
student sampling, or λ-intermediate + BAR/MBAR. See
`docs/results/ESS_STUDENT_TO_POLAR_2026-10-27.md`.
