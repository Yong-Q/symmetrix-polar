# 4G wide + ee4G rewrite (2026-10-26)

**Question:** can a wider element-resolved angular basis plus the ee4G
electrostatic-potential feature reduce the 4G error enough to approach the
5 meV/atom / 30 meV/Å gate?

**Answer: it is a real improvement, but not enough.** C2H4/C2H6 improve by
38–41 %, bulk energy drops from 39.4 to 31.0 meV/atom, and dense forces drop
from 338.8 to 275.2 meV/Å (c1 → wide; fixed → wide 293.8 → 275.2). H2O and CH4
regress, and the dense force is still 9.2× the gate.

## What changed

| item | old 4G | wide + ee4G |
|---|---|---|
| angular basis | 8 G4 | **24 G4** |
| radial basis | 12 RBF | **16 RBF** |
| angular cutoff | 3.5 Å | **4.5 Å** |
| descriptor dim | 128 | **304** |
| long-range feature | q only | **q + QEq potential φ** |
| vdW | on (previous ablation worse) | off for this run |
| training | fine-tune from c1 | **from scratch, 25 epochs** |

The ee4G feature is the QEq electrostatic potential at each site, appended to
the local energy head. It is not a full l=1 multipole model; it only lets the
local network see the global electrostatic environment.

## Overall result

Comparison uses the same `error_breakdown.py` pass for both models (mean
per-frame MAE), plus the training VALID values for gas.

| metric | fixed 4G | wide + ee4G final (epoch 24) | change |
|---|---:|---:|---:|
| bulk E (meV/atom) | 39.44 | **31.01** | −21 % |
| bulk F (meV/Å) | 81.71* | **75.06*** | −8 % |
| gas E (meV/atom) | 48.96 | **35.80** | −27 % |
| gas F (meV/Å) | 14.93 | **14.70** | −2 % |

\*The training VALID table uses a per-atom weighted force MAE and reports
77.48 (fixed) and 71.2 (wide, epoch 24); the breakdown table uses mean
per-frame MAE and reports 81.71 and 75.06. Both show the same direction.

Best-by-bulk-E checkpoint (epoch 9) is more energy-specialised:
bulk 27.14 meV/atom / 118.98 meV/Å; gas 32.17 / 56.23. The final epoch 24 is
the balanced checkpoint and is what the table above uses.

## Species

| species | fixed E | wide E | change |
|---|---:|---:|---:|
| C2H4 | 137.7 | **85.2** | −38 % |
| C2H6 | 125.4 | **74.2** | −41 % |
| CO2 | 14.2 | 12.1 | −15 % |
| NH3 | 5.7 | 5.8 | flat |
| CH4 | 9.8 | **19.4** | +98 % |
| H2O | 10.0 | **24.3** | +143 % |

The wide angular basis does what it was meant to do for the C–C molecules, but
it overfits or mis-weights the small molecules. The descriptor needs
element-resolved angular channels / a separate small-molecule basis, not just
more eta/zeta functions.

## Density-resolved force

| ρ (molecules/Å³) | fixed F | wide F |
|---|---:|---:|
| 1e-4 | 48.5 | 46.9 |
| 5e-4 | 30.6 | 28.9 |
| 2e-3 | 51.4 | 47.0 |
| 5e-3 | 88.9 | 80.3 |
| 1e-2 | 93.6 | 79.4 |
| **2e-2** | **293.8** | **275.2** |

Dense-state force improved ~6 % but remains **9.2×** the 30 meV/Å gate. This
is the l=1 monopole gap; ee4G is not a substitute for explicit multipoles.

## Verdict

- Wide + ee4G is a **real step**: bulk E −21 %, gas E −27 %, C2H4/C2H6 −38/−41 %.
- It is **not** a gate pass: bulk E 6.2×, bulk F 2.5×, gas E 7.2×; gas F passes.
- The remaining blockers are:
  1. explicit **l=1 multipoles / multipole Ewald** for dense forces;
  2. **element-resolved / small-molecule-balanced angular channels** for H2O/CH4;
  3. a less aggressive potential feature or a separate scaling for ee4G.
- The code is at `plan_c_4g/model.py`, `descriptors.py`, `train_bulk.py`
  (commit `a478f5e`). Checkpoints are in
  `bench/plan_c_4g_2026-10-23/results_4g_wide/`.

## Artifacts

- `bench/plan_c_4g_2026-10-23/results_4g_wide/train_4g_wide_ee4g_summary.json`
- `bench/plan_c_4g_2026-10-23/results_4g_wide/breakdown_final.json`
- `bench/plan_c_4g_2026-10-23/results_4g_wide/plan_c_4g_4g_wide_ee4g_final.pt`
- figure `docs/results/figures/PLAN_C_4G_WIDE_EE4G_2026-10-26.png`
