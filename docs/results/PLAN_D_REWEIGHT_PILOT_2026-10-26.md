# Plan D → MACE-POLAR reweighting pilot (2026-10-26)

**Question:** if Plan D is used as the MD proposal, is the importance-sampling
ESS better than the OFF→POLAR 500 ps result (σ_β=13.76, ESS=0.47%)?

**Answer: yes, much better in the pilot.** On 1,152 atoms / 10 frames,
Plan D→POLAR gives σ_β=5.90 and ESS=15.9% (1.59/10), versus 0.47% for
OFF→POLAR. The pilot is small and correlated, so the absolute ESS needs a
longer trajectory; the σ_β reduction from 13.76 to 5.90 is the robust signal.

## Why ASE, not the Symmetrix LAMMPS pair style

The Symmetrix direct/JIT path deadlocks at 1,152 atoms at the first neighbor
rebuild (process `D`, GPU 0%). It also hit a long-run startup deadlock at
5,184 atoms, although a 200-step 5,184 test passes. The generic/materialized
path is stable but ~0.25 s/step. For the pilot we used ASE/PyTorch MACE with
the Plan D bulk model; it is stable at 1,152 atoms and runs at 0.143 s/step.

## Protocol

| item | value |
|---|---|
| system | 1,152 atoms (384 H2O), same box as the OFF 500 ps run |
| proposal | Plan D bulk model (MACE, FP32) |
| seed | `seed_5ps.lammpstrj` (OFF 5 ps frame) |
| MD | ASE Langevin, 1 fs, 300 K, friction 0.01, fixcm=False |
| length | 5 ps (5,000 steps) |
| frames | 10 frames, one per 0.5 ps |
| target | MACE-POLAR-1-M, FP64 |

## Result

| quantity | Plan D→POLAR | OFF→POLAR (500 ps) |
|---|---:|---:|
| frames | 10 (5 ps) | 501 (500 ps) |
| ΔE/atom mean | 0.148 meV/atom | 417.42 meV/atom (constant offset) |
| ΔE/atom std | **0.132 meV/atom** | 0.309 meV/atom |
| ΔE total std | **0.153 eV** | 0.356 eV |
| σ_β | **5.90** | 13.76 |
| ESS | **1.59 / 10 (15.9%)** | 2.35 / 501 (0.47%) |
| verdict | usable (pilot) | collapse |

The constant ΔE offset (0.148 meV/atom) is harmless after weight
normalization. The per-atom fluctuation is 2.3× smaller than OFF, and the
extensive σ_β is 2.3× smaller.

## Caveats

- **10 frames is a pilot.** The ESS 15.9% has a large statistical error; the
  frames are 0.5 ps apart and correlated. A 50 ps run (100 frames) is needed
  for a production ESS.
- The MD is ASE/PyTorch, not the Symmetrix LAMMPS path; the trajectory is
  valid, but the engine speed advantage is not in this number.
- The 5 ps trajectory starts from the OFF seed; the first frames may not be
  fully equilibrated under Plan D.
- The Symmetrix 1,152-atom deadlock is an engine issue, not a Plan D model
  issue; it is recorded separately.

## Next

1. Extend the ASE Plan D trajectory to 20–50 ps (40–100 frames) and recompute
   ESS; expected ~50 min per 20 ps at 0.143 s/step.
2. Fix the Symmetrix 1,152-atom deadlock (launch-policy/non-persistent JIT
   artifact or a source-level rebuild) if the engine speed is required.
3. If the 50 ps ESS remains >10%, Plan D is a viable proposal for POLAR
   reweighting at 1,152 atoms.

## Artifacts

- `bench/plan_d_md_1152_ase_2026-10-26/traj_5ps.xyz`
- `bench/plan_d_md_1152_ase_2026-10-26/summary.json`
- `bench/plan_d_md_1152_ase_2026-10-26/frames.json`
- figure `docs/results/figures/PLAN_D_REWEIGHT_PILOT_2026-10-26.png`
