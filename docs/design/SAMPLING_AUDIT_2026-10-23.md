# Sampling pipeline audit — 2026-10-23

Scope: the bulk teacher-data generator used by the 43k campaign
(`generate_bulk.py`, a per-task copy under
`bench/path_b_bulk2_*/attempt-01/pkg/`). Requirement: cover the
density/temperature state space (PRL-like and beyond), include extreme
conditions, produce independent samples, and supply enough data to distil
PolarMACE to 5 meV/atom and 30 meV/Å in bulk.

## What the current script does

- 32 molecules per periodic box, random rotation + jittered 4x4x2 placement.
- Density grid 1e-4..1e-1 molecules/Å³; temperature grid 150..2000 K.
- One NVT Langevin trajectory per (species, density, temperature).
- 1,200 steps x 0.5 fs = **600 fs total**; a frame every 12 steps = **6 fs**;
  100 frames per state.
- Frames are written **once per species**, at species end.
- Velocities seeded 12345 and Langevin noise seeded 54321 for **every state**.
- A state is dropped if the minimum interatomic distance is below **0.9 Å**;
  there is no retry and no minimisation.

## Flaws (ordered by impact)

| # | Flaw | Evidence | Severity |
|---|---|---|---|
| 1 | **Not independent samples.** Frames are 6 fs apart, far below the ~0.1–1 ps correlation time; fixed seeds make the noise sequence identical across states. | 100 frames/state at 6 fs; seeds 12345/54321 | fatal for the "independent samples" requirement |
| 2 | **Finite-size violation.** 32 molecules gives L=9.86 Å at water liquid density and 6.84 Å at 0.1 molecules/Å³, below the L>=2*r_cut=12 Å criterion. Only rho<=~0.018 passes. | L=(32/rho)^(1/3); r_cut=6 Å | high; high-density states are not bulk |
| 3 | **No equilibration.** Sampling starts at step 12 (6 fs) from a jittered lattice with MB velocities; at 150–300 K the H-bond network needs ps to relax. | loop starts frames at step 12; 600 fs total | high; low-T frames are placement-biased |
| 4 | **Coverage holes exactly at high density.** 44/110 pilot states dropped (40%) because random placement overlaps; no retry. | pilot summary: H2O 44 dropped, etc. | high; dense states under-represented |
| 5 | **No inhomogeneous density profiles.** Homogeneous NVT boxes only; no slabs, interfaces, pores, or external potentials. PRL's validation includes 800 inhomogeneous densities. | script has no external field / slab builder | high if "PRL-like profiles" is required |
| 6 | **All-or-nothing output.** Per-species write only; a crash loses the whole species, and there is no per-state progress line. | A/B/C produced no file after ~1 h; cannot distinguish slow from stuck | medium-high for a 43k campaign |
| 7 | **43k frames != 43k samples.** Effective sample size is likely a few hundred to ~2k; the budget should be stated in ESS. | 432 states x 100 correlated frames | high for the accuracy gate |
| 8 | **No throughput batching.** 96 atoms per forward; 12.6 s/state at best; A100 is latency-bound. | pilot 835 s / 66 states | medium; fixes wall time, not correctness |
| 9 | **Low-T states are not equilibrium.** Plain Langevin at 150–250 K for 600 fs can be trapped; no replica exchange. | no REMD/enhanced sampling | medium |
| 10 | **Provenance gaps.** No model sha256, script hash, git commit, instantaneous T, or virial pressure in the summary. | summary.json fields | medium |
| 11 | **Fixed grid, no active selection.** Easy states receive as many frames as hard ones; no uncertainty or coverage-driven allocation. | fixed nested loops | medium |

## Requirement matrix

| Requirement | Status | Why |
|---|---|---|
| Broad homogeneous density/temperature coverage | **partial** | grid spans 1e-4..0.1 and 150..2000 K, but finite-size valid only <=0.018 |
| Extreme conditions | **partial** | high-density states are finite-size limited; low-T states are non-equilibrium |
| Independent samples | **fail** | 6 fs spacing, one trajectory per cell, fixed seeds |
| PRL-like / inhomogeneous density profiles | **fail** | no slab/interface/pore states |
| Enough data for the 5/30 bulk gate | **unproven, likely insufficient in ESS** | 43k frames, but ESS likely 1e2–2e3 |
| Reproducible from the repository | **fail** | script lives under gitignored `bench/`, copied per task |
| Throughput | **partial** | no batching; 96-atom latency-bound |

## Verdict

- **Can meet:** a first bulk fine-tune that repairs the catastrophic
  gas-to-dense distribution shift (Plan D's 2.27 eV/atom stress failure) and
  gives a coverage tranche across densities/temperatures.
- **Cannot meet:** 5 meV/atom + 30 meV/Å in bulk, independent-sample claims,
  PRL-like inhomogeneous profiles, or publication-grade sampling, as
  currently configured.

## Tranche 2 design (recommended)

1. **Boxes:** 64 molecules for rho<=0.033 (L>=12.4 Å), 128 for 0.05–0.07,
   192 for 0.1; or cap the high-density grid and say so.
2. **Seeds:** unique per (species, rho, T, replica); 2 replicas per cell.
3. **Equilibration/production:** discard 2–5 ps, produce 8–12 ps, save every
   0.5–1 ps; report ACF and ESS, not frame count.
4. **Placement:** species-specific exclusion, soft push or short
   minimisation, retry until N successes per cell; record success rate.
5. **Checkpointing:** append each frame and a per-state manifest
   (model sha256, script hash, git commit, instantaneous T, pressure).
6. **Batching:** 8–16 boxes per forward pass; this is the throughput lever.
7. **Inhomogeneous tranche:** liquid–vapour slabs at 300/400/500 K and
   confined slit states; bin density profiles.
8. **Budget:** target ~2k–4k independent configurations
   (e.g. 216 cells x 1–2 replicas x 8 ps at ~1 ps ACF), not 43k correlated
   frames. Estimate 35–90 GPU-hours before batching; 5–15 GPU-hours with
   8–16x batching.
9. **Low T:** replica exchange or label 150–250 K as quenched, not
   equilibrium.

Tranche 1 (the running 43k campaign) should finish and be used as coverage
and stress data; do not extend it with more frames of the same kind.
