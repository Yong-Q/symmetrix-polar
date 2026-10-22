# Path B committee active learning — 2026-10-22

**Task:** `t-20261022193021-r82s4`, queue004, 1×A100-80GB, deadline 4 h.
**Goal:** select teacher labels by committee disagreement instead of expanding
the dataset uniformly.

## Pipeline

1. **Committee.** Base 128-channel student (corrected-loss checkpoint) plus two
   additional 128-channel members trained from seeds 43 and 44 on the same
   457/107 six-molecule data, `loss: weighted`, forces weight 10, 100 epochs.
2. **Unlabeled candidates.** The base student runs NVT Langevin MD at 300, 600
   and 900 K (300 steps, 0.5 fs, one frame every 10 steps) for each of H2O,
   NH3, C2H4, CH4, C2H6, CO2, plus 50 Gaussian rattles per species. Candidates
   with minimum-image distance <0.7 Å or >8 Å are dropped.
3. **Uncertainty.** Every committee member scores every candidate. The
   selection score is the mean across atoms/components of the committee force
   variance; energy variance is recorded as a diagnostic. Top 40 frames per
   species are selected with a trajectory-spacing rule (skip frames within 3
   steps of an already selected frame from the same MD temperature).
4. **Teacher labels.** The selected ~240 frames are evaluated with the FP64
   PolarMACE teacher (`REF_energy` / `REF_forces`) and written to
   `active_selected.xyz`.
5. **Retrain.** `train.xyz` + `active_selected.xyz` (≈697 frames) train a
   192-channel student (`MLP_irreps 48x0e`, forces weight 10, 100 epochs);
   the held-out 107 frames remain untouched.

## Why this design

- The first student showed train≈valid error (9.4 vs 9.5 meV/atom; 24.4 vs
  24.5 with the corrected loss), i.e. underfitting. The primary suspect is
  model capacity (128 channels is 1/16 of the teacher polar head width), so
  the final model is 192 channels.
- Random expansion to ~2,166 frames is running separately; active learning
  uses the teacher budget on frames where the committee disagrees, which is
  more sample-efficient and tests whether data coverage is the secondary
  bottleneck.
- The 107 held-out frames are never used for selection or training, so the
  final E/F numbers remain comparable with the static 128/192 runs.

## Experiment matrix (all q4, 1 GPU each)

| Task | Variable | Purpose |
|---|---|---|
| `t-20261022191537-h7bf9` | 128, forces weight 5/10 | loss balance |
| `t-20261022192158-74mmf` | 192, same 457/107 | capacity control |
| `t-20261022191732-jsz8t` | random data → ~2,166 | coverage control |
| **`t-20261022193021-r82s4`** | **committee active selection + 192** | **sample-efficient data** |

## Decision rule

- If 192 static improves E substantially: capacity is primary; combine 192 +
  expanded data and run one more active round.
- If active-192 beats static-192 and random-192 at equal labels: keep the
  committee loop and scale it.
- If all remain above 5 meV/atom: the 4× compression gate is too aggressive;
  move to C=384 pruning + fine-tune or a 256-channel student.
