# Local 4G active-sampling round (2026-10-26)

The RUNNER/4G active-sampling task was moved off Q4 and run locally on the
shared A100 (the card was already ~35 GB used by another container, but the 4G
model is small).

## Round

| item | value |
|---|---|
| pool | `data_20261025_subset/bulk_train.xyz`, stride 3, 577 frames |
| committee | `c1`, `c2`, `c3` |
| selection | force-variance rank + ACF refractory + FPS |
| candidates after ACF | 240 |
| selected | 40 |
| selected density | 0.0001 (3), 0.0005 (5), 0.002 (7), 0.005 (9), 0.01 (5), 0.02 (10), 0.035 (1) |
| selected species | CO2 20, H2O 8, CH4 5, NH3 5, C2H4 1, C2H6 1 |
| merged train | 1,731 base + 40 selected = 1,771 frames |
| init | fixed 4G (`self-energy + Pearson χ/J`) |
| train | 5 epochs, batch 8, lr 1e-4, force weight 20 |

## Result

| model | bulk E (meV/atom) | bulk F (meV/Å) | gas E (meV/atom) | gas F (meV/Å) |
|---|---:|---:|---:|---:|
| fixed 4G, no AL | 39.44 | 77.48 | 48.96 | 14.93 |
| fixed 4G + local AL | **38.79** | **76.70** | **48.14** | 15.81 |

Best epoch was 3. The change is:

- bulk E: **−1.6 %**
- bulk F: **−1.0 %**
- gas E: **−1.7 %**
- gas F: **+5.9 %** (worse)

This is a marginal change, within seed/finite-sample noise. It confirms the
earlier conclusion: the bottleneck is not the amount of selected data but the
missing physics (teacher l=1 multipoles, thin angular descriptor) and the
incomplete charge-training stage. The selected set is also imbalanced —
C2H4/C2H6 contribute only one frame each.

## Artifacts

- `runner-4gal-01/results_remote/train_4g_al_r1_summary.json`
- `runner-4gal-01/active/remote_r1/select.json`
- `runner-4gal-01/results_remote/plan_c_4g_4g_al_r1.pt`
- figure `docs/results/figures/PLAN_C_4G_LOCAL_AL_2026-10-26.png`

The remote duplicate `t-20261026144930-jw8c7` was cancelled after the local
run started. Q4 remains full; the Plan D task `t-20261026135843-f6bq5` is still
queued.
