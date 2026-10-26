# Bulk + gas dataset rebuild and decorrelated training subset (2026-10-25)

**Scope:** rebuild the Path B / Plan C training set from the *completed* bulk2
shards, and produce a decorrelated subset that is honest about the ACF.

## 1. Why the old split was stale

`data/{train,valid}.xyz` were written at **2026-10-23 03:32** from a partial
campaign. The three bulk2 shards finished later (H2O/CO2 at 04:41, NH3/CH4 at
04:39, C2H4/C2H6 at 04:33), so the 15,641-frame snapshot used for the bulk
fine-tune attempts never contained the new dense-state data. The fine-tune
that OOMed and then diverged was therefore both under-sampled *and* initialized
from a gas-only student.

## 2. Rebuild (`data_20261025/`)

```bash
export PYTHONPATH=/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0:$PWD
python scripts/prepare_bulk_dataset.py --out-dir data_20261025
```

| quantity | old snapshot | **rebuilt** |
|---|---:|---:|
| frames read | 15,641 | **26,841** |
| frames kept | 15,641 | **26,841** |
| duplicates | 0 | **0** |
| invalid | 0 | **0** |
| atoms | 1,810,629 | **3,615,429** |
| min interatomic distance | 0.772 Å | **0.772 Å** (p01 0.859 Å) |
| train / valid frames | 13,576 / 2,065 | **23,376 / 3,465** |
| train / valid states | 1,098 / 176 | **1,100 / 176** |
| wall time | 352 s | 545 s |

Per-species bulk counts (states): C2H4 46 (3,432 frames), C2H6 43 (3,126),
CH4 55 (4,065), CO2 61 (4,480), H2O 68 (4,998), NH3 61 (4,574). Gas adds 6 ×
361 = 2,166 frames on the same 16 Å protocol as before.

The split is still a whole-state stratified holdout (no state appears on both
sides), so correlated frames never leak across the boundary.

## 3. The independent-sample problem

The new manifest recomputes the ACF on all trajectories:

| stratum | frames | N_eff (upper) | N_eff (estimable) | median tau_E | median tau_F |
|---|---:|---:|---:|---:|---:|
| all | 26,841 | 6,331 (23.6 %) | 4,450 (17.8 %) | 14.74 | 3.31 |

So 26,841 frames carry roughly **4,450 independent samples**. Training on all
of them over-weights whichever states happen to have the most dumped frames
(pilot states have 3 frames, shard states up to 103) and pays ~5× for
correlated duplicates in the loss.

## 4. Decorrelated subset (`data_20261025_subset/`)

```bash
python scripts/make_decorrelated_subset.py --stride 15 --gas-stride 5
```

Rule: keep 1 frame every `stride` frames **inside each trajectory**, using the
same trajectory key the ACF was measured on. `stride = 15` ≈ median
`tau_int(E) = 14.7`; gas MD trajectories use `stride = 5` because their tau is
~2–3 frames. Gas rattle/equilibrium configs are single-frame states and are
kept as-is.

| file | frames | composition |
|---|---:|---|
| `bulk_train.xyz` | **1,731** | 96–256 atoms; C2H4 241, C2H6 225, CH4 286, CO2 311, H2O 351, NH3 317 |
| `bulk_valid.xyz` | **253** | same species mix, 176 held-out states |
| `gas_train.xyz` | **990** | 165 configs/species |
| `gas_valid.xyz` | **168** | 28 configs/species |

Density coverage of the retained bulk training frames: 1e-4 (354), 5e-4 (348),
2e-3 (354), 5e-3 (338), 1e-2 (230), 2e-2 (98), 3.5e-2 (9). Temperature
coverage 150–2000 K with 11 values.

**This is the set to train on.** It is 1,731 decorrelated bulk frames instead
of 23,376 correlated ones — about the same information content, 13× cheaper per
epoch, and no bias toward high-frame-count states.

Caveat: stride-based thinning is a first-order fix. It does not re-weight the
states, so a state with 103 saved frames still contributes ~7 frames while a
3-frame pilot state contributes 1. The next iteration should either drop
states below a minimum N_eff or assign per-state weights.

## 5. 4G-HDNNP training on the decorrelated set

`plan_c_4g.train_bulk` was extended with `--bulk-train-file` /
`--bulk-valid-file` so it consumes a pre-split dataset instead of re-splitting
internally (the old path would have re-split the merged file and left
`bulk_frames` and `bulk_valid_frames` inconsistent).

Run (one A100, FP64, hidden 160, cg 40, batch 8, 30 epochs):

```bash
export PYTHONPATH=$PWD/bench/plan_c_4g_2026-10-23:$PWD
python -u -m plan_c_4g.train_bulk \
  --bulk-train-file data_20261025_subset/bulk_train.xyz \
  --bulk-valid-file data_20261025_subset/bulk_valid.xyz \
  --gas-train data_20261025_subset/gas_train.xyz \
  --gas-valid data_20261025_subset/gas_valid.xyz \
  --epochs 30 --batch-size 8 --valid-every 2 \
  --hidden 160 --cg-iters 40 --select mixed \
  --tag 4g_c1 --outdir bench/plan_c_4g_2026-10-23/results_al
```

Committee member 1 (`4g_c1`, seed 20261023) result:

| step | bulk E (meV/atom) | bulk F (meV/Å) | gas E (meV/atom) | gas F (meV/Å) |
|---:|---:|---:|---:|---:|
| 440 | 176.8 | 220.2 | 180.7 | 96.9 |
| 1,321 | 71.4 | 143.6 | 88.6 | 44.0 |
| 2,202 | 56.8 | 116.6 | 70.0 | 32.8 |
| 3,084 | 49.5 | 102.2 | 60.7 | 26.3 |
| 3,964 | 50.0 | 98.1 | 62.0 | 28.1 |
| **4,404 (best)** | **46.7** | **94.4** | **57.4** | **26.4** |
| 6,607 (final) | 43.0 | 86.9 | 54.6 | 16.5 |

**Verdict: still far from the 5 meV/atom · 30 meV/Å gate** (best bulk
43–47 meV/atom vs 5 required; force 87–94 vs 30). The new data and the
decorrelated subset fix the *data* problem, not the *model capacity* problem:
a 72k-parameter 4G-HDNNP with 2×15-ish hidden nodes does not have the
expressive power of a 15.2M-parameter PolarMACE teacher, and this is consistent
with the earlier Plan C conclusion (gas 38.2 meV/atom, bulk 73.2).
Escalation options are in `docs/results/PATH_B_BULK_PIPELINE_2026-10-23.md`
§6 and in the Plan C note.

## 6. Artifacts

- `data_20261025/{train,valid}.xyz`, `split_manifest.json`
- `data_20261025_subset/{bulk_train,bulk_valid,gas_train,gas_valid}.xyz`,
  `subset_manifest.json`
- `logs/prepare_dataset_20261025.log`, `logs/train_4g_c1.log`
- `scripts/make_decorrelated_subset.py`
- `bench/plan_c_4g_2026-10-23/results_al/plan_c_4g_4g_c1.pt`
