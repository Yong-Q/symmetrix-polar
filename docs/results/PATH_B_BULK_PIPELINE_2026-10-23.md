# Path B bulk-data pipeline — 2026-10-23

**Scope.** Plan B only (distilled 128L1 PolarMACE student). No Plan A/C/D
file was touched. Code is staged for review; nothing was committed and no
production fine-tune was launched.

Data snapshot used for the smoke test: **2026-10-23T03:32Z**. The 43k-frame
bulk campaign is still running (pilot complete; shard species H2O, NH3 and
C2H4 written, CO2, CH4 and C2H6 still pending at that time). All counts below
are measured on that snapshot; rerun `prepare_bulk_dataset.py` after the
shards finish to regenerate the final split. Do not treat these partial
counts as the final dataset size.

## 1. Code and commands

Three new scripts under `scripts/` (names do not collide with existing files):

| File | Purpose |
|---|---|
| `scripts/prepare_bulk_dataset.py` | merge gas + bulk extxyz, de-duplicate, validate, state-level split, coverage + ACF/N_eff report, write `data/{train,valid}.xyz` + `data/split_manifest.json` |
| `scripts/finetune_student_bulk.py` | stage an auditable MACE run dir (resolved YAML, `run.sh`, `command.txt`, `seed.txt`, `manifest.json`) and optionally launch with cell-grouped batching |
| `scripts/eval_student_bulk.py` | evaluate a student checkpoint on the state-level valid split, overall and per species / density bin / temperature bin / gas-only, optional speed in µs/atom/step |

Environment (MACE 0.3.16 + ASE + torch 2.5.1; `graph_electrostatics`
must be on `PYTHONPATH`):

```bash
PY=/opt/matos-tools/conda/envs/matos-compute/bin/python       # local
# PY=/vepfs-mlp2/c20250207/lvdingyang/matos-tools/conda/envs/matos-compute/bin/python  # q4
REPO=/workspace/group/shared/bza/symmetrix-polar
export PYTHONPATH=/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0:$REPO:${PYTHONPATH:-}
```

Prepare (defaults already point at the pilot, shards A/B/C and the 2,166 gas
configs; missing shard files are warnings, not errors):

```bash
$PY scripts/prepare_bulk_dataset.py
# -> data/train.xyz, data/valid.xyz, data/split_manifest.json
```

Stage the fine-tune (writes the run directory, does **not** train):

```bash
$PY scripts/finetune_student_bulk.py stage \
  --run-dir bench/path_b_bulk_ft_2026-10-23/attempt-01 \
  --train data/train.xyz --valid data/valid.xyz \
  --init-model bench/path_b_128L1_data2_q4_2026-10-22/attempt-01/student_run/models/path_b_student128L1_data2.model \
  --seed 42 --epochs 60 --batch-size 16 --lr 1e-3 \
  --energy-weight 1 --forces-weight 50 --device cuda --dtype float64
```

Evaluate:

```bash
$PY scripts/eval_student_bulk.py \
  --model <student.model> --valid data/valid.xyz \
  --device cuda --dtype float64 --speed \
  --out <results.json> --md <results.md>
```

### State definition used by the split

* **Bulk:** `(species, density_mol_A3, temperature_K[, seed/traj_id])`.
  Independent replicas written by the pilot and by a shard at the same
  thermodynamic point are **merged into one state**, because the current data
  carries no seed/trajectory id. A state is therefore never split across
  train/valid even when two runs covered the same density/temperature.
* **Gas MD:** `(species, md<temperature>K)`. The gas `source` field is a
  *frame index* (`md200K_00`, `md200K_01`, ...), not a trajectory id: all 35
  frames of one temperature come from one 800-step Langevin run. Grouping by
  `source` (or by the old `split=train/valid` flag) would leak correlated
  frames; the new split holds out whole `(species, md temperature)`
  trajectories.
* **Gas rattle / equilibrium:** each config is its own single-frame state
  (rattle displacements are independent by construction).

The split is a deterministic stratified whole-state holdout: within each
`(dataset, species, kind)` group, states are sorted in density-major /
temperature order and every `k = round(1/valid_fraction) = 7`-th state is
held out (seed 20261023). At least one state is held out when a group has
≥2 states. No state appears on both sides.

## 2. Data coverage (snapshot 2026-10-23T03:32Z)

Read **15,641 frames** (13,475 bulk + 2,166 gas), 1,810,629 atoms.
**0 exact duplicates** (SHA-256 of species + rounded positions + cell),
**0 invalid frames** (all `REF_energy`/`REF_forces` finite, valid cells).
Minimum-image minimum distance over all kept frames: **0.772 Å**
(1st percentile 0.852 Å; threshold 0.5 Å).

| dataset/species | states (tr/va) | frames (tr/va) | densities (molecules/Å³) | temperatures (K) |
|---|---:|---:|---|---|
| bulk/C2H4 | 46 (40/6) | 3,432 (3,014/418) | 1e-4 … 1e-2 (5 values) | 150–2000 (11) |
| bulk/C2H6 | 42 (36/6) | 126 (108/18) | 1e-4 … 5e-3 (4 values) | 150–2000 (11) |
| bulk/CH4 | 55 (48/7) | 165 (144/21) | 1e-4 … 1e-2 (5 values) | 150–2000 (11) |
| bulk/CO2 | 60 (52/8) | 180 (156/24) | 1e-4 … 2e-2 (6 values) | 150–2000 (11) |
| bulk/H2O | 68 (59/9) | 4,998 (4,371/627) | 1e-4 … 3.5e-2 (7 values) | 150–2000 (11) |
| bulk/NH3 | 61 (53/8) | 4,574 (3,953/621) | 1e-4 … 2e-2 (6 values) | 150–2000 (11) |
| gas/C2H4, C2H6, CH4, CO2, H2O, NH3 | 157 each (135/22) | 361 each (305/56) | 16 Å single-molecule box | 1 eq + 6 md temps + 150 rattle per species |

Bulk state counts pooled over species (rows = density, columns = temperature,
cells = `states(train/valid)`; `-` = not yet written):

```
rho\T          150       200       250       300       400       600       800      1000      1200      1500      2000
0.0001       6(6/0)   6(6/0)   6(6/0)   6(6/0)   6(6/0)   6(6/0)   6(0/6)   6(6/0)   6(6/0)   6(6/0)   6(6/0)
0.0005       6(6/0)   6(6/0)   6(0/6)   6(6/0)   6(6/0)   6(6/0)   6(6/0)   6(6/0)   6(6/0)   6(0/6)   6(6/0)
0.002        6(6/0)   6(6/0)   6(6/0)   6(6/0)   6(6/0)   6(0/6)   6(6/0)   6(6/0)   6(6/0)   6(6/0)   6(6/0)
0.005        6(6/0)   5(0/5)   6(5/1)   6(6/0)   6(6/0)   6(6/0)   6(6/0)   6(6/0)   5(0/5)   6(6/0)   6(5/1)
0.01         4(4/0)   4(4/0)   4(4/0)   4(4/0)   5(1/4)   4(4/0)   5(5/0)   4(4/0)   4(4/0)   4(4/0)   4(4/0)
0.02         3(0/3)        -   2(2/0)   2(2/0)   3(3/0)   2(2/0)   1(1/0)   3(3/0)   2(1/1)   1(1/0)   2(2/0)
0.035             -        -   1(1/0)        -        -        -        -        -        -   1(1/0)   1(1/0)
```

The full per-species density × temperature grid (including frame counts and
the train/valid state assignment) is in
`data/split_manifest.json` under `coverage["bulk/<species>"]["density_temperature_grid"]`.

**Nominal grid once all shards finish** (from the campaign design, not yet
measured): each shard species covers 9 densities
(1e-4, 5e-4, 2e-3, 5e-3, 1e-2, 2e-2, 3.5e-2, 7e-2, 1e-1) × 8 temperatures
(150, 250, 300, 400, 600, 800, 1200, 2000 K), 1200 steps/state, 100
frames/state. The pilot adds 10 densities × 11 temperatures at 3
frames/state. Actual frame counts will be below the 43,200 cap because
unstable states are dropped (the completed pilot dropped 40–62 % of its
states per species, mostly at high density; the shard summary will report its
own drops). Rerun the prepare command to get the final measured table.

## 3. State split counts (snapshot)

| split | frames | states | atoms |
|---|---:|---:|---:|
| train | 13,576 | 1,098 | 1,578,797 |
| valid | 2,065 | 176 | 231,832 |

By dataset: bulk train/valid = 11,746 / 1,729 frames (288 / 44 states);
gas train/valid = 1,830 / 336 frames (810 / 132 states). Overall the valid
split is 13.2 % of frames and 13.8 % of states.

Gas valid states per species: 1 whole MD trajectory (~35 frames) + 21 rattle
configs = 22 states / 56 frames. Rattle frames are independent single-frame
states; MD states are whole trajectories.

## 4. Independent-sample estimate (ACF)

Method: per **trajectory** (bulk trajectories are kept separate even when
their states are merged for the split), compute the normalized ACF of energy
per atom and of force RMS, estimate the integrated autocorrelation time
`tau = 1 + 2*sum(rho_k)` with initial-positive-sequence truncation and a
Sokal window, use the slower observable (`max(tau_E, tau_F)`), and report
`N_eff = n / tau`. Trajectories with fewer than 5 frames cannot support an
ACF estimate; their `N_eff` is set to the raw frame count and is an **upper
bound, not evidence of independence**. These short states include every
3-frame pilot state and every 1-frame rattle/equilibrium config.

Measured on the snapshot (6,441-frame run earlier gave the same picture; the
numbers below are the 15,641-frame snapshot):

| stratum | frames | estimable frames | N_eff (upper bound) | N_eff (estimable only) | median tau_E | median tau_F |
|---|---:|---:|---:|---:|---:|---:|
| all | 15,641 | 13,760 | 4,165.2 (26.6 %) | 2,284.2 (16.6 %) | 13.84 | 7.63 |
| bulk | 13,475 | 12,500 | 2,713.5 (20.1 %) | 1,738.5 (13.9 %) | 18.94 | 12.26 |
| gas | 2,166 | 1,260 | 1,451.7 (67.0 %) | 545.7 (43.3 %) | 2.55 | 1.99 |

Per-trajectory estimable N_eff: median 5.48, min 3.02, max 100.0. For bulk
species with a completed shard file the estimable-only fraction is
C2H4 7.3 %, H2O 17.1 %, NH3 15.4 %; C2H6, CH4 and CO2 only have 3-frame
pilot states so far and are not estimable.
Gas MD trajectories: 27.6–69.6 % (35 frames each); rattle states are
single-frame by construction. Trajectory-level values and per-state
`tau_E`, `tau_F`, `N_eff` are stored in `split_manifest.json`.

**Consequence:** the 15,641 frames are not 15,641 independent samples. Even
on the estimable subset, the ACF reduces the effective count to ~17 %. The
final ~25–30k-frame bulk set should be assumed to carry a similar effective
fraction until the ACF is recomputed; do not report raw frame counts as
independent data.

## 5. Smoke test (clearly labeled; not a gate result)

The smoke test used the existing **gas-only** 128L1 student
`bench/path_b_128L1_data2_q4_2026-10-22/attempt-01/student_run/models/path_b_student128L1_data2.model`
(3.8 meV/atom, 6.1 meV/Å on 432 gas frames; 91.64 / 95.11 µs/atom/step engine
speed). It is a pipeline test and an out-of-domain baseline, not a bulk gate.

* `prepare_bulk_dataset.py`: full run on the snapshot, 352 s, produced
  15,641 frames / 1,274 states.
* `eval_student_bulk.py`: deterministic spread of 32 valid frames (26 bulk +
  6 gas; 3,483 atoms), local A100, float32 (smoke only; production eval defaults to FP64). Energy in meV/atom, force in
  meV/Å, mean over all atoms/components (MACE `PerAtomMAE` convention).

| stratum | frames | MAE E | MAE F | median E | median F | frames >100 meV/atom |
|---|---:|---:|---:|---:|---:|---:|
| all | 32 | 4.32e10 | 2.60e14 | 525.2 | 837.1 | 16 |
| gas only | 6 | **4.60** | **4.93** | 4.74 | 4.45 | 0 |
| bulk, dilute (ρ < 1e-3) | 10 | 8.23 | 129.46 | 5.85 | 65.38 | 0 |
| bulk, 1e-3 ≤ ρ < 1e-2 | 10 | 4.73e5 | 4.44e7 | 4,042 | 17,178 | 10 |
| bulk, ρ ≥ 1e-2 | 6 | 2.24e11 | 1.35e15 | 2.27e8 | 1.94e10 | 6 |

The gas row (4.60 / 4.93 on 6 frames) is consistent with the published
3.8 / 6.1 on the 432-frame gas holdout. The dilute bulk row is close on
energy but ~10× over the 30 meV/Å force gate. Dense states are not merely
inaccurate: the gas-only student **diverges**. On one CH4 frame at
ρ = 0.01 molecules/Å³, T = 400 K the model returns
E_pred = 2.10e9 eV against REF = −3.53e4 eV (ΔE ≈ 1.31e7 eV/atom); the same
frame was checked directly outside the eval loop, so this is a genuine
out-of-domain prediction, not an evaluation-unit bug. The mean rows above
are dominated by these divergences; the medians and the >100 meV/atom counts
are the honest summary. A scan of all 13,576 training frames shows stable
per-species labels (max |E|/atom 1711 eV for CO2, no label outliers), so this
is the gas-only student diverging, not corrupted data. **The gas-only student is not usable on bulk and the
bulk fine-tune must be checked for numerical stability as well as accuracy.**

Fine-tune dry run (config validation only; no optimization):

* 13,576 train / 2,065 valid frames loaded; 4 elements; E0s applied.
* Foundation model loaded from the existing `.model`; extracted architecture
  matches `128x0e+128x1o`, r_max 6.0, max_ell 3, 2 interactions, 2 SCF steps.
* Loss `WeightedEnergyForcesLoss(energy_weight=1.000, forces_weight=50.000)`,
  `default_dtype=float64`, `loss: weighted`.
* Cell-grouped loader: 8 distinct cells at this snapshot, group sizes
  106–2,280, 853 batches/epoch at batch_size 16; a unit test confirmed every
  emitted batch contains a single cell and all indices are covered.
* `--dry_run` completed and stopped before training; no unrecognized YAML keys.

Speed: the eval `--speed` path was exercised, but the local A100 was shared
with co-tenant processes during the smoke test (`nvidia-smi` showed other
processes and fluctuating utilisation). The measured smoke value was
**307,836 µs/atom/step** on 7 gas frames (10.16 s total), dominated by
contention and cold start; it is **not a benchmark and not a speed claim**.
The only speed result for this student remains the clean q4 engine number,
91.64 / 95.11 µs/atom/step at 1,296 / 5,184 atoms.

## 6. Exact next command to launch the real fine-tune when shards finish

Run on q4 (same image/queue as the existing path_b tasks; the launcher will
create the run directory, write the resolved YAML/command/seed, and start
`run_train` with cell-grouped batching):

```bash
REPO=/vepfs-mlp2/c20250207/lvdingyang/matos-v2-state/runtime/scopes/users/yongqiu2023-163-com/workspace-state/shared/bza/symmetrix-polar
PY=/vepfs-mlp2/c20250207/lvdingyang/matos-tools/conda/envs/matos-compute/bin/python
export PYTHONPATH=/vepfs-mlp2/c20250207/lvdingyang/matos-v2-state/runtime/scopes/users/yongqiu2023-163-com/workspace-state/shared/bza/external-repos/graph_electrostatics-0.4.0:$REPO:${PYTHONPATH:-}
cd "$REPO"
$PY scripts/prepare_bulk_dataset.py && \
$PY scripts/finetune_student_bulk.py launch \
  --run-dir bench/path_b_bulk_ft_2026-10-23/attempt-01 \
  --train data/train.xyz --valid data/valid.xyz \
  --init-model bench/path_b_128L1_data2_q4_2026-10-22/attempt-01/student_run/models/path_b_student128L1_data2.model \
  --seed 42 --epochs 60 --batch-size 16 --lr 1e-3 \
  --energy-weight 1 --forces-weight 50 \
  --device cuda --dtype float64 --overwrite
```

After training, evaluate the produced model (the tag is
`path_b_student128L1_bulk`):

```bash
$PY scripts/eval_student_bulk.py \
  --model bench/path_b_bulk_ft_2026-10-23/attempt-01/student_run/models/path_b_student128L1_bulk.model \
  --valid data/valid.xyz --device cuda --dtype float64 --speed \
  --out bench/path_b_bulk_ft_2026-10-23/attempt-01/eval.json \
  --md bench/path_b_bulk_ft_2026-10-23/attempt-01/eval.md
```

Notes for the production run:

* The data snapshot is regenerated from scratch; any earlier split (including
  the smoke-test split) is replaced. Do not reuse an old checkpoint's
  validation numbers.
* Cost estimate (not measured): with the final bulk set around 20–30k frames,
  batch 16 gives ~1,300–1,900 updates/epoch; 60 epochs is ~6–13 h on one A100
  in FP64. `patience: 30` and `ReduceLROnPlateau` (factor 0.8, patience 15)
  are active; monitor early epochs for divergence on dense states and lower
  `--lr` (or raise `--batch-size`) if the validation energy error explodes.
* The launcher records `student128L1_bulk_config.yaml`, `command.txt`,
  `seed.txt`, `run.sh` and `manifest.json` (data/init hashes, resolved
  settings) in the run directory. The plain-MACE command is recorded as well;
  the cell-grouped wrapper only changes batch composition, not the model or
  loss. `foundation_filter_elements: true` is a recognized MACE 0.3.16 key
  used here to make `load_foundations_elements(..., load_readout=True)`
  transfer the readout heads as well as the backbone/electrostatics.

## 7. Artifacts

* `scripts/prepare_bulk_dataset.py`
* `scripts/finetune_student_bulk.py`
* `scripts/eval_student_bulk.py`
* `data/train.xyz`, `data/valid.xyz`, `data/split_manifest.json` (snapshot)
* `bench/path_b_bulk_pipeline_2026-10-23/smoke_eval.json`
* `bench/path_b_bulk_pipeline_2026-10-23/smoke_eval.md`
* `bench/path_b_bulk_pipeline_2026-10-23/smoke_speed_gas.json`
* `bench/path_b_bulk_ft_2026-10-23/attempt-01/` — staged run directory from the
  current snapshot (`student128L1_bulk_config.yaml`, `command.txt`,
  `seed.txt`, `run.sh`, `manifest.json`, empty `student_run/`); no training
  was launched. The production command in section 6 re-stages it with
  `--overwrite` so the paths and data hashes match q4.
* this document

Changes are left uncommitted for review.
