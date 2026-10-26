# Remote Q4 tasks (2026-10-26)

All tasks below are single-GPU `queue004` submissions. Q4 was fully allocated
(64/64, 0 free) at submission time, so the active ones are queued.

## 1. Plan D: expanded sampling + MACE-POLAR distillation

| item | value |
|---|---|
| task | `t-20261026135843-f6bq5` |
| config | `bench/plan_d_al_2026-10-26/configs/student_al_full_remote.yaml` |
| data | `data_20261025_al_full/train.xyz` (24,776 frames) |
| validation | `data_20261025/valid.xyz` (3,465 frames) |
| init | `bench/plan_d_bulk_2026-10-25/models/bulk/plan_d_bulk.model` |
| deadline | 50 h |
| visible workdir | `/workspace/group/shared/bza/symmetrix-polar/attempt-03` |
| status | queued |

## 2. RUNNER/4G: active sampling + distillation

| item | value |
|---|---|
| task | `t-20261026144930-jw8c7` |
| entrypoint | `scripts/remote_4g_al_round.sh` |
| pool | `data_20261025_subset/bulk_train.xyz` (stride 3, 600 frames) |
| committee | `checkpoints/c1.pt`, `c2.pt`, `c3.pt` |
| student init | `checkpoints/fixed.pt` (self-energy + Pearson χ/J) |
| train | merged pool selection + 5 epochs 4G distillation |
| deadline | 50 h |
| visible workdir | `/workspace/group/shared/bza/symmetrix-polar/runner-4gal-01` |
| status | cancelled (moved to local, see `PLAN_C_4G_LOCAL_AL_2026-10-26.md`) |

## 3. Clean 4G v2 baseline

| item | value |
|---|---|
| task | `t-20261026164227-vvhlh` |
| code | `plan_c_4g_v2/` (clean fork, not a patch of `plan_c_4g`) |
| entrypoint | `scripts/remote_4g_v2_baseline.sh` |
| data | `data_20261025_subset/` + teacher charge files |
| model | 24 G4 / 16 RBF / 4.5 Å + ee4G, 3-epoch charge stage + 5-epoch E/F |
| deadline | 50 h |
| visible workdir | `/workspace/group/shared/bza/symmetrix-polar/runner-4gv2-01` |
| status | queued |

## Notes

- The Plan D task is a **linear MACE** student distilling MACE-POLAR labels;
  it is not a 4G/4H HDNNP and is not initialised from PolarMACE.
- The 4G work is an independent PyTorch implementation of the RuNNer2 4G
  construction; the RUNNER repository remains the Slurm/VASP committee
  workflow and does not contain the GPU 4G model.
- Local Plan D was started when the local A100 freed up; the current local
  process is training the 24,776-frame expanded set, not a new acquisition
  round.
- The clean v2 baseline is queued on Q4; l=1 multipoles are the next milestone.
- A third queued allocation in the user quota (`t-20261026135739-j2wxb`) is
  not one of these tasks.
