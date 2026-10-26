# Remote Q4 tasks (2026-10-26)

Both tasks are single-GPU `queue004` submissions. Q4 was fully allocated
(64/64, 0 free) at submission time, so both are queued.

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
| status | queued |

## Notes

- The Plan D task is a MACE student distilling MACE-POLAR labels; it is not
  initialised from PolarMACE.
- The 4G task is an independent PyTorch implementation of the RuNNer2 4G
  construction; the RUNNER repository remains the Slurm/VASP committee
  workflow and does not contain the GPU 4G model.
- Local A100 was not free: ~35 GB used, 55 % utilisation, ~47 GB free, below
  the ~64 GB peak of the full Plan D fine-tune.
- A third queued allocation in the user quota (`t-20261026135739-j2wxb`) is
  not one of these two tasks.
