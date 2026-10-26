#!/bin/bash
# Remote Q4 round: 4G committee active selection -> merge -> 4G distillation.
set -euo pipefail

cd "$WORKDIR"
export TOOLCHAIN="$MATOS_EXTERNAL_TOOLCHAIN_ROOT"
export MLIP_ENV="$TOOLCHAIN/conda/envs/matos-compute"
export PATH="$MLIP_ENV/bin:$TOOLCHAIN/bin:$PATH"
export LD_LIBRARY_PATH="$MLIP_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

mkdir -p active/remote_r1 results_remote logs

echo "[4g-al] stage 1: committee pool selection" | tee logs/remote_4g_al.log
"$MLIP_ENV/bin/python" scripts/pool_active_select.py \
  --pool data_20261025_subset/bulk_train.xyz \
  --pool-stride 3 --pool-limit 600 \
  --students checkpoints/c1.pt checkpoints/c2.pt checkpoints/c3.pt \
  --outdir active/remote_r1 --max-select 40 --device cuda \
  >> logs/remote_4g_al.log 2>&1

cp active/remote_r1/selected.xyz active/remote_r1/selected_labeled.xyz

echo "[4g-al] stage 2: merge selected frames" | tee -a logs/remote_4g_al.log
"$MLIP_ENV/bin/python" -m plan_c_4g.active --stage merge --round 1 \
  --outdir active/remote_r1 \
  --base-train data_20261025_subset/bulk_train.xyz \
  >> logs/remote_4g_al.log 2>&1

echo "[4g-al] stage 3: 4G distillation" | tee -a logs/remote_4g_al.log
"$MLIP_ENV/bin/python" -m plan_c_4g.train_bulk \
  --bulk-train-file active/remote_r1/train_merged.xyz \
  --bulk-valid-file data_20261025_subset/bulk_valid.xyz \
  --gas-train data_20261025_subset/gas_train.xyz \
  --gas-valid data_20261025_subset/gas_valid.xyz \
  --init checkpoints/fixed.pt --physical-init \
  --epochs 5 --batch-size 8 --lr 1e-4 --force-weight 20 \
  --cg-iters 40 --valid-every 1 --seed 20261026 \
  --outdir results_remote --tag 4g_al_r1 \
  >> logs/remote_4g_al.log 2>&1

echo "[4g-al] done" | tee -a logs/remote_4g_al.log
