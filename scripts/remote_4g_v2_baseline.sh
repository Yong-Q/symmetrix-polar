#!/bin/bash
# Clean plan_c_4g_v2 baseline: wide/ee4G + 3-epoch charge stage + 5-epoch E/F.
set -euo pipefail

cd "$WORKDIR"
export TOOLCHAIN="$MATOS_EXTERNAL_TOOLCHAIN_ROOT"
export MLIP_ENV="$TOOLCHAIN/conda/envs/matos-compute"
export PATH="$MLIP_ENV/bin:$TOOLCHAIN/bin:$PATH"
export LD_LIBRARY_PATH="$MLIP_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

mkdir -p results_v2 logs

echo "[4g-v2] baseline training" | tee logs/remote_4g_v2_baseline.log
"$MLIP_ENV/bin/python" -m plan_c_4g_v2.train_bulk \
  --bulk-train-file data_20261025_subset/bulk_train.xyz \
  --bulk-valid-file data_20261025_subset/bulk_valid.xyz \
  --gas-train data_20261025_subset/gas_train.xyz \
  --gas-valid data_20261025_subset/gas_valid.xyz \
  --ang-preset 24 --n-rbf 16 --r-cut-ang 4.5 --use-ee4g --no-vdw \
  --physical-init --charge-stage-epochs 3 \
  --charge-train-file data_20261025_subset/bulk_train_charges.xyz \
  --charge-gas-file data_20261025_subset/gas_train_charges.xyz \
  --epochs 5 --batch-size 8 --lr 3e-4 --force-weight 20 \
  --cg-iters 40 --valid-every 1 --seed 20261026 \
  --outdir results_v2 --tag v2_baseline \
  >> logs/remote_4g_v2_baseline.log 2>&1

echo "[4g-v2] done" | tee -a logs/remote_4g_v2_baseline.log
