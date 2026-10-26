#!/bin/bash
set -euo pipefail
REPO=/workspace/group/shared/bza/symmetrix-polar
export PYTHONPATH=/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0:$REPO:${PYTHONPATH:-}
PY=/opt/matos-tools/conda/envs/matos-compute/bin/python
CKPT=/workspace/group/shared/bza/symmetrix-polar/bench/plan_d_linear_2026-10-23/models/off23ft/plan_d_linear_student_gas128.model
$PY -m mace.cli.run_train \
  --config "$REPO/bench/plan_d_bulk_2026-10-25/configs/student_bulk.yaml" \
  --name plan_d_bulk \
  --foundation_model "$CKPT" \
  --device cuda 2>&1 | tee "$REPO/bench/plan_d_bulk_2026-10-25/logs/train_bulk.log"
