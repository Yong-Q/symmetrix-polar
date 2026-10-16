#!/usr/bin/env bash
# rebuild_fast.sh — 快速迭代构建(跳过 wheel 打包/安装, 省 3-5 分钟; 2026-10-15)
#
# 原理: pip install 的 wheel 流程 = 编译 + 打 wheel + 卸旧 + 装新(打包环节对大 wheel
#       要 3-5 分钟)。本脚本直接 ninja 增量编译 build 目录, 然后把:
#         - 编译产物 .so  → 覆盖 site-packages/symmetrix/
#         - python 源     → 同步 site-packages/symmetrix/*.py (calculator.py 等)
#       省去 wheel 环节。产物与 pip 版逐字节同源(同一 build 目录).
#
# 用法: bash scripts/rebuild_fast.sh <tag>
# 回退: 任何可疑时用 scripts/rebuild_symmetrix.sh <tag>(完整 pip 流程)
set -u
TAG="${1:?usage: rebuild_fast.sh <tag>}"
MLIP=/workspace/group/shared/bza/task3_sampling_database/mlip
R=/workspace/group/shared/bza/external-repos/symmetrix-xl
BUILD=$R/symmetrix/build
SITE=/workspace/caches/conda/envs/polar/lib/python3.12/site-packages/symmetrix
LOG=$MLIP/logs/symmetrix_rebuild_$TAG.log

export PATH=/opt/matos-tools/cuda/12.9/bin:$PATH
export CXX=$R/libsymmetrix/external/kokkos/bin/nvcc_wrapper
export CUDACXX=/opt/matos-tools/cuda/12.9/bin/nvcc
export LIBRARY_PATH=/opt/matos-tools/conda/envs/matos-compute/lib:${LIBRARY_PATH:-}

cd "$BUILD" || exit 1
{
  echo "== ninja incremental ($(date '+%H:%M:%S')) =="
  /workspace/caches/conda/envs/polar/bin/ninja 2>&1 | tee "$LOG.ninja" | tail -5
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "NINJA_FAIL rc=$rc"
    echo "REBUILD_EXIT=$rc"
    exit "$rc"
  fi
  cp -f "$BUILD"/symmetrix.cpython-312-x86_64-linux-gnu.so "$SITE"/ 2>&1
  cp -f "$R"/symmetrix/source/symmetrix/*.py "$SITE"/ 2>&1
  # cli 子包(如存在变化)
  [ -d "$R/symmetrix/source/symmetrix/cli" ] && \
      cp -rf "$R"/symmetrix/source/symmetrix/cli/. "$SITE"/cli/ 2>/dev/null
  echo "COPIED so+py -> site-packages"
  echo "REBUILD_EXIT=0"
} > "$LOG" 2>&1
tail -3 "$LOG"
