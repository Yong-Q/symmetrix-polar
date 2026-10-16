#!/usr/bin/env bash
# Symmetrix-XL polar-support canonical rebuild (v11 起逐次验证)。
# 用法: bash scripts/rebuild_symmetrix.sh <tag>   # tag 例: v14
# 产物: logs/symmetrix_rebuild_<tag>.log; 结尾 REBUILD_EXIT=<code>
# 前置: 源码树 external-repos/symmetrix-xl 已改; polar venv = /workspace/caches/conda/envs/polar
set -u
TAG="${1:?usage: rebuild_symmetrix.sh <tag>}"
MLIP=/workspace/group/shared/bza/task3_sampling_database/mlip
R=/workspace/group/shared/bza/external-repos/symmetrix-xl

export PATH=/opt/matos-tools/cuda/12.9/bin:$PATH
export CXX=$R/libsymmetrix/external/kokkos/bin/nvcc_wrapper
export CUDACXX=/opt/matos-tools/cuda/12.9/bin/nvcc
export LIBRARY_PATH=/opt/matos-tools/conda/envs/matos-compute/lib:${LIBRARY_PATH:-}

cd "$R" && PIP_USER=0 /workspace/caches/conda/envs/polar/bin/python -m pip install \
  --no-deps --force-reinstall --no-build-isolation ./symmetrix \
  --config-settings=cmake.define.CMAKE_BUILD_TYPE=Release \
  --config-settings=cmake.define.SYMMETRIX_DEVICE_BACKEND=CUDA \
  --config-settings=cmake.define.SYMMETRIX_BACKEND_SELECTOR=cuda \
  --config-settings=cmake.define.SYMMETRIX_BACKEND_ARCHITECTURE=sm80 \
  --config-settings=cmake.define.SYMMETRIX_HOST_ARCH=none \
  --config-settings=cmake.define.SYMMETRIX_BLAS_LIBRARY=/opt/matos-tools/conda/envs/matos-compute/lib/libblas.so \
  --config-settings=cmake.define.SYMMETRIX_BLAS_INCLUDE_DIR=$R/libsymmetrix/external/cblas-prototypes \
  --config-settings=cmake.define.Kokkos_ENABLE_SERIAL=ON \
  --config-settings=cmake.define.Kokkos_ENABLE_CUDA=ON \
  --config-settings=cmake.define.Kokkos_ENABLE_OPENMP=OFF \
  --config-settings=cmake.define.KokkosKernels_INST_EXECSPACE_OPENMP=OFF \
  --config-settings=cmake.define.Kokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
  --config-settings=cmake.define.Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=OFF \
  --config-settings=cmake.define.Kokkos_ENABLE_IMPL_VIEW_LEGACY=ON \
  --config-settings=cmake.define.Kokkos_ARCH_NATIVE=OFF \
  --config-settings=cmake.define.SYMMETRIX_KOKKOS=ON \
  --config-settings=cmake.define.SYMMETRIX_SPHERICART_CUDA=ON \
  --config-settings=cmake.define.SPHERICART_ENABLE_CUDA=ON \
  > "$MLIP/logs/symmetrix_rebuild_$TAG.log" 2>&1
echo "REBUILD_EXIT=$?" >> "$MLIP/logs/symmetrix_rebuild_$TAG.log"
tail -1 "$MLIP/logs/symmetrix_rebuild_$TAG.log"
