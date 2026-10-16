#!/usr/bin/env bash
# build_status.sh — 构建实时状态(任何人可随时运行) 2026-10-15
# 用法: bash scripts/build_status.sh
R=/workspace/group/shared/bza/external-repos/symmetrix-xl
B=$R/symmetrix/build
LOG=/workspace/group/shared/bza/task3_sampling_database/mlip/logs
echo "==== 构建实时状态 $(date '+%H:%M:%S') ===="
echo "--- 剩余编译边(ninja -n) ---"
cd "$B" && timeout 45 /workspace/caches/conda/envs/polar/bin/ninja -n 2>/dev/null | grep -E "Building|Linking" | head -12
echo "--- 活跃 nvcc 进程 ---"
ps -eo etime,args | grep "[n]vcc " | grep -oE "mace_polar_kokkos|mace_nonlinear_kokkos_[a-z]+|symmetrix\.cpp|e3nn[a-z_]*|mace\.cpp" | sort | uniq -c
echo "nvcc 总数: $(ps -eo comm | grep -c nvcc)"
echo "--- 最近完成的 .o(近1小时) ---"
find "$B" -name "*.o" -newermt "-60 minutes" 2>/dev/null | sed "s|.*/||" | sort | head -15
echo "--- 最新构建日志 ---"
ls -t $LOG/symmetrix_rebuild_v*.log 2>/dev/null | head -1 | xargs tail -3 2>/dev/null
