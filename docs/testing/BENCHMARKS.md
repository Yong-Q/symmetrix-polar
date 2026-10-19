# Testing Module — Speed Protocol and Profiling

**中文摘要：** 测速必须同模型、同精度、同构建、交替 A/B、GPU 加锁、取中位数；
CUPTI/Kokkos 只测 kernel 时长和 launch 形状，occupancy 是静态估计。round 2
发现“已验收二进制”链接了旧 object，实际在跑老 kernel；因此每轮必须记录源码
状态、trace kernel 名和最终二进制 hash。

This module defines how a speed number is produced, so that two numbers in the
repository can be compared without guessing. It also records the profiling
limitations of the container.

## 1. Measurement protocol (single rank)

| Item | Value |
|---|---|
| GPU | one NVIDIA A100-SXM4-80GB (`CUDA_VISIBLE_DEVICES=0`) |
| Precision | FP64 |
| Workload | one energy+force evaluation per loop iteration |
| System | 128-atom NH3, 14.736 Å cubic, periodic |
| Warmup | 1 evaluation |
| Measured | 12 evaluations per round |
| Rounds | 2, interleaved: baseline, candidate, baseline, candidate |
| Statistic | median of the two per-round medians of LAMMPS `Loop time` |
| Locking | every run under `flock /workspace/group/shared/bza/.tmp/gpu.lock` |
| Env | `SYMMETRIX_BACKEND=cuda12-sm80`, descriptor path, FP64 model |

The interleaved A/B is deliberate: absolute times are contention-sensitive,
while same-run ratios are stable. A speed claim without an interleaved baseline
is not accepted.

### 432-atom protocol

The 432-atom N2 box uses the existing NVT input; the reported metric is the
**first 10-step loop** (`ms/step`), the same metric as the prior 293.34 ms/step
report. The second 10-step loop is pathologically slower in both binaries
(baseline 1.42–1.53 s/step, final 0.44–0.48 s/step); it is a pre-existing
behaviour and is not used as a headline number.

## 2. Commands

```bash
# Single-rank A/B (example harness; paths are in the round directory)
export MLIP=/workspace/group/shared/bza/task3_sampling_database/mlip
export CUDA_VISIBLE_DEVICES=0
export SYMMETRIX_BACKEND=cuda12-sm80
export SYMMETRIX_BACKEND_DESCRIPTOR_PATH=$MLIP/models/mace-polar/backend_cuda12_sm80.json
export LD_LIBRARY_PATH=/opt/matos-tools/cuda/12.9/lib64:/opt/matos-tools/conda/envs/matos-compute/lib:$LD_LIBRARY_PATH

# Interleaved A/B is launched through the round's Python harness, e.g.
#   bench/kernel_round2_2026-10-19/speed_ab.py
# under:
flock -w 10800 /workspace/group/shared/bza/.tmp/gpu.lock -c '<python ...>'
```

Each run directory contains `result.<tag>.json` with `median_ms`, `mean_ms`,
`std_ms`, `min_ms`, `max_ms`, and the raw samples. The summary JSON records
the protocol and the per-round medians.

## 3. Binary and object staleness — mandatory check

Round 2 found that the “accepted” binary `7e9c6bec…` had been linked from
**stale object files**: the source already contained the 2-D `dense_reverse`
and the fused gate-reverse path, but the runtime still executed the old
1-D/26 ms and unfused/13 ms kernels. A forced rebuild (touching the source,
content unchanged) activated them and saved ~39 ms/eval.

Rules that follow from this:

1. Before a speed claim, force a rebuild of the changed translation units (or a
   clean build) and verify the kernel names in a CUPTI trace.
2. Do not trust a binary hash alone as proof of the source state; the object
   files can be stale. Record the source hash, the object mtime, and the trace
   label.
3. Every accepted round states the final `lmp` sha256 and the trace that
   confirms the expected kernels are present.

## 4. Profiling: what is measured and what is not

**Measured (CUPTI Activity API + Kokkos Tools):**

- kernel name, start/end timestamps, duration;
- grid/block geometry, registers per thread, static/dynamic shared memory;
- Kokkos `parallel_for/reduce/scan` labels paired to CUPTI kernels;
- host stage spans from `MATOS_POLAR_PROFILE=1` (fenced marks).

**Not measured (blocked):**

- achieved occupancy, DRAM throughput, SM utilization counters;
- `ncu` is a wrapper without the Nsight Compute payload in this container, and
  the host driver has `RmProfilingAdminOnly: 1` with no capabilities. Hardware
  counters would fail with `ERR_NVGPUCTRPERM`.
- Therefore any “occupancy” or “SM coverage” number in this repository is a
  **static estimate** from launch geometry, not a measured achieved value.

The profiler shim is read-only (`LD_PRELOAD=libsymtracer.so`,
`KOKKOS_TOOLS_LIBS=libsymtracer.so`); it does not modify the engine.

## 5. Definitions used in the result tables

- **GPU busy:** \(\sum_i t_i\) over all kernels in the eval window; the eval
  window is between `eval:enter` and `backbone-reverse:done`.
- **SM-weighted busy:** \(\sum_i t_i O_i\) with
  \(O_i=\min(1,G_i/(108 B_i))\); see `../theory/PERFORMANCE_MODEL.md`.
- **Coverage:** \(O_i\) as a percentage.
- **Wall:** LAMMPS `Loop time` per evaluation (median).
- **Kernel time:** CUPTI duration per eval, from one warm eval with the
  profiler markers; profiling adds a few percent to wall but does not change
  kernel durations.

## 6. Gate-before-speed rule

No speed change is accepted unless:

1. the complete gate suite passes on the same final build;
2. the A/B is interleaved and the baseline is from the same build session;
3. the kernel-level attribution is consistent with the wall-time saving
   (e.g., round 1: 61.89 ms kernel saving vs 61.81 ms wall saving);
4. the final binary sha256 and the trace are recorded.

## 7. Artifacts

| Artifact | Path |
|---|---|
| Round-2 report | `bench/kernel_round2_2026-10-19/FINAL_REPORT.md` |
| Round-2 summary | `docs/examples/data/kernel_round2_summary.json` |
| Round-2 family table | `docs/examples/data/kernel_round2_families.csv` |
| Round-2 label table | `docs/examples/data/kernel_round2_labels.csv` |
| Round-2 traces | `bench/kernel_round2_2026-10-19/trace_runs/` and `analysis_exact/` |
| Round-1 report | `bench/e3lin_opt_2026-10-19/FINAL_REPORT_e3lin.md` |
| Round-1 traces | `bench/e3lin_opt_2026-10-19/trace_runs/` |
| H1 validation | `docs/examples/data/mpi_h1_validation.json` |
