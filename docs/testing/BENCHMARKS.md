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
| System | 128-atom NH3 = **32 NH3 molecules**, 14.736 Å cubic, periodic |
| Warmup | 1 evaluation |
| Measured | 12 evaluations per round |
| Rounds | 2, interleaved: baseline, candidate, baseline, candidate |
| Statistic | median of the two per-round medians of LAMMPS `Loop time` |
| Locking | every run under `flock /workspace/group/shared/bza/.tmp/gpu.lock` |
| Env | `SYMMETRIX_BACKEND=cuda12-sm80`, descriptor path, FP64 model |

### Round-3 final result (build 8, 2026-10-20)

| Item | Value |
|---|---:|
| 128 atoms (32 NH3), full PolarMACE | **48.74 ms/eval** |
| Same binary, all round-3 env switches reverted | 53.54 ms/eval |
| Same-binary speedup | **1.0985×** |
| Speedup vs round-2 accepted 56.50 ms | **1.159×** |
| Speedup vs v29 Kokkos engine 183.8 ms | **3.77×** |
| 1024 atoms (256 NH3), full PolarMACE | **312.1 ms/eval** |
| Final binary sha256 | `c7608f35845548ef037341833c138da87515484ce51c04a257a5920bb9424631` |

A/B files: `bench/kernel_round3_2026-10-19/speed_ab_r3final2.json`; trace:
`bench/kernel_round3_2026-10-19/analysis_exact/r3c_full_*`; parity:
`bench/kernel_round3_2026-10-19/r3_gate_lammps_parity_build8.log`. The
same-binary fallback protocol disables every round-3 slice through its
documented env switch, so the ratio is not confounded by a different
LAMMPS/Kokkos build.

### Plan B-lite final result (2026-10-20)

| Item | Value |
|---|---:|
| 128 atoms (32 NH3), full PolarMACE | **40.95 ms/eval** |
| Replay-cache-only (B1) | 45.09 ms/eval |
| Round-3 build 8 (pre-Plan-B) | 48.74 ms/eval |
| Speedup vs build 8 | **1.190×** |
| Speedup vs v29 | **4.49×** |
| Final binary sha256 | `18d5fc94fda3df98837520b885de48956afa59fb4c81223f9fe5d3f2e7833c64` |
| Gates | module 5–8, 32/32 end-to-end, LAMMPS 1-rank and 2-rank exact: **PASS** |

Slices: B1 replay cache **+3.39 ms**, B4 reverse-gather fusion **+4.13 ms**
(same-binary A/B after the write-through fix); B3 merged unpack is exact but
**−5.97 ms** (rejected, opt-in); B5 is neutral. Defaults:
`MATOS_REPLAY_CACHE=1`, `MATOS_REV_GATHER_FUSE=1`, `MATOS_UVU_X2_MB_SMEM=0`.
Artifacts: `bench/plan_b_lite_2026-10-20/` and
`docs/results/PLAN_B_LITE_FINAL.md`.

**Plan B-core cross-check (2026-10-21):** frozen-tree binary `ed331d23…`
measures **38.73 ms/eval**, all gates PASS (module 5–8, 32/32, LAMMPS 1-rank
and 2-rank exact). The accepted 38.66 ms build includes the write-through fix;
the 0.2% difference is run noise. Repeated evaluations are mandatory for any
FP64 MH-1 team build. Artifacts: `bench/plan_b_core_2026-10-20/` and
`docs/results/PLAN_B_CORE_FINAL.md`.

### Plan B-full final result (2026-10-21)

| Item | Value |
|---|---:|
| 128 atoms (32 NH3), full PolarMACE | **31.33 ms/eval** |
| Same-session B-core baseline | 38.51 ms/eval |
| Saving | **−7.18 ms (−18.6%)** |
| MH-1 FP64 team | +3.10 ms |
| UVU reverse d1 specialization | +3.83 ms |
| reverse cutoff team | +2.03 ms |
| UVU forward/LE d1 split | +0.74 ms (rejected, <1 ms bar) |
| Final binary sha256 | `bcecbf9051cf8ab830e6b3af303fa44e6f0911f95768cfe57a652f7ebaef14ec` |
| Gates | module 5–8, 32/32, LAMMPS 1/2-rank, repeated-eval: **PASS** |

Artifacts: `bench/plan_b_full_2026-10-20/` and
`docs/results/PLAN_B_FULL_FINAL.md`. Remaining gap to 30 ms: **1.33 ms**.

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
| **Round-3 final report** | `docs/results/KERNEL_ROUND3_FINAL.md` |
| **Round-3 A/B** | `bench/kernel_round3_2026-10-19/speed_ab_r3final2.json` |
| **Round-3 trace** | `bench/kernel_round3_2026-10-19/analysis_exact/r3c_full_*` |
| **Round-3 parity** | `bench/kernel_round3_2026-10-19/r3_gate_lammps_parity_build8.log` |
| Round-2 report | `bench/kernel_round2_2026-10-19/FINAL_REPORT.md` |
| Round-2 summary | `docs/examples/data/kernel_round2_summary.json` |
| Round-2 family table | `docs/examples/data/kernel_round2_families.csv` |
| Round-2 label table | `docs/examples/data/kernel_round2_labels.csv` |
| Round-2 traces | `bench/kernel_round2_2026-10-19/trace_runs/` and `analysis_exact/` |
| Round-1 report | `bench/e3lin_opt_2026-10-19/FINAL_REPORT_e3lin.md` |
| Round-1 traces | `bench/e3lin_opt_2026-10-19/trace_runs/` |
| H1 validation | `docs/examples/data/mpi_h1_validation.json` |
