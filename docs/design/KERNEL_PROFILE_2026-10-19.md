# Symmetrix-XL PolarMACE GPU Kernel Profile — 2026-10-19

Scope: one warmed energy+force evaluation of the full PolarMACE model and of the
polar-off backbone on the 128-atom NH3 system. Read-only profile; no engine
code or model files were modified.

- Engine binary profiled: `symmetrix/build-lammps-aacd036df381800a/lmp`
  (loaded at 08:29 UTC; the working tree was concurrently edited/rebuilt during
  this study — see **Caveat** below).
- Models: `models/mace-polar/polar_symmetrix_probe.json` (full),
  `models/mace-polar/polar_symmetrix_probe_polaroff.json` (backbone).
- System: `bench/lammps_parity/engine_compare/nh3_128.data` (128 atoms, 10,852
  full neighbors), `pair_style symmetrix/mace/kk charge 0.0 spin 1.0`.
- GPU: single A100-SXM4-80GB; every run serialized with
  `flock /workspace/group/shared/bza/.tmp/gpu.lock`.
- Clean wall times (unprofiled, same inputs): full **183.5 ms/eval**,
  backbone-only **87.2 ms/eval**.

## Profiling blockers and method

**`ncu` is unusable in this container.** The only installed path,
`/opt/matos-tools/cuda/12.9/bin/ncu`, is a 3.8 KB wrapper with no payload:

```
ERROR : nsight-compute directory is not found under
/opt/matos-tools/cuda/12.9/bin/../ or /opt/nvidia.
Nsight Compute is not installed on your system.
```

There is no `nsight-compute` directory under `/opt`, `/usr/local`,
`/workspace/caches`, and no `nsys`. In addition, the host driver is
535.129.03 with `RmProfilingAdminOnly: 1`, the process runs as
`uid=1000(openvscode-server)` with `CapEff=0000000000000000`, so hardware-
counter collection would fail with `ERR_NVGPUCTRPERM` even if `ncu` were
installed.

Fallback used (no fabricated numbers):

- **CUPTI Activity API** (tracing, allowed without root): captured every kernel
  with name, start/end timestamps, grid/block, registers/thread, static and
  dynamic shared memory. Implemented as a read-only LD_PRELOAD shim,
  `bench/kernel_profile_2026-10-19/symtracer.cpp`.
- **Kokkos Tools callbacks**: the same shim exports `kokkosp_*`, so every
  `parallel_for/reduce/scan` is paired with its engine label. Kernel execution
  order and host label order are aligned with a causality-constrained sequence
  alignment (single CUDA stream).
- **Engine stage marks** (`MATOS_POLAR_PROFILE=1`): fenced host marks, mapped
  to the CUPTI clock through a `CLOCK_MONOTONIC` calibration file. These are
  the stage boundaries used below.
- CUPTI timestamps and kernel records give real GPU durations. Occupancy and
  "SM coverage" are **static estimates** from registers/block/grid, not
  measured achieved occupancy. Memory-throughput counters (DRAM %, SM %) are
  unavailable because of the driver restriction.

Profiled runs: `MATOS_POLAR_PROFILE=1` adds one fence + fprintf per stage,
raising the loop time from 183.5 to 198.5 ms; kernel durations themselves are
unaffected. Percentages below use the clean 183.5 ms denominator.

## Eval decomposition

Full model, warm eval (second `eval:enter .. backbone-reverse:done` block):

| stage | host span (ms) | GPU busy (ms) | SM-weighted busy (ms) | launches |
|---|---:|---:|---:|---:|
| pre-pipeline: backbone+stage1 | 39.67 | 35.32 | 33.21 | 173 |
| stage6:coulomb (SCF/Coulomb block) | 15.84 | 13.40 | 4.94 | 109 |
| pipeline:end (local-electron UVU) | 5.94 | 5.07 | 1.45 | 35 |
| rev1.5b: uvu-reverse | 7.60 | 7.29 | 3.04 | 12 |
| rev5a: replay (x2) | 15.59 | 13.40 | 4.95 | 101 |
| rev5d: mlp-reverse (x2) | 26.60 | 26.18 | 0.02 | 26 |
| rev5e: embedding-reverse (x2) | 11.98 | 11.36 | 5.80 | 30 |
| rev5f: descriptor-reverse (x2) | 3.78 | 2.72 | 0.004 | 33 |
| rev9: node-forces | 2.85 | 2.73 | 2.73 | 6 |
| backbone-reverse:done (backbone backward) | 55.92 | 51.81 | 31.55 | 117 |
| **total** | **196.6** | **176.4** | **95.7** | **755** |

`SM-weighted busy` multiplies each kernel duration by
`min(1, grid_blocks / (108 × blocks_per_SM_static))`. It estimates how much of
the device the launch could actually occupy. The full eval is GPU-busy 96% of
the clean wall (176.4/183.5 ms), but **SM-weighted utilization is only 52%**
(95.7/183.5 ms). The engine is not launch-overhead bound (~7 ms of host gaps);
it is bound by kernels that use one or a few SMs.

The worst offenders are the two `rev5d:mlp-reverse` stages: 26.2 ms of GPU busy
time with **0.02 ms** of SM-weighted time. `rev5f:descriptor-reverse` is
similar (2.72 ms GPU, 0.004 ms SM-weighted). At the other extreme,
`rev9:node-forces` is 100% SM-covered.

## Top-10 kernels/regions, full model, warm eval

Sorted by total GPU time. `% eval` uses 183.5 ms. `occ` is theoretical
per-SM occupancy; `SM cov` is the static grid coverage defined above.

| # | kernel / region (Kokkos label) | family | total ms | % eval | launches | avg µs | regs | block / grid | occ / SM cov | bottleneck note |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---|
| 1 | `e3 linear small batch` | e3nn linear | 33.78 | 18.4% | 147 | 230 | 31 | 256 / 4x256 | 100% / 100% | scalar FP64 K-loop, weight reload per sample; ~3% of FP64 peak |
| 2 | `e3 reverse linear small batch` | e3nn linear | 32.21 | 17.6% | 97 | 332 | 32 | 256 / 4x256 | 100% / 100% | same, source-major policy → strided/uncoalesced weight reads |
| 3 | `polar reverse dense` | polar pipeline | 25.74 | 14.0% | 8 | 3218 | 40 | 128 / 1x1x1 | 75% / 0.1% | one 128-thread block for a full dense layer; redundant atomics |
| 4 | `reverse nonlinear normalization` | backbone nonlinear | 12.74 | 6.9% | 2 | 6368 | 52 | 128 / 1x1x1 | 56% / 0.1% | one block; two serial passes over `pre_gate_width` per node |
| 5 | `polar reverse uvu mode0 x2` | polar pipeline | 10.90 | 5.9% | 10 | 1090 | 48 | 256 / 2x128 | 62% / 47% | small per-thread dot products, half-device grid |
| 6 | `polar rev replay uvu mode0` | polar pipeline | 5.76 | 3.1% | 6 | 961 | 32 | 256 / 2x128 | 100% / 30% | replay UVU; 30% grid coverage |
| 7 | `polar uvu mode0` | polar pipeline | 5.76 | 3.1% | 6 | 960 | 32 | 256 / 2x128 | 100% / 30% | forward UVU; 30% grid coverage |
| 8 | `polar reverse uvu mode0 x1` | polar pipeline | 5.45 | 3.0% | 10 | 545 | 32 | 256 / 2x128 | 100% / 30% | paired with #5; separate launch |
| 9 | `polar le uvu mode0` | polar pipeline | 4.59 | 2.5% | 4 | 1147 | 32 | 256 / 2x128 | 100% / 30% | local-electron UVU; 30% grid coverage |
| 10 | `e3 tensor reverse mh1 fused channels` | tensor product | 4.32 | 2.4% | 2 | 2158 | 48 | 256 / 73x32 | 62% / 100% | regular FP64 contraction; compute-bound |

Top 10 = **141.2 ms, 77.0% of the clean eval**, 292 of 755 launches.

Family roll-up (sum of all labels; SM-weighted time shows how much of the
duration could occupy the device):

| family | total ms | % eval | launches | SM-weighted ms | owner |
|---|---:|---:|---:|---:|---|
| polar pipeline (SCF/Coulomb/LE/UVU/dense) | 73.36 | 40.0% | 238 | 12.84 | `mace_polar_impl.tpp` |
| e3nn linear (scalar forward/reverse + mask) | 66.68 | 36.3% | 302 | 66.68 | `e3nn_kokkos.cpp` |
| backbone nonlinear (gates, gather, normalization) | 19.03 | 10.4% | 46 | 4.18 | `mace_nonlinear_kokkos_impl.tpp` |
| tensor product | 10.91 | 5.9% | 18 | 9.14 | `e3nn_kokkos.cpp` (MH-1 fused paths) |
| MLP/dense | 3.44 | 1.9% | 65 | 1.12 | `affine_mlp_kokkos.cpp` |
| cuBLAS/external | 1.40 | 0.8% | 51 | 0.63 | cuBLAS DMMA / sphericart |
| other / init / radial / pair | 1.58 | 0.9% | 30 | 1.09 | various |

The pair force/virial reduction is **not** a bottleneck: only 0.002 ms of pair
kernels fall inside the model eval, and the post-model reduction/virial kernels
(including the atomics) total <0.05 ms. `e3nn_product_kokkos.cpp` contraction
kernels (`compiled product contraction/reverse`) contribute ~1.5 ms;
`affine_mlp_kokkos.cpp` contributes ~3.4 ms. The dominant cost is the
`e3nn_kokkos.cpp` scalar linear path and the polar `mace_polar_impl.tpp`
kernels.

Also notable: `e3 tensor product mh1 fused` 3.31 ms (100% cov),
`e3 tensor reverse mh1 fused harmonics` 3.08 ms (45% cov),
`polar scf project features` 2.32 ms and `polar rev replay project` 2.30 ms
(both 1x1x1 grids, 0.1% cov), `gather edge message adjoint` 2.10 ms (100% cov),
`reverse convolution cutoff` 1.53 ms (2.9% cov), `e3 linear mask` 0.68 ms /
58 launches, cuBLAS/external 1.40 ms / 51 launches (cuBLAS DMMA kernels are
negligible because the packed path is not used — see below), memsets+memcpys
2.18 ms / 304 operations.

## Backbone-only model and polar delta

Backbone-only warm eval: 91.6 ms measured (87.2 ms clean), GPU busy 83.8 ms,
SM-weighted 64.3 ms, 268 kernels. Its top kernels are the same e3 linear
forward/reverse (24.2/24.3 ms), `reverse nonlinear normalization` (12.7 ms),
and the tensor-product kernels (~10.7 ms). The polar blocks add ~92.6 ms of GPU
time. Largest positive deltas (full − backbone):

| delta ms | label |
|---:|---|
| 25.74 | `polar reverse dense` |
| 10.90 | `polar reverse uvu mode0 x2` |
| 9.58 | `e3 linear small batch` (polar adapter linears) |
| 7.93 | `e3 reverse linear small batch` |
| 5.76 | `polar rev replay uvu mode0` |
| 5.76 | `polar uvu mode0` |
| 5.45 | `polar reverse uvu mode0 x1` |
| 4.59 | `polar le uvu mode0` |
| 2.32 | `polar scf project features` |
| 2.30 | `polar rev replay project` |
| 1.16 | `polar project features` |
| 1.01 | `polar desc ab reverse` |

## Bottleneck classification

- **Launch-count / tiny-duration bound?** Only partially. `e3 linear mask`
  (58 × 11.8 µs) and `affine linear epilogue` (32 × 32.5 µs) are launch-heavy
  but total ~1.7 ms. The 304 memset/memcpy operations total 2.18 ms. The real
  problem is not launch overhead but kernel shape.
- **Low occupancy / grid-starved:** `polar reverse dense`, `reverse nonlinear
  normalization`, `polar scf project features`, `polar rev replay project`,
  `polar project features`, `polar rev phase and source`, and
  `polar desc proj phase reverse` all launch **1x1x1 grids** (128 threads).
  Their per-SM theoretical occupancy is 56–75%, but only one block exists, so
  the device is effectively idle. Across the 39 such launches in the eval this
  pattern accounts for **47.8 ms of GPU time and 0.042 ms of SM-weighted
  time** — 26% of the eval spent on one SM.
- **Atomic contention:** `polar reverse dense` uses `Kokkos::atomic_add` on
  `i_adj(node,i)` although each `(node,i)` has a unique writer; it is pure
  overhead. Pair force/virial atomics are fine (<0.05 ms at 128 atoms).
- **Memory vs compute:** the e3 linear scalar kernels are load/latency-bound:
  the weight matrix is small enough to sit in L2, but each sample re-reads it
  and the K loop is serial per thread; measured throughput is a few percent of
  the A100 FP64 peak. The tensor-product kernels (`73x128`, `73x32` grids) are
  the only large compute-bound FP64 kernels. The UVU kernels are small-dot-
  product latency-bound. cuBLAS is not a bottleneck because it is barely used.

## Caveat: concurrent tree edits

During this study the engine tree was modified and relinked by another agent:

- `mace_polar_impl.tpp` mtime 08:33 UTC; `mace_nonlinear_kokkos_impl.tpp`
  mtime 08:35 UTC; `libsymmetrix.a` 08:31; `lmp` relinked 08:34.
- The full trace (08:29) used the **older binary** with the 1-D
  `polar reverse dense` kernel. The current tree already contains a 2-D
  rewrite at `mace_polar_impl.tpp:1998-2010` (`MDRangePolicy<Rank<2>>` over
  `(num_nodes, in_w)`, atomics removed). **Rebuild and re-profile before
  treating #3 below as outstanding.**
- The backbone trace (08:33) used a binary from before the 08:34 relink; its
  polar-off path is unaffected by the `polar reverse dense` rewrite.

## Top-3 accuracy-preserving optimizations (current tree)

All keep FP64 and all physical terms. None uses float32 or drops a term.

### 1. e3 linear small-batch forward/reverse — 66.0 ms, 36% of eval

Targets: `libsymmetrix/source/e3nn_kokkos.cpp:420-437` (`evaluate` scalar
kernel), `:1214-1231` (`reverse` scalar kernel), forced on CUDA by
`:383-391` (`use_scalar_backend` returns `true` unconditionally under
`KOKKOS_ENABLE_CUDA`). Packed/cuBLAS variants already exist at
`:541-597` + `:579` (forward) and `:1573-1608` + `:1603` (reverse), with the
`evaluate_packed_to_packed` / `reverse_packed_to_packed` entry points at
`:964-999` / `:1573-1608`.

Actions:
- **Safe (bitwise):** keep the scalar per-output accumulation order but stage
  the weight column and input row in shared memory once per block, and
  transpose the reverse weight access (`source`-major policy currently reads
  `weights[source*M_out+target]` with stride `M_out` between adjacent threads).
  Expected 2–4x on these kernels → save **30–45 ms**.
- **Larger (needs parity re-run):** route the same layers through the existing
  `packed_gemm` path (`KokkosBlas::gemm`, cuBLAS FP64 DMMA). The 51 cuBLAS
  kernels currently in the trace cost only 1.4 ms, so 244 batched launches
  should remain cheap. This changes FP64 summation order but no math; re-run
  `bench/polar_lammps_parity.py` and `bench/polar_endtoend_parity.py`.

### 2. `reverse nonlinear normalization` — 12.7 ms, 6.9% of eval

Target: `libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp:5975-5992`.
One 128-thread block; each thread makes two serial passes over
`pre_gate_width` (the `linear_adj` write and the `density_adj` reduction).
Actions: parallelize the `linear_adj` write with
`MDRangePolicy<Rank<2>>({0,0},{num_nodes,pre_gate_width})`, and compute the
per-node `density_adj` with a `TeamPolicy` (one team per node) or a
`parallel_reduce` over `(node,k)` with a deterministic FP64 reduction. This
removes the 6.4 ms single-block launches; expected **11–12 ms saved**.

### 3. UVU family — 35.0 ms, 19% of eval

Targets: `mace_polar_impl.tpp:860-878` (`polar uvu mode0`),
`:1412-1427` (`polar le uvu mode0`), `:1841-1877` (reverse x1/x2),
`:2354-2372` (replay mode0); mode1 variants at the adjacent lambdas.
All are 2x128 grids (30–47% SM coverage), and x1/x2 are separate launches over
the same operands. Actions: fuse the x1 and x2 adjoint updates into one
`MDRangePolicy` pass (each output keeps the same per-thread accumulation
order), and use a shared-memory tile for the small per-thread dot products.
Expected **5–10 ms saved**. (If the concurrent `polar reverse dense` rewrite
lands, the measured #3 hot spot disappears; re-profile first.)

Secondary, lower payoff: parallelize the 1x1x1 `polar scf project features`
(`:937-990`) and `polar rev replay project` (`:2251-2312`) over `(node,m)`
with a k-reduction (~3–4 ms); expand `reverse convolution cutoff`
(`mace_nonlinear_kokkos_impl.tpp:7098`) from 37 blocks to edge-level
parallelism (~1 ms); fuse `e3 linear mask` (`e3nn_kokkos.cpp:438`) into the
last instruction finalization (~0.7 ms + 58 launches).

## Raw artifacts

Root: `/workspace/group/shared/bza/task3_sampling_database/mlip/bench/kernel_profile_2026-10-19/`

Full model (08:29 binary):
- `runs/full_v29_diag/cupti_kernels.csv` — 3,900 kernel+mem records, 755 in-eval kernels
- `runs/full_v29_diag/kokkos_events.csv` — 38,462 label/region events
- `runs/full_v29_diag/clock_sync.csv`, `stderr_stamped.full_v29_diag`
- `analysis_final/full_labels.csv`, `full_stages.csv`, `full_stage_labels.csv`

Backbone-only:
- `runs/backbone_polaroff/cupti_kernels.csv`, `kokkos_events.csv`,
  `clock_sync.csv`, `stderr_stamped.backbone_polaroff`
- `analysis_final/backbone_labels.csv`, `backbone_stages.csv`

Cross-model:
- `analysis_final/polar_delta.csv`, `summary.json`
- `analysis_final/top_labels.png`, `stage_stack.png`, `delta.png`

Tooling (read-only shim, no engine changes):
- `symtracer.cpp`, `libsymtracer.so`, `run_trace.sh`, `stamp_stderr.py`,
  `analyze_trace.py`, `pair_labels.py`, `final_analysis.py`

Reproduce: run `run_trace.sh <tag> <input>` under
`flock /workspace/group/shared/bza/.tmp/gpu.lock`, then
`python final_analysis.py` with the matos-compute Python.

## Figures

![Top kernels](../../../task3_sampling_database/mlip/bench/kernel_profile_2026-10-19/analysis_final/top_labels.png)

![Stage spans vs GPU busy](../../../task3_sampling_database/mlip/bench/kernel_profile_2026-10-19/analysis_final/stage_stack.png)

![Polar-specific delta](../../../task3_sampling_database/mlip/bench/kernel_profile_2026-10-19/analysis_final/delta.png)

Full paths:
`/workspace/group/shared/bza/task3_sampling_database/mlip/bench/kernel_profile_2026-10-19/analysis_final/top_labels.png`
`/workspace/group/shared/bza/task3_sampling_database/mlip/bench/kernel_profile_2026-10-19/analysis_final/stage_stack.png`
`/workspace/group/shared/bza/task3_sampling_database/mlip/bench/kernel_profile_2026-10-19/analysis_final/delta.png`
