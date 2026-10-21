# CUDA / HPC Speedup Techniques 2026 for FP64 PolarMACE

**Date:** 2026-10-20
**Scope:** read-only literature and repository scan. No GPU jobs, no builds, no engine source edits, no git commit/push.
**Current best:** 253.84 ms/eval, 1,296 H2O atoms, 8.55 GiB (S1.1 node-contiguous 16k-edge tiled streaming + A1 UVU tile). 128 NH3: 30.33 ms. 5,184 atoms: 986 ms at 19.21 GiB with `MATOS_MH1_CHANNEL_VEC=1` (default environment is 1,549 ms because of the A1 32x16 tile / reverse vec=4 interaction).
**Prior survey:** [CUDA_STREAMING_OPTIMIZATION_SURVEY.md](CUDA_STREAMING_OPTIMIZATION_SURVEY.md), 474 lines. Its T1-T19 are not repeated unless new 2026 quantitative evidence is added. The two places where that happens are explicitly marked **“new evidence”** below.
**Invariants:** FP64, exact full-k Ewald and the same GTO tables, 2 SCF steps, Fukui/density update order, mask semantics, polar guard. No FP16/BF16/TF32/FP8, no approximate Ewald, no new atomics in deterministic reductions, no change to the physics.

---

## 1. Chinese executive summary (6 lines)

1. **基线：1,296 H₂O 为 253.84 ms / 8.55 GiB，128 NH₃ 为 30.33 ms，5,184 最佳 986 ms / 19.21 GiB；Ewald 仅占 GPU busy 的 1.67%，所以本轮不动 Ewald。**
2. **最大新杠杆：polar UVU 的 14 个 `[512 x N x 512]` 路径 GEMM 可重排成 grouped FP64 GEMM（W 在 N 方向只读一次，DMMA 是精确 IEEE FP64），而当前这组在 pre-A1 CUPTI 中是 105.83 ms，只有约 0.1 TFLOP/s。**
3. **第二杠杆：去掉 2 步 SCF/Fukui 的 24 次 path-table D2H 拷贝、host 串行 Fukui 归约和 1x1x1 小核，改成设备端确定性归约 + cooperative/graph 编排；这不改变 2 步定点。**
4. **第三杠杆：4x A100 上用 NVSHMEM/NCCL Device API 的单 kernel `pack/put/signal/unpack` halo 交换；A100 用 LSA 与 GIN proxy，不用 multimem/NVLS。**
5. **第四/第五：FlashTP 的 CG 稀疏 + path 聚合（CG 矩阵 71-86% 零系数）和 e3j 的 destination-tiled、>80% HBM 融合卷积；两者只动 backbone UVU，不碰 SCF/Ewald。**
6. **4 周目标（投影，非承诺）：1,296 ≤205 ms、128 ≤27 ms、5,184 ≤850 ms，4x A100 5,184 ≥2.5x；全程 dE≤1e-9 eV、dF≤1e-8 eV/Å，不引入低精度。**

### Top 5 concrete techniques (not in the prior survey)

| # | Technique | One-line reason | Projection at 1,296 |
|---|---|---|---|
| 1 | **Polar UVU as grouped FP64 GEMM + fused Hadamard/mask epilogue** | W is shared by all nodes; the current per-node UVU is a latency-bound batched GEMV at ~0.1 TFLOP/s | **30-60 ms saved** |
| 2 | **Device-resident 2-step SCF/Fukui orchestration** (no path-table D2H, device scalar reductions, cooperative/graph replay) | Removes 24 syncs/eval, host-serial Fukui, and one-block launches; keeps SCF state on device | **5-15 ms saved** |
| 3 | **GPU-initiated NVSHMEM/NCCL-LSA fused halo exchange for 4x A100** | Halo dominates multi-GPU communication; exact Ewald only needs a ~11 KB rho(k) allreduce | **1.5-2.2x at 1,296; 2.5-3.2x at 5,184** |
| 4 | **FlashTP-style CG-sparse, path-aggregated UVU** | Skip exact zero CG coefficients and reuse inputs across paths in MH-1/UVU | **5-15 ms saved** |
| 5 | **e3j-style HBM-saturating destination-tiled fused conv** (new 2026 evidence; extends T1/T5) | >80% HBM and >2x cuE conv forward show the edge path is still memory-schedulable | **8-15 ms saved** |

> N5 is a new-evidence extension of the prior survey's T1/T5, included under the brief's explicit exception for new quantitative evidence. It is not counted as a new algorithmic family; N1-N4 are new relative to T1-T19.

---

## 2. Technique table

Legend: **S** = result quoted from the cited source; **P** = projection for this engine, not measured. Effort is engineering days for one experienced Kokkos/CUDA engineer, including parity work, not queue time.

| ID | Technique | Source (paper/repo URL + exact file/kernel) | Measured source result | Applicability to 253.84 ms / 1,296 and 5,184 | Expected gain (P unless marked S) | Risk / effort |
|---|---|---|---|---|---|---|
| N1 | **Polar UVU -> grouped FP64 GEMM + epilogue** | Internal: `external-repos/symmetrix-xl-s1/libsymmetrix/source/mace_polar_impl.tpp:918-1182` (`launch_uvu`), `:2249-2680` (`uvu_reverse`); math in `docs/design/POLAR_PAPER_OPTIMIZATION_PLAN.md` §3.1. External: NVIDIA GTC 2020 S21819 (A100 19.5 TFLOPS FP64 TC vs 9.7 DFMA), CUTLASS `cutlass_80_tensorop_d884gemm_64x32_16x4_nn_align1`, `mma.sync.aligned.m8n8k4.f64.f64.f64.f64` | A100 DMMA peak **19.5 TFLOPS**, cuBLAS large DGEMM ~18-19 TFLOPS; small batched DGEMM ~2.4 TFLOPS (secondary, unverified). Current polar UVU **105.83 ms/eval** in the pre-A1 CUPTI trace (`kernel_labels_1296.json`), ~9.5 GFLOP -> ~90 GFLOP/s, ~0.5-1% of DMMA peak | **Direct.** 4 UVU tables x 2 paths x d1 in {1,3} -> 14 GEMMs `M=512, N=1296/5184, K=512` (mode0 d1=3 needs 3; mode1 d1=3 needs 1); W 16 MiB shared; forward + reverse; one reusable `Z[512,N]` workspace (21 MB at 5,184); no edge tensors. Works at 128 by batching the 14 groups | **P: 30-60 ms at 1,296 (12-24%); 150-300 ms at 5,184; 3-8 ms at 128.** Roofline: 9.5 GFLOP / 2.4 TFLOPS = 4.0 ms at 1,296 even at small-batch DGEMM efficiency | Medium / 7-10 d. FP64 summation order changes; cuBLAS determinism; epilogue fusion; verify path table is dense 512x512 |
| N2 | **Device-resident SCF/Fukui orchestration** | Internal: `mace_polar_impl.tpp:250` (`run_polar_pipeline`), `:882-889` path-table D2H, `:2201-2204` Fukui D2H, `:937-990` one-block project; `POLAR_PAPER_OPTIMIZATION_PLAN.md` §3.4. External: PERKS arXiv:2204.02064; CPU-free multi-GPU arXiv:2610.15878; GPUOS arXiv:2604.17861; CUDA Cooperative Groups | PERKS: **2.12x** 2D stencil, **1.24x** 3D, **4.86x/1.43x** CG/SpMV geomean; CPU-free: **1.63x** CG on 8x A100, **58.8%** lower 3D-stencil comm latency; GPUOS: up to **15.3x** on small-op pipelines (per-op 8 -> 3.1 us in secondary extract); `grid.sync()` ~1-2 us | **Direct.** Polar pipeline has 238 launches and 39 one-block launches = 47.8 ms of 183.5 ms at 128; 24 path-table D2H syncs/eval; host gap 8.65 ms at 1,296. 2 SCF steps, global sums over N. Keep node-local tiling; global fixed point remains exact | **P: 5-15 ms at 1,296 (2-6%); 3-6 ms at 128 (10-20%); 20-40 ms at 5,184.** Best case is host-sync and one-block removal, not a 2x kernel | High / 5-10 d. Cooperative launch must fit co-resident grid; determinism; Kokkos/stream interop; conflicts with cuBLAS calls in the same loop |
| N3 | **GPU-initiated NVSHMEM/NCCL-LSA fused halo exchange** | GROMACS NVSHMEM, SC25 PAW-ATM, DOI 10.1145/3731599.3767508; NVIDIA blog “A Practical Guide to GPU-Initiated Communication for MD at Scale”; NCCL Device API docs; “Every Microsecond Matters” arXiv:2607.16100; DistMLIP arXiv:2506.02023 | GROMACS: **1.5x** intra-node vs GPU-aware MPI, **2x** multi-node NVLink, **1.3x** InfiniBand, **46%** (1,649 vs 1,126 ns/day) at 45k atoms / 4 GPUs; 12 blocking waits per step -> one fused kernel per step (all pulses via `blockIdx.y`). NCCL small-message AllReduce **11.0 -> 2.37 us** on 4x GB200 (8.7% lower ITL). DistMLIP **8x faster / 3.4x larger** than prior multi-GPU methods on 8 GPUs | **Direct for 4x A100.** Halo ratio estimate: 324 atoms/GPU at 1,296 -> ~2.8x ghost/owned; 1,296 atoms/GPU at 5,184 -> ~1.5x. Exact direct-k Ewald needs only a ~11 KB `rho(k)` allreduce at 1,296 (~44 KB at 5,184, because Nk grows), so communication is halo-dominated. One fused kernel for all pulses; pack/put/signal/unpack fused; dependency partition | **P: 1.5-2.2x at 1,296; 2.5-3.2x at 5,184; 4-GPU capacity ~4x memory.** A100 uses LSA and GIN proxy only; do not expect GB200 2.37 us | High / 15-20 d. A100 has no multimem/NVLS/SHARP; GIN `get` unsupported on Ampere; MPI parity; small-system halo overhead |
| N4 | **CG-sparse, path-aggregated UVU (FlashTP)** | PMLR 267:33143 (ICML 2025) https://proceedings.mlr.press/v267/lee25l.html; code https://github.com/SNU-ARC/flashTP; API `flashTP_e3nn.uvu_TP` | Kernel up to **41.6x vs e3nn**, **60.8x vs cuEquivariance**; SevenNet-l3i5 **4.2x inference**, **3.5x training**, **6.3x** lower inference memory; CG matrices **71-86% sparse**; 8-bit unique-CG index; input reuse across paths | **Partial.** MH-1 forward **17.77 ms**, reverse **76.73 ms** pre-A1; path/instruction loop. This checkpoint has only 2 paths per UVU, so the gain depends on the CG instruction sparsity, not on the 2-path table. Exact zero skipping only; no approximation | **P: 5-15 ms at 1,296 (2-6%); 20-50 ms at 5,184; 1-3 ms at 128.** Audit must show >=15-20% instruction reduction or stop | Medium / 5-10 d. FlashTP uses atomic scatter; ours needs deterministic fixup. Limited sparsity in this checkpoint is the main kill risk |
| N5 | **HBM-saturating destination-tiled fused conv (e3j) — new evidence** | arXiv:2609.35099 (e3j); Pallas-GPU message-passing convolution kernel; CUDA TP kernel | **>80% of H100 HBM** on TP; message-passing conv **>2x forward** vs prior backends; MACE water-box NPT up to **34% faster** than cuEquivariance. Secondary extract: 1.6-1.9 TB/s vs 0.75-0.83 TB/s (unverified) | **Direct for A1 UVU tile / backbone edge path.** CTA owns a destination tile; stage node rows + phi/harmonics in shared; `cp.async` double buffer; channel-last vectorized `double4`; one write per node. This is a dataflow upgrade of T1/T5, not a new algorithm family | **P: 8-15 ms at 1,296 (3-6%); 30-60 ms at 5,184; 1-2 ms at 128.** If the edge path is already >70% HBM, gain collapses | Medium / 5-10 d. H100 result, not A100; no ncu counters in this container -> use CUPTI duration + byte model |
| N6 | Packed FP64 GEMM/DMMA for e3 linear small-batch | Internal `KERNEL_PROFILE_2026-10-19.md`; `e3nn_kokkos.cpp:383-391`, `:420-437`, `:1214-1231`, `:541-597`, `:964-999`, `:1573-1608`; GTC 2020 S21819 | At 128 atoms: `e3 linear small batch` 33.78 ms + `e3 reverse linear small batch` 32.21 ms = **66.68 ms of 183.5 ms (36%)**, ~3% of FP64 peak; packed/cuBLAS DMMA path exists but is unused because `use_scalar_backend` returns true under CUDA | **Partial.** Strong at 128; at 1,296 the scalar linears are part of the 35.83 ms cuBLAS/external group. Route through existing packed GEMM; no new math | **P: 30-45 ms in the 183.5 ms profile; the current 30.33 ms already includes other fixes, so re-profile before claiming 128-atom gain. At 1,296 P: 5-15 ms** | Medium / 3-5 d. FP64 summation order; cuBLAS determinism; parity |
| N7 | Green Contexts SM partitioning | NVIDIA blog “Control How Your GPU Shares Work with Green Contexts”; Berkeley EECS-2026-206; arXiv:2501.16909; arXiv:2601.07600 (A100) | H200: victim kernel **1.00x** with GC vs **4.2x** degradation shared; latency variance 0.8 us vs 8,068 us. Blackwell: critical kernel **0.007 ms** with 8-SM GC vs **0.140 ms** high-priority stream. **No L2/HBM isolation**; up to **2.15x** L2-contention slowdown | **Low-medium.** Reserve 8-16 SMs for Ewald/global reduction/NCCL progress while UVU uses the rest. Ewald is only 1.67% of GPU busy, so the ceiling is small. Memory-bound kernels still share HBM | **P: 1-3% at 1,296; more at 128 if launch/sync jitter dominates** | Low / 2-3 d. Driver API CUDA 12.4+; A100 tested in the edge study [30]; static partition, no L2 isolation |
| N8 | Per-instruction L2 cache hints + `cuda::annotated_ptr` | CUDA C++ guide L2 cache control; `createpolicy.fractional.L2::evict_last`, `ld.global.L2::cache_hint`, `cuda::access_property::persisting` | No direct end-to-end number in the surveyed sources; the H100 GC study shows **2.15x** sensitivity to L2 capacity contention. Prior survey T13 only used a coarse access-policy window | **Medium.** Pin node features, UVU weights (16 MiB), path plan, and cuBLAS workspace in L2; mark edge tiles `evict_first`. At 5,184 the node working set is 81 MB (>40 MB L2), so tile it | **P: 1-3% at 1,296; 2-5% at 5,184** | Low / 1-2 d. Per-architecture hint support; A100 L2 is 40 MB; do not pin >L2 |
| N9 | `cudaMallocAsync` pool + graph memory nodes | CUDA C++ guide “Stream-Ordered Memory Allocator”; `cudaMemPoolSetAttribute`, `cudaGraphAddMemAllocNode` | No reliable public end-to-end number for this workload; the S1.1 profile reports 304 memsets/memcpys = 2.18 ms at 128 and host gap 8.65 ms at 1,296 | **Medium-low.** Pre-allocate the polar GEMM Z workspace and edge arena once; avoid `cudaFree` synchronization during graph capture. Do not use managed memory | **P: 0.5-2% at 1,296; 1-3 ms at 128** | Low / 1 d. Graph capture and Kokkos allocator interaction |
| N10 | CUDA graph batching, with new 2026 caution | Ekelund et al., PDP 2025, arXiv:2501.09398; PyGraph arXiv:2503.19779; prior survey T12 | Ekelund: **>1.4x** end-to-end, up to **1.5x** small workloads, crossover after ~2-3 iteration batches. PyGraph: **more than one-third of 400+ graphs hurt** because of creation/parameter/replay overhead | **Medium.** Good for the 128-atom 755-launch case and for the 5,184 multi-tile case; at 1,296 launch overhead is only ~1-3 ms. Capture after N1/N2 stabilize shapes and pointers | **P: 1-3% at 1,296; 3-6% at 128; 2-4% at 5,184** | Low-medium / 1-2 d. Do not graph-capture allocations or MPI; check graph replay overhead |
| N11 | NCCL copy-engine collectives (supporting N3) | NCCL 2.28 release notes / NVIDIA blog “Fusing communication and compute with new Device API and copy-engine collectives” | Offloads small/medium collectives to DMA/copy engines so SMs stay on compute; no independent A100 number in the surveyed sources | **Medium for 4x A100.** The `rho(k)` allreduce is ~11 KB and latency-bound; a CE collective can overlap it with the next node stage. Halo exchange still needs SM-side pack/unpack | **P: 0.5-2 ms at 1,296; 5-15 ms at 5,184** | Medium / 3-5 d. CE collective support and topology; measured on H100/Blackwell class |
| N12 | Exact direct-k Ewald on 4 GPUs (no FFT) | Engine `polar_cosines/sines [Nk,N]`, `rho(k)` global sum; Ewald 9.56 ms pre-A1; user-reported 1.67% of current GPU busy | No distributed FFT is needed: each rank builds local `rho(k)`, allreduce ~11 KB at 1,296 (~44 KB at 5,184), then node-local project. P3M/HeFFTe is therefore **not** required and would change the algorithm | **High architectural fit.** Keep exact full-k; the multi-GPU Ewald cost is a tiny allreduce, not an FFT | **P: <0.5 ms at 1,296; <2 ms at 5,184** | Low / 1-2 d. Deterministic sum order across ranks; do not replace with PPPM/PSWF |

---

## 3. Deep dives on the top five

### 3.1 N1: Rewrite the polar UVU as grouped FP64 GEMMs

#### 3.1.1 Why the current kernel is slow

The polar UVU is not an edge contraction. For each node and each path it computes (see `POLAR_PAPER_OPTIMIZATION_PLAN.md` §3.1):

```
mode0, d1:
  pair(u,v) = sum_c x1[node, in1_off + u*d1 + c] * x2[node, in2_off + v*d1 + c]
  mixed(u)  = sum_v pair(u,v) * W[u,v]
  out[node, u] += pw * inv_sqrt_d1 * mixed(u) * mask[u]

mode1, d1:
  mixed(u)  = sum_v x2[node, in2_off + v] * W[u,v]
  out[node, u*d1+c] += pw * inv_sqrt_d1 * mixed(u) * x1[node, in1_off + u*d1 + c] * mask[...]
```

The checkpoint has 4 UVU tables (`dot_products`, `tp_out`, LE `q`, LE `v`), 2 paths each, `d1 in {1,3}`, and `W` is a shared 512x512 FP64 matrix per path (16 MiB total). The current implementation launches a team per `(node, out_tile)` and loops `v` and `c` per thread. In the pre-A1 kernel W is re-read from L2 for every node; the A1 tile already stages W for a small node tile, but the per-thread dot products remain low-ILP. At 1,296 the polar UVU group is 105.83 ms in the pre-A1 CUPTI trace, while the useful work is only ~9.5 GFLOP (14 path-GEMMs of 512x1296x512). That is ~90 GFLOP/s, roughly 0.5-1% of the A100's 19.5 TFLOPS FP64 tensor-core peak.

#### 3.1.2 The GEMM reformulation

For every path `p`, define `X2_p` as the node-major `[N, K]` block (`K = 512`; for `d1=3`, one GEMM per `c`), and `W_p` as row-major `[M=512, K=512]`. Compute:

```
Z_p = W_p @ X2_p^T                  // [M, N], computed by cuBLAS DGEMM
```

Then the epilogue is elementwise:

```
mode0, d1=1: out[n, out_off+u]      = (sum over paths of pw_p * inv_sqrt_d1 * Z_p[u,n] * X1_p[n,u]) * mask[u]
mode0, d1=3: out[n, out_off+u]      = (sum over paths of pw_p * inv_sqrt_d1 * sum_c Z_p^(c)[u,n] * X1_p[n,u*d1+c]) * mask[u]
mode1, d1=3: out[n, out_off+u*d1+c] = (sum over paths of pw_p * inv_sqrt_d1 * Z_p^(c)[u,n] * X1_p[n,u*d1+c]) * mask[...]
```

This is exact: it is the same bilinear form, just with the `v` reduction done by a GEMM tile instead of a per-thread loop. The only numerical change is the FP64 summation order over `v` (and, for `d1=3` mode0, the `c` reduction is applied to the three GEMM results in the epilogue instead of inside the `v` loop); the gates allow `dE<=1e-9`, `dF<=1e-8`, and the current parity margin is ~5.9e-13.

Why the shape is good:

| Quantity | 1,296 | 5,184 |
|---|---:|---:|
| Nodes N | 1,296 | 5,184 |
| Path-GEMMs | 14 | 14 |
| Each GEMM M x N x K | 512 x 1,296 x 512 | 512 x 5,184 x 512 |
| Total MACs (forward) | 4.76 G | 19.0 G |
| Total FLOPs (forward) | 9.5 GFLOP | 38.1 GFLOP |
| At 2.4 TFLOPS (small-batch DGEMM) | 4.0 ms | 15.9 ms |
| At 10 TFLOPS | 1.0 ms | 3.8 ms |
| Current pre-A1 UVU (forward+reverse) | 105.8 ms | ~423 ms (scaled) |

Even at the pessimistic 2.4 TFLOPS small-batch number, the forward GEMMs are ~4.0 ms at 1,296 instead of tens of ms. The reverse is the same set of GEMMs transposed (`dX2 = W^T @ dZ`; `dX1` is elementwise), so the total is ~8 ms at 2.4 TFLOPS, ~1.9 ms at 10 TFLOPS. This is the single largest headroom in the engine.

#### 3.1.3 Kernel structure and memory layout

- **W layout.** Keep each `W_p` row-major `[M=512, K=512]`. cuBLAS is column-major, so pass it as column-major `[K, M]` with `op(A)=CUBLAS_OP_T`, `lda=K`.
- **X2 layout.** The node-major `[N, K]` block is exactly column-major `[K, N]` with `ldb=K`; use `op(B)=CUBLAS_OP_N`. No transpose copy.
- **Z layout.** `Z` is column-major `[M, N]`, which is node-major `[N, M]` with `ldc=M`; the epilogue reads `Z[u*N + n]` with a warp over `n`, fully coalesced.
- **Epilogue.** One CTA per `(u_tile, n_tile)`, e.g. `u_tile=32`, `n_tile=128`; each warp owns 32 consecutive `u` for one node. Read `Z`, `X1`, `mask`; write `out` once. Fuse path accumulation in registers, then one mask multiply. No zero kernel, no per-path RMW.
- **Batching.** At 128 atoms a single path is only ~4 tiles of 128x128; batch all 14 path-GEMMs with `cublasGemmBatchedEx` (pointer arrays) or `cublasGemmGroupedBatchedEx` (if available in the target CUDA; otherwise a short host loop) so the GPU sees ~64 tiles, or use 64x64 tiles/split-K to reach ~256. At 1,296/5,184, 14 separate `cublasDgemm` calls are also fine (14 launches ~0.1 ms), but grouping keeps the graph small.
- **Workspace.** One `Z[512, N]` buffer reused across paths: 5.3 MB at 1,296, 21.2 MB at 5,184. This replaces the current reverse UVU partials (`[4,N,mul1*d1]`, 243 MiB at 5,184) and `mixed_bar`, so peak memory should not increase.
- **Reverse reuse.** The reverse needs `Z` (for `dX1`) and `X1` (for `dZ`). Either store the 14 `Z` tiles when the existing replay cache allows (74 MB at 1,296, 297 MB at 5,184) or recompute them with the same 14 GEMMs. Budget one extra `Z` workspace if double-buffering.
- **Determinism.** Use `cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED)` and a fixed algorithm (no split-K) for repeat-eval stability. cuBLAS DGEMM is deterministic for fixed shapes and architecture, but verify with 5x `run 0`.
- **Precision.** DMMA is IEEE-compliant FP64 FMA with FP64 accumulate; it is not TF32/FP16. Do not enable TF32 or reduced-precision math modes.

C++/cuBLAS sketch:

```cpp
// W row-major [M,K], X2 node-major [N,K], Z column-major [M,N]
cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K,
            &alpha, W_p, K,      // lda = K, column-major [K,M]
            X2_p, K,             // ldb = K, column-major [K,N]
            &beta, Z_p, M);      // ldc = M, column-major [M,N]
// epilogue: out[n, out_off+u] += pw*inv_sqrt_d1*Z_p[u,n]*X1_p[n,u]*mask[u]
```

Kokkos/CUDA port:

- Get raw pointers with `view.data()`; the layouts must be checked. If a View is `LayoutLeft`, either pass the appropriate `op`/`ld` or create a `LayoutRight` mirror once per evaluation (the mirror cost is small relative to the GEMM).
- Use the Kokkos CUDA execution-space stream for cuBLAS (do not create a second stream). If your Kokkos version does not expose it, use a single application-owned stream and make Kokkos use it; otherwise event-synchronize once per stage.
- Keep the existing per-node UVU kernel as `MATOS_POLAR_UVU_GEMM=0` fallback until the new path passes all gates.
- Build the 14 `(W, X1, X2, mask, out_off, pw, d1)` descriptors once from `parse_uvu_paths` and cache them on device. This also removes the 8 path-table D2H copies per step (N2).

#### 3.1.4 Risks and acceptance

- **Numerical:** FP64 reassociation over `v`. Gate at 128/1,296: `dE<=1e-9`, `max dF<=1e-8`; repeat-eval `<=1e-12` at 128 and no first-eval ULP surprise at 1,296.
- **Path table:** Verify that each `W` is dense 512x512. If 70-86% of coefficients are zero (as in FlashTP's CG matrices), a dense GEMM still wins because it replaces the slow per-node loop, but the ideal gain is smaller. Consider N4 afterwards.
- **Weights:** Inference does not need `dW`; do not compute it.
- **Epilogue:** The mask and the `x1` Hadamard factor must be applied once to the final sum, matching the current `(a+b)*mask` order. Keep path order ascending in the accumulator.

### 3.2 N2: Device-resident 2-step SCF/Fukui orchestration

#### 3.2.1 What is host-synchronizing today

From the polar plan and CUPTI profile:

- 24 `create_mirror_view_and_copy` + `std::vector` path-table copies per eval (`mace_polar_impl.tpp:882-889`, `:1809-1825`, `:3183-3188`, `:3953-4059`).
- `fukui_reverse` does two `[N,8]` D2H copies (`:2201-2204`) and a serial host loop over owned nodes.
- The polar pipeline has 238 launches; 39 launches are `1x1x1` grids (128 threads), totaling 47.8 ms of GPU time but only 0.042 ms of SM-weighted time at 128 atoms.
- Host gap at 1,296 is 8.65 ms in the pre-A1 trace.

The global SCF/Fukui fixed point cannot be edge-streamed; node tiles are legal only within one SCF step (global sums between steps). The point of N2 is not to change the fixed point, but to keep the two steps and their reductions on device.

#### 3.2.2 Design

Two variants, in increasing risk:

**Variant A (low risk, recommended first): CUDA graph + device reductions.**
- Cache the `PolarUvuPlan` and path tables on device at model load; never copy them per step.
- Replace the host-serial Fukui reduction with a device `parallel_reduce` into a 2-element `fbar`, then a tiny kernel that applies the correction. For MPI, allreduce only the 2-element `fbar` (and the `rho(k)` vector, which is ~11 KB).
- Parallelize the `1x1x1` kernels with `MDRangePolicy<Rank<2>>` or `TeamPolicy` (the plan already identifies `polar scf project features`, `polar rev replay project`, `polar reverse dense`, `reverse nonlinear normalization`).
- Capture the whole 2-step sequence (GEMMs, epilogues, reductions, Ewald, LE) in a CUDA graph; update only pointers/positions between evals via `cudaGraphExecKernelNodeSetParams`. No allocations inside capture.
- This alone removes most host gaps and all path-table syncs. It is an extension of T12, but the new part is the device-side scalar reduction and the removal of the 24 D2H copies.

**Variant B (high risk): cooperative persistent SCF kernel.**
- Launch `cudaLaunchCooperativeKernel` with a grid that fits co-residently: `blocks = SMs * blocks_per_sm` from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, 256-512 threads.
- Phases: `fukui -> kgeom -> potential -> SCF step 0 -> SCF step 1 -> coulomb -> LE -> readout`. Between global phases, `cooperative_groups::this_grid().sync()`; between node-local phases, no global barrier.
- Work distribution: a static `(node_tile, out_tile)` map or an atomic work queue; tiles are node-contiguous so each node has one writer.
- Global reduction: each CTA writes a partial `[CTA, 8]` (or `[CTA, Nk]` for `rho(k)`) to global scratch, `grid.sync()`, CTA 0 reduces in fixed CTA order, writes the scalar/vector, `grid.sync()`, all CTAs read. Fixed order preserves determinism; the association differs from the current tree, so gate it.
- Keep `W` and the path plan in L2; keep `Z` tiles and node features in shared memory. Use double buffering for `Z`.
- If cuBLAS GEMMs must stay outside, capture them in a graph and use Variant A; a fully fused cooperative kernel requires custom DMMA/CUTLASS in-kernel GEMMs, which is a larger project.

Launch config and synchronization:

```cpp
namespace cg = cooperative_groups;
__global__ void polar_scf_kernel(/* ... */) {
  cg::grid_group grid = cg::this_grid();
  // phase 1: fukui + kgeom, one CTA per node tile
  fukui_kgeom(...);
  grid.sync();
  for (int step = 0; step < 2; ++step) {
    // local rho(k) partials -> scratch[cta]
    rho_partials(...);
    grid.sync();
    if (blockIdx.x == 0) deterministic_reduce_and_broadcast(...);
    grid.sync();
    // project, dot, tp, readout, density update
    scf_step(...);
    grid.sync();
  }
  coulomb_le_readout(...);
}
```

`grid.sync()` on A100 is ~1-2 us for a co-resident grid; the win comes from removing host launches and keeping intermediates on chip, not from the barrier itself.

#### 3.2.3 Expected effect and risk

- **P: 5-15 ms at 1,296 (2-6%)**, **3-6 ms at 128 (10-20%)**, **20-40 ms at 5,184**.
- Risk: cooperative launch occupancy (high register/shared use shrinks the resident grid); determinism of the new reduction tree; Kokkos stream/capture interaction; MPI allreduce must stay outside the graph.
- Kill criterion: if the cooperative kernel cannot fit at least 2 CTAs/SM with the SCF state, use Variant A only.

### 3.3 N3: GPU-initiated communication for 4x A100

#### 3.3.1 Why halo, not Ewald

The engine's Ewald is a direct full-k sum: `cos/sin [2,Nk,N]`, local `rho(k) = sum_n cos/sin * density`, then a node-local project. There is no distributed FFT. On 4 ranks the only global Ewald communication is a `rho(k)` allreduce (~11 KB at 1,296, ~44 KB at 5,184), which is latency-bound but negligible (~5-20 us). The multi-GPU bottleneck is the LAMMPS halo/ghost exchange and the per-layer node-feature exchange, not Ewald.

Halo arithmetic (uniform water, 6 A cutoff, cube domains; geometric estimate):

| Atoms/GPU | Owned atoms | Cube side | Halo atoms | Halo/owned |
|---:|---:|---:|---:|---:|
| 324 (1,296 on 4 GPUs) | 324 | 21.3 A | ~900 | ~2.8x |
| 1,296 (5,184 on 4 GPUs) | 1,296 | 33.8 A | ~1,900 | ~1.5x |

So 4-GPU at 1,296 atoms is halo-dominated; do not expect 4x. At 5,184 the halo is still large but manageable. A graph-level zero-redundancy partition (DistMLIP) would avoid ghost replication but is a multi-month rewrite of the LAMMPS pair style; keep it as a longer-term alternative, not a 4-week item.

Reference scaling: SevenNet's spatial decomposition reports **>80% weak-scaling efficiency on 32 GPUs**, and MatterScale reports **94% at 32,768 GPUs** with ~34.5k atoms/GPU [14][15]. Both have far more atoms per GPU than our 4x1,296 case (324 atoms/GPU), which is why the 1,296 multi-GPU projection is only 1.5-2.2x; 5,184 (1,296 atoms/GPU) is closer to their regime but still small.

#### 3.3.2 Design: one fused kernel per step

GROMACS 2026 replaced 12 blocking MPI waits per step (six coordinate + six force pulses) with GPU-initiated NVSHMEM kernels, and fuses all pulses into one kernel:

- Pack owned boundary coordinates/features into a symmetric buffer.
- `nvshmem_putmem_signal_nbi` (or the `nvshmemx_` variant; direct NVLink stores via `nvshmem_ptr` when available) + system-scope release signal.
- Receiver waits on the signal (`nvshmem_signal_wait_until`) and unpacks into ghost rows.
- Process pulses in `Z -> Y -> X` for coordinates and reverse for forces; independent pulses are packed/sent immediately; dependent atoms wait only on the specific earlier-pulse signals (`dependencyAtomOffset`).
- Elect the last block per pulse with a release atomic counter; only the elected block issues the system-scope signal.
- Fuse all pulses into one kernel with `blockIdx.y` as the pulse id; one launch per step instead of 6-12.

For this engine, the same pattern applies to the LAMMPS ghost exchange: coordinates and node features (width 2048/512) are rank-local; the halo is a few thousand rows. Use NVSHMEM 3.x on A100:

- `nvshmem_malloc` symmetric heap for send/recv buffers and signals.
- A100 supports LSA load/store over NVLink/PCIe and GIN `put/signal` via the proxy backend; **do not** request multimem/NVLS (SM 9.0+) and **do not** use `gin.get` on Ampere (unsupported).
- Fallback: GPU-aware MPI with `cudaStreamWaitEvent` / `MPI_Isend`/`MPI_Irecv` (or `MPI_Ialltoall`) on a dedicated stream; measure both. NVIDIA's GROMACS result is 1.5x intra-node, so even the fallback should be benchmarked.

NCCL Device API alternative for the tiny allreduce:

- Register `rho(k)` and `fbar` in symmetric windows (`ncclMemAlloc`, `ncclCommWindowRegister` with `NCCL_WIN_COLL_SYMMETRIC`).
- Use `ncclDevCommCreate`, `ncclPutSignal`, `ncclWaitSignal`, and `ncclLsaBarrierSession`; or a one-shot push AllReduce: each rank writes its partial to all peers' scratch, then reduces locally.
- The 2026 low-latency paper reports **11.0 -> 2.37 us** for small-message AllReduce on 4x GB200, using barrier-free push and LL/sentinel completion. A100 cannot use multimem and was not tested; treat 2.37 us as an upper-bound target, not an A100 promise.
- Do not use the NCCL ring for the 11 KB allreduce; use a one-shot LSA kernel or copy-engine collective.

#### 3.3.3 Expected effect and risk

- **P: 1,296 1.5-2.2x; 5,184 2.5-3.2x.** The 1,296 result is limited by the 2.8x halo/owned ratio, not by NVSHMEM.
- **Capacity:** 4x A100 80 GB at ~6-8 GiB/GPU for 5,184 leaves room for ~20k atoms at the same per-GPU memory, but halo overhead grows. Do not claim linear capacity.
- Risk: high. MPI parity at 2 ranks is already gated (`dE<=5e-5`, `dF<=1e-4` official); 4-rank needs new gates. NVSHMEM adds a dependency and a symmetric heap; the NCCL Device API was still experimental in 2.28 with no ABI guarantee, so pin versions. Do not mix communication with CUDA graph capture.
- Kill criterion: if 4-GPU 5,184 <2.2x, report the negative result and keep 2-rank; do not hide it with a larger baseline.

### 3.4 N4: FlashTP-style CG sparsity and path aggregation

FlashTP (ICML 2025) reports that Clebsch-Gordan coefficient matrices are **71-86% sparse**, stores only nonzeros in a COO-like format, uses **8-bit indices into a small array of unique CG values**, groups paths by shared input so `x1` is read once, and fuses the edge scatter. On SevenNet-l3i5 it reports **4.2x inference**, **3.5x training**, **6.3x** lower memory; kernel-level up to **41.6x vs e3nn** and **60.8x vs cuEquivariance**.

What transfers to PolarMACE:

- **Exact zero skipping.** The MH-1 instruction list and the UVU path table contain masks. Any coefficient that is exactly zero can be skipped without changing the result. Audit the checkpoint first: count zero paths / zero CG blocks for the 2-path UVU tables. If <10% are zero, stop.
- **Path aggregation.** Reorganize the loop from output-major to input-major: load `x1` once, iterate all paths that consume it, and accumulate into per-thread registers. This reduces repeated `x1` reads in the MH-1 forward/reverse. It does not change the math.
- **CG value compression.** Store the unique CG values once and index with 8-bit offsets; this reduces constant/L2 traffic. It is exact.
- **Deterministic scatter.** FlashTP uses `atomicAdd`; our reverse requires a fixed order. Use a per-destination segmented reduction with a deterministic fixup (the OEQ pattern, prior survey T5), not atomics.

Applicability caveat: PolarMACE's polar UVU has only 2 paths per table, so the large FlashTP gains are unlikely to transfer directly. The more plausible target is the backbone MH-1 instruction loop (forward 17.77 ms, reverse 76.73 ms pre-A1), where the CG instruction set is wider. Projection **5-15 ms at 1,296**, **20-50 ms at 5,184**, **1-3 ms at 128**. Effort 5-10 d; kill if the audit shows no meaningful exact sparsity.

### 3.5 N5: e3j HBM-saturating destination-tiled conv (new evidence)

e3j (arXiv:2609.35099, Sept 2026) reports that its Pallas-GPU message-passing convolution reaches **>80% of H100 HBM bandwidth**, more than **2x** the forward throughput of prior backends, and **up to 34% faster** MACE water-box NPT end-to-end than cuEquivariance. A secondary extract reports 1.6-1.9 TB/s vs 0.75-0.83 TB/s for cuEquivariance at Lmax=2 (unverified). The transferable design, not the H100-specific code:

- **Destination-tiled.** One CTA owns a destination node tile (e.g. 8-16 receivers). It stages the receiver's `up`/message rows and the tile's `phi`/harmonics in shared memory. Edges are sorted by destination, so the CTA walks a contiguous edge range and writes each output row once.
- **Channel-last, vectorized.** Layout node rows as `[node][channel]` with `double4` (32 B) loads/stores; a warp covers 32 channels. This is the layout the A1 UVU tile already approximates.
- **Async double buffer.** `cp.async` 16 B per thread prefetches the next edge window's `phi`/harmonics/`up` while the current window is multiplied. Two stages, `__syncthreads()` (or `cuda::barrier`) between stages. A100 supports `cp.async`, not TMA; do not copy the Hopper TMA variant.
- **Persistent CTA work loop.** Launch `4 * SM_count` CTAs x 256 threads; each CTA grid-strides over destination tiles, keeping the weight/type tables and shared-memory allocation alive. This is the same structure as Sobek's direct kernel (prior survey T1/T11).
- **One write per node.** Accumulate the 16/40 output components in registers, then one `st.global` per component. Avoid per-edge global `+=`; that is the remaining weakness of the current receiver-owned kernel.

Expected: **8-15 ms at 1,296 (3-6%)**, **30-60 ms at 5,184**, **1-2 ms at 128**. The gain collapses if the edge path already runs at >70% HBM; measure with CUPTI duration and a byte model, since `ncu` counters are unavailable in this container. Risk is medium (5-10 d), and this is explicitly a **new-evidence upgrade of T1/T5**, not a new algorithm family.

---

## 4. Ranked 4-week experiment plan

Protocol: one A100-80GB per single-GPU run, `flock /workspace/group/shared/bza/.tmp/gpu.lock`; same-binary interleaved A/B, 2 rounds x 4 samples, median of medians; `MATOS_REPLAY_CACHE=1`, `MATOS_CONV_CUTOFF_BACKEND=team`, `MATOS_MH1_TEAM_SIZE=64`, `MATOS_MH1_TEAM_CHANNEL_VEC=1`; NVML at 20 ms; do not relax parity. All items are implementation-team work; this survey runs none of them.

**Assumption:** two engineers for four weeks (20 working days each): **A** owns single-GPU kernels; **B** owns multi-GPU/NVSHMEM. The effort estimates in §2 sum to more than 20 days, so this is a ranked portfolio, not a commitment to finish all five. With one engineer, run ranks 1 and 3 plus the E6b audit; defer rank 2 (multi-GPU) and rank 4 (e3j prototype) to phase 2.

| Rank | Owner | Week / days | Experiment | Change | Acceptance | Kill / fallback |
|---:|---|---|---|---|---|---|
| 1 | A | W1 d1 | **E0 Re-profile 253.84 ms** | CUPTI tracer + `MATOS_POLAR_PROFILE=1`; split polar UVU fwd/rev, MH-1, cuBLAS, host gaps at 1,296 and 128 | Attributes >=80% of 253.84 ms to named kernels; confirms current polar UVU ms | If attribution <50%, add stage marks before optimizing |
| 1 | A | W1 d2-4 | **E1 Polar UVU GEMM microbenchmark** | Extract `W/X1/X2/mask/pw` for the 4 tables at 128/1,296; implement 14 grouped DGEMMs + epilogue in a standalone harness; compare outputs to `launch_uvu` | max abs diff <=1e-10; GEMM+epilogue >=5x faster than current forward UVU; no NaN/Inf | If <2x, stop and audit W density/path table |
| 1 | A | W1 d5-W2 d4 (8 d) | **E2+E3 Forward and reverse polar GEMM** | `MATOS_POLAR_UVU_GEMM=1`; one `Z` workspace; batched/grouped cuBLAS; `dX2 = W^T dZ`; elementwise `dX1`; no `dW`; Kokkos stream | 1,296 <=215 ms; 128 <=29 ms; 5,184 <=900 ms; dE<=1e-9, dF<=1e-8; repeat <=1e-12; peak <=45 GiB | If forward parity fails twice, keep fallback; if reverse fails, ship forward-only and report |
| 2 | B | W1-W4 (18 d) | **E5 4x A100 NVSHMEM/NCCL** | W1: symmetric buffers + LSA microbenchmark. W2: one-kernel halo exchange vs MPI. W3: 2-rank full eval. W4: 4-rank full eval and overlap tuning | 5,184 >=2.5x vs 1 GPU; 1,296 >=1.6x; overlap >=70%; 2/4-rank parity dE<=5e-5, dF<=1e-4; <=45 GiB/GPU | If W2 halo microbenchmark is <1.2x vs MPI, stop and keep 2-rank |
| 3 | A | W3 d11-14 | **E4 Device-resident SCF (Variant A)** | Remove 24 path-table D2H; device `fbar` reduction; parallelize 1x1x1 kernels; CUDA graph capture of the 2-step loop | 1,296 <=205 ms; 128 <=27.5 ms; trace shows 0 D2H in loop; repeat stable | If graph capture conflicts with Kokkos/MPI, do device reductions only; cooperative Variant B is phase 2 |
| 4 | A | W3 d15-16 | **E6a e3j HBM tile prototype** | Destination-tiled, `cp.async` double buffer, `double4`, one write per node; bench on A1 UVU tile | Edge-path kernel >=15% faster; full 1,296 <=200 ms; parity | If HBM model shows >70% already, stop |
| 5 | B/offline | W1 d2 (no GPU) | **E6b FlashTP sparsity audit** | Count exact zero CG coefficients/paths for this checkpoint; estimate MH-1 instruction reduction; do not integrate unless the audit passes | >=15-20% instruction reduction projected; deterministic-fixup design sketched | If <10% zero, stop; reallocate the 2 days to E6a |
| — | A+B | W4 d17 | **E7 Final A/B and gates** | 128/1,296/5,184, 1/2/4-rank; same-binary interleaved; memory, hashes, logs | 1,296 <=205 ms (target <=190); 128 <=27 ms; 5,184 <=850 ms (target <=750); 4-GPU 5,184 <=400 ms; dE/dF; no FP32/TF32/FP16 | If 1,296 >215 ms, publish the negative result and keep the best gated arm |
| — | A+B | W4 d18-20 | Buffer | Fix parity/perf regressions, finalize docs, rerun gates | Same as E7 | Do not start new structural work in the buffer |

**Overall acceptance**

- 1,296: <=205 ms (stretch <=190 ms), peak <=20 GiB, default remains materialized, generic opt-in.
- 128: <=27 ms, repeated-eval drift <=1e-12.
- 5,184: <=850 ms (stretch <=750 ms), peak <=45 GiB, finite energy, 3 repeats stable.
- 4x A100: 5,184 >=2.5x, 1,296 >=1.6x, overlap >=70%, per-GPU <=45 GiB, 2/4-rank parity.
- Parity: 128/1,296 `dE<=1e-9 eV`, `max dF<=1e-8 eV/Å`; 2-rank `dE<=5e-5`, `dF<=1e-4`; repeat-eval `<=1e-12` at 128.
- Logs: `streamed_edges='generic'`, `polar_uvu_gemm=1`, no D2H in the SCF loop, no E-sized allocation >2.6 GiB in receiver-owned mode, no new atomics.

**Recommended order:** A: E0 -> E1 -> E2+E3 -> E4 -> E6a -> E7; B: E5 in parallel, with the W2 halo microbenchmark as the go/no-go. E6b is a 1-day offline audit, not a GPU task. If only one structural change is possible, do E1-E3 (polar GEMM); if only one cheap change is possible, do E4 (device reductions) after E0. If the team has one engineer, defer E5 and E6a to phase 2.

---

## 5. Do not copy

| Do not copy | Why |
|---|---|
| FP16/BF16/TF32/FP8, or FP64-emulation through low-precision tensor cores (Ozaki-style splitting) | The gates are FP64 `dE<=1e-9`, `dF<=1e-8`; even accurate emulation changes rounding and is unverified. DMMA is IEEE FP64 and is allowed; low-precision emulation is not |
| Approximate Ewald, PPPM, PSWF/ESP splitting, k-space truncation, RBE sampling | The engine requires exact full-k Ewald with the same GTO tables and `k0` mask; Ewald is only 1.67% of GPU busy, so there is no reason to touch it. The 2026 PSWF Ewald 2.5-7x result is not applicable |
| Dense N x N neighbor/attention matrices | O(N^2) memory defeats the 5,184/11M-atom capacity. Note: the N1 GEMM is 512 x N x 512, i.e. O(512N), not N x N |
| Root-rank gather / centralized edge processing | Kills one-rank-per-GPU scaling and MPI balance; all edge data must stay rank-local |
| Training-only caches (Sobek receiver aggregates for `grad_B`, stale node caches, historical embeddings) | Inference forces need `phi_adj` and geometric adjoints, not model-weight gradients; the caches add memory without helping the eval |
| Hopper TMA/WGMMA, warp-specialized FA3, thread-block clusters, `setmaxnreg` | A100 is SM80: no TMA, no WGMMA, no clusters. Use `cp.async` and ordinary shared memory |
| Device graph launch and conditional graph nodes in a device-launched graph | Requires compute capability 9.0+; A100 cannot use them. Host-launched CUDA graphs are fine |
| NVLS/multimem/NVLink SHARP collectives | SM 9.0+ (Hopper/Blackwell); A100 cannot. Use LSA load/store and GIN proxy, not multimem |
| NCCL `gin.get` on Ampere | Specifically unsupported on Ampere/directNIC; use `put/signal` or the proxy backend |
| Stream-K for these GEMMs | The polar GEMMs are small but numerous (14 groups), and the MLP GEMMs are large-M; Stream-K targets a different tail regime. Grouped/batched GEMM is the right tool |
| FlashTP's `atomicAdd` scatter | Nondeterministic; our reverse needs a fixed order. Use a deterministic segmented reduction/fixup |
| GROMACS' TMA variant of the NVSHMEM halo kernel | Uses `cp_async_bulk`/TMA, Hopper-only; port the ordinary `cp.async` path |
| "CUDA graphs everywhere" | PyGraph found more than one-third of 400+ graphs hurt due to creation/parameter/replay overhead; graph only after the shapes and pointers stabilize |
| Kahan/compensated summation in the FP64 hot path | Unnecessary at FP64, changes the summation order, and adds instructions. Use fixed-order FP64 accumulation |
| `cudaMallocManaged`/UVMM oversubscription for capacity | Page-fault thrash destroys MD throughput; capacity comes from tiling and multi-GPU, not oversubscription |
| Multi-stream concurrency that reorders reductions | Introduces new nondeterminism; keep one stream for the deterministic reductions, or use events with a fixed order |
| New atomics in the polar/SCF reductions | The current polar reverse already has unique writers; atomics add nondeterminism and are not needed |
| Reordering LAMMPS atom IDs or globally permuting the CSR | Breaks `map_kokkos`, ghost/owned semantics, force dumps, and MPI; only rank-local edge permutations within the existing order are safe |
| Caching full `[E,2048]` weights at 5,184 without a budget | +10.8 GiB for both layers, erodes the capacity claim; use the receiver-owned/streamed path or a budgeted cache |
| Sobek FP32-only codegen or cuEquivariance binary wheels without FP64 validation | Precision/contract mismatch; port patterns, not precision assumptions |
| `mh1_fast_path` / `direct` / pair-spline / FP32 for PolarMACE | Would disable materialized, enable FP32/pair spline, and break the accepted baseline; S1.1 keeps `mh1_fast_path=false` |

---

## 6. Citations

[1] NVIDIA, “Optimizing Applications for NVIDIA Ampere GPU Architecture,” GTC 2020, S21819. A100 FP64: 9.7 TFLOPS DFMA, 19.5 TFLOPS FP64 tensor core; DGEMM benchmark at 4096^3. https://developer.download.nvidia.cn/video/gputechconf/gtc/2020/presentations/s21819-optimizing-applications-for-nvidia-ampere-gpu-architecture.pdf

[2] NVIDIA A100 SXM4 80 GB specification (9.7 TFLOPS FP64, 19.5 TFLOPS FP64 TC, 2,039 GB/s HBM2e). https://portal.mlcloud.uni-tuebingen.de/user-guide/_specs/gpu/a100/

[3] NVIDIA CUTLASS, FP64 tensor-op kernels (`cutlass_80_tensorop_d884gemm_64x32_16x4_nn_align1`), `mma.sync.aligned.m8n8k4.f64.f64.f64.f64`. https://github.com/NVIDIA/cutlass

[4] NVIDIA cuBLAS documentation, `cublasDgemm`, `cublasGemmBatchedEx`, `cublasGemmGroupedBatchedEx`, `cublasSetAtomicsMode`, FP64 compute modes. https://docs.nvidia.com/cuda/cublas/

[5] NVIDIA, CUDA C++ Programming Guide, “Cooperative Groups” and `grid.sync()`. https://docs.nvidia.com/cuda/cuda-programming-guide/

[6] PERKS: “Persistent Kernels for Iterative Memory-bound GPU Applications,” arXiv:2204.02064. 2.12x 2D stencil, 1.24x 3D stencil, 4.86x/1.43x CG/SpMV geomean. https://arxiv.org/abs/2204.02064

[7] I. Ismayilov et al., “Multi-GPU Communication Schemes for Iterative Solvers: When CPUs are Not in Charge,” ICS 2023 / arXiv:2610.15878. 1.63x CG on 8x A100, 58.8% lower 3D-stencil communication latency. https://arxiv.org/abs/2610.15878. Code: https://github.com/ParCoreLab/CPU-Free-model

[8] “GPUOS: A GPU Operating System Primitive for Transparent Operation Fusion,” arXiv:2604.17861. Up to 15.3x on small-op pipelines; device work queue. https://arxiv.org/abs/2604.17861

[9] M. Doijade et al., “Redesigning GROMACS Halo Exchange: Improving Strong Scaling with GPU-initiated NVSHMEM,” SC25 PAW-ATM, DOI 10.1145/3731599.3767508. https://dl.acm.org/doi/10.1145/3731599.3767508

[10] NVIDIA, “A Practical Guide to GPU-Initiated Communication for Molecular Dynamics at Scale,” 2026. GROMACS: up to 1.5x intra-node, 2x multi-node NVLink, 1.3x InfiniBand, 46% at 45k atoms/4 GPUs; 12 blocking waits -> one fused kernel per step. https://developer.nvidia.com/blog/a-practical-guide-to-gpu-initiated-communication-for-molecular-dynamics-at-scale/

[11] NVIDIA NCCL, “Device-Initiated Communication” (LSA, multimem, GIN; `ncclDevCommCreate`, `ncclPutSignal`, `ncclWaitSignal`, `ncclLsaBarrierSession`). https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html

[12] “Every Microsecond Matters: Achieving Near Speed-of-Light Latency in GPU Collectives,” arXiv:2607.16100. Small-message AllReduce 11.0 -> 2.37 us on 4x GB200, 8.7% lower ITL; barrier-free one-shot push + LL/sentinel. https://arxiv.org/abs/2607.16100

[13] DistMLIP: “A Distributed Inference Platform for Machine Learning Interatomic Potentials,” ICLR 2026, arXiv:2506.02023. Zero-redundancy graph-level partitioning, all-to-all boundary features, up to 8x faster and 3.4x larger on 8 GPUs. https://arxiv.org/abs/2506.02023

[14] SevenNet: “Scalable Parallel Algorithm for Graph Neural Network Interatomic Potentials in Molecular Dynamics Simulations,” J. Chem. Theory Comput. 2024, DOI 10.1021/acs.jctc.4c00190. >80% weak-scaling efficiency on 32 GPUs. https://doi.org/10.1021/acs.jctc.4c00190

[15] MatterScale / MatterSim, SC26 presentation abstract: 1.13B atoms on 32,768 GPUs, 94% weak-scaling efficiency. https://sc26.conference-program.com/presentation/?id=pap459&sess=sess236

[16] FlashTP: “Fused, Sparsity-Aware Tensor Product for Machine Learning Interatomic Potentials,” ICML 2025, PMLR 267:33143-33156. 41.6x vs e3nn, 60.8x vs cuEquivariance, 4.2x inference, 6.3x memory; CG matrices 71-86% sparse. https://proceedings.mlr.press/v267/lee25l.html. Code: https://github.com/SNU-ARC/flashTP

[17] e3j: “An Efficient and Open-Source Backend for Euclidean Equivariant Operations on GPU and TPU,” arXiv:2609.35099. >80% H100 HBM on TP, >2x conv forward, up to 34% faster MACE NPT than cuEquivariance. https://arxiv.org/abs/2609.35099

[18] NVIDIA, “Control How Your GPU Shares Work with Green Contexts,” 2026. Critical kernel 0.007 ms with 8-SM GC vs 0.140 ms high-priority stream; no L2/HBM isolation. https://developer.nvidia.com/blog/control-how-your-gpu-shares-work-with-green-contexts/

[19] Berkeley EECS-2026-206, Green Contexts on H200: 1.00x victim with GC vs 4.2x degradation shared; API overhead. http://www2.eecs.berkeley.edu/Pubs/TechRpts/2026/EECS-2026-206.pdf

[20] “Understanding GPU Resource Interference One Level Deeper,” arXiv:2501.16909. Green-context L2 contention up to 2.15x; no HBM isolation. https://arxiv.org/abs/2501.16909

[21] NVIDIA, CUDA C++ Programming Guide, “L2 Cache Control,” `cudaAccessPolicyWindow`, `cuda::annotated_ptr`, `cuda::access_property`. https://docs.nvidia.com/cuda/cuda-programming-guide/

[22] NVIDIA, CUDA C++ Programming Guide, “Stream-Ordered Memory Allocator,” `cudaMallocAsync`, `cudaMemPoolSetAttribute`, graph memory nodes. https://docs.nvidia.com/cuda/cuda-programming-guide/

[23] J. Ekelund, S. Markidis, I. B. Peng, “Boosting Performance of Iterative Applications on GPUs: Kernel Batching with CUDA Graphs,” PDP 2025, arXiv:2501.09398. >1.4x end-to-end, up to 1.5x small workloads. https://arxiv.org/abs/2501.09398

[24] PyGraph: “Robust Compiler Support for CUDA Graphs in PyTorch,” arXiv:2503.19779. More than one-third of 400+ graphs hurt. https://arxiv.org/abs/2503.19779

[25] NVIDIA, CUDA C++ Programming Guide, “CUDA Graphs”; device graph launch and conditional nodes require compute capability 9.0+. https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html

[26] A. Bartolomeu, U. Halver, G. Sutmann, “Scaling and performance aspects of the P3M method on multiple GPUs,” Procedia Computer Science 286 (2026) 14-23, DOI 10.1016/j.procs.2026.08.015. Kokkos+Cabana+HeFFTe, near-ideal scaling for large systems; small systems latency-bound. https://doi.org/10.1016/j.procs.2026.08.015

[27] Ewald PSWF/ESP, Nature Communications 2026, DOI 10.1038/s41467-026-73232-8. Fourier grid up to 8x smaller, 2.5-7x MD speedup — **not applicable** because it changes the Ewald splitting. https://www.nature.com/articles/s41467-026-73232-8

[28] NVIDIA, “Fusing communication and compute with new Device API and copy-engine collectives in NCCL 2.28,” 2025-2026. https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/

[29] NVIDIA NCCL release notes 2.30.4: `gin.get` requires GDAKI and is not supported on Ampere/directNIC; symmetric kernels, Device API timeouts. https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2323/release-notes/rel_2-30-4.html

[30] “Performance Isolation for Inference Processes in Edge GPU Systems,” arXiv:2601.07600. Green Contexts on A100 and Jetson Orin; fine-grained SM allocation, no memory isolation. https://arxiv.org/abs/2601.07600

[31] Internal references:
- `docs/results/S1_1_A1_INTEGRATION.md` — 253.84 ms / 8.55 GiB at 1,296; 5,184 `MATOS_MH1_CHANNEL_VEC=1` 986 ms / 19.21 GiB.
- `docs/results/OPTIMIZATION_LEDGER.md` — 30.33 ms at 128; optimization ladder.
- `docs/results/data/kernel_labels_1296.json` — polar UVU 105.83 ms, MH-1 reverse 76.73 ms, MH-1 forward 17.77 ms, cuBLAS 35.83 ms, Ewald 9.56 ms, host gap 8.65 ms (all pre-A1).
- `docs/design/KERNEL_PROFILE_2026-10-19.md` — 755 launches, 52% SM-weighted utilization at 128; 39 one-block launches = 47.8 ms; `e3 linear` 66.68 ms.
- `docs/design/POLAR_PAPER_OPTIMIZATION_PLAN.md` — polar dataflow, §3.1 UVU math, §3.4 24 D2H copies, §4 node tiling.
- `external-repos/symmetrix-xl-s1/libsymmetrix/source/mace_polar_impl.tpp:918-1182,2249-2680` — `launch_uvu` and `uvu_reverse`.
- `external-repos/symmetrix-xl-s1/libsymmetrix/source/e3nn_kokkos.cpp:383-437,1214-1231,1573-1608` — scalar e3 linear and packed path.
- `external-repos/symmetrix-xl-s1/libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp` — generic streamed forward/reverse.

---

## 7. Unverified claims / caveats

- **[U1]** All **P** numbers are projections from the engine's memory/flop ledger and cited kernel ratios, not measurements. The only measured engine numbers are 253.84 ms / 8.55 GiB (1,296), 30.33 ms (128), 986 ms / 19.21 GiB (5,184 with `MATOS_MH1_CHANNEL_VEC=1`), and the pre-A1 CUPTI kernel breakdown.
- **[U2]** The polar UVU 105.83 ms and the 14-GEMM/9.5-GFLOP estimate are derived from the pre-A1 `kernel_labels_1296.json` and the path table in `POLAR_PAPER_OPTIMIZATION_PLAN.md` §3.1. A1's UVU tile already saved ~24 ms on the materialized path (290.2 -> 266.2 ms), and the combined generic path is now 253.84 ms; re-profile before treating 105.83 ms as current. Verify the checkpoint's actual path count, `d1`, masks, and whether W is dense.
- **[U3]** The small-batch DGEMM ~2.4 TFLOPS figure is a secondary summary of batched DGEMM studies, not an independent measurement for 512x1296x512 on this A100. The 19.5 TFLOPS peak and ~18-19 TFLOPS large-DGEMM numbers are NVIDIA/vendor data; actual clocks and cuBLAS version matter.
- **[U4]** e3j's 1.6-1.9 TB/s vs 0.75-0.83 TB/s is from a search extract; the arXiv abstract only states >80% HBM and >2x conv forward. Its code was not inspected. The result is on H100, not A100.
- **[U5]** FlashTP's 71-86% CG sparsity, 8-bit unique-CG indexing, and path aggregation are from the PMLR abstract plus secondary analyses/README; the exact CUDA kernel was not inspected. FlashTP uses atomics; do not copy that part.
- **[U6]** GROMACS NVSHMEM numbers are H100/GH200 (Eos) and NVLink/InfiniBand; A100 was not tested. The 2x multi-node and 46% at 45k atoms/4 GPUs are from the NVIDIA blog and SC25 paper; the SC25 PDF itself returned HTTP 403 during this scan.
- **[U7]** NCCL 2.37 us AllReduce is on 4x GB200; A100 cannot use multimem/NVLS and was not tested. The 8.7% ITL and 11.0 -> 2.37 us are from the arXiv HTML/search extract, not a full independent reproduction.
- **[U8]** CPU-free 1.63x is CG on 8x A100; PERKS numbers are stencil/CG/SpMV, not MLIP. They support the persistent-kernel pattern, not a direct 1.63x/2.12x projection for PolarMACE.
- **[U9]** The halo/owned ratios (~2.8x at 324 atoms/GPU, ~1.5x at 1,296 atoms/GPU) are geometric estimates for uniform water at 1 g/cm3 with a 6 A cutoff and cube domains; actual LAMMPS ghost counts depend on the decomposition, cutoff, and rank layout.
- **[U10]** The 4x A100 projections assume NVLink/NVSwitch peer access and a working NVSHMEM/NCCL build on the target driver; PCIe-only or MIG would change them. Determine the actual topology before scheduling E5.
- **[U11]** The 5,184 baseline 986 ms is the best measured environment (`MATOS_MH1_CHANNEL_VEC=1`), not the default 1,549 ms; fix `uvu_tile_shape()` size-adaptivity (32x16 <=2k atoms, 64x4 larger) before using 5,184 as a default target.
- **[U12]** MatterScale (SC26) and the e3j secondary TB/s numbers are from conference abstracts/search extracts; treat them as context, not reproduced results.

---

## 8. Bottom line

The engine's remaining single-GPU time is not in Ewald (1.67%) and not primarily in edge streaming after S1.1+A1. The largest unexploited structure is the polar UVU: 4 tables x 2 paths x d1 in {1,3} are 14 shared-weight `512 x N x 512` bilinear forms that the current per-node kernel evaluates at ~0.1 TFLOP/s. Recasting them as grouped FP64 GEMMs with a fused Hadamard/mask epilogue is exact, uses the same `W`, removes the per-node weight re-read and the reverse UVU partials, and is the only change in this survey with a credible 30-60 ms headroom at 1,296. Device-resident SCF orchestration, NVSHMEM multi-GPU halo exchange, FlashTP-style exact sparsity, and e3j-style HBM-saturating tiling are the next four; all keep FP64, exact Ewald, and the two-step fixed point intact.
