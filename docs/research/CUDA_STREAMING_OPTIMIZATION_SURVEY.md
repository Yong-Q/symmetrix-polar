# CUDA / Tiled-Edge Streaming Optimization Survey for S1.1

**Date:** 2026-10-21
**Scope:** web/read-only survey. No GPU jobs, no builds, no engine source edits, no git commit/push. The protected trees `bench/plan_b_micro_2026-10-21` and `bench/plan_a_uvu_tile_2026-10-21` were not touched.
**Baseline:** S1 backbone streaming (`streamed_edges='generic'`), FP64 PolarMACE, 1,296 atoms: 290.0 ms materialized vs 397.2 ms streamed (1.3695x); peak VRAM 39.9 vs 9.0 GiB; 5,184 atoms 1,552.3 ms at 19.65 GiB.
**Purpose:** identify CUDA/streaming techniques that can recover the 37% S1 penalty without losing the 4.4-7.5x capacity gain, and map them to concrete `C++20/Kokkos/CUDA` changes.

> **Evidence status.** The measured 397.2 ms binary is `bench/s1_backbone_2026-10-21/bin/lmp_s1` (SHA256 `9aeab9...`), built from `external-repos/symmetrix-xl-s1` at `7b4cf10`. While this survey was being written, that worktree was on branch `s1.1-tile-cache` with **uncommitted** changes (node-contiguous tiles, fused B4 reverse gather, `s1:*` profile marks). Those changes are not part of the measured 397.2 ms number and are treated here as in-flight S1.1 work, not as results. All performance numbers below are either quoted from a cited source or explicitly labeled a projection.

---

## 1. Chinese executive summary (8 lines max)

1. **结论：S1 的 37% 惩罚主要不是"流式"本身，而是每条边仍被物化成大张量、反向逐块重算、以及 S1 仍按 layer 重复分配 edge arena；上游 FP32 tiled streaming 相对 standard streaming 的实测代价只有约 +4%。**
2. **最高杠杆的发现：引擎里已经存在 receiver-owned UVU 前向核（`try_evaluate_execution_uvu_from_nodes`）和 source-owned 确定性反向核（`try_reverse_execution_uvu_from_nodes`），S1 的 generic 路径因为 `execution_layer=false` 从未调用它们。**
3. **这两个核在飞地做 final radial projection、直接累加到 node messages、反向按 source 归约且不用 atomics；接上它们可同时去掉 `[E,weights]`、`[E,messages]`、`[E,edge_up]`、message gather 和 `edge_message_adj`。**
4. **第二便宜的修复：把两个 layer、前后两个 sweep 的 edge 临时区做成一个共享 arena，并把 tile 容量 C 从 16,384 提到 65,536；1,296 峰值约 14 GiB（2 tiles），5,184 约 24 GiB（7 tiles）。**
5. **第三批：把 materialized 路径已有的 B4 融合反向 gather、team cutoff、4-wide message/density gather、raw→weights×cutoff 融合移植到 generic 路径；部分已在未提交的 S1.1 slice 中出现。**
6. **用 `evaluate_conditioned_indexed` 去掉 `[C,2048]`/`[C,1280]` 贡献表；把 final linear 权重按 phi-major 存放，并向量化 128-bit load/store。**
7. **最后再考虑 persistent tile loop + CUDA graph + cp.async 双缓冲；A100 没有 TMA/WGMMA，FA3 式 warp specialization 收益有限，grouped GEMM/Stream-K 对当前大 M GEMM 预计 <2%。**
8. **S1.1 目标：1,296 ≤1.05× materialized（目标 ≤0.95×），5,184 ≤1,552 ms 且 ≤30 GiB，保持 dE≤1e-9 eV / dF≤1e-8 eV/Å，默认仍为 materialized。**

### Top 5 actionable items

| # | Action | Why first | Effort | Projected effect |
|---|---|---|---|---|
| A1 | Wire `try_evaluate_execution_uvu_from_nodes` into the generic streamed forward for both layers | Reuses an existing precision-generic receiver-owned kernel; removes the largest edge tensors and gather | 3-5 d | 1,296 1.37→1.05-1.15; arena <1 GiB |
| A2 | Wire `try_reverse_execution_uvu_from_nodes` with a per-graph source CSR | Removes `edge_message_adj`, `up_adj` atomics, and the generic conv reverse | 2-4 d | 10-20 ms at 1,296; deterministic |
| A3 | One shared edge arena across layers/sweeps + C=65,536 | Reuses dead buffers; 7→2 tiles at 1,296, 27→7 at 5,184 | 1-3 d | 5-15%; peak 14/24 GiB |
| A4 | Indexed conditioning + fused raw→weights×cutoff + vectorized/coalesced density gather | Removes `[C,2048]` tables and extra copies; low risk | 1-3 d | 2-6% |
| A5 | Persistent tile loop + CUDA graph + cp.async double buffer | Cuts 200-650 launches and hides gathers; A100-safe | 3-5 d | 2-5% at 1,296; more at 5,184 |

---

## 2. Technique table

Legend: **S** = source claim, **P** = projection for this engine (unverified). Effort is engineering days for one experienced Kokkos/CUDA engineer, including parity work but not queue time.

| ID | Technique | Source (repo/paper + exact file/function) | What it does | Applicability to S1 / tiled-edge (1,296 speed vs 5,184 capacity) | Expected impact (source number, then projection) | Risk / effort |
|---|---|---|---|---|---|---|
| T1 | Receiver-owned UVU forward with fused final radial projection | Engine: `libsymmetrix/source/e3nn_kokkos.cpp:3627` `E3TensorProductKokkosT::try_evaluate_execution_uvu_from_nodes`; design in Sobek `sobek/kernels/uvu/templates/forward_direct.cu.in` and `forward_codegen.py` (`_direct_kernel`, `_direct_path`) [1][2] | One thread/warp owns `(receiver, channel)`; loops the receiver's CSR edges, computes `W = bias + contribution + phi·B` on the fly, applies harmonics/cutoff, accumulates into node messages. Never materializes `[E,weight_numel]` or `[E,message_width]`. | **Direct fit.** Engine kernel is precision-generic and already initialized for PolarMACE's conv (`mh1_fast_path=true`, `mh1_direct_node_layout_=true`); only the engine-level `execution_layer` gate excludes it from `generic`. | Sobek 1.2-49.7x over 75 capacity-matched comparisons, up to -99% peak memory [1]. **P:** removes the 44 GiB edge-message buffer stream (write+read, fwd+rev) and 5.5 GiB edge-weight stream at 1,296, replacing them with receiver-local accumulation; ≈15-30 ms plus gather/launch savings; 1.37→1.05-1.15. | Medium / 3-5 d. FP64 path is code-inspected, not yet run; contribution table and prefix plumbing. |
| T2 | Source-owned deterministic UVU reverse | Engine: `e3nn_kokkos.cpp:3749` `try_reverse_execution_uvu_from_nodes`; Sobek `templates/backward_source.cu.in`, `backward_edge.cu.in` [1][2] | Loops source owners x channels x input components, traverses outgoing CSR, accumulates `up_adj` in registers; separate edge-parallel kernels write `phi_adj`, harmonic adjoint, cutoff adjoint. No atomics; fixed order. | **Direct fit for FP64.** Removes the generic reverse's `[C,message_width]` `edge_message_adj` gather and `up_adj` atomics. Needs a source CSR, which the engine already builds graph-wide for the execution path (`prepare_execution_source_schedule`, `impl:2927-3078`). | Sobek deterministic reverse is part of the 1.2-49.7x result; OEQ's deterministic path is reported faster than its atomic path [1][3]. **P:** 10-20 ms at 1,296; removes the 22 GiB edge-adjoint buffer stream and atomics. | Medium / 2-4 d. Build the graph-wide source CSR once per neighbor rebuild; slice per block; FP64 parity. |
| T3 | Fixed-workspace tiling + one arena reused across sweeps; larger C | Symmetrix-XL paper §Tiled [2]: "the same temporary workspace is reused across three sweeps", 8,192 workspace receivers; repo `https://github.com/bonan-group/symmetrix-xl` [U]; local `docs/design/TILED_EDGE_PLAN.md` | Bounds all edge temporaries by the largest tile and reuses the same buffer across forward/reverse and both layers. S1 currently sums two per-layer arenas and never shares them. | **High fit.** S1 arena 37,460 doubles/edge = 4.57 GiB at C=16,384. Shared max is 20,177 doubles/edge = 2.46 GiB. C=65,536 gives 9.85 GiB shared arena, 2 tiles at 1,296, 7 at 5,184. | Symmetrix-XL: tiled streaming costs **~+4% per atom** vs standard streaming while accommodating ~9x atoms [2]. **P:** 5-15% at 1,296 from 7→2 tiles; peak 14.1 GiB / 23.9 GiB. | Low-medium / 1-3 d. Lifetime proof: streaming reverse recomputes all edge state, so forward buffers are dead after each layer. |
| T4 | Model-specialized generated kernels (NVRTC/JIT) | Symmetrix-XL paper §Model-specialized execution, repo [2]; local `bench/paper_2610.01036/symmetrix-xl_paper.tex:339-354` | Compiles checkpoint-specific sparse path coefficients, layouts, and channel structure into generated C++/CUDA; removes runtime interpretation. | **Partial.** PolarMACE fails the v4 direct/JIT contract (`NonLinearBiasReadoutBlock`, vector product output) and the pair style rejects direct. Do not flip `mh1_fast_path`. | Symmetrix-XL end-to-end 3.1-5.0x vs ML-IAP + cuEquivariance, FP32 [2]. Not directly transferable to the FP64 PolarMACE v4 contract. | High / 10-25 d for PolarMACE JIT; out of S1.1 scope. |
| T5 | Fused TP + graph convolution with deterministic segmented reduction | OpenEquivariance `openequivariance/openequivariance/templates/loop_unroll_conv_det.cuh` (`forward`, `fixup_forward`, `kahanAdd`), `loop_unroll_conv_atomic.cuh`; paper arXiv 2501.13986 [3] | Warp accumulates into shared memory, stores once per destination row, then a fixup kernel merges warp partials by destination. Avoids atomics and edge-wise intermediates. | **High for the remaining density/harmonic gathers** if T1/T2 are not adopted; redundant once receiver-owned kernels are used. | OEQ: up to 1.3x over cuEquivariance, >10x over e3nn; FP64 MACE inference up to 6.2x over the original unoptimized version [3]. | Medium / 2-4 d. Workspace/fixup sizing; FP64. |
| T6 | Fused GSpMM / SDDMM+SpMM templates | DGL `src/array/cuda/ge_spmm.cuh` (`GESpMMKernel`), `src/array/cuda/gspmm.cu`, `dgl.ops.gspmm` docs [4]; FeatGraph SC'20 [5] | Fuses edge-wise message computation and destination reduction; no edge materialization; feature tiling and shared-memory staging. | **Conceptual.** Our conv+evaluate+gather is an SDDMM+SpMM. The receiver-owned kernel is the stronger version for UVU. Use templates for density and future non-UVU paths. | DGL docs: two steps fused into one kernel, O(|N|D) space [4]. GE-SpMM 1.41x vs cuSPARSE, 1.81x vs GraphBLAST, 3.67x end-to-end [6]; FeatGraph up to 7x GPU [5]. | Medium / 2-4 d. |
| T7 | Grouped / strided-batched small GEMM | CUTLASS `include/cutlass/gemm/kernel/gemm_grouped.h`, `gemm_grouped_problem_visitor.h`, `examples/57_hopper_grouped_gemm`; OpenEquivariance `openequivariance/openequivariance/extension/group_mm.hpp` (`group_gemm_blas` -> `cublasDgemmStridedBatched`, FP64 branch) [3][7] | Batches many small GEMMs/many path groups into one kernel or one strided-batched call; reduces launch and tail. | **Low-medium.** PolarMACE's MLP GEMMs are already large (M=16k-111k, N=64/2048, K=8/64); per-path TP groups are already fused by the MH-1 instruction kernel. Only useful if profiling shows small-GEMM tail. | Source: CUTLASS grouped GEMM is the standard small-GEMM batching path; no end-to-end number for this workload. **P:** <2%, possibly negative. | Medium / 3-5 d. |
| T8 | Stream-K work-centric decomposition | CUTLASS `include/cutlass/gemm/kernel/gemm_universal_streamk.h`, `sm90_tile_scheduler_stream_k.hpp`, `examples/47_ampere_gemm_universal_streamk`; paper arXiv 2301.03598 [7][8] | Splits the aggregate K-iterations evenly across CTAs, eliminating tile-quantization tail for small/irregular GEMMs. | **Low.** Our per-tile MLP GEMMs have thousands of tiles (M/128); Stream-K's target is the opposite regime. Per-path GEMMs are tiny but already fused. | Stream-K: up to 14x/6.7x peak, better average across 32,824 GEMM geometries [8]. **P:** ~0 to -2% here. | Medium / 3-5 d; not recommended for S1.1. |
| T9 | `cp.async` multi-stage double buffering | FlashAttention-2 `csrc/flash_attn/src/flash_fwd_kernel.h` [9]; CUDA `cuda::memcpy_async`/`cuda::pipeline` [10] | Prefetches the next tile's radial/harmonics/phi/up rows into shared memory while computing the current tile; hides global latency. | **Medium.** A100 supports `cp.async` (16 B per thread), not TMA. Useful for the receiver-owned kernel's phi/harmonics/up staging. | FA2: 2-4x vs optimized baselines, 25-40% peak [9]. **P:** 2-5% at 1,296; more at 5,184. | Medium / 2-3 d. |
| T10 | Warp-specialized producer/consumer | FlashAttention-3 `hopper/flash_fwd_kernel_sm90.h` (producer/consumer warp groups, pipeline barriers, TMA/WGMMA) [11] | One warp group issues async loads, another computes; register reallocation (`setmaxnreg`), ping-pong scheduling. | **Low on A100.** Requires Hopper TMA/WGMMA to pay off; Kokkos does not expose warp specialization. A cp.async producer/consumer variant is possible but high effort. | FA3: 1.5-2.0x over FA2 on H100, 840 TFLOP/s BF16 at ~85% [11]. | High / 5-10 d; defer unless target moves to Hopper/Blackwell. |
| T11 | Persistent / grid-stride tile kernel | Sobek `templates/aggregate.cu.in` and `forward_direct.cu.in` (grid-stride work loop, `sm_count*4` blocks, 256 threads); FA3 persistent scheduler [1][11] | One launch per sweep; CTAs grid-stride over tiles, keeping occupancy and reusing shared memory/type tables. | **Medium-high.** S1's per-tile loop launches hundreds of kernels; a persistent loop removes launch/tail and enables shared-memory reuse. | Sobek uses this structure in its fastest path [1]. **P:** 2-5% at 1,296; more at 5,184. | Medium / 3-5 d. |
| T12 | CUDA Graph capture of the host-free subgraph | CUDA docs §4.2 CUDA Graphs, `cudaGraphInstantiate`, `cudaGraphExecKernelNodeSetParams` [12]; Sobek `TPConvCUDAGraph`; PyTorch `torch/_inductor/cudagraph_trees.py` [13] | Captures the tile loop and replays it with one launch; update kernel params for changing pointers without re-instantiation. | **Medium.** Topology changes only on neighbor rebuild; positions/pointers can be updated per eval. Good for 128-atom/1,296 launch-bound cases. | CUDA Graph batching paper: optimal iteration batch exists; gains grow with launch count [14]. **P:** 1-3% at 1,296; 3-6% at 128. | Low-medium / 1-2 d. Allocator/stream capture and MPI interaction need care. |
| T13 | L2 persistence / access-policy window | CUDA docs §4.13 L2 Cache Control, `cudaAccessPolicyWindow`, `cudaStreamSetAttribute` [15] | Reserves L2 for a contiguous window (type tables, final linear weights, CSR offsets) and marks streaming data as streaming. | **Low.** Type tables are 64 KB, final weights ~1 MB, CSR ~0.5 MB; they already fit in L2. Main edge streams are far larger than L2. | Docs quantify no speedup; **P:** <1-2%. | Low / 0.5 d. MIG disables L2 set-aside. |
| T14 | Vectorized 128-bit loads + shared-memory staging | Sobek README: `phi_dim` multiple of 4 enables `float4` radial loads [1]; OEQ `templates/common.cuh` `ROW_OPERATION`, `load_ir_segments` [3]; CUB `cub/cub/block/block_load.cuh` [16] | Loads/stores `double2`/`double4`; stages type tables and per-edge phi/sh in shared memory; coalesces channel-major accesses. | **High, cheap.** The generic path's 1-wide gathers and scalar stores leave bandwidth on the table; materialized path already uses 4-wide message gather. | OEQ reports large gains from shared-memory/ILP management [3]. **P:** 2-6% at 1,296. | Low / 1-2 d. Alignment and layout conversions. |
| T15 | Node-aligned edge tiling (never split a CSR segment) | Symmetrix-XL receiver tiling [2]; local `docs/design/TILED_EDGE_PLAN.md` §3.1; in-flight `s1.1-tile-cache` `StreamEdgeTile` | Tiles cover complete target-node CSR ranges, preserving per-node FP64 summation order and enabling `assignment` instead of `+=`. | **High for parity and arena sharing.** S1's raw edge-range blocks split nodes; the in-flight branch already adds node-contiguous tiles. | TILED design: preserves per-node order; required for bitwise energy. **P:** parity improvement, no direct speed claim. | Low / 1-2 d (already in flight). |
| T16 | Edge/graph reordering for locality | VLDB 2024 graph-reordering study [17]; IO-aware GNN layers arXiv 2605.31500 [18]; GraphTensor destination-centric layout [19] | Reorders nodes/edges (community, Morton, RCM) to improve gather/scatter cache locality. | **Low for water.** Uniform 3D water at 6 Å already has spatial locality; reordering atoms in LAMMPS breaks mapping/MPI and force dumps. Only internal edge permutations are safe. | Reordering is effective overall, but lightweight reordering is better for GPUs and hyperparameters matter [17]; it helps gather-dominated kernels more than feature-parallel ones [18]. **P:** 0-5%; high risk. | High / 3-5 d; defer. |
| T17 | Selective reverse cache / activation checkpointing | PyTorch `torch/utils/checkpoint.py` [20]; Sobek optional retained receiver aggregates for backward; Symmetrix-XL planner selects retention vs recomputation [1][2] | Stores selected forward values so the reverse need not recompute them; trades memory for time. | **Medium, but not Sobek's receiver aggregates.** Our inference reverse needs `phi_adj`, not model-weight gradients, so receiver aggregates are training-oriented. Cache final weights + `h2` (64-wide) or full MLP activations instead. | PyTorch checkpoint documents the memory/recompute tradeoff [20]. **P:** 5-10 ms at 1,296; +3-12 GiB at 5,184. Only if profiling shows prefix/MLP reverse hot. | Medium / 2-4 d. Memory budget must keep 5,184 ≤45 GiB. |
| T18 | Fused epilogue + indexed conditioning; remove `[C,weight]` tables | Engine `affine_mlp_kokkos.cpp:160-205` phi-major final weight; `affine_mlp_kokkos.cpp:590-700` `evaluate_conditioned` / `evaluate_conditioned_indexed` / `evaluate_conditioned_prefix[_indexed]`; cuEquivariance `FusedTP3`/`FusedTP4` in `cuequivariance_torch/cuequivariance_torch/primitives/segmented_polynomial_fused_tp.py` [21] | Adds type contributions inside the final-linear epilogue from `source_type`/`target_type` indices instead of materializing a `[C,2048]` table; fuses the final projection into the consumer; provides a phi-major weight pack. | **High, cheap.** Generic path builds `[C,2048]` + `[C,1280]` per block and does an extra MLP-output copy; `evaluate_conditioned_indexed` and the phi-major packer already exist. | cuEquivariance fuses segmented TP to avoid intermediates [21]; **P:** 2-5% at 1,296. | Low / 1-2 d. Same FP64 addition order (`bias + (src+tgt)`); verify. |
| T19 | Occupancy / launch tuning | Internal CUPTI profile `docs/design/KERNEL_PROFILE_2026-10-19.md`: 755 launches, 52% SM-weighted utilization at 128 atoms; CUDA occupancy API `cudaOccupancyMaxActiveBlocksPerMultiprocessor` [22] | Measures per-kernel duration/occupancy and tunes block size, grid, tile C, and launch count. | **High diagnostic value.** S1 adds per-tile launches and smaller grids; the profile is the only way to attribute the 107 ms. | Internal profile: full eval GPU-busy 96% but SM-weighted 52% [internal]. **P:** identifies >=60% of the 107 ms. | Low / 1-2 d (no engine change). |

### Techniques surveyed but not selected

| Technique / project | Why not selected |
|---|---|
| xFormers fused attention | Attention-specific; no edge-conditioned equivariant TP or scatter pattern to port. |
| cuGraph / WholeGraph | Multi-GPU feature storage and sampling; root-rank/partition gather conflicts with one-rank-per-GPU and exact neighbor semantics. |
| MACE / mace-torch | No streamed-edge execution path found in this survey; MACE accelerates via cuEquivariance/torch.compile. The streaming formulation is in Symmetrix-XL and Sobek [1][2]. Marked unverified absence. |
| DGL GraphBolt neighbor sampling | Training-oriented mini-batch sampling; full-graph MD inference needs all edges and exact forces. |
| FlashAttention softmax/ping-pong | The pipelining pattern is useful (T9/T10), but the softmax/rescaling math is irrelevant. |
| Triton/Inductor GNN kernels | Good for host-fusion prototypes, but our hot path is Kokkos/CUDA with FP64 and LAMMPS; generated Triton cannot call the existing UVU kernels. |

---

## 3. Deep dives on the top five

### 3.1 Receiver-owned UVU forward: use the kernel that is already in the engine

#### 3.1.1 What S1 does today

Per block, the generic forward (`7b4cf10:libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp:4876-5218`) does:

1. Build `convolution_contributions[C, weight_size]` and `density_contributions[C, density_width]` from type tables.
2. Evaluate the conditioned MLP on `radial[C,8]` plus the row contribution table -> `raw_weights[C,2048/1280]`.
3. Copy `raw -> weights`, apply cutoff.
4. Gather `edge_up[C,512]` from node `up`.
5. `interaction.convolution.evaluate(edge_up, harmonics, weights, edge_messages[C,8192/5120])`.
6. Gather `edge_messages` into `messages[N,message_width]`.
7. Density MLP + `tanh(raw^2)` + gather into `densities`.

Steps 3-6 materialize and re-read `[C, weight_size]`, `[C, up_width]`, and `[C, message_width]` every block. At 1,296, `message_width` 8,192/5,120 makes this the dominant traffic: forward+reverse message write/read is ~44 GiB (projection; see §3.1.5).

#### 3.1.2 The existing receiver-owned kernel

`E3TensorProductKokkosT<Precision>::try_evaluate_execution_uvu_from_nodes` (`e3nn_kokkos.cpp:3627`) already implements exactly the Sobek direct-kernel reassociation:

- Work decomposition: `MDRangePolicy<Rank<2>>({0,0},{active_receivers.extent(0), multiplicity})`.
- Each thread owns one `(receiver, channel)` pair.
- It walks `receiver_offsets(receiver)..receiver_offsets(receiver+1)`, clipped to `[first_edge, first_edge+samples)`.
- For each edge, it computes
  `weight = bias[channel] + contribution[local_edge,channel] + sum_q linear_weight[channel,q] * phi[local_edge,q]`,
  applies cutoff, then for each instruction/component computes the sparse Wigner product and does `node_messages(receiver, output_index) += scale * mask * product`.
- No `[E, weight]`, no `[E, message]`, no edge gather. `source_node_values` is the materialized node `up`; `edge_input_2` is the resident harmonics block.

The kernel is templated on `Precision`, with no `if constexpr(float)` branch in its body. It is gated only by `supports_execution_direct_uvu()` (`e3nn_kokkos.hpp:192`), which is `mh1_fast_path && mh1_direct_node_layout_` **on the convolution object**, not the nonlinear engine's family flag.

For PolarMACE, the conv object satisfies this:
- `conv_tp.weight.shape = [0]` -> `internal_weights.extent(0)==0`.
- Every instruction is `uvu`, `has_weight`, `path_shape=[m,1]`, second-input multiplicity 1, output multiplicity `m` (verified by parsing the model JSON for both layers).
- Layer 0 input multiplicity 512; layer 1 input blocks both have multiplicity 128 -> `mh1_direct_node_layout_=true`.

So `supports_execution_direct_uvu()` is true for both PolarMACE layers. The generic path never calls the kernel because `execution_layer = mace_uses_prepared_execution(streamed_edges)` is false for `generic` (`impl:4716`).

#### 3.1.3 Exact argument mapping for S1.1

| Kernel argument | S1.1 value | Notes |
|---|---|---|
| `source_node_values` | `state.up` `[N,512]` | Same buffer the gather reads today; no copy. |
| `source_indices` | `neigh_indices` | Global directed-edge source. |
| `target_indices` | `local_targets` | Global directed-edge target. |
| `active_receivers` | node indices in the tile | For node-contiguous tiles, an iota subview `[first_node,last_node)`; no host build. |
| `receiver_offsets` | `local_offsets` `[N+1]` | Existing CSR. |
| `first_edge` | `tile.first_edge` | Clips the receiver loop to the tile. |
| `edge_phi` | new block view `[C,64]` | Output of `interaction.convolution_weights.evaluate_conditioned_prefix_indexed(...)`; 64 = final-linear input width. The storage hook `state.execution_embedding_storage` already exists. |
| `linear_weight` | `interaction.convolution_weights.final_linear_weight(num_bessel)` | `[weight_size,64]`; already used by the execution path. |
| `linear_bias` | `final_linear_bias()` | May be empty; kernel handles it. |
| `edge_linear_contribution` | `state.convolution_contributions` `[C,weight_size]` | Built from the pair-conditioning type tables exactly as today. |
| `edge_input_2` | `harmonics_block` `[C,16]` | Resident E-sized harmonics; subview only. |
| `edge_cutoff_scale` | `local_cutoffs` | Pass `apply_edge_cutoff_scale = !apply_cutoff`. |
| `node_messages` | `state.messages` `[N,message_width]` | Must be zeroed before the tile loop (it already is for the `+=` generic gather). |

The call is a drop-in branch inside the existing `for(first_edge...)` loop:

```cpp
const bool receiver_owned =
    interaction.convolution.supports_execution_direct_uvu()
    && !spline_layer && !execution_layer;
if (receiver_owned) {
    // 1) prefix: 8 -> 64 -> 64, no final 64->2048
    interaction.convolution_weights.evaluate_conditioned_prefix_indexed(
        radial_block, neigh_types, local_targets, node_types, first_edge,
        convolution_source, convolution_target, state.execution_embedding);
    // 2) fused final linear + TP + receiver accumulation
    const bool ok = interaction.convolution.try_evaluate_execution_uvu_from_nodes(
        state.up, neigh_indices, local_targets, active_receivers,
        local_offsets, first_edge, state.execution_embedding,
        interaction.convolution_weights.final_linear_weight(num_bessel),
        interaction.convolution_weights.final_linear_bias(),
        state.convolution_contributions, harmonics_block, local_cutoffs,
        !apply_cutoff, state.messages);
    if (!ok) throw std::logic_error("receiver-owned UVU forward did not dispatch");
} else {
    // existing edge-owned evaluate + gather
}
```

Add a new capability accessor `supports_receiver_owned_uvu()` (do not reuse `supports_streamed_edges()`), and log it so the A/B harness can assert the path.

#### 3.1.4 Kernel structure, memory layout, launch config

- **Memory layout.** `phi` `[C,64]` row-major, all threads of a receiver/edge read the same 64 values (broadcast). `linear_weight` is currently `[weight_size,64]`; threads over `channel` read different rows with stride 64 doubles, which is uncoalesced. The engine **already has** a phi-major packer: `prepare_final_linear_weight_phi_major` / `final_linear_weight_phi_major` (`affine_mlp_kokkos.cpp:160-205`) returns `[phi_dimension,weight_size]`. Pass that view and change the kernel index to `linear_weight(phi,weight_index)` so a warp over channel reads 32 contiguous weights for each `q`. The type contribution table `[C,weight_size]` is coalesced across channel. The existing kernel writes `node_messages` in `mul_ir` (`channel*output_width+component`); for fully coalesced stores, add an `ir_mul` variant (`component*multiplicity+channel`) and convert once per node with the existing `mh1_output_mul_to_ir` mapping. Sobek uses the native `ir_mul` layout [1]. Start with the existing `mul_ir` writes and measure before changing the layout.
- **Launch config (Kokkos).** Current `MDRangePolicy<Rank<2>>` with `{0,0}->{R, multiplicity}`. Add an inner tile `{1,64}` or `{1,128}` so a warp covers consecutive channels of one receiver. For layer 0, R≈195-700 nodes/tile, 512 channels -> 100k-360k threads; for layer 1, 128 channels -> 25k-90k. Block size 128/256. If the kernel is rewritten in raw CUDA, use a persistent grid of `4 * SM_count` blocks x 256 threads, warp = 32 channels, and register accumulators for up to 16/40 components (Sobek's direct kernel) [1].
- **Register accumulation.** The existing kernel does `node_messages(...) +=` inside the edge loop, so it still performs one global read-modify-write per component per edge. A second version should keep the 16 (layer 0) / 40 (layer 1) component accumulators in registers, loop edges, and store once per `(receiver,channel)`. This is the Sobek pattern and is where the last 10-20% of the win is. Layer 1's 40 doubles/lane = 80 registers; split components into two 20-wide passes or use 128-thread blocks.
- **Shared memory.** Stage the type tables (`types x weight_size` = 64 KB layer 0, 40 KB layer 1) in shared memory per CTA; they are reused by every edge. `linear_weight` is 1 MB (layer 0) / 0.65 MB (layer 1), keep in L2 / persisting window (T13).

#### 3.1.5 Expected impact and risk

- **Traffic removed at 1,296:** the `[E,8192/5120]` edge-message buffer stream (write+read, forward+reverse) is ~44 GiB; the edge-weight buffers ~5.5 GiB; edge_up ~1.8 GiB. The receiver kernel replaces the 44 GiB stream with per-receiver accumulation of node messages; the node rows (64 KB per node for layer 0) are reused across that node's edges and can stay in L1/L2, so the DRAM saving is large but not literally 44 GiB. Add the gather-kernel and launch savings. Projection 15-30 ms, before occupancy effects; the measured gap is 107 ms, so this is necessary but not alone sufficient.
- **Projection:** 1.37 -> 1.05-1.15 with T1 alone; 5,184 peak can drop toward the ~14-16 GiB floor because the edge arena collapses to prefix/adjoint buffers (<1 GiB).
- **Risks:** (i) the FP64 path is code-inspected but not run; (ii) `evaluate_conditioned_prefix_indexed` must produce the same 64-wide prefix as the existing MLP; (iii) the contribution table is still `[C,weight_size]` (T18 removes it); (iv) per-edge global `+=` limits the gain until register accumulation is added.

### 3.2 Source-owned deterministic reverse: unlock the existing kernel

#### 3.2.1 What S1 does today

The committed generic reverse (`7b4cf10:impl:6790-7205`) recomputes the conditioned MLP, then:

1. Gathers `edge_up[C,512]`.
2. Materializes `edge_message_adj[C,8192/5120]` from `message_adj[target]`.
3. Calls `interaction.convolution.reverse` with the edge-indexed adjoint.
4. Scatters `edge_up_adj` into `up_adj` with `Kokkos::atomic_add`.
5. Runs a scalar per-edge cutoff loop (FP64; the team path is only used when `direct_node_reverse`, which is FP32-only at `impl:3427-3440`).

#### 3.2.2 The existing source-owned kernel

`try_reverse_execution_uvu_from_nodes` (`e3nn_kokkos.cpp:3749`) does:

- `MDRangePolicy<Rank<3>>({0,0,0},{num_source_owners, multiplicity, angular_dimension})`.
- Each thread owns `(source, channel, input_component)` and loops the source's outgoing edges in `source_edge_offsets`.
- It accumulates `up_adj(source, component)` in a register and writes once: **no atomics, fixed order**.
- Three edge-parallel passes then write `edge_phi_adjoint[C,64]`, `edge_input_2_adjoint[C,16]`, and `edge_cutoff_scale_adjoint[C]`.
- It recomputes `weight` from `bias + contribution + phi·B` and cutoff; no stored weights.

This is exactly the Sobek backward-source/edge split [1], but already written in Kokkos for this engine, precision-generic.

#### 3.2.3 Argument mapping and source CSR

| Kernel argument | S1.1 value |
|---|---|
| `source_node_values` | `state.up` |
| `source_indices`, `target_indices` | `neigh_indices`, `local_targets` |
| `source_edge_offsets`, `source_edge_indices` | graph-wide source CSR + per-block owner segments; reuse `prepare_execution_source_schedule_host` (`impl:2951-3078`) with the generic `neigh_indices`/`local_targets`, cache per graph generation, and slice `source_offsets` by `execution_block_source_segments_host` |
| `edge_phi` | prefix `[C,64]` recomputed for the tile (cheap: 8->64->64) |
| `linear_weight`, `bias`, `contribution`, `harmonics`, `cutoff` | same as forward |
| `target_node_output_adjoint` | `state.message_adj` |
| `source_node_input_adjoint` | `state.up_adj` |
| outputs | `edge_phi_adjoint[C,64]`, `edge_input_2_adjoint[C,16]`, `edge_cutoff_adjoint[C]` |

After the call:

1. `reverse_prefix_from_tape(edge_phi_adjoint, edge_feature_adj)` (or a custom prefix reverse) gives the 8-wide radial adjoint; add the density feature adjoint and accumulate into `radial_adjoint_values`.
2. Sum `edge_input_2_adjoint` into `harmonic_adjoint_values` (E-sized, resident).
3. Add `edge_cutoff_adjoint` into `cutoff_adjoint_values`.
4. The density path remains separate (T5/T18).

The prefix tape is block-local: call `evaluate_conditioned_prefix_indexed` immediately before `reverse_prefix_from_tape` for the same tile, as the current generic reverse does for the MLP tape. Never let another tile overwrite the tape between those two calls.

The source CSR is graph-wide (all blocks concatenated in `source_edges`, with `block_source_segments` giving each block's owner range), so it is built once per neighbor rebuild, O(E log E) host time. Cache it with the tile table. At 5,184, E=434,456 ints = 1.7 MB; negligible.

#### 3.2.4 Launch config and layout

- `Rank<3>` with `{1,32,1}` or `{1,64,1}` inner tiles so warps cover consecutive channels; each thread accumulates one input component in a register.
- Source CSR arrays are int32; `edge_phi`/`edge_input_2` read per edge (broadcast across channels/components); `linear_weight` phi-major for coalescing.
- The three edge-parallel passes can be fused into one kernel per edge with a warp/team reduction, or kept separate (they are bandwidth-bound and cheap).

#### 3.2.5 Expected impact and risk

- Removes `edge_message_adj` write+read (~22 GiB at 1,296) and all `up_adj` atomics; replaces the generic conv reverse with a source-owned reduction. Projection 10-20 ms at 1,296, plus better run-to-run determinism.
- Risks: graph-wide source CSR build/caching and per-block slicing; FP64 parity; ensuring `message_adj` is complete before the reverse (it is, after the polar reverse); source CSR must not span ranks in MPI (tiles are rank-local).

### 3.3 One shared edge arena + larger tile C

#### 3.3.1 Memory model

S1's per-edge arena is the sum of both layers and both sweeps because `states[layer]` keeps its own storages:

| Set | Doubles/edge |
|---|---:|
| L0 forward (weights, raw, up, messages, contributions, density) | 14,913 |
| L1 forward | 9,537 |
| L0 reverse-only | 4,641 |
| L1 reverse-only | 3,105 |
| shared conditioned-MLP workspace (value+adjoint) | 5,264 |
| **S1 total** | **37,460** |

In streaming mode the reverse recomputes all edge state and the forward edge buffers are dead after each layer's node block. A single arena only needs the **maximum** set plus the MLP workspace:

`max(14,913, 9,537, 4,641, 3,105) + 5,264 = 20,177 doubles/edge`.

| C (edges) | S1 arena (GiB) | S1 peak 1,296 / 5,184 (GiB) | Shared arena (GiB) | Shared peak 1,296 / 5,184 (GiB) | Tiles 1,296 / 5,184 |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 4.57 | 8.83 / 18.63 | 2.46 | 6.72 / 16.52 | 7 / 27 |
| 32,768 | 9.15 | 13.41 / 23.21 | 4.93 | 9.19 / 18.99 | 4 / 14 |
| **65,536** | 18.29 | 22.55 / 32.35 | **9.85** | **14.11 / 23.91** | **2 / 7** |
| 98,304 | 27.44 | 31.70 / 41.50 | 14.78 | 19.04 / 28.84 | 2 / 5 |
| 131,072 | 36.58 | 40.84 / 50.64 | 19.70 | 23.96 / 33.76 | 1 / 4 |

The floor is 4.26/7.49/14.06 GiB (rows 15-25 of the TILED ledger: node, polar, Ewald, readout, model, CUDA/LAMMPS base, resident E-sized geometry). The measured S1 peak is ~5.5% above the model (9.00 vs 8.83; 19.65 vs 18.63); apply that margin to the projections. C=65,536 without arena sharing is 22.6/32.4 GiB modeled, ~23.8/34.2 GiB with margin, still under the 45 GiB gate. With arena sharing it is ~14.9/25.2 GiB.

#### 3.3.2 Why C=65,536 is the cheap experiment

- 1,296: 7 tiles -> 2 (65,536 + 45,776); 5,184: 27 -> 7.
- The MLP GEMM batch grows from 16k to 65k rows, closer to the materialized 111k; fewer per-block launches, conditioning builds, and gather passes.
- The arena still fits the 40/45 GiB gates even without sharing. It is a one-constant or one-runtime-parameter experiment (change `streamed_edge_block_size` / add `set_streamed_edge_capacity`).
- If 65,536 does not recover enough, 98,304 (2/5 tiles) is feasible without sharing at the 45 GiB gate but tight (41.5 modeled, ~43.8 with margin); use only with NVML monitoring.

#### 3.3.3 Shared arena implementation

- Add one `StreamEdgeArena` with subviews for the maximum widths; `ensure_view` targets the arena rather than `states[layer]` storages.
- Keep node buffers (`up`, `messages`, `densities`, `output`, adjoints) in `states[layer]`; only edge temporaries move.
- Prove lifetime with the S1 dependency analysis: forward edge buffers are consumed by the node block and never read by the reverse; the reverse recomputes weights/phi/harmonics. The MLP workspace is reused sequentially.
- Add assertions: arena high-water <1 GiB (receiver-owned) or <10 GiB (edge-owned C=65k); no E-sized allocation >2.6 GiB in receiver-owned mode.
- MPI: tiles are rank-local; the arena is per rank; no halo change.

#### 3.3.4 Expected impact and risk

- 5-15% at 1,296 from 7->2 tiles and larger batches; 5,184 benefits more from 27->7.
- Risks: buffer aliasing bugs (use a single arena with explicit offsets and a debug build that poisons/fences); parity; C must be a multiple of the tile plan; node-contiguous tiles must not split a node.

### 3.4 Fused epilogue / indexed conditioning + coalesced density

#### 3.4.1 Remove the `[C,weight_size]` contribution table

`AffineMLPKokkosT::evaluate_conditioned_indexed` (`affine_mlp_kokkos.cpp:600`) already adds `source_contribution[stype,row] + target_contribution[ttype,row]` inside the final-linear epilogue. The generic path currently calls `evaluate_conditioned(radial, convolution_contributions, raw_weights)`, which requires the `[C,weight_size]` table. Switch to:

```cpp
interaction.convolution_weights.evaluate_conditioned_indexed(
    radial_block, neigh_types, local_targets, node_types, first_edge,
    convolution_source, convolution_target, state.raw_weights);
```

Addition order is the same (`bias + (source + target)`); verify with the 1e-9/1e-8 gates. This removes the table build kernel and its write/read, and removes the `density_contributions` table similarly.

#### 3.4.2 Fuse the raw->weights*cutoff copy

The materialized path already fuses `raw -> weights * cutoff` in one kernel (`impl:5480-5500`). The generic path does `ordered_kokkos_deep_copy(weights, raw_weights)` then a separate cutoff scale (`impl:5207-5230`). Replace with the same one-pass kernel; at C=65,536 this avoids a full `[C,2048]` read+write per block per sweep.

#### 3.4.3 Density path

The density MLP is small (8->64->1) but the per-block gather over all `N` nodes is a serial edge loop per node. Two options:

- Node-contiguous tiles + one lane per node (32 nodes per warp), each lane loops its node's edges and writes one scalar density; this coalesces the density stores and removes the per-block `N`-wide launch. This is the minimal change if the receiver-owned density kernel is deferred.
- Receiver-owned density kernel: one thread per receiver, loop its edges, compute the conditioned density MLP on the fly, `tanh(raw^2)`, apply cutoff, accumulate; write `densities(receiver)` once. This removes `density_matrix[C]` and the gather entirely. Cost ~4k MACs/edge; at 111k edges ~0.5 GFLOP, negligible. This is the natural extension of T1 to the scalar density path.

#### 3.4.4 Vectorization and layout

- Use `double2`/`double4` 16/32-byte loads for `phi` (64 doubles = 8 `double4` per edge), `harmonics` (16 doubles = 2 `double4`), and `up` rows (512 doubles = 64 `double4`).
- Repack `linear_weight` phi-major `[64,weight_size]` as in §3.1.4.
- Stage type tables in shared memory (64 KB / 40 KB).
- Use `ir_mul` for output stores; convert once per node.

Expected: 2-6% at 1,296, low risk. The indexed-conditioning change is independent of T1 and can land first.

### 3.5 Persistent tile loop + CUDA graph + cp.async; grouped GEMM/Stream-K

#### 3.5.1 Persistent tile loop

S1's tile loop issues ~6-8 kernels per tile per sweep; at 1,296 C=65k there are 2 tiles, at 5,184 7 tiles. After T1/T2, the per-tile kernel count drops (forward: prefix + UVU; reverse: prefix + UVU reverse + harm/cutoff adjoints + density). A persistent kernel over tiles can:

- Launch `4 * SM_count` CTAs x 256 threads once per sweep; each CTA grid-strides over `(tile, receiver, channel)` work.
- Keep type tables/linear weights in shared memory/L2 across tiles.
- Avoid per-tile launch overhead and tail effects.

Sobek's `aggregate.cu.in`/`forward_direct.cu.in` use exactly this grid-stride pattern with `sm_count*4` blocks and 256 threads [1]. Port it as a raw CUDA kernel if Kokkos `TeamPolicy` cannot express the persistent work loop efficiently; Kokkos can express a grid-stride `parallel_for` over a flat work index, but shared-memory persistence across teams is harder.

#### 3.5.2 CUDA Graph capture

- Capture the per-sweep sequence after the neighbor list rebuild; instantiate once per `(graph_generation, tile_plan, model)`.
- On each eval, update only pointers/positions via `cudaGraphExecKernelNodeSetParams` (kernel params, grid) or `cudaGraphExecUpdate`; do not recapture.
- Keep allocations out of the capture (pre-allocate the arena; no `Kokkos::realloc` during capture). Kokkos' default stream must be captured, and MPI/comm must stay outside the graph.
- At 1,296, launch overhead is ~1-3 ms; at 128 atoms it is a larger fraction. The S1 profile at 128 reports 755 launches for the full eval; graph capture is most valuable there and for the many-tile 5,184 case.

#### 3.5.3 cp.async double buffering (A100-safe)

- A100 (SM80) supports `cp.async` 16-byte copies, not TMA/WGMMA. Use `cuda::memcpy_async`/`cuda::pipeline` or inline `cp.async.cg.shared.global`.
- Prefetch the next tile's `phi` rows, harmonics, cutoff, and source `up` rows into shared memory while computing the current tile. For the receiver-owned kernel, per-edge `phi` (64 doubles = 512 B) and harmonics (128 B) are small; stage a window of 8-16 edges, not the whole tile.
- `up` rows are 4 KB each; stage only the source rows needed by the current window.
- Double-buffer with two stages; use `__syncthreads()` or `cuda::barrier` between producer/consumer. On Ampere, a simple two-stage `cp.async` pipeline is enough; warp specialization is not required.

#### 3.5.4 Grouped GEMM / Stream-K: honest assessment

- Our MLP GEMMs are `M=C` (16k-111k), `N=64/2048`, `K=8/64`; they have thousands of M-tiles, so Stream-K's tail problem does not apply. Stream-K is for small/irregular GEMMs (source: up to 14x peak, but for different shapes) [8]. Projected gain here ~0 to negative.
- Grouped GEMM applies to the 4/10 tensor-product instruction groups, but the MH-1 fused kernel already loops instructions inside one kernel; splitting into groups would add launches. OpenEquivariance's `group_gemm_blas` (`cublasDgemmStridedBatched`, FP64 branch) is a useful reference if a future path needs batched per-path GEMM, but it is not the S1 bottleneck [3].
- Recommendation: do not spend S1.1 time on T7/T8 unless profiling shows a GEMM tail >5% of the 107 ms.

#### 3.5.5 Expected impact and risk

- Persistent + graph + cp.async: 2-5% at 1,296, 3-8% at 5,184; largest benefit at 128 atoms and for many-tile cases.
- Risks: persistent kernels need careful occupancy/register tuning; CUDA graph capture with Kokkos/MPI can be fragile; `cp.async` shared-memory budget and alignment.

---

## 4. Do not copy

| Do not copy | Why |
|---|---|
| FP16/BF16/TF32/FP8 or reduced-precision accumulation | Sobek is float32/tf32 only [1]; cuEquivariance/FA3 use low precision for speed [11][21]; Symmetrix-XL results are FP32 [2]. PolarMACE gates are FP64 `dE<=1e-9`, `dF<=1e-8`. |
| Approximate Ewald / PPPM / k-space truncation | The engine requires exact Ewald; capacity gain comes from edge state, not from changing electrostatics. |
| Dense `N x N` neighbor or attention matrices | O(N^2) memory and compute; defeats the 5,184/11M-atom capacity goal. |
| Root-rank gather / centralized edge processing | Kills one-rank-per-GPU scaling and MPI balance; all edge data must stay rank-local. |
| Training-only caches (historical embeddings, stale node caches, Sobek receiver aggregates for grad_B) | Inference forces need `phi_adj` and geometric adjoints, not model-weight gradients; receiver aggregates target training and add memory without helping S1. |
| Graph sampling / neighbor dropout / pruning | Changes the physical model and forces; not acceptable for MD parity. |
| Reordering LAMMPS atom IDs or globally permuting CSR | Breaks `map_kokkos`, ghost/owned semantics, force dumps, and MPI; only internal edge permutations within a rank and within the existing CSR order are safe. |
| TMA / WGMMA / FA3 warp specialization on A100 | A100 is SM80: no TMA, no WGMMA. `cp.async` is the available primitive. |
| Stream-K for these large-M GEMMs | Tail is not the problem; may add overhead. |
| Caching full `[E,2048]` weights at 5,184 without a budget | +10.8 GiB (both layers), erodes the capacity claim; use a memory-budgeted switch or the fused receiver kernel instead. |
| Sobek's FP32-only codegen or cuEquivariance binary wheels without FP64 validation | Numerical contract mismatch; port patterns, not precision assumptions. |
| `mh1_fast_path` / `direct` / pair-spline / FP32 for PolarMACE | Would disable materialized, enable FP32/pair spline, and break the accepted baseline; S1 explicitly keeps `mh1_fast_path=false`. |
| Compensated (Kahan) summation in the FP64 hot path | Unnecessary at FP64 and changes the summation order; OEQ uses it for float [3]. |
| Parallel tree reductions that change per-node FP64 order | Use ascending edge order or deterministic per-source fixup; the gates are tolerance-based but not free. |
| `cudaMallocManaged`/UVMM oversubscription for capacity | Page-fault thrash destroys MD throughput; capacity must come from tiling, not oversubscription. |
| Multi-stream concurrency that reorders reductions | Would introduce new nondeterminism; keep one stream and fixed tile order. |

---

## 5. Ranked two-week S1.1 experiment plan

All GPU work is for the implementation team; this survey runs none. Use the accepted environment (`MATOS_REPLAY_CACHE=1`, `MATOS_CONV_CUTOFF_BACKEND=team`, `MATOS_MH1_TEAM_SIZE=64`), one A100-80GB, `flock /workspace/group/shared/bza/.tmp/gpu.lock`, same-binary interleaved A/B (2 rounds x 4 samples, median of medians), NVML at 20 ms. Do not relax parity thresholds.

| Step | Days | Change | Acceptance gate | Kill / fallback |
|---|---:|---|---|---|
| **E0 Profile** | 1-2 | Run S1 generic vs materialized with `MATOS_POLAR_PROFILE=1` and the CUPTI tracer (`bench/kernel_profile_2026-10-19/symtracer.cpp`); use the in-flight `s1:*` marks if available. Produce per-kernel delta and launch counts. | Attributes >=60% of the 107 ms to named kernels; no code change. | If attribution <40%, add finer marks before optimizing. |
| **E1 Capacity sweep** | 2-3 | Add a runtime `set_streamed_edge_capacity(C)` (or rebuild constants) and test C in {16,384, 32,768, 65,536, 98,304}; one shared edge arena if feasible. | 1,296 median ratio <=1.25 without arena sharing; 5,184 rc=0 and peak <=45 GiB; parity G1/G2. | If peak >45 GiB or parity fails, fall back to C=32,768. |
| **E2 Receiver-owned forward** | 4-6 | Wire `try_evaluate_execution_uvu_from_nodes` for both layers with `evaluate_conditioned_prefix_indexed`; add `supports_receiver_owned_uvu` log; use node-contiguous tiles. | 128/1,296 dE<=1e-9, dF<=1e-8; 1,296 ratio <=1.15; arena <1 GiB; 5,184 peak <=35 GiB. | If parity fails twice, keep the edge-owned path and report; do not relax. |
| **E3 Source-owned reverse** | 6-8 | Wire `try_reverse_execution_uvu_from_nodes` with a cached graph-wide source CSR and per-block owner segments; remove `edge_message_adj` and `up_adj` atomics; prefix reverse to radial adjoint. | 1,296 ratio <=1.10; repeat-eval max dF <=1e-12; 2-rank G5; 5,184 peak <=35 GiB. | If source CSR rebuild >2 ms, keep it cached with the tile plan; if parity fails, revert. |
| **E4 Indexed + coalesced cleanup** | 8-9 | `evaluate_conditioned_indexed`; fuse raw->weights*cutoff; vectorized density gather or receiver-owned density; phi-major weights; shared type tables. | 1,296 ratio <=1.05; no new allocations; parity. | If addition-order parity fails, keep the table path. |
| **E5 Launch/overlap** | 10-12 | Persistent tile loop, CUDA graph capture with parameter updates, cp.async double buffer; C auto-tune. | 1,296 ratio <=1.00 (target 0.95); launch count >=5x lower; 5,184 <=1,400 ms. | If graph capture is unstable with MPI/Kokkos, keep persistent kernels only. |
| **E6 Final gates + report** | 12-14 | Same-binary A/B, 128/1,296/5,184, 1/2-rank, NVML, hashes, docs. | 1,296 ratio <=1.05 (hard <=1.15); 5,184 <=1,552 ms and <=30 GiB; dE/dF gates; default materialized; no FP32/TF32. | If 1,296 >1.15, ship as capacity-only opt-in and publish the negative speed result. |

**Overall S1.1 acceptance**

- 1,296: generic <=1.05x materialized (stretch <=0.95x), peak <=40 GiB (target <=20 GiB).
- 5,184: rc=0, median <=1,552 ms (target <=1,300 ms), peak <=30 GiB (hard <=45 GiB), finite energy, 3 repeats stable.
- Parity: 128/1,296 `dE<=1e-9 eV`, `max dF<=1e-8 eV/Å`; 2-rank official `dE<=5e-5`, `dF<=1e-4`; repeat-eval `<=1e-12` at 128.
- Log assertions: `streamed_edges='generic'`, `supports_receiver_owned_uvu=true`, tile count/C, arena bytes, no E-sized allocation >2.6 GiB in receiver-owned mode.
- Default remains `materialized`; no FP16/TF32/FP8; no new atomics; no change to the polar guard or Ewald.

**Recommended order within the two weeks:** E0 -> E1 -> E2 -> E3 -> E4 -> E5 -> E6. E1 and E4 are cheap and can run in parallel with E2/E3 coding. If only one structural change is possible, do E2+E3 (receiver/source-owned kernels); if only one cheap change is possible, do E1 (shared arena + C=65,536).

---

## 6. Citations

[1] V. Chorošajev, C. Bény, "Sobek: Streaming Equivariant Tensor Product Convolutions," arXiv:2607.18074, 2026. https://arxiv.org/abs/2607.18074. Code: https://github.com/Cortex-Discovery/sobek. Files verified: `sobek/kernels/uvu/templates/forward_direct.cu.in`, `forward_direct_path.cu.in`, `forward_group_launcher.cu.in`, `aggregate.cu.in`, `backward_source.cu.in`, `backward_edge.cu.in`, `sobek/kernels/uvu/forward_codegen.py`, `sobek/kernels/config.py`.

[2] L. Fu et al., "Train for Accuracy, Execute at Scale: Architecture-Preserving Inference for Equivariant Atomistic Foundation Models" (Symmetrix-XL), arXiv:2610.01036, 2026. https://arxiv.org/abs/2610.01036. Code availability: https://github.com/bonan-group/symmetrix-xl [U]. Local source: `bench/paper_2610.01036/symmetrix-xl_paper.tex`; local engine `external-repos/symmetrix-xl-s1`.

[3] V. Bharadwaj, A. Glover, A. Buluç, J. Demmel, "An Efficient Sparse Kernel Generator for O(3)-Equivariant Deep Networks," SIAM ACDA 2025, arXiv:2501.13986. https://arxiv.org/abs/2501.13986. Code: https://github.com/PASSIONLab/OpenEquivariance. Files verified: `openequivariance/openequivariance/templates/loop_unroll_conv_det.cuh`, `loop_unroll_conv_atomic.cuh`, `openequivariance/openequivariance/extension/group_mm.hpp`.

[4] DGL, `dgl.ops.gspmm` documentation ("fuses two steps into one kernel"). https://docs.dgl.ai/en/2.2.x/generated/dgl.ops.gspmm.html. CUDA source: `src/array/cuda/ge_spmm.cuh` (`GESpMMKernel`), `src/array/cuda/gspmm.cu`; introduced in commit `071cba1f2bc9f856040ccebfde706ec9b2cd4434`. https://github.com/dmlc/dgl.

[5] Y. Hu et al., "FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems," SC'20. https://doi.org/10.1109/SC41405.2020.00075.

[6] G. Huang et al., "GE-SpMM: General-Purpose Sparse Matrix-Matrix Multiplication on GPUs for Graph Neural Networks," SC'20. https://doi.org/10.1109/SC41405.2020.00076.

[7] NVIDIA CUTLASS. https://github.com/NVIDIA/cutlass. Files verified: `include/cutlass/gemm/kernel/gemm_grouped.h`, `gemm_grouped_problem_visitor.h`, `gemm_universal_streamk.h`, `sm90_tile_scheduler_stream_k.hpp`; examples `24_gemm_grouped`, `47_ampere_gemm_universal_streamk`, `57_hopper_grouped_gemm`.

[8] M. Osama, D. Merrill, C. Cecka, M. Garland, J. D. Owens, "Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU," arXiv:2301.03598, 2023. https://arxiv.org/abs/2301.03598.

[9] T. Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," arXiv:2307.08691, 2023. https://arxiv.org/abs/2307.08691. Code: https://github.com/Dao-AILab/flash-attention (`csrc/flash_attn/src/flash_fwd_kernel.h`).

[10] NVIDIA, CUDA C++ Programming Guide, "Asynchronous Data Copies" and `cuda::pipeline`/`cuda::memcpy_async`. https://docs.nvidia.com/cuda/cuda-programming-guide/ and https://nvidia.github.io/libcudacxx/extended_api/async/pipeline.html.

[11] J. Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision," NeurIPS 2024, arXiv:2407.08608. https://arxiv.org/abs/2407.08608. Code: https://github.com/Dao-AILab/flash-attention (`hopper/flash_fwd_kernel_sm90.h`).

[12] NVIDIA, CUDA C++ Programming Guide, "CUDA Graphs"; Runtime API Graph Management (`cudaGraphInstantiate`, `cudaGraphExecKernelNodeSetParams`). https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html; https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html.

[13] PyTorch, CUDA Graph Trees. https://github.com/pytorch/pytorch/blob/main/torch/_inductor/cudagraph_trees.py.

[14] J. Ekelund, S. Markidis, I. B. Peng, "Boosting Performance of Iterative Applications on GPUs: Kernel Batching with CUDA Graphs," PDP 2025. https://doi.org/10.1109/pdp66500.2025.00019.

[15] NVIDIA, CUDA C++ Programming Guide, "L2 Cache Control"; Runtime API `cudaAccessPolicyWindow`, `cudaStreamSetAttribute`. https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html; https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaAccessPolicyWindow.html.

[16] NVIDIA CCCL/CUB. https://github.com/NVIDIA/cccl; `cub/cub/block/block_load.cuh`, `block_reduce.cuh`, `cub/cub/device/device_scan.cuh`, `cub/cub/device/device_reduce.cuh`.

[17] N. Merkel, P. Toussing, R. Mayer, H.-A. Jacobsen, "Can Graph Reordering Speed Up Graph Neural Network Training? An Experimental Study," PVLDB 2024. https://doi.org/10.14778/3705829.3705846.

[18] D. Fomina et al., "On Efficient Scaling of GNNs via IO-Aware Layers Implementations," arXiv:2605.31500, 2026. https://arxiv.org/abs/2605.31500.

[19] "GraphTensor: Comprehensive GNN-Acceleration Framework for Efficient Parallel Processing of Massive Datasets," IEEE IPDPS 2023. https://doi.org/10.1109/IPDPS54959.2023.00011.

[20] PyTorch, `torch.utils.checkpoint`. https://pytorch.org/docs/stable/checkpoint.html.

[21] NVIDIA cuEquivariance. https://github.com/NVIDIA/cuEquivariance; docs https://docs.nvidia.com/cuda/cuequivariance/; fused TP `cuequivariance_torch/cuequivariance_torch/primitives/segmented_polynomial_fused_tp.py` (`FusedTP3`, `FusedTP4`). Vendor performance claims (10x end-to-end MACE, 200x symmetric contraction, 100k atoms/GPU): https://developer.nvidia.com/cuequivariance [U].

[22] NVIDIA, CUDA Runtime API, occupancy functions (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`). https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html.

### Internal references

- `docs/design/S1_BACKBONE_STREAMING_POC.md` — S1 design, dependencies, memory model, gates.
- `docs/results/S1_BACKBONE_STREAMING_POC_RESULTS.md` — measured 397.2 ms / 1.3695x, 5,184 1,552.3 ms / 19.65 GiB.
- `docs/design/TILED_EDGE_PLAN.md` — tiled-edge memory ledger, node-aligned tiles, arena.
- `docs/design/KERNEL_PROFILE_2026-10-19.md` — CUPTI profile: 755 launches, 52% SM-weighted utilization at 128 atoms.
- `external-repos/symmetrix-xl-s1/libsymmetrix/source/e3nn_kokkos.cpp:3627,3749` — receiver-owned forward and source-owned reverse UVU kernels.
- `external-repos/symmetrix-xl-s1/libsymmetrix/source/affine_mlp_kokkos.cpp:590-700` — conditioned/indexed/prefix MLP entry points.
- `external-repos/symmetrix-xl-s1/libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp:4876-5218,6790-7205` — generic streamed forward/reverse (7b4cf10 numbering).

---

## 7. Unverified claims / caveats

- [U1] The contents of `https://github.com/bonan-group/symmetrix-xl` were not fetched; only the paper's code-availability statement is used.
- [U2] Vendor performance claims for cuEquivariance (10x MACE, 200x symmetric contraction, 100k atoms/GPU) are from NVIDIA's product page, not independently reproduced.
- [U3] All **P** numbers are projections from the S1/TILED memory ledger and traffic counts, not measurements. The measured S1 points are only 1,296/5,184 as cited.
- [U4] The existing receiver-owned/source-owned UVU kernels are precision-generic by code inspection; they have **not** been run on PolarMACE FP64. This is the first thing E2/E3 must validate.
- [U5] The absence of a streamed-edge path in MACE/mace-torch is a survey result, not a proof; MACE may have related work outside the searched repositories.
- [U6] The stage attribution of the 107 ms (messages, gather, MLP, launches) is a hypothesis from code-diff and traffic analysis; E0 must profile it.
- [U7] The `s1.1-tile-cache` worktree had uncommitted changes at the time of writing; none of its results are claimed here.
- [U8] Some file paths (DGL CUDA sources, CUTLASS `cute` headers) were verified via directory listings/search rather than a full raw fetch; the function names and repository locations are reported as found.

---

## 8. Bottom line

S1's 37% penalty is largely self-inflicted by the generic path, not by streaming. The engine already contains the two kernels that implement the Sobek/Symmetrix-XL receiver-owned schedule for UVU: `try_evaluate_execution_uvu_from_nodes` and `try_reverse_execution_uvu_from_nodes`. They are precision-generic, initialized for PolarMACE's convolution, and excluded only by the engine-level `execution_layer` gate. Wiring them into `streamed_edges='generic'`, then sharing one edge arena and raising C to 65,536, is the shortest credible route to 1,296 near the materialized time (target ≤1.05x, stretch ≤0.95x) while keeping 5,184 well under 30 GiB. The remaining work (indexed conditioning, vectorized density, persistent/CUDA graph, cp.async) is incremental; grouped GEMM/Stream-K and warp specialization are not where the time is.
