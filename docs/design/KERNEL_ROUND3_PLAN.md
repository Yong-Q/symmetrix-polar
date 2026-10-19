# Symmetrix-Polar GPU kernel — Round 3 implementation plan

Status: **planning only** (2026-10-19). No engine source was edited, nothing was
built, no GPU run was made. The shared build tree
`symmetrix/build-lammps-aacd036df381800a` and the GPU lock are currently owned by
the MPI agent. The implementation agent must use a **private build directory**
and the shared GPU lock (`flock /workspace/group/shared/bza/.tmp/gpu.lock`).

Round-2 accepted state: **56.50 ms/eval** (128-atom NH3, full polar),
GPU busy 49.97 ms, static SM-weighted 22.93 ms.

---

## 中文摘要

Round-2 后剩余的热点是三块 MH-1 tensor product（10.81 ms）、UVU 反向
x1/x2（6.76 ms）、低覆盖的 affine/卷积核（4.33 ms）以及 e3 linear 的
pack/unpack + 小 GEMM（约 6.5 ms，其中 3.80 ms 是 cuBLAS）。
**核心结论：MH-1 tensor product 不能改成 DMMA/GEMM**——每个输出的 K 只有
1–21 个非零项、N≤7，实测只有约 0.12 TFLOP/s，瓶颈是标量指令与输出写带宽，
不是浮点吞吐；正确路线是通道分块（每线程 4 个 channel）+ 反向寄存器累加
+ 去掉冗余 zero-fill，预计这三项合计从 10.81 ms 降到 4.6–6.1 ms，且可保持
bitwise。按本计划全部实施，128-atom 全模型预计 **41–45 ms/eval**（对 round 2
再快 1.25–1.38×）。

## Executive summary

Round 3 should not chase DMMA for the MH-1 tensor products. The measured
contractions are tiny and sparse-per-output: interaction 0 has 16 nonzero
Wigner entries per channel (4 instructions, M=512), interaction 1 has 86
(10 instructions, M=128), with output width ≤7. The forward kernels reach only
≈0.12 TFLOP/s and the reverse kernels ≈0.2 TFLOP/s because they are
instruction- and store-bound, not FLOP-bound. A batched GEMM/Khatri-Rao
reformulation materialises 0.6–2.1 GB of intermediates and is 5–12× below
the DMMA ridge point. The real wins are: (i) process 4 channels per thread to
amortise plan/index/`input_2` loads, (ii) accumulate `input_1_adjoint` in
registers per instruction instead of a global read-modify-write per entry
(bitwise, ≤4× fewer stores), (iii) write the forward output through (`=`
instead of zero-fill + `+=`) because the conv_tp output blocks are disjoint and
the mask is all ones. In parallel, widen the three narrow kernels (affine
LayerNorm+SiLU, reverse cutoff) with team/MDRange policies and split the UVU
reverse reductions to raise occupancy from 39.5% to ~100%. Fusing the e3-linear
pack/unpack into a strided in/out GEMM is the largest single "remove work" item
after that, but carries real GEMM-performance risk and should come last.

Expected cumulative saving 11–16 ms/eval, i.e. **41–45 ms/eval**, 1.25–1.38×
over round 2, with the first slice (MH-1 reverse channels) alone worth
2.0–2.7 ms and fully bitwise.

---

## 0. Verified shapes and accounting

All numbers below are derived from the model JSON
(`polar_symmetrix_probe.json`, 316 MB), the round-2 CUPTI traces
(`bench/kernel_round2_2026-10-19/trace_runs/r5_full/`) and the exact label
analysis (`analysis_exact/r5_full_*`). Since `load_kernels()` filters
`MEMCPY*`/`MEMSET*`, the reported 49.97 ms GPU-busy excludes **2.03 ms** of
copies/memsets in the same eval window; the tensor-product output zeroing is
0.153 + 0.096 ms of that (see §2.1.5).

### 0.1 Tensor products (MH-1 fast path, double precision)

The three "e3 tensor * mh1 fused *" labels are **not** in
`e3nn_product_kokkos.cpp`. That file implements `E3ProductBasisKokkosT`
(correlation-3 symmetric contraction; labels `product contraction` 0.50 ms,
`compiled product reverse` 0.99 ms, family "other"). The MH-1 kernels are the
`interaction.convolution` (`conv_tp`) objects in `e3nn_kokkos.cpp`.

| object | interaction | E (edges) | M (channels) | instructions | input-1 widths | Wigner nnz / channel | output dim | fwd ms | rev-ch ms | rev-h ms |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| P0 | `interactions[0].conv_tp` | 4652 | 512 | 4 | 1 | 16 | 8192 | 1.813 | 1.935 | 1.062 |
| P1 | `interactions[1].conv_tp` | 4652 | 128 | 10 | 1,3 | 86 | 5120 | 1.524 | 2.352 | 2.036 |
| **total** | | | | | | | | **3.337** | **4.288** | **3.097** |

Launch geometry confirms the interpretation: `MDRangePolicy<Rank<2>>({0,0},
{E,M})` with tile `{64,4}` gives grid `73×128` for M=512 and `73×32` for
M=128 (`73 = ceil(4652/64)`, `grid_y = ceil(M/4)`). Reverse harmonics uses
`{E, input_2_dim=16}` → grid `73×4`. The output masks are **all ones** for both
conv_tp objects, and the instruction `i_out` lists are `0..3` and `0..9`
respectively, i.e. the output blocks are **disjoint**.

### 0.2 Occupancy / geometry of the other targets

| label | launches/eval | grid × block | static coverage | measured ms |
|---|---:|---|---:|---:|
| `polar reverse uvu mode0 x1` | 10 | 256 × 256 | 39.5% | 3.207 |
| `polar reverse uvu mode0 x2` | 10 | 256 × 256 | 39.5% | 3.552 |
| `affine fused layer norm silu` | 22 (16×37 blk + 6×1 blk) | `ceil(R/128)`, R=4652 or 128 | 3.4% | 1.262 |
| `affine fused reverse layer norm silu` | 11 (8×37 + 3×1) | same | 3.4% | 1.539 |
| `reverse convolution cutoff` | 2 | 37 × 128 | 2.9% | 1.532 |
| `e3 linear pack` | 147 | 16×64 etc. | 7.4–100% | 0.658 |
| `e3 linear unpack` | 147 | 16×64 etc. | 100% | 1.012 |
| `e3 reverse linear pack` | 97 | 16×64 etc. | 100% | 0.599 |
| `e3 reverse linear unpack` | 97 | 1×64 etc. | 7.4–100% | 0.409 |

UVU: `num_nodes = 128`, `mul1 = mul2 = 512`, `d1 ∈ {1,3}`, two paths (dot, tp).
Grid `256 = ceil(128/64) × ceil(512/4)`; teams are `64 nodes × 4 channels`
(256 threads), 6 resident blocks/SM, 65,536 threads for 65,536 work items — no
wasted lanes, but only 30% of the A100's 221,184 thread slots.

Affine: edge MLPs (`conv_tp_weights`, `density_fn`) have R=4652 rows × W=64;
the local-electron MLP has R=128 × W=128. Forward uses one thread per row and
three serial passes over W; reverse uses four passes. `affine linear epilogue`
(0.984 ms, 32 launches) is already at 100% coverage and is not a target.

### 0.3 Formulas used throughout

Amdahl:

\[
S_{\rm tot}=\frac{1}{(1-p)+p/s},\qquad \Delta T = T_0\left(1-\frac{1}{S_{\rm tot}}\right),
\qquad p=\frac{T_{\rm target}}{T_{\rm wall}} .
\]

Static coverage (round-2 convention, N_SM=108):

\[
O_i=\min\!\left(1,\frac{G_i}{N_{\rm SM}B_i}\right).
\]

DMMA ridge point on A100:

\[
P_{\rm DMMA}=19.5\ \text{TFLOP/s},\quad B_{\rm mem}\approx2.0\ \text{TB/s},
\quad I_{\rm ridge}=\frac{P}{B}\approx9.75\ \text{FLOP/B}.
\]

---

## 1. Prioritized roadmap

| # | Target | current ms/eval | expected saving | effort | risk | gate strategy | removes work or widens? |
|---|---|---:|---:|---|---|---|---|
| 1 | MH-1 reverse channels: register accumulation + 4-channel blocking | 4.288 | **2.0–2.7** | M | Low | bitwise PE/force + full suite | widens + removes global RMW |
| 2 | MH-1 forward: 4-channel blocking + write-through output, drop zero-fill | 3.337 | **1.4–2.0** | M | Low | bitwise PE/force + full suite | removes zero-fill and output read |
| 3 | UVU reverse x1/x2: split the reduction (S=4) + shared `mixed_bar` | 6.759 | **2.4–3.4** | L | Medium | full suite (reordered sums) | widens |
| 4 | Affine LayerNorm+SiLU forward/reverse: team per row | 2.801 | **2.0–2.4** | M | Medium | full suite (tree reduction) | widens |
| 5 | Reverse convolution cutoff: 2-D (edge,k) + deterministic 2-stage sum | 1.532 | **1.0–1.2** | S–M | Medium | full suite (grouped sum) | widens |
| 6 | MH-1 reverse harmonics: unroll 4 channels per thread | 3.097 | **0.9–1.4** | S | Low | bitwise + full suite | fewer instructions |
| 7 | e3-linear: fuse pack/GEMM/unpack into strided in/out FP64 GEMM | 2.678 | **1.5–2.5** | L | Medium | full suite | removes 488 launches + transpose traffic |
| 8 | Group the tiny cuBLAS calls (`gemvx`/`gemmk1`/`splitK`) | 0.393 | **0.2–0.3** | M | Low | full suite | fewer launches |
| 9 | CUDA Graph capture | (≈6.5 ms launch gap) | 0–3, defer | L | High | full suite + determinism | launch gap only |

Sum of #1–#8: **11.4–16.0 ms**, i.e. 40.5–45.1 ms/eval (1.25–1.39×). A
mid-range outcome (13 ms) is **43.5 ms/eval, 1.30×**.

Do **#1 and #2 first**: they are in one function family, are bitwise, and
remove ~3.4–4.7 ms. #3/#4/#5 are independent and can be done in parallel by a
second agent if build isolation is respected.

---

## 2. Per-target design

## 2.1 MH-1 tensor products (10.72 ms; top priority)

### 2.1.1 What the kernels actually compute

For one edge \(e\), channel \(u\), instruction \(i\) with input widths
\(A=2l_1+1\), \(B=2l_2+1\), output width \(C=2l_3+1\), the MH-1 fast path
computes

\[
z_{e,u,c}=\sum_{(a,b)\in \mathcal T_i(c)}
 W^{\rm 3j}_{i}(a,b,c)\;x_{e,u,a}\;y_{e,b}\;w_{e,u,i},
\qquad c=0\ldots C-1,
\]

where \(y\) is the edge spherical-harmonic block (`input_2`, multiplicity 1),
\(x\) is the edge-gathered node feature (`input_1`, M channels), and
\(w\) is the per-edge convolution weight (one scalar per channel and
instruction). \(\mathcal T_i(c)\) is the engine's `nonzero_wigner` list.

**Key fact (verified in the model JSON):** the Wigner tensors are sparse per
output even though their bounding boxes are dense:

| object | per-instruction nnz | Σ nnz | output components ΣC |
|---|---:|---:|---:|
| P0 | 1, 3, 5, 7 | 16 | 16 |
| P1 | 1, 3, 3, 3, 11, 5, 11, 21, 7, 21 | 86 | 40 |

The fused forward kernel is called twice (P0 grid `73×128`, P1 `73×32`); the
reverse is split into a channel kernel (writes `input_1_adjoint` and
`weights_adjoint`) and a harmonics kernel (writes `input_2_adjoint`, reducing
over channels). Source:

- `e3nn_kokkos.cpp:2116` — zeroes the forward output.
- `e3nn_kokkos.cpp:2169-2200` — `e3 tensor product mh1 fused`.
- `e3nn_kokkos.cpp:2218-2220` — zeroes all three adjoints.
- `e3nn_kokkos.cpp:2321-2358` — `e3 tensor reverse mh1 fused channels`.
- `e3nn_kokkos.cpp:2359-2387` — `e3 tensor reverse mh1 fused harmonics`.
- Plan construction: `e3nn_kokkos.cpp:1828-2058` (`mh1_fast_path` at 1948,
  sparse entries at 1884-1908, mask identity flag at 1919).
- Data members: `e3nn_kokkos.hpp:312-333`.

The CUDA TeamPolicy variants in the same function
(`e3nn_kokkos.cpp:2126-2167`, `2234-2319`) are compiled **only for float**
(`if constexpr(std::is_same_v<Precision,float>)`). The double model always
takes the MDRange path. There is no double "direct node" path either
(`try_reverse_from_nodes`, `2400-2563`, is float-only).

### 2.1.2 Why DMMA / batched GEMM is not viable (with evidence)

Write the per-instruction contraction as a Khatri–Rao product
\(P_{e,u,(a,b)}=x_{e,u,a}y_{e,b}\). Then
\(z_{e,u,:}=P_{e,u,:}\,G_i\) with \(G_i\in\mathbb R^{K\times C}\),
\(K=AB\le21\), \(C\le7\).

- **Batch shapes:** the only batch axis of useful size is the edge axis
  (4652) or the channel axis (512/128). Per batch item the GEMM is
  \(1\times C\times K\) (edge batch) or \(M\times C\times K\) with
  \(K\le21, C\le7\) (channel batch). An 8×8×4 DMMA tile wastes ≥8× in M or N
  and ≥12.5% in K.
- **Arithmetic intensity (fused, no intermediate):**
  \[
  I=\frac{2CK}{8(A+B+C)}.
  \]
  For the largest instruction (A=3,B=7,C=5): \(I=2\cdot5\cdot21/(8\cdot15)
  =1.75\) FLOP/B. For P0 (A=1,B=7,C=7): \(I=0.82\) FLOP/B. Both are
  **5–12× below the 9.75 FLOP/B DMMA ridge**, so tensor cores would idle.
- **Materialised Khatri–Rao is worse:** P1's worst instruction needs
  \(4652\times128\times21\times8=100\) MB per instruction; the 10 instructions
  need ≈0.6 GB of write+read before any math. Using the alternative
  \(T=xG_c\) intermediate (M×B per edge per component) needs ≈2.1 GB.
- **Measured throughput proves the bottleneck is not math:** forward
  \(P_0\): \(2\cdot16\cdot16\) FLOP per (edge,channel) ⇒ ≈0.23 GFLOP in
  1.813 ms = **0.13 TFLOP/s**; \(P_1\) ≈0.18 GFLOP/1.524 ms = **0.12 TFLOP/s**
  (1.2% of CUDA-core FP64 peak, 0.6% of DMMA peak). Reverse channels:
  0.18–0.20 TFLOP/s. Reverse harmonics: 0.18–0.25 TFLOP/s.
- The forward output traffic is the dominant real cost: P0 writes
  \(4652\times512\times16\times8=305\) MB, P1 writes 191 MB; the kernel does
  `+=`, so it also reads them (610 + 381 MB). At 1.8/1.5 ms this is
  ≈390/295 GB/s — high enough to matter, but still not HBM-bound.

The round-1 packed-GEMM machinery
(`e3nn_kokkos.cpp:391-409` backend switch, pack at `622-705`/`1435-1497`,
`KokkosBlas::gemm` at `637`/`816`/`1451`) is the right **template** for the
e3-linear fusion in §2.2.3, but it cannot be reused for the MH-1 tensor
products: the per-channel weight is a scalar multiplier, so the contraction
across the packed columns is not a matrix multiply.

**Verdict: do not implement DMMA or cuBLAS for these three kernels.** The
correct alternative is a custom register-blocked FP64 CUDA-core kernel that
(a) processes 4 channels per thread, (b) hoists the channel-invariant
quantities, (c) accumulates in registers, and (d) removes the redundant
zero-fill.

### 2.1.3 Slice 1 — reverse channels (largest single kernel, bitwise)

Current inner loop (`e3nn_kokkos.cpp:2341-2357`): for every Wigner entry it
does a global read-modify-write of `input_1_adjoint` and a global read of
`output_adjoint`/`input_1`/`input_2`/`dynamic_weights`. For P0 all 16 entries
of a channel target the same `a=0` slot; for P1 the 86 entries map to only 4
distinct global `(channel,a)` slots (1 for the 0e block, 3 for the 1o block).
Flushing once per instruction reduces the global RMWs from 16/86 to 4/22,
i.e. 4–3.9× fewer.

Proposed kernel:

```
vec = 4 (env MATOS_MH1_CHANNEL_VEC, 1 restores the old path)
flow = MDRangePolicy<Rank<2>>({0,0},{E, ceil(M/vec)})     // tile {64,4}
per (sample, cb):
  for instruction i:
    in1_acc[vec][3] = 0          // width_1 <= 3 by official_layout
    weight_value[vec] = 0
    for entry in component_offsets[component_base .. +output_width]:
      second = input_2(sample, in2_off + b)             // shared by vec
      entry_coeff = paths(i) * values(entry)            // same product as before
      for t in 0..vec-1:
        ch = cb*vec+t
        oi = out_off + ch*output_width + c
        common = (entry_coeff * mask(oi)) * output_adjoint(sample,oi)
        first  = input_1(sample, in1_off + ch*input_width + a)
        in1_acc[t][a] += (common*weight(ch))*second      // exact old order
        weight_value[t] += (common*first)*second
    for t, a: input_1_adjoint(sample, in1_off+ch*input_width+a) += in1_acc[t][a]
    for t:    weights_adjoint(sample, w_off+ch) = weight_value[t]
```

Bitwise argument: each `(channel,a)` has exactly one writer; the adjoint is
zeroed before the kernel; the entry order inside an instruction and the
instruction order are unchanged. Flushing once per instruction is therefore
bitwise-identical to the current per-entry `+=` (global starts at zero,
`0+x=x`). Do **not** accumulate across instructions: that changes the grouping.

Launch shape: P0 `ceil(512/4)=128 → grid 73×32` (2336 blocks); P1
`ceil(128/4)=32 → grid 73×8` (584 blocks). With registers ≤80, B≥3, coverage
stays 100%.

Expected: 4.29 → 1.6–2.2 ms (1.95–2.7×).

Fallback: `MATOS_MH1_CHANNEL_VEC=1` runs the current code unchanged.
Gate first with the LAMMPS PE/force bitwise comparison (expect max|dF|=0).

### 2.1.4 Slice 2 — forward (write-through + channel blocking)

Two independent changes in `e3nn_kokkos.cpp:2169-2200`:

1. **Write-through.** Because (i) `output_mask_is_identity_ == true` and
   (ii) the instruction output blocks are disjoint, set output with `=`
   instead of `+=` and delete the `ordered_kokkos_deep_copy(output,0)` at
   line 2116. Add a host-side check at construction (`mh1_output_disjoint_`,
   verified from `i_out`/`output_offset`), otherwise keep the old path.
   Saves the two memsets (0.153 + 0.096 ms, currently hidden in the
   `MEMCPY/MEMSET` bucket) plus the 305+191 MB RMW read; the kernel then
   writes each output element exactly once.
   This is bitwise (`0 + scale*result = scale*result`).
2. **4-channel blocking** with the same `vec` switch: hoist the entry
   coefficient (`paths*values`), indices, mask and `input_2` out of the
   per-channel loop; keep `result[vec]` per component. This is bitwise because each channel's accumulation order is
   unchanged.

Launch shape after: `73×32` and `73×8`. Expected: 3.34 → 1.6–2.0 ms.

### 2.1.5 Slice 3 — reverse harmonics (channel unroll)

`e3nn_kokkos.cpp:2359-2387`: one thread per `(edge, component b)` reduces over
512/128 channels. Unroll the channel loop by 4 with four independent
`result[4]` accumulators and one `pv = paths*harmonic_values(term)` hoisted per
term (same product, same rounding). The channel reduction order is preserved
within each accumulator group only if channels are visited in ascending order;
to stay bitwise, unroll as `ch, ch+1, ch+2, ch+3` and add the four partials in
ascending order at the end — **this changes the grouping**, so first implement
the pure ILP variant (keep a single accumulator, interleave four independent
loads) which is bitwise, and only then try the grouped version behind a gate.

Expected: 3.10 → 2.0–2.4 ms (bitwise variant), 1.7–2.1 ms (grouped variant,
gated).

### 2.1.6 Adjoint zero-fill elision (cross-cutting, "removes work")

- `weights_adjoint` is zeroed at `e3nn_kokkos.cpp:2220`, but the channel kernel
  assigns **every** weight index (`= weight_value`). When all instruction
  `path_shape=[M,1]` blocks tile the weight array (verify at construction),
  the zero is redundant: ≈0.24 ms (76.5 + 47.8 MB).
- `input_2_adjoint` is assigned by the harmonics kernel for every component;
  its zero is redundant (small).
- The forward output zero is removed by write-through (0.25 ms).
- Total: **0.5–0.6 ms** of work removal. Add `mh1_adjoint_full_coverage_` flags
  and keep the zeroing as the fallback.

### 2.1.7 Gate plan for §2.1

1. Rebuild in a private tree; run `run 0` on `nh3_128.data` with old/new and
   compare PE and all 384 force components (`bench/kernel_round2_2026-10-19/
   bitwise/` tooling). Require exact 0.0 difference for slices 1/2/6.
2. Full suite on the final tree: module 5, 6, 7, 8, end-to-end 32/32, LAMMPS
   1-rank parity (`docs/testing/GATES.md` §2).
3. Trace confirmation: labels unchanged; `e3 tensor reverse mh1 fused channels`
   grids become `73×32` and `73×8`, time < 2.4 ms; `e3 tensor product mh1
   fused` < 2.2 ms; `e3 tensor reverse mh1 fused harmonics` < 2.4 ms; no new
   large `MEMSET` for the 305/191 MB buffers.

---

## 2.2 cuBLAS / external 3.80 ms (312 launches)

### 2.2.1 Where the time actually is

`analysis_exact/r5_full_summary.json` top-kernels and the exact external
breakdown:

| kernel | launches/eval | ms/eval | call site |
|---|---:|---:|---|
| `cutlass_80_tensorop_d884gemm_32x32_16x5_nt` | 96 | 1.095 | e3-linear forward (`e3nn_kokkos.cpp:637,816,1051,1094,1225`) |
| `cutlass_80_tensorop_d884gemm_32x32_16x5_nn` | 69 | 1.034 | e3-linear reverse (`:1451,1547,1669,1705,1749,1815`) |
| `sm80_xmma_gemm_f64...32x64...tn` | 16 | 0.513 | conditioned edge MLPs / density (`affine_mlp_kokkos.cpp:479/482`) |
| `cutlass_80_tensorop_d884gemm_64x32_16x4_nn` | 11 | 0.355 | affine reverse (`affine_mlp_kokkos.cpp:776/779`) |
| `sm80_xmma_gemm_f64...64x64...tn` | 4 | 0.304 | final conv-weight linear, M=2048/1280, N=4652 |
| `gemvx` | 44 | 0.228 | tiny e3-linears (M≤16, N=128) |
| `gemmk1` | 41 | 0.110 | tiny e3-linears + split-K |
| `xmma_32x64_nt` + `cutlass_32x32_tn` | 11 | 0.101 | e3-linear / affine |
| `splitKreduce` | 19 | 0.055 | split-K of the tiny GEMMs |
| `spherical_harmonics_kernel` | 1 | 0.009 | trace-only |

The "312 tiny launches" framing is only partly right: 104 launches
(`gemvx`/`gemmk1`/`splitKreduce`) cost 0.39 ms, while the 182 `cutlass` and
25 `xmma` launches cost 3.40 ms. The medium GEMMs are already FP64 DMMA
(`d884` = 8×8×4 tensor op) and are the expensive part.

### 2.2.2 GEMM shapes (batch = 128 nodes)

The model has 121 linear instructions; per eval 147 forward + 97 reverse
e3-linear GEMMs (some layers are replayed in the reverse pass). The dominant
shapes (\(M\)=out multiplicity, \(N=128(2l+1)\), \(K\)=in multiplicity) are:

| class | K | M | N | count | FLOPs each |
|---|---:|---:|---:|---:|---:|
| interaction linear_1/2, l=1..3 | 512 | 512 | 128,384,640,896 | many | 67–470 MFLOP |
| interaction linear_res/1 (0e) | 512 | 2048 | 128 | 4 | 268 MFLOP |
| interaction-1 input linears | 128 | 512/2048 | 384/128 | 6 | 50–67 MFLOP |
| embeddings/readouts | 83/1/2/16/32 | 1–512 | 128 | many | <10 MFLOP |

63 of 121 instructions are <1 MFLOP and together contribute only 0.04 GFLOP
(0.6% of the 6.8 GFLOP per pass); they are pure launch/tile overhead
(`gemvx`/`gemmk1` at 4–6 µs each).

Example: \(M{=}512,N{=}896,K{=}512\) on a 32×32×16 DMMA tile is
\(16\times28=448\) blocks — 4.1 blocks/SM, fine. A tiny
\(M{=}16,N{=}128,K{=}512\) is \(1\times4=4\) blocks — 0.4% of the GPU.
Grouping tiny GEMMs helps the second class, not the first.

### 2.2.3 Recommendations

1. **Fuse pack + GEMM + unpack into one custom strided FP64 GEMM** (priority
   #7). Layout: weight \(W[K,M]\) row-major contiguous; input
   `x[sample, channel*width+component]`; output `y[sample, out*width+comp]`.
   A tiled kernel can load a `(K_tile × N_tile)` input tile by computing the
   feature index per column (gather) and write the output tile with the inverse
   scatter, applying `path_weight`, mask and (reverse) `+=` in the epilogue.
   This removes 2.68 ms of pack/unpack kernels and 488 launches. Risk: a
   hand-written GEMM can lose to cuBLAS on the medium shapes; budget a
   fallback `MATOS_E3_LINEAR_CUDA_BACKEND=packed_gemm` that keeps the current
   path. Expected net saving **1.5–2.5 ms**.
2. **Group the tiny GEMMs** (`M≤16` or `N≤32`) into one grouped/batched kernel
   (one block per instruction, or `cublasGemmGroupedBatchedEx` if the CUDA
   version exposes it). These are 104 launches / 0.39 ms; expected saving
   **0.2–0.3 ms**, low risk.
3. **Do not pad/merge the medium GEMMs into a single call.** Their
   `(K,M,N)` differ per irrep block; padding to a common shape wastes FLOPs
   and breaks the block-diagonal structure. They already use DMMA.
4. Correctness constraints: the current path already reorders FP64 sums
   (round-1 gate); a fused kernel adds a different but deterministic order.
   Keep every term, mask, bias and `path_weight`; no FP32 accumulation.

---

## 2.3 Low-coverage affine and convolution kernels (4.33 ms at ~3%)

### 2.3.1 Affine fused LayerNorm+SiLU (forward 1.262, reverse 1.539)

Source: `affine_mlp_kokkos.cpp:450-469` (forward) and `730-767` (reverse).
Current: one thread per row; forward does 3 serial passes over W, reverse 4
passes plus two `exp` evaluations per column. For R=4652, W=64 that is 37
blocks × 128 threads; for R=128, W=128 it is one block.

Proposed: `TeamPolicy` with one team per row, `team_size = W` capped at 128
(W=64 or 128 in this model):

\[
T_{\rm serial}\sim 3W,\qquad
T_{\rm team}\sim \frac{3W}{T}+2\log_2 T,\qquad T=\min(W,128).
\]

- Mean and variance from `Kokkos::parallel_reduce` over the team (fixed tree
  order → run-to-run deterministic, but not bitwise vs the serial sum).
- Normalize+SiLU and the reverse's `scaled` pass are elementwise, one column
  per thread; write directly.
- Reverse: reuse the two reductions; do not recompute `exp` more than twice
  per element (the current code computes it three times).
- Launch: R teams (4652 or 128). For R=4652, 64-thread teams give
  4652/(108×24) > 1 → 100% coverage. For R=128 the kernel is still small
  (128 teams), but per-row serial work drops 128×, which is the actual win.
- Bitwise fallback `MATOS_AFFINE_LN_PARALLEL=elementwise`: keep the serial
  mean/variance per row and parallelize only the elementwise Silu/normalize
  writes. That preserves every output value exactly and still removes most of
  the `exp`/store serialization.

Expected: forward 1.26 → 0.18–0.35, reverse 1.54 → 0.22–0.45; saving
**2.0–2.4 ms**. Gate with the full suite; the reduction order changes, so do
not claim bitwise.

### 2.3.2 Reverse convolution cutoff (1.532 ms)

Source: `mace_nonlinear_kokkos_impl.tpp:7162-7170`. One thread per edge (4652
threads in 37 blocks), serial loop over `k=0..weight_size-1` (2048 for P0,
1280 for P1) doing

\[
c_e=\sum_k a_{e,k}\,r_{e,k},\qquad
b_{e,k}=a_{e,k}\,\ell_e ,
\]

plus `raw_weight_adj` stores. Work is \(E\cdot K\) = 9.5M + 6.0M products.

Proposed two-stage deterministic reduction:

1. `MDRangePolicy<Rank<2>>({0,0},{E, ceil(K/16)})`: each thread handles 16
   consecutive `k`, writes the 16 `raw_weight_adj` values (exactly as now) and
   a chunk partial `partial[e,chunk]` summed in ascending `k`.
2. One thread per edge sums chunks in ascending chunk order and adds to
   `cutoff_adjoint_values(edge)`.

The grouping changes (chunked vs fully serial), but the result is
deterministic. The change survives the gates (round-2 already accepted reordered
FP64 reductions). If bitwise is required, stage 1 writes per-`k` products to a
workspace and stage 2 sums them serially — 0.5–0.7 ms instead of 0.3–0.5 ms.

Traffic is only ≈250 MB per eval, so the kernel should fall to
**0.3–0.5 ms** (saving 1.0–1.2). Fallback:
`MATOS_CONV_CUTOFF_BACKEND=scalar`.

---

## 2.4 UVU reverse x1/x2 (6.76 ms, 39.5%)

Source: `mace_polar_impl.tpp:1969-2066` (mode 0, `d1<=8`), fallback at
`2067-2102`. Forward/LE/replay equivalents at `858-942`, `1486-1536`,
`2559-2660`.

### 2.4.1 Can x1 and x2 share one pass?

No, not profitably. x1 owns output `(node,u)` and reduces over `v`; x2 owns
`(node,v)` and reduces over `u`. Both read the same W and `mixed_bar`, but a
fused thread per `(u,v)` would need two shared-memory reductions:
`x1_acc[u][c] += pair*x2[v][c]` and `x2_acc[v][c] += pair*x1[u][c]`.
That is \(2 M_1 M_2 d_1\) shared atomics per node per path
(≈1.6M per node for \(d_1=3\)), which is slower than the current two kernels.
Keep two kernels.

### 2.4.2 The real problem: only 256 blocks

Work items: \(N\cdot M_1 = 128\times512 = 65{,}536\) threads; the A100 can
host 108×2048 = 221,184. Coverage is 39.5% because there are only 256 teams
of 256 threads (6 resident blocks/SM). The `64×4` tile has no wasted lanes
(128 nodes = 2 exact tiles), so tiling is not the issue — parallelism is.
Shrinking `tile_m`/`tile_n` does **not** help: smaller teams lower the resident
blocks/SM \(B_i\) (e.g. 128-thread teams, B=12 → coverage 29.6% with 384
blocks), so the fixed 256-thread team is kept and the reduction is split
instead.

**Split-reduction:** split the reduction dimension into `S=4` chunks:

- x1: each team handles `(node,u, v-chunk)`; write
  `partial[s, node, u, c]`; a second kernel sums the four partials in
  ascending `s` and adds to `x1_adjoint`.
- x2: symmetric over `u`.
- Workspace per path: \(S\,N\,M_1 d_1\) doubles = 2.1 MB (d1=1) / 6.3 MB
  (d1=3); allocate the max once. Reduction kernel: 196K threads.

\[
\text{threads} = S\cdot N\cdot M_1 = 262{,}144\;(S=4),\qquad
O\approx 1.0 .
\]

Expected 1.8–2.5× on the two large launches; small launches unchanged. With
reduction/store overhead, plan on 6.76 → 3.4–4.4 ms (saving **2.4–3.4 ms**);
the best case (S=4, perfect scaling of the two large launches) is 2.9–3.6 ms
(saving 3.2–3.9), which is not assumed in the roadmap.

### 2.4.3 Extend `mixed_bar` hoisting

`mixed_bar(node,u)` is computed and stored in the x1 pass, then read from
global in x2 once per `(node,v,u)` — \(128\times512\times512=33.5\)M loads
(268 MB). Stage a `mixed_bar` tile in shared memory per x2 team: the team
covers 64 nodes × 4 `v`; loop the `u` dimension in chunks of 32 and stage
`64×32` doubles = 16 KB (plus the 16.4 KB Wᵀ tile). This cuts those loads 4×
and extends the round-2 hoist. Expected additional 0.2–0.4 ms.

Fallback: `MATOS_UVU_REV_SWEEP` env, with S=1 and no staging reproducing the
current code. Gates: full suite; sums are reordered, so no bitwise claim.

### 2.4.4 Optional extension

The forward (`polar uvu mode0` 1.20), replay (1.20) and LE (0.91) kernels
have the same 256-block geometry. Applying the same split-over-`v` template
is another **1.0–1.5 ms**; schedule it after the reverse slice is gated.

---

## 2.5 CUDA Graph (defer, lower priority)

Round 2 has 1194 launches and 49.97/56.50 = 88.4% GPU busy; the gap is
≈6.5 ms, but part of that is CPU-side host work, not launch overhead. A graph
capture is **not feasible without a larger refactor**:

- the reverse polar pass performs host↔device scalar round trips
  (`create_mirror_view_and_copy` of `dbar`, `polar_global_sum_host`,
  host-computed Fukui coefficients at `mace_polar_impl.tpp:1898-1948`), which
  synchronise and cannot be captured;
- workspaces use `ensure_view`/`Kokkos::realloc`, which would need to be
  pre-sized and frozen;
- the e3-linear path reallocates packed buffers if shapes grow.

A backbone-only graph is possible but saves at most 2–4 ms and adds a second
execution path to gate. Do not do it in round 3. The cheaper launch-gap wins
are already in #7 (488 launches) and #8 (104 launches).

---

## 3. First implementation slice (independent, bitwise, gated)

**Target:** `e3 tensor reverse mh1 fused channels` (`e3nn_kokkos.cpp:2321-2358`),
4.288 ms/eval (7.6% of wall), the largest single kernel.

**Changes:**

1. Add `mh1_channel_vec_` (default 4; `MATOS_MH1_CHANNEL_VEC=1` restores the
   old path) in `e3nn_kokkos.hpp` / constructor.
2. Change the MDRange extent to `{E, ceil(M/vec)}`; keep tile `{64,4}`.
3. Register-accumulate `input_1_adjoint` per instruction (`in1_acc[vec][3]`)
   and `weights_adjoint` (`weight_value[vec]`), flushing once per instruction
   in the original order.
4. Hoist `entry_coeff = paths(instruction)*values(entry)` and
   `second = input_2(...)` out of the channel loop; keep the exact expression
   order `((entry_coeff*mask)*out_adj)`, `(common*weight)*second`.
5. Add the adjoint zero-fill elision for `weights_adjoint` behind
   `mh1_weight_adjoint_fully_written_` (verified in the constructor from
   `path_shape`/`weight_offset` tiling).

**Expected:** 4.29 → 1.6–2.2 ms; grid `73×32` (P0) and `73×8` (P1).

**Why this slice first:** one function, one file, one env switch; every channel
keeps its own accumulator and entry order, so PE and all forces must be
bitwise identical. It validates the riskiest performance hypothesis (channel
blocking) on the largest kernel before the forward/harmonics variants.

**Fallback:** `MATOS_MH1_CHANNEL_VEC=1` executes the unmodified loop; if
register pressure hurts, try `vec=2` before abandoning the approach.

**Gate sequence:**

1. Private build; `run 0` A/B on `nh3_128.data`; require PE bitwise and
   max|dF| = 0.0.
2. Full suite: modules 5–8, end-to-end 32/32, LAMMPS 1-rank parity.
3. Interleaved speed A/B (1 warmup + 12 measured, 2 rounds), polar-off and
   432-atom N2 as regression checks.
4. CUPTI trace: confirm the two new grids and the label time; confirm the
   76.2/47.6 MB weight-adjoint `MEMSET` is gone (zero-fill elision).

Only after this is accepted should #2 (forward) and #6 (harmonics) be applied;
they share the same `vec` switch and the same bitwise test.

---

## 4. Acceptance criteria

A slice is accepted only if all of the following hold on a **clean rebuild of
the final tree**:

1. **Numerical gates** (`docs/testing/GATES.md`):
   - module 5: dens abs ≤5e-6, rel ≤1e-6;
   - module 6: dE rel ≤1e-6, abs ≤5e-6 eV;
   - module 7: stage rel ≤1e-6 / out abs ≤1e-6;
   - module 8: FD ≤1e-3, parity ≤1e-3 eV/Å;
   - end-to-end: 32/32 frames, dE rel ≤1e-6, dE abs ≤5e-6 eV,
     dF ≤1e-4 eV/Å;
   - LAMMPS 1-rank parity: dE ≤5e-5 eV, max dF ≤1e-4 eV/Å.
   - 2-rank H1 parity only after the MPI agent's tree is stable; the current
     H1 rework is not accepted until that gate passes again.
2. **Bitwise check** for every slice claimed as bitwise (#1, #2, #6): LAMMPS
   PE identical and max|dF| = 0.0 on the 128-atom input, using the round-2
   `bitwise/` tooling.
3. **Speed protocol** (`docs/testing/BENCHMARKS.md`): interleaved A/B, same
   session, 1 warmup + 12 measured × 2 rounds, median of per-round medians,
   under `flock .../gpu.lock`; report full polar, polar-off, and 432-atom N2.
   Accept if the wall saving is within ±25% of the attributed kernel saving
   and no secondary workload regresses by >2%.
4. **Trace labels** (CUPTI, one warm eval):
   - `e3 tensor reverse mh1 fused channels`: 2 launches, grids `73×32`/`73×8`,
     total < 2.4 ms;
   - `e3 tensor product mh1 fused`: 2 launches, grids `73×32`/`73×8`,
     total < 2.2 ms after slice #2;
   - `e3 tensor reverse mh1 fused harmonics`: 2 launches, unchanged grid,
     < 2.4 ms after #6;
   - `polar reverse uvu mode0 x1/x2`: new partial/reduce labels present, each
     total < 2.4 ms after #3;
   - `affine fused [reverse] layer norm silu`: grid = team count, not 37
     blocks; forward < 0.35 ms, reverse < 0.45 ms after #4;
   - `reverse convolution cutoff`: grid ≫37; < 0.5 ms after #5;
   - total launches reduced (target ≤ 950 after #7, ≤ 850 after #8).
5. **Provenance:** record the `lmp` sha256, the source file sha256/mtime, the
   object mtime, the trace tag, and the final gate JSONs in the round
   directory. Do not trust a binary hash alone (round-2 stale-object lesson,
   `BENCHMARKS.md` §3).

---

## 5. Do not do

- **No FP32/TF32/FP16** anywhere: no downcast, no "FP32 compute with FP64
  storage", no mixed-precision GEMM. The FP64 contract is absolute.
- **No dropped terms or approximate Wigner/UVU contractions**, including
  "small coefficient" pruning; the sparse entry lists are exact model data.
- **No DMMA/GEMM rewrite of the MH-1 tensor products** (§2.1.2): it is
  memory-bound, 5–12× below the DMMA ridge, and materialises 0.6–2.1 GB.
- **No padding of medium GEMMs** to a common shape to merge them; it wastes
  real FLOPs and breaks block-diagonal structure.
- **No atomics on output tensors** and no unordered reductions. Deterministic
  two-stage or per-owner reductions only.
- **No redundant ghost or edge work** to raise occupancy. Widening must
  parallelise the actual reduction, not duplicate it.
- **No touching `pair_symmetrix/*`** or any MPI-owned file.
- **No build or GPU use in the shared tree / without the GPU lock** while the
  MPI agent is active: use a private build directory and `flock`.
- **No CUDA Graph in round 3** (§2.5).
- **No removal of the `MATOS_*` fallbacks**: every slice keeps an env switch
  that restores the round-2 code path.

---

## 6. References

Round-2 artifacts:

- `bench/kernel_round2_2026-10-19/FINAL_REPORT.md`
- `docs/examples/data/kernel_round2_summary.json`
- `docs/examples/data/kernel_round2_families.csv`
- `docs/examples/data/kernel_round2_labels.csv`
- `bench/kernel_round2_2026-10-19/analysis_exact/r5_full_{labels,families,kernels,summary}.{csv,json}`
- `bench/kernel_round2_2026-10-19/trace_runs/r5_full/{cupti_kernels.csv,kokkos_events.csv}`
- `bench/kernel_round2_2026-10-19/gates/`, `bitwise/`

Theory and testing modules:

- `docs/theory/PERFORMANCE_MODEL.md` (Amdahl, coverage, launch overhead)
- `docs/theory/TENSOR_ALGEBRA.md` (e3 linear, tensor product, UVU, kernel-shape rules)
- `docs/theory/ENERGY_MODEL.md`, `docs/theory/MPI_H1.md`
- `docs/testing/GATES.md`, `docs/testing/BENCHMARKS.md`
- `docs/design/KERNEL_PROFILE_2026-10-19.md`
- `docs/design/MH1_FASTPATH_FEASIBILITY.md` (channel counts 512/128, family limits)
- `bench/e3lin_opt_2026-10-19/FINAL_REPORT_e3lin.md` (packed GEMM path, 147/97 launches, scalar revert)

Engine source references:

- `/workspace/group/shared/bza/external-repos/symmetrix-xl/libsymmetrix/source/e3nn_kokkos.cpp`
  (MH-1 plan 1828-2058; forward 2116-2200; reverse 2218-2387; e3-linear
  GEMM 637/816/1451/1547/1669/1705/1749/1815; pack/unpack 622-705, 1435-1497)
- `.../e3nn_kokkos.hpp` (MH-1 members 312-333)
- `.../affine_mlp_kokkos.cpp` (fused LN+SiLU 450-469, 730-767; GEMMs 479/482, 776/779)
- `.../mace_nonlinear_kokkos_impl.tpp` (reverse cutoff 7162-7170; two-pass
  reduction precedent 6023-6051)
- `.../mace_polar_impl.tpp` (UVU reverse 1969-2102; forward 858-942; LE
  1486-1536; replay 2559-2660)
- `.../e3nn_product_kokkos.cpp` (correlation-3 product basis; not an MH-1 target)

---

## Appendix A — Expected total (Amdahl check)

Round-2 \(T_0 = 56.50\) ms. Using the mid-range savings:

| item | ΔT (ms) |
|---|---:|
| MH-1 reverse channels (#1) | 2.35 |
| MH-1 forward + zero-fill (#2) | 1.70 |
| UVU reverse split (#3) | 2.90 |
| affine forward+reverse (#4) | 2.20 |
| reverse cutoff (#5) | 1.10 |
| MH-1 harmonics (#6) | 1.15 |
| e3-linear pack/unpack fusion (#7) | 1.75 |
| tiny-GEMM grouping (#8) | 0.25 |
| **total** | **13.40** |

\[
T_{\rm new}=56.50-13.40=43.1\ \text{ms},\qquad
S=\frac{56.50}{43.1}=1.31 .
\]

Low end (11.4 ms saved): 45.1 ms, 1.25×. High end (16.0 ms): 40.5 ms,
1.39×. The band **41–45 ms/eval (1.25–1.38×)** is the round-3 target.

## Appendix B — Why the front-end theorem "100% coverage ⇒ compute-bound" does not hold here

`docs/theory/TENSOR_ALGEBRA.md` says the MH-1 kernels are "compute-bound".
The trace geometry (100% block coverage) is consistent, but the measured math
rate is 0.12–0.25 TFLOP/s, i.e. 1–2.5% of FP64 CUDA-core peak. "Coverage"
only says enough blocks were launched; it says nothing about instructions per
FLOP. The correct roofline for these kernels is dominated by the
`plan`/`indices`/`values`/`mask`/`weight` loads and the output stores, not by
the 16–86 FMAs per channel. That is why round 3 widens/removes instruction
work instead of adding tensor cores.

## Appendix C — Measurement caveat

`analyze_exact.py` (via `final_analysis.load_kernels`) drops `MEMCPY*` and
`MEMSET*`. In the r5 eval window there are 249 such operations totalling
**2.031 ms**, including the 305 MB and 191 MB tensor-product output memsets
(0.153 + 0.096 ms) and 76.5/47.8 MB weight-adjoint memsets/copies. The round-2
GPU-busy figure 49.97 ms is therefore a compute-kernel sum, not the true
device-busy time; round 3 reports both and counts the zero-fill elision as a
real saving. The two big `MEMSET`s are 0.25 ms of pure removable work.

---

## 7. Stage 2 — beyond 41–45 ms toward 20–30 ms

The slice plan in §1–§3 is the **first stage** and is honestly bounded at
**41–45 ms/eval** (mid-range 43.1). The requested 20–30 ms is **not reachable
by the Stage-1 slices alone**. The planning analysis also rules out DMMA for
the MH-1 tensor products (§2.1.2), so the remaining path is instruction/store
reduction, work removal, and host-gap elimination.

### 7.1 Additional Stage-2 levers

| Lever | Current (post-Stage-1 estimate) | Expected saving | Risk |
|---|---:|---:|---|
| Gather fusion: `gather edge message adjoint` + `nonlinear gather messages` (3.56 ms; both 100% covered but separate passes) | ~3.6 ms | 1.0–2.0 ms | Medium (layout/ownership) |
| SCF/Coulomb/project occupancy: 1×1×1 and low-block launches in the polar forward | ~5.7 ms | 1.0–2.0 ms | Medium (reductions) |
| Host-gap refactor: remove per-eval host round trips; partial CUDA Graph only if a capture-safe subgraph exists | ~6.5 ms gap | 3.0–4.0 ms | High |
| UVU forward/replay/LE split-reduction (same template as reverse) | ~3.3 ms | 1.0–1.5 ms | Medium |
| Tensor-product second pass: 8-channel blocking, plan/index hoisting, fewer `values` loads | ~4.6–5.2 ms | 1.0–2.0 ms | Low–Medium |
| Replay elimination / stage fusion (UVU+normalization+affine) | ~1–3 ms | 1.0–2.0 ms | Medium |
| **Stage-2 total** | | **8.0–13.5 ms** | |

Stage 1 mid (43.1) minus Stage 2 high/low gives **29.6–35.1 ms**. Reaching
**30 ms** therefore requires most of Stage 2 to land. Reaching **20–25 ms**
would require removing another ~5–10 ms beyond Stage 2; the only honest
candidates are algorithmic (fewer SCF iterations within the density tolerance,
caching more invariants across MD steps) or a larger persistent-kernel
refactor. Those are not promised here without a separate gate/accuracy study.

### 7.2 Stage-2 acceptance

Same contract as §4: clean rebuild, full gate suite, bitwise where claimed,
interleaved A/B, CUPTI attribution, pinned binary/source/object hashes. A
Stage-2 slice that saves <1 ms or fails a gate is reverted and documented; the
20–30 ms target is not met by lowering the gate.

### 7.3 Bottom line

- **Stage 1 (this plan):** 41–45 ms, 1.25–1.39× over round 2, high confidence.
- **Stage 2:** 30–35 ms, medium confidence, requires the seven levers above.
- **20–25 ms:** low confidence at 128 atoms; requires work removal/algorithmic
  changes beyond the current kernel set. The static current-work SM-weighted
  floor is 22.93 ms, so 20 ms is below it.

