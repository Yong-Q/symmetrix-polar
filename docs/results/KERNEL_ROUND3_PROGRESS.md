# Kernel Round 3 — implementation progress (2026-10-19)

**中文摘要：** round 3 已进入实现阶段。第一批切片已经落到引擎源码：
MH-1 反向通道（4-channel 寄存器累加 + 按指令 flush）、MH-1 前向
（4-channel + 写穿输出 + 去 zero-fill）、MH-1 harmonics 4 路 ILP、
UVU reverse x1/x2 确定性分块归约（S=4）、affine LayerNorm+SiLU 的
team-per-row 版本、reverse convolution cutoff 两阶段归约。每片都保留
round-2 标量回退环境变量。当前私有 build 正在编译；测速要等 MPI agent
释放共享 GPU lock。数值门和速度数字尚未产生，本文件只记录已实现的代码切片。

## Status

| Item | State |
|---|---|
| Round-2 accepted baseline | 56.50 ms/eval (GPU busy 49.97 ms) |
| Round-3 target | ≤30 ms primary, 20–25 ms stretch |
| Code slices implemented | 6 (MH-1 ×3, UVU reverse, affine LN, conv cutoff) |
| Private build | `symmetrix/build-kernel-r3` (sm80, CUDA 12.9), in progress |
| Speed A/B | blocked on the shared GPU lock (MPI H1 debug session) |
| Numerical gates | not yet run on this build |
| GitHub sync | this commit (patch snapshot, implementation WIP) |

## Implemented slices

### 1. MH-1 reverse channels (`e3nn_kokkos.cpp`)

- Four channels per thread; per-instruction `(channel,a)` slots are loaded
  once, updated in the original entry order, and flushed once.
- Because the slot value is loaded at instruction entry and entries are added
  in the same order, the result is bitwise-identical to the per-entry global
  read-modify-write.
- Launch: `MDRangePolicy<Rank<2>>({0,0},{E,ceil(M/4)})`, grid `73×32` (P0) and
  `73×8` (P1).
- Fallback: `MATOS_MH1_CHANNEL_VEC=1`.

### 2. MH-1 forward (`e3nn_kokkos.cpp`)

- Four channels per thread; channel-invariant `values`/`input_2` loads shared.
- Host-side coverage check proves the instruction output blocks tile the
  complete output exactly once and the mask is identity; in that case the
  zero-fill and the `+=` read-modify-write are replaced by a write-through
  store.
- Launch: `73×32` / `73×8`; output zero-fill removed.
- Fallbacks: `MATOS_MH1_CHANNEL_VEC=1`, `MATOS_MH1_WRITE_THROUGH=0`.

### 3. MH-1 reverse harmonics (`e3nn_kokkos.cpp`)

- Four-channel ILP with four independent accumulators; partials combined in
  ascending channel order. Deterministic, not bitwise vs the serial sum.
- Fallback: `MATOS_MH1_HARMONIC_UNROLL=1`.

### 4. MH-1 adjoint zero-fill elision

- `weights_adjoint` zero-fill removed when the host-side weight intervals tile
  `[0, weight_size_)`.
- `input_2_adjoint` zero-fill removed for the MH-1 path because the harmonics
  kernel assigns every component.
- Fallback: `MATOS_MH1_WEIGHT_ZERO_ELIDE=0`.

### 5. UVU reverse x1/x2 (`mace_polar_impl.tpp`)

- Split the reduction dimension into `S=4` contiguous chunks over `v` (x1) and
  `u` (x2); each team writes a per-chunk partial, a second kernel folds the
  partials in ascending chunk order and adds to the adjoint.
- Workspace: `(S,N,mul*d1)` doubles, grown but not shrunk across paths.
- Launch: 1024 teams per x1/x2 pass instead of 256, 100% static SM coverage.
- Fallback: `MATOS_UVU_REV_SWEEP=1` (round-2 kernel).

### 6. Affine fused LayerNorm+SiLU (`affine_mlp_kokkos.cpp`)

- Forward: one team per row, deterministic tree reductions for mean/variance,
  one column per thread for the elementwise SiLU.
- Reverse: same team mapping; normalized value and SiLU derivative are staged
  in team scratch, so `exp()` is evaluated once per element.
- Fallback: `MATOS_AFFINE_LN_PARALLEL=0` (round-2 kernel).

### 7. Reverse convolution cutoff (`mace_nonlinear_kokkos_impl.tpp`)

- Deterministic two-stage reduction: 16 consecutive `k` per thread write
  `raw_weight_adj` and a chunk partial; one thread per edge folds the chunks in
  ascending order and adds to `cutoff_adjoint_values`.
- Fallback: `MATOS_CONV_CUTOFF_BACKEND=scalar`.

## Expected effect (not yet measured)

| Slice | Current ms/eval | Planned saving |
|---|---:|---:|
| MH-1 reverse channels | 4.288 | 1.8–2.7 |
| MH-1 forward + zero-fill | 3.337 (+0.25 memset) | 1.4–2.0 |
| MH-1 harmonics | 3.097 | 0.9–1.4 |
| UVU reverse x1/x2 | 6.759 | 2.4–3.4 |
| Affine LN forward+reverse | 2.801 | 2.0–2.4 |
| Reverse convolution cutoff | 1.532 | 1.0–1.2 |

Sum ≈ 9.5–13.1 ms kernel time. This is Stage A of
`docs/design/KERNEL_ROUND3_TARGET.md`; Stage B/C are still required for
≤30 ms.

## Next steps

1. Finish the private build and verify the kernel labels/grids with CUPTI.
2. Interleaved A/B vs `bench/kernel_round2_2026-10-19/lmp_r5`.
3. Run module 5–8, 32/32 end-to-end, LAMMPS 1-rank parity on a rebuilt Python
   snapshot.
4. Continue with Stage B: UVU forward/replay/LE split, e3 pack/unpack fusion,
   gather fusion, and the low-coverage SCF/project kernels.
5. Stage C: host-gap/launch reduction. CUDA Graph remains deferred per the
   round-3 plan.

## Provenance

- Round-3 plan: `docs/design/KERNEL_ROUND3_PLAN.md`
- Target/acceptance contract: `docs/design/KERNEL_ROUND3_TARGET.md`
- Engine patch snapshot: `patches/symmetrix-xl-polar-support.patch`
  (includes the base polar support and the round-1/2/3 kernel edits)
- Private build: `external-repos/symmetrix-xl/symmetrix/build-kernel-r3`
