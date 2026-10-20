# Kernel Round 3 — implementation progress (2026-10-19/20)

**中文摘要：** round 3 第一批切片已经实现并完成干净 A/B 测速。
同一二进制、全部 round-3 开关回退 vs 默认：**54.45 → 49.79 ms/eval**
（**1.0935×，省 4.66 ms**）；相对 round-2 验收的 56.50 ms 为 **1.135×**。
CUPTI kernel 时间总和 **49.97 → 43.91 ms**。真正有效的切片是
MH-1 反向通道（-1.63 ms）、MH-1 harmonics（-0.65 ms）、
affine LayerNorm+SiLU 正反向（合计 -1.92 ms）、UVU reverse x1（-0.46 ms）、
k 空间 `desc ab` / `desc proj phase` 分块（合计 -1.59 ms）和
MH-1 前向写穿（-0.11 ms）。计划中预期的 MH-1 前向 4-channel 和
reverse cutoff 两阶段归约实测**变慢**，已分别改回 scalar 默认（保留 opt-in）；
UVU reverse x2 的分块归约基本中性。构建 7 还加入了 `polar rev phase and source`
分块和三个 polar LayerNorm+SiLU 的 team-per-row 版本；前者已编译，后者待下一次
构建生效。e3-linear 的 **merged unpack**（block-plan，一次 launch 完成全部
unpack）已实现，但在多贡献 block 上触发 illegal memory access，因此默认关闭，
仅保留为 `MATOS_E3_LINEAR_MERGED_UNPACK=1` 实验开关；关闭时 parity gate PASS
（dE 2.91e-11 eV，max dF 2.54e-14 eV/Å）。离 20–30 ms 仍有明显差距，下一阶段需要
继续做 Stage B/C（e3 pack/unpack 合并、UVU x2 共享内存、gather 融合、
host gap/CUDA Graph）。

## Measured result (clean interleaved A/B)

Protocol: one binary (`build-kernel-r3/lmp`, sha256
`936440bf69fef369f6de2e935bc5c5d8a9310ab82c22f75ba5080c60d84374e0`),
baseline = every round-3 slice reverted through its env switch
(`MATOS_MH1_CHANNEL_VEC=1`, `MATOS_MH1_WRITE_THROUGH=0`,
`MATOS_MH1_HARMONIC_UNROLL=1`, `MATOS_MH1_WEIGHT_ZERO_ELIDE=0`,
`MATOS_UVU_REV_SWEEP=1`, `MATOS_UVU_FWD_SWEEP=1`,
`MATOS_AFFINE_LN_PARALLEL=0`, `MATOS_CONV_CUTOFF_BACKEND=scalar`).
1 warmup + 12 measured, 2 interleaved rounds, median of per-round medians.

| Configuration | ms/eval | Note |
|---|---:|---|
| Round-3 defaults | **49.79** | per-round medians 49.86, 49.72 |
| Round-2 fallbacks (same binary) | **54.45** | per-round medians 54.15, 54.75 |
| Saving | **4.66 ms** | **1.0935×** |
| Round-2 accepted headline | 56.50 | **1.135× vs round 2** |

CUPTI kernel-time comparison (same 128-atom load, one eval):

| kernel | round 2 | round 3 | delta |
|---|---:|---:|---:|
| e3 tensor reverse mh1 fused channels | 4.288 | 2.662 | **-1.626** |
| e3 tensor reverse mh1 fused harmonics | 3.097 | 2.449 | **-0.649** |
| affine fused reverse layer norm silu | 1.539 | 0.490 | **-1.049** |
| affine fused layer norm silu | 1.262 | 0.392 | **-0.870** |
| polar desc ab reverse (+partial) | 1.043 | 0.077 | **-0.966** |
| polar desc proj phase reverse (+partial) | 0.846 | 0.221 | **-0.625** |
| polar reverse uvu mode0 x1 | 3.207 | 2.749 | -0.458 |
| polar uvu mode0 | 1.201 | 1.092 | -0.110 |
| e3 tensor product mh1 fused | 3.337 | 3.231 | -0.106 |
| polar reverse uvu mode0 x2 | 3.552 | 3.612 | +0.060 |
| **GPU-busy total** | **49.97** | **43.91** | **-6.06** |

Wall 49.79 vs GPU busy 43.91 leaves a ≈5.88 ms host/launch gap.

## Numerical validation on this build

- LAMMPS 1-rank parity vs the round-2 Python reference (`pybase`):
  **PASS**, dE 2.910e-11 eV (≤5e-5), max dF 2.603e-14 eV/Å (≤1e-4).
- Bitwise check round-2 `lmp_r5` vs the Stage-A build on the 128-atom input:
  PE identical, **max |dF| = 0.0** over all 384 components. The k-space
  ab/phase split was added afterwards and is covered by the parity gate above.
- The pre-existing `cudaErrorCudartUnloading` abort at interpreter teardown
  (rc 134) follows the printed GATE PASS, as in every prior round.

## Status

| Item | State |
|---|---|
| Round-2 accepted baseline | 56.50 ms/eval (GPU busy 49.97 ms) |
| Round-3 current | **49.79 ms/eval** (GPU busy 43.91 ms) |
| Round-3 target | ≤30 ms primary, 20–25 ms stretch |
| Code slices implemented | 9 (MH-1 ×3, UVU reverse, UVU forward, affine LN, conv cutoff, ab/phase split, phase-source split) |
| Private build | `symmetrix/build-kernel-r3` (sm80, CUDA 12.9), clean build |
| Speed A/B | PASS, interleaved same-binary protocol |
| Numerical gates | LAMMPS 1-rank parity PASS; full module 5–8 / 32-frame gates still require a rebuilt Python snapshot |
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

- Host-side coverage check proves the instruction output blocks tile the
  complete output exactly once and the mask is identity; in that case the
  zero-fill and the `+=` read-modify-write are replaced by a write-through
  store. Measured saving ≈0.13 ms.
- Four-channel blocking is implemented but **not the default**: it measured
  4.04 ms vs 3.34 ms for the scalar kernel. `MATOS_MH1_FWD_CHANNEL_VEC=4`
  enables it; the shared `MATOS_MH1_CHANNEL_VEC` still overrides both
  directions when set.
- Fallbacks: `MATOS_MH1_FWD_CHANNEL_VEC=1`, `MATOS_MH1_WRITE_THROUGH=0`.

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
- **Not the default:** the chunked version measured 2.87 ms vs 1.53 ms for the
  serial per-edge loop. Scalar is the default; set
  `MATOS_CONV_CUTOFF_BACKEND=chunked` to opt into the experiment.

### 8. `polar desc ab reverse` / `polar desc proj phase reverse` split
(`mace_polar_impl.tpp`)

- Both per-k/per-node serial reductions are split into four contiguous chunks
  with deterministic ascending folding. This is the next build (the current
  measured binary still has the serial versions).

## Planned vs measured

| Slice | Plan saving | Measured saving | Verdict |
|---|---:|---:|---|
| MH-1 reverse channels | 1.8–2.7 | **1.63** | keep |
| MH-1 harmonics | 0.9–1.4 | **0.65** | keep |
| Affine LN forward+reverse | 2.0–2.4 | **1.92** | keep |
| UVU reverse x1+x2 | 2.4–3.4 | **0.46** (x1 only) | x2 neutral |
| k-space ab / phase split | — | **1.59** | keep |
| MH-1 forward | 1.4–2.0 | **0.11** | 4-channel regressed; scalar + write-through |
| Reverse convolution cutoff | 1.0–1.2 | **-1.34** (regression) | scalar default |
| UVU forward split | — | **0.11** | keep |
| **Total GPU busy** | 9.5–13.1 | **6.06** | |

The plan's Stage-A estimate was too optimistic for the MH-1 forward and the
reverse cutoff. The measured Stage-A+B result is **49.79 ms**, not the planned
41–45 ms. The remaining 5.88 ms host/launch gap and the large
MH-1/UVU/gather/cuBLAS kernels are the path to ≤30 ms; 20–25 ms would require
CUDA Graph/launch elimination plus tensor-core or work-removal changes that
the round-3 plan itself flags as low confidence.

## Next steps

1. Rebuild once more to activate the `polar rev phase and source` split
   (implemented, not yet in the measured binary); expected ≈0.4 ms.
2. e3-linear pack/unpack launch merge (488 launches → a few) to attack the
   5.88 ms host gap; expected 1–2 ms wall.
3. UVU reverse x2 shared-memory `mixed_bar` staging; expected 0.3–0.5 ms.
4. Gather fusion into the MH-1 forward/reverse; expected 0.5–1.0 ms.
5. Rebuild the Python snapshot and run module 5–8 / 32-frame gates.
6. CUDA Graph only after the launch count and host syncs are reduced.

## Provenance

- Round-3 plan: `docs/design/KERNEL_ROUND3_PLAN.md`
- Target/acceptance contract: `docs/design/KERNEL_ROUND3_TARGET.md`
- Engine patch snapshot: `patches/symmetrix-xl-polar-support.patch`
  (includes the base polar support and the round-1/2/3 kernel edits)
- Private build: `external-repos/symmetrix-xl/symmetrix/build-kernel-r3`
- Measured binary: sha256 `936440bf69fef369f6de2e935bc5c5d8a9310ab82c22f75ba5080c60d84374e0`
- Speed A/B: `bench/kernel_round3_2026-10-19/speed_ab_r3final1.json`
- Trace: `bench/kernel_round3_2026-10-19/analysis_exact/r3c_full_*`
- Parity gate: `bench/kernel_round3_2026-10-19/r3_gate_lammps_parity_build5.log`
