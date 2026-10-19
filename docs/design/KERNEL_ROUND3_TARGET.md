# Kernel Round 3 — Target, Acceptance Contract, and Iteration Loop

**中文摘要：** round 3 的目标是把 128 原子全模型从 **56.50 ms** 压到
**20–30 ms**，同时保持 FP64 和全部数值门通过。详细规划
（`KERNEL_ROUND3_PLAN.md`）有一个重要结论：**MH-1 tensor product 不能改成
DMMA/GEMM**（每输出只有 1–21 个非零项，实测 0.12–0.25 TFLOP/s，瓶颈是标量
指令和输出写带宽，不是浮点吞吐，DMMA ridge 的 5–12× 以下）。因此 Stage 1
（通道分块、寄存器累加、删 zero-fill、UVU/affine/cutoff 加宽）的现实目标是
**41–45 ms**；**30 ms 需要 Stage 2**（gather/SCF/host-gap/UVU forward/fusion
等额外 8–13.5 ms）；**20–25 ms** 还需要 Stage 2 之外的算法级删工作，置信度低。
任何一片不过 gate 就回退，换下一片，不降低精度门；达不到目标不停止迭代。

## 1. Baseline and goal

| Item | Value |
|---|---:|
| Round-2 accepted wall (128 atoms, full) | **56.50 ms/eval** |
| Round-2 GPU busy | 49.97 ms |
| Round-2 static SM-weighted | 22.93 ms |
| **Round-3 primary target** | **≤30.0 ms/eval** |
| **Round-3 stretch** | **20–25 ms/eval** |
| Hard constraint | all gates PASS, FP64, no dropped terms |

The target is deliberately stated as a **wall-time** target, not an occupancy
target. Occupancy is a means; the gate is the contract.

### Feasibility statement (honest)

| Level | What it requires | Confidence |
|---|---|---|
| **41–45 ms** | Stage 1: MH-1 channel blocking/register accumulation/zero-fill elision, UVU reverse split, affine/convolution widening, e3 pack/unpack fusion, tiny-GEMM grouping | high |
| **30–35 ms** | Stage 2: gather fusion, SCF/Coulomb occupancy, host-gap refactor, UVU forward/replay/LE, tensor second pass, replay/stage fusion | medium |
| **20–25 ms** | Stage 2 plus algorithmic work removal (fewer SCF iterations within tolerance, caching invariants, persistent-kernel refactor); below the current-work static floor 22.93 ms | low but pursued |

We will not fake 20 ms by relaxing gates, dropping terms, or using FP32/TF32.
If the measured floor is above 20 ms, the result will say so with the exact
blocking kernels and the evidence.

## 2. Budget (staged, from the r5 profile and the round-3 plan)

| Stage | Slice | Current | Expected saving | Cumulative |
|---|---|---:|---:|---:|
| 1 | MH-1 reverse channels: channel blocking + register accumulation | 4.29 ms | 2.0–2.7 ms | 53.8–54.5 |
| 1 | MH-1 forward: channel blocking + write-through, drop zero-fill | 3.34 ms | 1.4–2.0 ms | 51.8–53.1 |
| 1 | UVU reverse x1/x2: split reduction + `mixed_bar` staging | 6.76 ms | 2.4–3.4 ms | 48.4–50.7 |
| 1 | Affine LayerNorm+SiLU forward/reverse: team per row | 2.80 ms | 2.0–2.4 ms | 46.0–48.7 |
| 1 | Reverse convolution cutoff: 2-D (edge,k) | 1.53 ms | 1.0–1.2 ms | 44.8–47.7 |
| 1 | MH-1 reverse harmonics: channel unroll | 3.10 ms | 0.9–1.4 ms | 43.4–46.8 |
| 1 | e3-linear pack/unpack fusion | 2.68 ms | 1.5–2.5 ms | 40.9–45.3 |
| 1 | Tiny cuBLAS grouping | 0.39 ms | 0.2–0.3 ms | **40.5–45.1** |
| 2 | Gather fusion (adjoint + messages) | ~3.6 ms | 1.0–2.0 ms | 38.5–44.1 |
| 2 | SCF/Coulomb/project occupancy | ~5.7 ms | 1.0–2.0 ms | 36.5–43.1 |
| 2 | Host-gap refactor / partial graph | ~6.5 ms gap | 3.0–4.0 ms | **32.5–40.1** |
| 2 | UVU forward/replay/LE split | ~3.3 ms | 1.0–1.5 ms | 31.5–38.6 |
| 2 | Tensor second pass (8-channel, plan hoisting) | ~4.6–5.2 ms | 1.0–2.0 ms | 29.5–37.6 |
| 2 | Replay elimination / stage fusion | ~1–3 ms | 1.0–2.0 ms | **27.5–36.6** |
| 3 | Algorithmic work removal / persistent kernels | residual | 3–8 ms | **20–33** |

The stages land one at a time. If a slice fails the gate or yields <1 ms,
revert it and take the next; the target does not move. **Stage 1 alone does
not reach 30 ms; 30 ms requires Stage 2; 20–25 ms requires Stage 3.**

## 3. Acceptance contract

A slice is accepted only if:

1. **Interleaved A/B**: same build session, 1 warmup + 12 measured evals,
   2 rounds, median of medians, GPU `flock`.
2. **Kernel attribution**: CUPTI confirms the intended kernel changed and the
   wall saving is consistent with the kernel saving.
3. **All gates**: module 5–8, 32/32 end-to-end, LAMMPS 1-rank parity; for any
   MPI-affecting change, H1 2-rank parity as well. Pre-existing teardown exit
   134 is accepted only with the PASS marker and JSON.
4. **Bitwise or explicitly gated**: prefer bitwise; if a packed GEMM reorders
   FP64 sums, record the max |Δ| and pass the full gate suite.
5. **Final binary**: record the sha256 of the pinned snapshot and the trace tag.

## 4. Iteration loop (do not stop until the target is met)

```
profile -> pick the largest valid slice -> implement -> build -> A/B -> gates
      -> accept + document + commit
      -> if wall > 30 ms: pick the next slice
      -> if wall <= 30 ms: continue toward 20-25 ms while a valid slice remains
      -> if a slice fails: revert, document the negative result, pick the next
```

- The first slice is chosen by the round-3 plan (DMMA if feasible, otherwise
  cuBLAS batching).
- Each accepted slice gets a short result section and a git commit; the docs
  are updated with the new table, figure, and gate numbers.
- A rejected slice is documented (why, gate metric, revert) rather than
  silently dropped.
- The target is not relaxed to make a slice look successful.
- At the end, if 20–30 ms is not reached, the report names the measured floor,
  the blocking kernels, and the evidence; it does not substitute a different
  metric.

## 5. Out of scope / forbidden

- FP32/TF32 or any precision reduction; dropping physical terms; approximate
  RBE/1-hop modes as production; redundant ghost computation to raise
  occupancy; changing gate tolerances.
- Editing `pair_symmetrix/*` in the kernel round (owned by the MPI workstream).

## 6. Current state

- Round-2 is accepted and committed (`f25c655`, `3b51131`, `fef3e03`).
- Round-3 planning is **complete** (`docs/design/KERNEL_ROUND3_PLAN.md`): it
  rules out DMMA for MH-1 tensor products with measured arithmetic-intensity
  evidence and defines the Stage-1 slices (41–45 ms) and Stage-2 levers
  (30–35 ms).
- MPI H1 is still being reworked; it is not part of the round-3 speed target
  until the 2-rank parity gate passes again.
- The remote 1-GPU probe is queued on queue004; the 2-GPU task is pending
  reconciliation. Neither blocks the single-GPU round-3 work.
