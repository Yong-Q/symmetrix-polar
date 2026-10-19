# Kernel Round 3 — Target, Acceptance Contract, and Iteration Loop

**中文摘要：** round 3 的目标是把 128 原子全模型从 **56.50 ms** 压到
**≤50 ms（冲刺 ≤45 ms）**，同时保持 FP64 和全部数值门通过。按当前 profile，
预算来自 tensor product DMMA（2–4 ms）、cuBLAS 批处理（2–3 ms）、UVU reverse
加宽（2–3 ms）、低覆盖 affine/convolution（2–3 ms）。任何一片不过 gate 就回退，
换下一片，不降低精度门；达不到目标不停止迭代。

## 1. Baseline and goal

| Item | Value |
|---|---:|
| Round-2 accepted wall (128 atoms, full) | **56.50 ms/eval** |
| Round-2 GPU busy | 49.97 ms |
| Round-2 static SM-weighted | 22.93 ms |
| **Round-3 target** | **≤50.0 ms/eval** |
| **Round-3 stretch** | **≤45.0 ms/eval** |
| Hard constraint | all gates PASS, FP64, no dropped terms |

The target is deliberately stated as a **wall-time** target, not an occupancy
target. Occupancy is a means; the gate is the contract.

## 2. Budget (from the r5 profile)

| Slice | Current | Expected saving | Cumulative target |
|---|---:|---:|---:|
| FP64 tensor products (DMMA/batched GEMM) | 10.81 ms | 2–4 ms | 52.5–54.5 |
| cuBLAS 312 small launches | 3.80 ms | 2–3 ms | 49.5–52.5 |
| UVU reverse x1/x2 | 6.76 ms | 2–3 ms | 46.5–50.5 |
| low-coverage affine/convolution | 4.33 ms | 2–3 ms | 43.5–48.5 |

The four slices are independent enough to land one at a time. If one slice
fails the gate or yields <1 ms, revert it and take the next; the target does
not move.

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
      -> if wall > 50 ms: pick the next slice
      -> if a slice fails: revert, document the negative result, pick the next
```

- The first slice is chosen by the round-3 plan (DMMA if feasible, otherwise
  cuBLAS batching).
- Each accepted slice gets a short result section and a git commit; the docs
  are updated with the new table, figure, and gate numbers.
- A rejected slice is documented (why, gate metric, revert) rather than
  silently dropped.
- The target is not relaxed to make a slice look successful.

## 5. Out of scope / forbidden

- FP32/TF32 or any precision reduction; dropping physical terms; approximate
  RBE/1-hop modes as production; redundant ghost computation to raise
  occupancy; changing gate tolerances.
- Editing `pair_symmetrix/*` in the kernel round (owned by the MPI workstream).

## 6. Current state

- Round-2 is accepted and committed (`f25c655`, `3b51131`, `fef3e03`).
- Round-3 planning is in progress (`docs/design/KERNEL_ROUND3_PLAN.md`).
- MPI H1 is still being reworked; it is not part of the round-3 speed target
  until the 2-rank parity gate passes again.
- The remote 1-GPU probe is queued on queue004; the 2-GPU task is pending
  reconciliation. Neither blocks the single-GPU round-3 work.
