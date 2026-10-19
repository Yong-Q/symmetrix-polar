# Kernel Round 3 — Target, Acceptance Contract, and Iteration Loop

**中文摘要：** round 3 的目标是把 128 原子全模型从 **56.50 ms** 压到
**20–30 ms**，同时保持 FP64 和全部数值门通过。30 ms 是必须达到的主目标，
20–25 ms 是冲刺；20 ms 已经低于“当前工作量 100% SM 覆盖”的静态下界
（22.93 ms），因此必须同时做到 DMMA/张量核、删除冗余工作、压缩 host gap，
不是单靠把现有 kernel 摊宽。任何一片不过 gate 就回退，换下一片，不降低精度门；
达不到目标不停止迭代。

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
| **30 ms** | DMMA tensor products + cuBLAS batching + UVU/low-coverage widening + significant host-gap reduction | high if DMMA lands |
| **25 ms** | the above plus near-optimal UVU/low-coverage and launch elimination | medium |
| **20 ms** | also remove work (replay/gather/redundant passes) and/or exceed the current-work static floor via tensor cores; small-grid/dependency floor at 128 atoms | low but pursued |

We will not fake 20 ms by relaxing gates, dropping terms, or using FP32/TF32.
If the measured floor is above 20 ms, the result will say so with the exact
blocking kernels and the evidence.

## 2. Budget (staged, from the r5 profile)

| Stage | Slice | Current | Expected saving | Cumulative |
|---|---|---:|---:|---:|
| A | FP64 tensor products (DMMA/batched GEMM) | 10.81 ms | 4–6 ms | 50.5–52.5 |
| A | cuBLAS 312 small launches | 3.80 ms | 2–3 ms | 47.5–50.5 |
| A | UVU reverse x1/x2 | 6.76 ms | 2–3 ms | 44.5–48.5 |
| A | low-coverage affine/convolution | 4.33 ms | 2–3 ms | 41.5–46.5 |
| B | UVU forward/replay/LE + mode1 | ~5.3 ms | 2–3 ms | 38.5–44.5 |
| B | e3 pack/unpack fusion | 1.76 ms | 1–1.5 ms | 37–43.5 |
| B | gather fusion (adjoint + messages) | 3.56 ms | 1–2 ms | 35–42.5 |
| B | SCF/project 1×1×1 kernels | ~5.7 ms | 1–2 ms | 33–41.5 |
| C | host gaps / CUDA Graph / launch elimination | ~6.5 ms | 3–4.5 ms | **28.5–38.5** |
| C | replay elimination / persistent fusion | ~1.2 ms + fusion | 1–2 ms | **26.5–37.5** |
| D | DMMA at tensor-core peak / work removal | residual | 2–5 ms | **20–35.5** |

The stages land one at a time. If a slice fails the gate or yields <1 ms,
revert it and take the next; the target does not move. Stage A alone does not
reach 30 ms; **30 ms requires Stage B + C**, and 20–25 ms requires Stage D.

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
- Round-3 planning is in progress (`docs/design/KERNEL_ROUND3_PLAN.md`).
- MPI H1 is still being reworked; it is not part of the round-3 speed target
  until the 2-rank parity gate passes again.
- The remote 1-GPU probe is queued on queue004; the 2-GPU task is pending
  reconciliation. Neither blocks the single-GPU round-3 work.
