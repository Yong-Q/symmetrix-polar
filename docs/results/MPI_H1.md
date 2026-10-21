# Results — MPI H1: One-Rank and Two-Rank Fixes Accepted (2026-10-20)

**中文摘要：** H1 的 1-rank 缺陷已修复：把 1-rank 图折叠到 local atoms，
H1 1-rank 从 **418.64 ms 降到 50.20 ms**，与默认路径的 50.84 ms 持平
（0.99×），同二进制 A/B 为 176.77 → 54.62 ms（3.24×）。1-rank parity
dE=0、max dF 1.41e-15 eV/Å，20-step NVT 稳定。**2-rank 的两个引擎侧
blocker 也已修复**：default 2-hop 从 OOM（峰值 80,966 MiB）降到
**63,088 MiB（~30.8 GiB/rank）**，parity dE=0、max dF 1.687e-6 eV/Å；
H1 2-rank 从 SIGSEGV 降到 **6,120 MiB（~3 GiB/rank）**，parity dE=0、
max dF 1.01e-14 eV/Å；两条路径的 20-step NVT 都稳定。2-GPU 测试现在
**没有引擎 blocker**，可以一 rank 一卡运行。

## 1. Status summary

| Item | State |
|---|---|
| H1 1-rank, final binary | **50.20 ms** (default 50.84; **0.99x**) |
| H1 1-rank, baseline snapshot | 418.64 ms (default 122.80; 3.41x slower) |
| Same-binary A/B of the one-rank fold | 176.77 → 54.62 ms (**3.24x**) |
| H1 1-rank absolute gain | **8.34x** (baseline snapshot → final binary) |
| H1 1-rank parity | **PASS**: dE = 0.0 eV, max dF = 1.41e-15 eV/Å |
| Exact 2-hop 1-rank parity | **PASS**: dE = 0.0, max dF = 1.54e-15 eV/Å |
| 20-step NVT, H1 1-rank | **PASS**: 128 atoms, no lost atoms, drift 4.74e-14 Å / 4.70e-14 eV/Å |
| **H1 2-rank** | **PASS**: dE=0, max dF **1.01e-14 eV/Å**; 6,120 MiB peak; 20-step NVT stable |
| **Exact 2-hop 2-rank** | **PASS**: dE=0, max dF **1.687e-6 eV/Å**; 63,088 MiB peak |
| 2-rank NVT | **PASS**: 128/128 atoms, no lost, Etot span 0.0603 eV (both paths) |
| 1-rank regression | **PASS**: modules 5–8, force dumps ≤1e-14 eV/Å |
| Final 2-rank-fixed binary | `2a999fe6202ca612a79762b9ec0a711b41431eaffdc2a6d956be1baab3de6857` |

![H1 pre-fix diagnosis](../figures/17_mpi_h1_diagnosis.png)

![H1 forward/adjoint schematic](../figures/18_mpi_h1_schematic.png)

![H1 final one-rank result](../figures/19_mpi_h1_final.png)

![2-rank memory/parity fix](../figures/20_mpi_2rank_fix.png)

## 2. Root cause and fix (one rank)

The H1 mode built a local+ghost engine graph even on one rank. LAMMPS'
periodic self-swaps made it legal but useless: **1170 node rows** (128 local +
1042 periodic images) instead of 128, i.e. 9.1x the node-only work of the
production one-rank path. The pre-fix cost decomposition (per compute, ms):

| component | default (128 nodes) | H1 pre-fix (1170 nodes) | delta |
|---|---:|---:|---:|
| entry | 2.02 | 2.19 | +0.18 |
| forward backbone + 2 H1 syncs | 10.70 | 42.93 | +32.23 |
| polar forward (stages 2–7, Coulomb, LE) | 19.21 | 95.65 | +76.44 |
| polar reverse (UVU, replay, embedding, MLP) | 62.44 | 225.64 | +163.20 |
| backbone reverse + 1 H1 sync | 29.57 | 57.77 | +28.20 |
| **sum** | **124.0** | **424.2** | **+300.3** |

The fix (pair-style only; `pair_symmetrix_mace_kokkos.cpp`):

- one-rank graph folded onto local atoms through the atom map
  (`:2640-2645`, edge remap `:2741-2745`);
- no communication-cutoff extension on one rank (`:803`);
- exchange hooks remain installed but are zero-sized when there are no ghost
  rows (`:2762-2792`), avoiding LAMMPS periodic self-swap writes;
- H1 exchange uses device-side Kokkos pack/unpack with
  `reverse_comm_device = 1` (`:497`, `:959/1057/1158/1279`; host fallbacks
  `:889/1003/1105/1210`).

The same-binary A/B isolates the change: the old 1170-node graph costs
**176.77 → 54.62 ms (3.24x)** even after device staging. The final stage
profile is flat versus the default path; the H1 hooks are 0.13% of the compute
on one rank.

## 3. Two-rank fix (accepted)

Two independent engine-side defects blocked 2 ranks; both are fixed.

### 3.1 Default exact 2-hop OOM — duplicate edge workspaces

The 2-hop graph is large by design: **2481/2581 nodes and 74,472/78,156 edges**
per rank. Every per-edge workspace is linear in `edges`; the defect was two
full-size buffers whose live ranges do not overlap with their forward
counterparts:

- `state.edge_message_adj` (a pure gather of `message_adj`, same shape as the
  already-dead forward `state.edge_messages`);
- `state.raw_weight_adj` (exactly `weight_adj * cutoff`).

Together they added ~8.9 GiB/rank and pushed the peak to **80,966 MiB** on an
80 GiB A100. The fix reuses `edge_messages_storage` for the adjoint and aliases
`raw_weight_adj` to `weight_adj` in place; the MLP reverse consumes
`weight_adj`. Peak: **80,966 MiB (OOM) → 63,088 MiB (~30.8 GiB/rank)**, rc=0.

### 3.2 H1 SIGSEGV — communication-width mismatch

`node_feature_width()` returned the 512-wide embedding. The pair style copies
it into `comm_forward`/`comm_reverse`, and LAMMPS sizes its send/recv buffers
once in `Comm::init`. The layer-1 H1 hook exchanges the **2048-wide**
product-block output, so pack/unpack overran the 512-wide buffers (SIGSEGV or
`cudaErrorIllegalAddress` depending on the build). The fix returns
`max(embedding, products[*].output_dimension())` and adds multi-rank extent +
per-layer sync-width guards. Peak H1 2-rank: **6,120 MiB (~3 GiB/rank)**, rc=0.

### 3.3 Parity / NVT / regression

| Check | Result |
|---|---|
| 2-rank default exact 2-hop | dE = 0.0 eV, max dF **1.687e-6 eV/Å** (reproduces the accepted value) |
| 2-rank H1 | dE = 0.0 eV, max dF **1.01e-14 eV/Å** |
| Official LAMMPS parity 1-rank | dE 2.910e-11 eV, max dF **2.60e-14 eV/Å** |
| Official LAMMPS parity 2-rank exact | dE 2.910e-11 eV, max dF **1.687475435e-6 eV/Å** |
| 20-step NVT, both 2-rank paths | 128/128 atoms, no lost, Etot span **0.0603 eV** |
| H1 2-rank vs 1-rank thermo | max difference **1e-10** |
| 1-rank regression | modules 5–8 PASS; force dumps vs accepted ≤1e-14 eV/Å |

**Remaining blocker for the 2-GPU test: none.** Run one rank per GPU; per-rank
footprints are ~31 GiB (default) and ~3 GiB (H1). The shared `lmp` was restored
to the validated 2-rank-fixed binary; the final integration rebuild must rerun
the 2-rank gates after merging the latest round-3 kernel edits.

## 4. What is inherent on one shared A100

All ranks share one GPU, so multi-rank cannot win wall-clock by construction.
The H1 design runs the complete node pipeline (including the polar k-space
Coulomb chain) on every ghost row on every rank. On the 128-atom system each
2-rank process carries ~836 ghost rows against ~64 local rows; the aggregate
node work is ~1.5x the one-rank H1 graph and the GPU serializes it: 2 ranks
measure ~2x the one-rank stage times despite fewer nodes per rank. No
pair-style change can remove that duplication without engine support; the only
pair-side win was the one-rank folded graph, now in place.

Ghost compaction cannot help: a geometric reconstruction shows all 861 ghosts
within the 8 Å communication cutoff are within \(r_c=6\) Å of a local atom
(0% unused). The real multi-rank win requires one rank per GPU and a larger
system (432/1024 atoms) where the ghost ratio is smaller.

## 5. Planned 2-GPU test

- Queue: `devinstance-queue3`, flavor `ml.pni2.4xlarge` (2×A100-80G, 16C).
- Systems: 128 (worst case), 432, 1024 atoms.
- Modes: H1 exact, exact 2-hop control; 1/2 ranks; parity + 20-step NVT.
- Metrics: ms/eval or ms/step, speedup vs 1 rank, ghost ratio, communication
  fraction, per-rank utilization.
- Acceptance: dE ≤5e-5 eV, max dF ≤1e-4 eV/Å, no lost atoms.
- **Prerequisite satisfied:** the 2-rank workspace/width fix above passes
  parity on one GPU. Per-rank footprint is ~31 GiB (default) and ~3 GiB (H1),
  well inside an 80 GiB A100 per rank.
- Launch one rank per GPU (launcher-specific device binding). The one-GPU
  2-rank wall (H1 0.518 s/step, default 1.478 s/step vs 1-rank 0.137 s/step)
  is the shared-SM floor, not a correctness issue.

## 6. Operational state

- **1-GPU remote probe `t-20261019191724-4zbd7` (`queue004`): FINISHED,
  Success (2026-10-20 06:23 UTC).** The remote package runs on a real
  A100-SXM4-80GB and reproduced the local round-2-era 1-rank timings:
  128 atoms **0.611576 s / 5 steps = 122.3 ms/step**, 432 atoms **1.47169 s
  / 5 = 294.3 ms/step**, 1024 atoms **3.05202 s / 5 = 610.4 ms/step**
  (`bench/mpi_scaling_2026-10-19/remote/attempt-02/`). This validates the
  remote runtime, not the Plan B build: the probe predates Plan B.
- The 2-GPU submit to `devinstance-queue3` returned `SubmitOutcomeUnknown`
  (Volc CLI exit 255). Its visible workdir `remote/attempt-01/` contains only
  inputs and no logs or dumps, so no remote instance ran; remote acceptance
  remains unconfirmed. The local reservation `pending-7dea5a64…` still holds
  2 GPUs and has no remote task id, so the structured status tool cannot
  reconcile it; this is a MatOS Volc-MCP gap.
- Queue state on 2026-10-21: `devinstance-queue3` has **0 free GPUs**
  (32/32 allocated), and `queue004` has **5 free GPUs** but a hard 1-GPU/task
  policy with no 2-GPU claim option. `volc_vemlp_queues` returns no
  `recommendedQueues` for `minGpuCount=2`.
- The 2-rank engine fix removes the last correctness blocker for the 2-GPU
  test; only queue capacity and the stale pending reservation remain
  operational. The 2-GPU test is ready to resubmit as soon as queue3 frees
  two A100s.

## 7. Artifacts

- `bench/mpi_h1_opt_2026-10-19/FINAL_REPORT.md` (full agent report)
- `docs/results/MPI_H1_AGENT_REPORT.md` (copy)
- `docs/examples/data/mpi_h1_opt_summary.json`
- `docs/figures/17_mpi_h1_diagnosis.png`, `18_mpi_h1_schematic.png`,
  `19_mpi_h1_final.png`, `20_mpi_2rank_fix.png`
- Final one-rank binary: `bench/mpi_h1_opt_2026-10-19/lmp_mpi_final`
  (`08cb7847…`)
- **Final 2-rank-fixed binary**: `2a999fe6…` (snapshot
  `bench/mpi_2rank_fix_2026-10-19/lmp_2rank_fixed_final`)
- Gate JSONs: `bench/mpi_h1_opt_2026-10-19/gates/gate_final/parity.json`,
  `gates/gate_fix2/`, `nvt/`; 2-rank gates
  `bench/mpi_2rank_fix_2026-10-19/gates/`
- 2-rank memory trace: `bench/mpi_h1_opt_2026-10-19/gates/sc2/memwatch.log`
- 2-rank fix report: `bench/mpi_2rank_fix_2026-10-19/FINAL_REPORT.md`
  (copy: `docs/results/MPI_2RANK_FIX_AGENT_REPORT.md`),
  `docs/examples/data/mpi_2rank_fix_summary.json`
