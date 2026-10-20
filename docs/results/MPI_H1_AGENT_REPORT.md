# MPI / H1 optimization round — final report (2026-10-20)

## Headline

The one-rank H1 path (`pair_style symmetrix/mace/kk polar_mpi_message_passing`)
built a local+ghost engine graph even on one rank. LAMMPS' periodic self-swaps
made that graph legal but useless: 1170 node rows (128 local + 1042 periodic
image ghosts) instead of 128, i.e. 9.1x the node-only work of the production
one-rank path. On one rank the graph is now folded onto local atoms through the
atom map (same trick as `no_domain_decomposition`), and the exchange hooks are
kept but are zero-sized because no ghost rows exist. Net effect:

| metric (128-atom NH3, 1 rank) | before | after |
|---|---:|---:|
| default `no_domain_decomposition` | 122.80 ms | 50.84 ms |
| H1 `polar_mpi_message_passing` | 418.64 ms | 50.20 ms |
| **H1 / default** | **3.41x slower** | **0.99x** |
| H1 absolute speedup | — | **8.34x** (baseline snapshot → final binary) |

The default-path gain (122.8 → 50.8 ms) is the concurrent kernel round, not
this work; the MPI change is the 3.41x → 0.99x ratio and the same-binary A/B
below (176.8 → 54.6 ms, 3.24x, one binary, one engine).

Two-rank wall-clock could not be re-validated: the **current engine source is
broken on 2 ranks** (OOM on the 2-hop graph, SIGSEGV on the H1 graph). This is
engine-side and predates these pair-style changes; details, evidence and a
patch proposal are in section 6.

## 1. Files and lines changed (pair-style only)

`pair_symmetrix/pair_symmetrix_mace_kokkos.cpp` (owned), header
`pair_symmetrix/pair_symmetrix_mace_kokkos.h` (owned). No engine file was
edited.

| what | where |
|---|---|
| opt-in `MATOS_MPI_PROFILE` stage timers (per-call graph / engine / reduce / sync) | `.h:117-130`, `.cpp:36-56, 80, 2604-2620, 2915-2960` |
| H1 exchange uses device-side Kokkos pack/unpack (`reverse_comm_device = 1`) | `.cpp:497`; `pack_forward_comm_kokkos` `.cpp:959`, `unpack_forward_comm_kokkos` `.cpp:1057`, `pack_reverse_comm_kokkos` `.cpp:1158`, `unpack_reverse_comm_kokkos` `.cpp:1279`; host fallbacks `.cpp:889/1003/1105/1210` |
| no comm-cutoff extension on one rank | `.cpp:803` |
| one-rank folded graph (`num_nodes = nlocal`, atom-map source remap) | `.cpp:2640-2645, 2741-2745` |
| exchange hooks installed but skipped when the graph has no ghost rows (LAMMPS self-swaps would otherwise write out of bounds) | `.cpp:2762-2792` |

The two-rank graph (`num_nodes = nlocal + nghost`, sources may be ghosts) is
unchanged; so is the exact two-hop mode (`polar_mpi_ghost two_hop`) and the
rejected one-hop mode (`polar_mpi_ghost one_hop`), which still runs its
original code path.

## 2. Protocol

* 128-atom NH3 (`mlip/bench/lammps_parity/polar_parity.data`), PolarMACE probe
  model, `run_style verlet/kk`, `neighbor 2.0 bin`.
* One warmup `run 1` in the process, then separate one-step `run 1` loops;
  median of the one-step samples; 2 process repetitions per config (3 for the
  baseline). A single multi-step loop is pathologically slower in *every*
  binary (old and new), so it is not used.
* Every GPU run under `flock gpu.lock`, every build under `flock build.lock`.
* A/B is reported two ways: (a) the required frozen baseline snapshot
  `lmp_mpi_baseline` = `7e9c6bec…` (built 09:22, before the kernel rounds),
  vs the final binary; (b) a **same-binary** A/B that isolates this change
  using the `MATOS_MPI_KEEP_GHOSTS` diagnostic build (folded graph vs old
  ghost graph, identical engine).

## 3. Cost decomposition (pre-fix, baseline snapshot, 1 rank)

Aggregated from `MATOS_POLAR_PROFILE=1` marks (`profile/polarprof_*`). The
engine-stage sum reproduces the wall difference (300.3 vs 295.8 ms) within
profiler-fence noise.

| component (ms/compute) | default, 128 nodes | H1 pre-fix, 1170 nodes | delta |
|---|---:|---:|---:|
| entry (`eval:enter`) | 2.02 | 2.19 | +0.18 |
| forward backbone + 2 H1 syncs | 10.70 | 42.93 | +32.23 |
| polar forward (stages 2-7, coulomb, LE) | 19.21 | 95.65 | +76.44 |
| polar reverse (uvu, replay, embedding, mlp) | 62.44 | 225.64 | +163.20 |
| backbone reverse + 1 H1 sync | 29.57 | 57.77 | +28.20 |
| **sum** | **124.0** | **424.2** | **+300.3** |

The dominant term is node-scaled polar work: Coulomb, reverse UVU, replay and
embedding-reverse each grow 5-7x with the 9.1x node count. With the folded
graph the same decomposition is flat: all five buckets differ by <1 ms between
H1 and the default path (final profiler, `profile/final_*`).

Internal timers on the final H1 one-rank path:
`graph = 0.19 ms`, `engine = 55.5 ms`, reductions `= 0.17 ms`, forward syncs
`0.001 ms` each (2 per compute), reverse sync `0.07 ms` (1 per compute). The
H1 hooks are 0.13% of the compute on one rank. The same-engine A/B still shows
+0.9 ms for H1 (54.62 vs 53.71, one round); the final two-round medians put H1
at 0.99x default (50.20 vs 50.84), so the difference is within run noise and
pair-side reduction structure, not communication.

## 4. A/B tables

### Baseline snapshot vs final binary (required)

| config | baseline `7e9c6bec` | final `08cb7847` |
|---|---:|---:|
| default 1-rank | 122.80 ms (reps 122.8 / 132.8 / 122.2) | 50.84 ms (reps 51.27 / 50.41) |
| H1 1-rank | 418.64 ms (reps 418.6 / 487.2 / 417.0) | 50.20 ms (reps 50.69 / 49.71) |
| H1 2-rank | 794.70 ms (reps 830.2 / 759.2) | **blocked** (engine, §6) |
| exact 2-hop 2-rank | 2536.29 ms (1 rep; 2nd rep OOM even on baseline) | **blocked** (engine OOM) |

One baseline rep of each config was noisy (GPU contention); medians are shown
and all reps are in `runs/ab_baseline2/`, `runs/final_bench/`, `runs/round0/`.

### Same-binary A/B of this change (engine held fixed, sha `5d2bdc94`)

| config | ms/compute |
|---|---:|
| default 1-rank (reference) | 53.71 |
| H1 1-rank, folded 128-node graph | 54.62 |
| H1 1-rank, pre-fix 1170-node ghost graph | 176.77 |
| **speedup of the fix** | **3.24x** |

This is the cleanest attribution: the old graph costs 3.24x even after the
device-staged exchange, and the fix removes essentially all of it.

### Two-rank cost structure (pre-fix baseline, per rank, ms/step)

H1 2-rank 794.7 ms with ~64 local + ~836 ghost rows per rank. Dominant engine
stages on rank 0: Coulomb 111.4, reverse UVU 98.4, replay 54.4 x2, pipeline end
66.5, embedding-reverse 50.4 x2, mlp-reverse 28.4 x2, UVU product 20.5 x2,
backbone reverse 98.7. Each stage is ~2x the one-rank H1 value although the
per-rank graph has 23% fewer nodes: both ranks run the whole node pipeline over
their ghost copies on the *same* GPU. The exchange itself is a small fraction
(forward hook ≈7 ms/sync); duplicated ghost node work dominates. Exact 2-hop is
3.2x slower still because it also carries ~2500 nodes and 75-78k edges per rank.

## 5. Gates (final binary `08cb7847…`)

| gate | result | verdict |
|---|---|---|
| H1 1-rank vs default 1-rank, 128-atom single point | dE = 0.0 eV, max dF = 1.41e-15 eV/Å | PASS (tol 5e-5 / 1e-4) |
| exact 2-hop 1-rank vs default | dE = 0.0 eV, max dF = 1.54e-15 eV/Å | PASS |
| device pack/unpack exercised through real ghost self-swaps (1170-node graph, proof build) | max dF = 1.03e-15 eV/Å vs reference | PASS |
| 20-step NVT, H1 1-rank vs default 1-rank | 128 atoms, no lost atoms, max position drift 4.74e-14 Å, max dF 4.70e-14 eV/Å, identical thermo trajectory | PASS |
| H1 2-rank parity | SIGSEGV on my builds; `cudaErrorIllegalAddress` on kernel build `86bb29b1` | BLOCKED (engine) |
| exact 2-hop 2-rank parity | 1.34 GiB CUDA OOM on my builds; `cudaErrorIllegalAddress` on `86bb29b1` | BLOCKED (engine) |

Gate artifacts: `gates/gate_final/parity.json`, `nvt/{h1,default}/`,
`profile/`, `gates/gate_fix2/h1_1rank/forces_keepghosts.dump` (device-staging
validation build).

The 20-step NVT loop times (H1 6.81 s, default 4.85 s) are dominated by the
known multi-step slowdown that affects every binary, not by H1 overhead: a
separate 5-step run measured 2.46 s (H1) vs 2.49 s (default) with identical
thermo output, and the one-step protocol above shows 50.20 vs 50.84 ms.

## 6. Two-rank blocker in the current engine source (needs the kernel owner)

While the accepted 09:22 binary runs two ranks, every build from the current
engine source (kernel-round snapshots `lmp_r2`…`lmp_r5`, my builds, and the
kernel agent's round-3 build `86bb29b1` at 02:18) fails:

* **Exact/default 2-hop path (production reference):** for my builds, peak GPU
  memory 80,966 MiB of 81,920 MiB; abort on a 1.28-1.34 GiB allocation. An
  `LD_PRELOAD` cudaMalloc tracer shows six multi-GB workspaces per rank:
  2.03 + 2.13 + 3.20 + 3.36 + 5.12 + 5.37 GiB ≈ 21 GiB/rank before the rest,
  allocated from
  `MaceNonlinearKokkosT::compute_node_energies_forces` →
  `ensure_view<double**,LayoutRight>` and
  `AffineMLPKokkosT::prepare` / `forward_impl`.
  Graph at 2 ranks: nodes 2481/2581, edges 74472/78156 (rank 0/1).
* **Kernel round-3 build (`86bb29b1`, 02:18, includes this pair style):** both
  the default 2-hop path and the H1 path now abort with
  `cudaErrorIllegalAddress` ("illegal memory access") instead of OOM, so the
  workspace sizing changed but the multi-rank path is still not runnable.
  The default path does not execute any of the H1 pair code, so this failure
  is engine-side.
* **H1 path on my builds:** ~3.6 GB peak, SIGSEGV after graph construction.
  The same input on the pre-change pair style with the current engine also
  exits SIGSEGV, and the 09:22 binary runs it, so the regression is
  engine-side.

Patch proposals (not applied — files owned by the kernel agent):

1. Size the `ensure_view` workspaces in `compute_node_energies_forces` (and the
   `AffineMLPKokkosT` forward buffers) by owned/local rows and local edges, or
   add a capacity guard that scales them down for multi-rank graphs. The
   current sizing behaves super-linearly in nodes/edges and makes the exact
   path unusable above one rank on an 80 GB GPU.
2. Add an extent assertion at the multi-rank engine entry that every per-node
   and per-edge view covers `num_nodes`/`num_edges` before the first kernel;
   the H1 SIGSEGV is consistent with an extent mismatch introduced by the new
   multi-rank code.

Until then, H1 2-rank and exact 2-hop 2-rank cannot be measured or gated, and
the device-staged exchange (validated on one rank with real self-swaps) remains
unvalidated over real MPI.

## 7. Rejected ideas (measured, not pursued)

* **Ghost compaction** (keep only ghost rows that are edge sources): a
  geometric reconstruction of the 2-rank decomposition shows 861 ghosts within
  the 8 Å comm cutoff and **all 861 are within r_cut = 6 Å of a local atom**
  (0% unused), so compaction cannot help.
* **Skipping the layer-forward sync:** required; ghost features feed the
  neighbour interaction.
* **Narrower exchange:** the engine passes the full 512-wide node feature view
  and the next `linear_up` consumes all columns.
* **Reducing pair-side all-reduces:** the only global sum is the engine's
  k-space callback; the pair-side reductions are device-local and measure
  0.17-0.22 ms (<0.5%).
* **Layer sync removal at layer 0 / merging layers:** would need engine-side
  changes; forward ghost node work is the real target (proposal below).
* **Engine-side, additional:** skip node-only *forward* stages for rows
  `>= owned_nodes` (their features are overwritten by the next sync) and reuse
  the owner's activations. This would cut the one genuinely redundant part of
  the multi-rank H1 pipeline; reverse stages on ghost rows are still needed.

## 8. What is inherent on one shared A100

All ranks share one GPU, so multi-rank cannot win wall-clock by construction.
More specifically, the H1 design runs the complete node pipeline (including the
polar k-space Coulomb chain) on every ghost row on every rank. On the 128-atom
system each 2-rank process carries ~836 ghost rows against ~64 local rows, so
the aggregate node work is ~1.5x the one-rank H1 graph and the GPU serializes
it: 2 ranks measure ~2x the one-rank stage times despite fewer nodes per rank.
No pair-style change can remove that duplication without engine support; the
only pair-side win was the one-rank folded graph, which is now in place.

## 9. Artifacts and final binary

Final shared `lmp`
(`/workspace/group/shared/bza/external-repos/symmetrix-xl/symmetrix/build-lammps-aacd036df381800a/lmp`):
**sha256 `08cb7847ffb1a804306907c56fcee800fd795e15da63105ed6cc296f8a7a69c8`**
(197,223,288 bytes), snapshot `lmp_mpi_final`.

Build provenance (engine files at build time, unchanged across the build):

```
7e5f6e056ed7e8470078c9a8967df3261dc8a4fcf34fb01c0d0215cee0451285  mace_nonlinear_kokkos.hpp
c66e3eb980b03d2bbefc084788f5291e2340cad96e69e807439e37042a6e4040  mace_nonlinear_kokkos_impl.tpp
6f74b3908763e31febd9b45bc9fd68a93524fad91f6b20b2f220f5838a0bbdc7  mace_polar_impl.tpp
af80930fe5be7aad31b442c2f6d1910cf7994dcbba6ebe1457273d6a179ee76d  e3nn_kokkos.cpp
5a2992b8d08b42363031332c019b49bf5d3c21e10481c70d303dacb5084070d9  e3nn_kokkos.hpp
```

The concurrent kernel agent may relink this path again; `lmp_mpi_final` is the
exact binary measured here.

Artifacts under this directory:

* `FINAL_REPORT.md`, `summary.json`, `mpi_h1_summary.png` (12.0x5.2 in, dpi 160)
* `bench.py`, `parity_gate.py`, `make_summary.py`
* `runs/{round0,ab_baseline,ab_baseline2,ab_current_fixed,ab_current_keepghosts,final_bench}/`
* `profile/{polarprof_*,polarprof_h1_2rank,final_default,final_h1,fix1_h1_1rank}/`
* `gates/{gate_fix1,gate_fix2,gate_final}/`, `nvt/{h1,default}/`
* `logs/build_*.log`, `logs/engine_hashes_*.txt`
* `lmp_mpi_baseline` (`7e9c6bec…`), `lmp_mpi_final` (`08cb7847…`)
* `gates/sc2/memwatch.log` (2-rank GPU memory trace, peak 80,966 MiB)
* `cudalloc_trace.so` + `gates/sc2/err.trace` (LD_PRELOAD cudaMalloc tracer
  used to locate the multi-GB workspaces)
