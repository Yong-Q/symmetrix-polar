# MPI 2-rank engine fix — final report (2026-10-19)

## Headline

Both 2-rank paths now run to completion on one A100 and pass the force gate:

| 2-rank config | before | after |
|---|---|---|
| default exact 2-hop (`no_mpi_message_passing`) | OOM at a 1.279/1.343 GiB allocation, peak 80,966 MiB | **rc=0**, dE = 0.0 eV, max dF = 1.687e-6 eV/Å |
| H1 (`polar_mpi_message_passing`) | SIGSEGV (host) / `cudaErrorIllegalAddress` (round-3 build), peak 3.6–4.0 GiB | **rc=0**, dE = 0.0 eV, max dF = 1.01e-14 eV/Å |

The two failures had different causes:

1. **Default 2-rank OOM — duplicate edge workspaces, not super-linear
   sizing.** The exact 2-hop graph is large by design (2481/2581 nodes,
   74,472/78,156 edges per rank). Every per-edge workspace is linear in
   `edges`; the defect was that the materialized reverse kept two full-size
   buffers whose live ranges do not overlap with their forward counterparts:
   `state.edge_message_adj` (a pure gather of `message_adj`, equal in shape to
   the already-dead forward `state.edge_messages`) and `state.raw_weight_adj`
   (exactly `weight_adj * cutoff`). Together they added 8.9–9.3 GiB per rank
   and pushed the peak to the A100 limit.
2. **H1 SIGSEGV — communication-width mismatch.** `node_feature_width()`
   returned the 512-wide embedding width. The pair style copies it into
   `comm_forward`/`comm_reverse`, and LAMMPS sizes its send/recv buffers once,
   in `Comm::init`. The layer-1 H1 hook exchanges the 2048-wide product-block
   output, so `pack_forward_comm`/`unpack_forward_comm` overran the 512-wide
   LAMMPS buffers. Depending on the build this corrupts the host heap
   (SIGSEGV) or trips an illegal device access at the following
   `Kokkos::deep_copy`. The old pair style declared
   `num_LM*num_channels = 8192`, which is why it never hit this.

## 1. File:line changes

`libsymmetrix/source/mace_nonlinear_kokkos.hpp`

* **379-395** — `node_feature_width()` now returns
  `max(node_embedding.output_dimension(), products[*].output_dimension())`
  (512 → 2048 for this model) and documents that LAMMPS sizes its comm
  buffers from this value at setup. This is the H1 fix.
* **6** — added `<algorithm>` for `std::max`.

`libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp`

* **4237-4279** — multi-rank entry guard: when a layer sync hook is installed
  (and execution is not the rejected generated-node program), assert that
  `features`, `node_energies`, `readout_contribution` cover `num_nodes` and
  `edge_harmonics`/`cutoffs`/`radial` cover `edges` before the first kernel.
* **4263-4279, 4474-4477, 7490-7493** — per-layer `validate_sync_view`
  asserts `extent(0)==num_nodes` and `extent(1)<=node_feature_width()` on
  every forward/reverse hook. This guard would have converted the H1 crash
  into a diagnostic exception.
* **7210-7216** — reverse `edge_message_adj` reuses
  `state.edge_messages_storage` (the forward messages are dead after the
  gather; the streamed-edge path at 7033 already used this invariant). Removes
  one `[edges, message_width]` buffer per layer.
* **7287-7293** — `raw_weight_adj` aliases `state.weight_adj`; the scaled
  adjoint is written in place (each element is read before it is rewritten)
  and the MLP reverse consumes `state.weight_adj`. Removes one
  `[edges, weight_size]` buffer per layer.
* **7360-7363** — the two `convolution_weights.reverse*` calls now take
  `state.weight_adj`; the `apply_cutoff` copy is skipped when the views alias.
* **7366** — the non-fast `reverse` path still recomputes the MLP from
  `edge_features`; no tape semantics changed.

No pair-style, `e3nn_kokkos.cpp`, FP64, physical-term, or gate change. The
accepted 1-rank math is untouched (1-rank forces differ from the accepted
dump by ≤1.7e-15 eV/Å, i.e. round-off).

## 2. Memory before/after (per rank, default exact 2-hop)

The node workspaces are `[num_nodes, width]` and are small (≤0.2 GiB each;
`num_nodes` is 2481/2581) — no node workspace participates in the OOM. All
rows are genuinely needed: ghost rows are edge sources, and in the exact
2-hop path the ghost receivers' own messages must be computed to make the
model exact, so no ghost rows were dropped and no per-node view was shrunk.
Every per-edge workspace is linear in the graph's `edges`; all 74,472/78,156
edges are processed. The defect was duplicate buffers, not a global or
squared extent.

Allocation trace (`LD_PRELOAD` cudaMalloc tracer, threshold 1 GiB) at the
2-rank graph (rank 0 / rank 1 edges 74,472 / 78,156):

| workspace | before | after |
|---|---:|---:|
| `edge_messages` layer 0, `[edges, 8192]` | 5.117 / 5.370 GiB | 5.117 / 5.370 GiB (reused for the adjoint) |
| `edge_messages` layer 1, `[edges, 5120]` | 3.198 / 3.357 GiB | 3.198 / 3.357 GiB (reused for the adjoint) |
| `edge_message_adj` layer 0 | 5.117 / 5.370 GiB | **0 (same storage)** |
| `edge_message_adj` layer 1 | 3.198 / 3.357 GiB | **0 (same storage)** |
| affine-MLP value + adjoint | 2×2.029 / 2×2.129 GiB | unchanged |
| `raw_weights` + `weights`, layer 0 `[edges,2048]` | 2×1.279 / 2×1.343 GiB | unchanged |
| `weight_adj` + `raw_weight_adj`, layer 0 `[edges,2048]` | 2×1.279 / 2×1.343 GiB | **1×** (aliased) |

Layer 1 uses `[edges,1280]` for the same buffers (0.84 GiB per rank); it is
below the 1 GiB tracer threshold but scales with the same factors.

Peak GPU memory:

| | before | after |
|---|---:|---:|
| 2-rank default, total | 80,966 MiB (OOM; reproduced 78,818) | **63,088 MiB on an idle GPU** (runs with ~6–7 GiB of external GPU usage measured 58,138–64,319 MiB incremental) |
| 2-rank default, per rank | ≈40.5 GiB | **≈30.8 GiB** |
| 2-rank H1 | SIGSEGV at 3.6–4.0 GiB | **6,120 MiB (6.0 GiB)** |
| 1-rank default | 3.1 GiB (accepted binary) | **2,193 MiB** |
| 1-rank H1, same ~6.2 GiB background | 3,129 MiB incremental (08cb) | **2,739 MiB incremental** |

The 2-rank default peak fell by **17,878 MiB (−22%)** and now leaves ~18 GiB
of headroom on one 80 GiB A100; on two GPUs each rank (~31 GiB default, ~3 GiB
H1) has ample headroom.

## 3. Two-rank parity and NVT

Reference: 1-rank `default` dump on the fixed binary (PE
−49242.4564153149 eV); tolerance dE ≤5e-5 eV, max dF ≤1e-4 eV/Å.

| config | atoms | dE (eV) | max dF (eV/Å) | verdict |
|---|---:|---:|---:|---|
| 1-rank default | 128 | 0.0 | 1.01e-14 | PASS |
| 1-rank H1 | 128 | 0.0 | 1.00e-14 | PASS |
| **2-rank default exact 2-hop** | 128 | **0.0** | **1.687e-6** | **PASS** |
| **2-rank H1** | 128 | **0.0** | **1.01e-14** | **PASS** |

The 2-rank default value reproduces the accepted pre-fix exact-2-hop result
(1.6874754352676705e-06 eV/Å) to 2e-15; the H1 value is 7 orders of magnitude
inside the gate and matches the accepted 1.4e-15 level. The official gate
runners also pass on the fixed binary:

| gate | result | verdict |
|---|---|---|
| `polar_lammps_parity_2rank` (exact 2-hop, 2 ranks, fixed binary vs engine) | dE 2.910e-11 eV, max dF 1.687475435e-6 eV/Å, `GATE PASS` | **PASS** |

20-step NVT (300 K, 0.5 fs, 128 atoms), all `rc=0`, 128/128 atoms at the end,
no lost atoms, finite thermo:

| config | steps | Etot span (eV) | PE range (eV) | T mean (K) |
|---|---:|---:|---|---:|
| 2-rank default | 20 | 0.0603 | −49242.76 … −49240.53 | 242.9 |
| 2-rank H1 | 20 | 0.0603 | −49242.76 … −49240.53 | 242.9 |
| 1-rank H1 | 20 | 0.0603 | −49242.76 … −49240.53 | 242.9 |

H1 2-rank vs H1 1-rank max thermo difference over the 20 steps:
**1.0e-10**; default 1-rank vs H1 1-rank: 2.0e-11.

## 4. One-rank regression

| gate | result | verdict |
|---|---|---|
| LAMMPS parity 1 rank (fixed binary vs accepted engine snapshot) | dE 2.910e-11 eV, max dF 2.604e-14 eV/Å, `GATE PASS` | **PASS** |
| module 5 (density/stages) | 12 density items, 11 stage items all in tolerance, `GATE PASS` | **PASS** |
| module 6 (Coulomb/SCF) | ρ̂ rel 1.42e-15, dE rel 1.09e-15, abs 5.55e-16 eV | **PASS** |
| module 7 (local electron) | 16/16, out max 4.55e-7 | **PASS** |
| module 8 (forces) | FD max 5.783e-4, parity max 7.191e-7 | **PASS** |
| 1-rank default/H1 force dump vs accepted `final_1rank.forces.dump` | max dF 1.01e-14 / 1.00e-14 eV/Å, pos drift 0.0 | **PASS** |

The known Kokkos teardown abort (exit 134 after `GATE PASS`) is unchanged and
does not affect the verdicts.

## 5. Wall time (one shared A100; still slower than 1 rank by design)

One warm-up + three one-step runs, median of steady samples (measured on the
intermediate fixed build `e4597810`, whose code paths are identical to the
final rebuild):

| config | s/step | ms/eval |
|---|---:|---:|
| 1-rank default (fixed) | 0.137 | 137 |
| 1-rank H1 (fixed) | 0.137 | 137 |
| **2-rank default** | **1.478** | **1,478** |
| **2-rank H1** | **0.518** | **518** |

The H1 2-rank measurement was taken with an idle GPU (4 MiB background). The
default 2-rank samples had ~7 GiB of background usage from another GPU
process; the median still excludes the one 2.2 s contention outlier. The
1-rank numbers in this session are ~2.7× the MPI report's 50.8 ms, but the
accepted `08cb` binary also measures 0.141 s/step under the same background
load, and the source-baseline binary measures 0.150/0.152 s: this fix adds no
1-rank cost, the difference is session/GPU contention. On one GPU, 2 ranks
share the SMs and duplicate ghost work; the 2-rank run is for
correctness/portability, the speedup target is one rank per GPU.

A later bench of the final rebuild under heavy external GPU contention
measured 0.94 s/step (H1 2-rank, 20-step NVT) and 2.27 s/step (default
2-rank); those numbers are not comparable to the clean one-step table above.

## 6. Final binary and artifacts

Final shared `lmp`
(`.../symmetrix/build-lammps-aacd036df381800a/lmp`), snapshot
`lmp_2rank_fixed_final`:

```
sha256 2a999fe6202ca612a79762b9ec0a711b41431eaffdc2a6d956be1baab3de6857
size   197,800,320 bytes
```

Source hashes at the final build:

```
1bf0f36fdb9336ff2be49a568ad207c214b9bf1c6cb2b99b93e966ecec182a9f  mace_nonlinear_kokkos.hpp
546efe04da021031e546e3a7641c045e38ca850b0b215d5c1f2ccebf6f7e3b69  mace_nonlinear_kokkos_impl.tpp
c094fe492549820677f4c8249f8de7d0d7c2130be01c7fcc91bd9fb3a6a55f64  mace_polar_impl.tpp
b91ee03fb13baafdf060ac1600ee8b5b23c8b8d904115a1df6ebec90bb55db35  e3nn_kokkos.cpp
8d9e83d795d0a9c982a48f1e817d71e0003db2163e4075b1692446d4bd406e92  e3nn_kokkos.hpp
```

The first fixed build (`e4597810`, from the same `node_feature_width`/
workspace changes but before the concurrent round-3 source update) produced
identical 1-rank and 2-rank energies/forces to the final rebuild; its
artifacts are kept as `lmp_2rank_fixed`.

Snapshots: `lmp_2rank_baseline` (`08cb7847…`, pre-fix accepted binary),
`lmp_src_baseline` (`bdb2ec19…`, current shared source without this fix).

Artifacts under this directory:

* `FINAL_REPORT.md`, `summary.json`, `mpi_2rank_fix.png`
* `run_case.sh`, `run_trace.sh`, `analyze.py`, `make_summary.py`
* `runs/{fix_default_1rank,fix_h1_1rank,fix_default_2rank,fix_h1_2rank}`
* `runs/{fix_nvt_default_1rank,fix_nvt_h1_1rank,fix_nvt_default_2rank,fix_nvt_h1_2rank}`
* `runs/{bench_default_1rank,bench_h1_1rank,bench_default_2rank,bench_h1_2rank,bench_*_src}`
* `runs/{src_default_2rank,src_h1_2rank}` — pre-fix reproductions
* `runs/{fix2_*,final_clean_*,final_trace_*,final_bench_*}` — final rebuild
* `runs/{trace_default_2rank,fix_trace_default_2rank,src_trace_default_1rank,fix_trace_default_1rank}` — cudaMalloc traces
* `runs/{base_default_1rank,base_h1_1rank,ref_h1_2rank_0922}` — accepted-binary references
* `gates/polar_lammps_parity_{1rank,2rank}.json`,
  `gates/final_polar_lammps_parity_{1rank,2rank}.json`,
  `gates/polar_module{5,6,7,8}_gate*`
* `src_snapshot/` — exact source files built by the final rebuild
* `logs/build_src_baseline.log`, `logs/build_fix.log`,
  `logs/build_fix_final.log`, `logs/gate_*.log`,
  `logs/final_gate_*.log`, `logs/engine_hashes_*.txt`

## 6b. Source drift during the fix

The shared tree was concurrently updated by the round-3 kernel work while this
fix was being validated. The first fixed build (`e4597810`) used
`hpp 1bf0f36f` / `impl cf9a1328` (our edits) but an older
`mace_polar_impl.tpp`. The tree then stabilized at
`hpp 1bf0f36f` / `impl 546efe04` / `polar c094fe49` / `e3nn b91ee03f`;
the final binary `2a999fe6` was rebuilt from that state (the exact files are
in `src_snapshot/`) and re-validated: 1-rank and 2-rank parity are identical
to `e4597810` to the last digit, both official LAMMPS parity gates pass, both
2-rank NVT paths complete, and the clean 2-rank default peak is 63,088 MiB.

While this report was being written, `e3nn_kokkos.cpp` changed twice more
(`b91ee03f` → `366bc331` → `03973af9`, 06:10–06:26) as the round-3 agent
continued integrating. An attempted rebuild during that window has ambiguous
provenance, so the shared `lmp` was restored to the validated `2a999fe6`
binary. **The two source changes that constitute this fix
(`node_feature_width` and the workspace reuse/alias blocks) are unchanged in
the tree.** Whoever produces the final round-3 integration binary must
rebuild and re-run the 1-rank and 2-rank parity gates; this fix is
independent of the e3nn kernel slices and no further engine change is needed
for either 2-rank path.

## 7. Remaining blocker for the 2-GPU test

None from the engine. Both 2-rank paths are runnable and gated. The 2-GPU test
must run **one rank per GPU** (e.g. `CUDA_VISIBLE_DEVICES=0,1` with the
launcher binding rank 0→GPU 0, rank 1→GPU 1); `--bind-to` / `numactl` or
`mpiexec` device binding is launcher-specific. Expected per-rank footprint is
~31 GiB for the default exact 2-hop path and ~3 GiB for H1, well inside an
80 GiB A100. The default path's one-GPU 2-rank slowness is inherent (full
2-hop graph on both ranks); H1 is the path intended for the scaling study.
