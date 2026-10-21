# Plan B-lite — gated replay cache, gather fusion and slice audit

**Status:** gated result on the isolated B tree (`symmetrix-xl-b` /
`lammps-b`). **Not the accepted headline.** The coordinator integrates the
accepted slices; the accepted result remains build 8 at **48.74 ms/eval**.
**Date:** 2026-10-20
**System:** 128-atom NH3 (32 molecules), full PolarMACE, FP64, A100.

## 0. Headline

| Configuration | ms/eval | Δ vs prior B default | Verdict |
|---|---:|---:|---|
| Prior B tree default (cache off, UVU x2 smem on) | 49.69 | — | — |
| Accepted build 8 (reference) | 48.74 | — | gated |
| **B1 replay cache default (cache=1, smem=0)** | **42.79** | **−6.90** | accepted |
| **B1 + B4 reverse-gather fusion (final)** | **38.66** | **−11.03** | **accepted, gated** |

Final speedup vs accepted build 8: **1.261×** (48.74 → 38.66 ms).

**2026-10-21 update — write-through fix.** The FP64 team MH-1 forward kernel
unconditionally used `+=` while `evaluate()` skipped the zero-fill under
write-through; repeated evaluations at fixed positions accumulated into the
previous result. Single-point gates could not see it. The fix branches on
`write_through` (see `MH1_TEAM_FP64.md`). Re-measured after the fix, the same
protocol gives **42.79 ms (gather0) → 38.66 ms (B4)**, three rounds,
median-of-medians: gather0 42.62 / 43.28 / 42.79; fused1 39.15 / 38.30 /
38.66. B4 = **−4.13 ms (9.6%)**. The old 40.95 ms figure was measured with the
buggy kernel; it is superseded. Repeat-evaluation drift before the fix:
−49242.46 → −48812.11 → +47.3M eV; after the fix all three evaluations agree
to 1e-15.

The B1 row is the same-binary cache+smem default (`MATOS_UVU_X2_MB_SMEM=0`);
the pre-fix B1 intermediate was 45.09 ms. Build 10's shared-memory staging
was a regression in the integration and is opt-in here.
The final configuration is `MATOS_REPLAY_CACHE=1` (default),
`MATOS_UVU_X2_MB_SMEM=0` (default), `MATOS_REV_GATHER_FUSE=1` (default);
`MATOS_E3_LINEAR_MERGED_UNPACK=0` and `MATOS_POLAR_SMALL_FUSE=0` remain
opt-in.

Binary snapshot `snapshot/lmp_b_final_wtfix` sha256
**`1e324f592198b8575e673d7a5979af331a3dd954cd009d36116f31649f5b8773`**.
The pre-fix binary was
`18d5fc94fda3df98837520b885de48956afa59fb4c81223f9fe5d3f2e7833c64`
(no longer accepted). Python module sha256
`509daed05b54db5d4c0a0bba72fc1379283dc0d9bc1db0acf7e1e43279e74ac8`
(built from the same B-tree source, Python 3.12; the fix is C++-only, so the
Python module is unchanged).

## 1. Gate suite (final binary, defaults)

The complete 1-rank suite and both LAMMPS parities were rerun on the final
build. `rc=134` is the pre-existing Kokkos teardown abort **after**
`GATE PASS`.

| Gate | Metric | Result | Verdict |
|---|---|---|---|
| module 5 | dens abs / rel, stage rel | 6.689e-8 / 4.218e-8 / 7.127e-8 | **PASS** |
| module 6 | ρ̂ rel, ΔE rel, ΔE abs | 1.270e-15, 1.018e-15, 7.772e-16 eV | **PASS** |
| module 7 | 16/16, stage rel, out | 1.174e-7 (stage), 4.546e-7 (out) | **PASS** |
| module 8 | FD max, parity max | 5.783e-4, 7.191e-7 | **PASS** |
| end-to-end | 32 frames | 32/32; dE abs max 1.034e-6 eV; dF max 1.624e-7 eV/Å | **PASS** |
| LAMMPS parity, 1 rank | dE, max dF | 2.910e-11 eV, 2.594e-14 eV/Å | **PASS** |
| LAMMPS parity, 1 rank, post-fix binary | dE, max dF | 2.910e-11 eV, 2.606e-14 eV/Å | **PASS** |
| LAMMPS parity, 2 ranks exact 2-hop | dE, max dF | 2.910e-11 eV, 1.6874754349e-6 eV/Å | **PASS** |
| repeated evaluation, fixed positions | 3 × `run 0` | dPE 0, max dF 1.0e-15 | **PASS** |

These are the build-8 metrics to round-off. B4 and B5 were additionally
checked bitwise against the prior binary with 17-digit force dumps: dPE = 0,
max dF ≤ 7.3e-16 for the fused defaults, i.e. cross-process noise. The
repeated-evaluation gate is new: the earlier suite was single-point and did
not detect the write-through bug.

## 2. Slice-by-slice A/B (same binary, interleaved 2 rounds × 12)

All arms use the same binary; only the env switch changes. GPU `flock` per
run, 1 warmup + 12 measured, median of per-round medians.

| Slice | Off (ms) | On (ms) | Saving | Decision |
|---|---:|---:|---:|---|
| **B1** replay cache | 48.48 | 45.09 | **+3.39** | **accepted, default on** |
| **B4** reverse gather fused into MH-1 reverse | 44.96 | 40.95 | **+4.01** | **accepted, default on** |

Post-fix re-measure (2026-10-21, 3 rounds × 12, same protocol):

| Slice | Off (ms) | On (ms) | Saving | Decision |
|---|---:|---:|---:|---|
| **B4** reverse gather fused into MH-1 reverse | 42.79 | **38.66** | **+4.13** | **accepted, default on** |
| **B5** small-polar zero-fill / store fusion | 41.07 | 41.06 | +0.01 | **neutral, reverted to opt-in** |
| **B3** e3-linear merged unpack | 40.93 | 46.89 | **−5.97** | **rejected, opt-in only** |

The B1 A/B is clean on an idle GPU. B3/B4/B5 were run back-to-back; the B4
`gather0` round-1 arm had one external-contention outlier (max/min 1.52) but
its median is consistent with round 0. The final full-polar check (prior vs
final) gave 44.96 vs 41.33 ms in its clean round 0; its round 1 was externally
contended (median 279.98 ms) and is excluded. A separate 432-atom round was
contended in one of two final arms (see §4).

## 3. B3 — fixed, but a measured regression

The merged-unpack path had two defects:

1. the flat-buffer cursor `merged_offset` was not rewound after the plan pass
   (illegal memory access), and
2. block writers are interleaved in instruction order, so the contribution
   ranges were not contiguous for multi-writer blocks (wrong sums).

Both are fixed in the B tree: the cursor is rewound, and instructions are
stable-sorted by output/input offset before the plan and GEMM passes, making
each block's contributions contiguous (the same construction the shared tree
uses). With the fix the path is exact: dPE = 0, max dF = 6.66e-16.

It is nevertheless **5.97 ms slower** on this workload (46.89 vs 40.93 ms).
The per-call plan construction and the single merged unpack kernel cost more
than the ~40 unpack launches it removes. It remains opt-in
(`MATOS_E3_LINEAR_MERGED_UNPACK=1`) and is not in the accepted configuration.

## 4. Cross-checks

| Configuration | Prior | Final | Saving |
|---|---:|---:|---:|
| polar-off, 128 atoms (ms/eval) | 23.69 | **19.93** | 3.76 ms (1.189×) |
| N2 432 atoms, first 10-step loop (ms/step) | 87.82 | **83.47** | 4.35 ms (clean round 0) |

The polar-off path has no polar blocks, so its gain is B4 alone; it confirms
that the gather fusion removes a backbone-wide pass. The 432-atom final
round 1 was externally contended (95.86 ms) and is excluded; round 0
(83.47 ms, matching its 84.04 ms warmup) is used.

## 5. Kernel attribution

CUPTI trace of the final configuration (`trace/final_fused`, one warm eval;
the host was externally loaded, so absolute kernel milliseconds are not used
for the wall delta):

| Quantity | build-8 trace (r3c) | cache=1 trace | final trace |
|---|---:|---:|---:|
| launches | 1228 | 1079 | **1077** |
| Kokkos launches | 916 | 807 | **805** |
| GPU busy (ms) | 43.91 | 40.74 | 33.92 |
| `polar rev replay *` marks | present | **absent** | **absent** |
| `gather edge message adjoint` | present (2.02 ms, 2 launches) | present (2.58 ms) | **absent** |

- B1 removes the forward replay: **−149 launches** and ~3.2 ms GPU busy; the
  `rev5a:cache-load` marks replace `rev5a:replay`.
- B4 removes the two gather launches and the edge-indexed message copy; the
  MH-1 reverse kernels now read `state.message_adj` through `local_targets`.
  The gather kernel alone was 2.0–2.6 ms GPU busy; the measured wall saving is
  4.01 ms because it also removes the copy's launch and memory pass.
- Top final kernels: `polar reverse uvu mode0 x2` 3.52 ms; MH-1 forward
  3.18 ms; `polar reverse uvu mode0 x1` 2.74 ms; MH-1 reverse channels
  1.80 ms; MH-1 reverse harmonics 1.69 ms; reverse convolution cutoff
  1.53 ms.

## 6. What was not landed

- **B2 host-gap / CUDA Graph**: not attempted. The reverse still contains
  `polar_global_sum_host`, host `create_mirror_view_and_copy` of dbar/sources
  and Fukui coefficients, and `ensure_view` reallocations. Capturing the
  forward/backbone alone would need those calls frozen first; no graph is
  claimed.
- **B7 MH-1 work removal**: not attempted. Inspection shows the path weight is
  already applied once per instruction/per channel block, not per entry in the
  forward scalar kernel; pre-scaling Wigner values removes only the reverse
  channel/harmonic multiplies, so the 2–4 ms audit estimate is not supported by
  the code structure. No speculative change was merged.
- **B9 SCF/k-space fusion**: no safe design within this session.
- **B5**: implemented and bitwise-correct, but neutral (+0.01 ms); left opt-in.

## 7. Remaining gap to 30 ms

**8.66 ms** (38.66 → 30 ms; 1.289×). The remaining budget is dominated by
polar pipeline ≈14.4 ms (UVU reverse x1/x2 ≈6.3 ms), MH-1 ≈6.8 ms, cuBLAS
≈3.6 ms, e3-linear ≈2.4 ms, and the host/launch gap. Closing it needs the
unlanded work removal (B7/B9) plus launch elimination (B2); the audit's
33–38 ms envelope for B-lite+B2 is consistent with what landed here
(38.66 ms without B2/B7). **20–25 ms remains not credible** under the current
FP64 exact-physics contract.

## 8. Artifacts

| Artifact | Path |
|---|---|
| Final binary + hashes | `snapshot/lmp_b_slices1.sha256`, `snapshot/source_hashes_final.txt` |
| Prior binary/source snapshot | `snapshot/lmp_b_prior*`, `snapshot/source_hashes_prior.txt` |
| A/B JSON and runs | `ab/*.json`, `ab/runs/` |
| Gate JSON/PNGs/logs | `gates/final/`, `logs/gate_final_*` |
| CUPTI traces | `trace/cache1/`, `trace/final_fused/` |
| Exactness probes | `probe/`, `logs/bitwise_new.log` |
| Summary / figure | `summary.json`, `plan_b_lite_speed.png` |
| Build scripts/logs | `build_pymodule_b.sh`, `rebuild_all.sh`, `logs/` |
