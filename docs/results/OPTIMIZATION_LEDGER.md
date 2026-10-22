# Optimization ledger — every reported number and its source

**Date:** 2026-10-21
**Convention:** `µs/atom = 1000 × ms/eval / atoms`. All rows are 1,296-atom
H₂O, full PolarMACE, FP64, one A100-80GB, unless stated otherwise.
**Protocol:** same-binary interleaved A/B, 1 warm-up + 4 measured `run 1`,
median of per-round medians, `flock` on the shared GPU lock.

This file exists so that every number in
[`docs/report/SYMMETRIX_POLAR_REPORT_2026-10-21.md`](../report/SYMMETRIX_POLAR_REPORT_2026-10-21.md)
maps to a committed source. Raw logs and CUPTI traces live under the
gitignored `bench/` tree; the extracted summary JSONs are committed under
[`docs/results/data/`](data/).

## 1. Optimization ladder

| Step | ms/eval | µs/atom | Source |
|---|---:|---:|---|
| fused MH-1 baseline | 599.54 | 462.6 | `MH1_TEAM_FP64.md` §"Measured result" |
| + FP64 team MH-1 | 403.97 | 311.3 | `MH1_TEAM_FP64.md` §"Measured result" |
| + B4 reverse-gather, write-through fixed | 348.97 | 269.3 | `MH1_TEAM_FP64.md` §"B4 + team"; 348.97/1.296 = 269.27 |
| + message-gather tile | 333.17 | 257.1 | `README.md`; `data/ab_gather_tile_1296.json` |
| + edge-feature tile | 320.97 | 247.7 | `README.md`; `data/ab_edge_features_1296.json` |
| + path-grouped MH-1 forward | 299.15 | 230.5 | `README.md`; `data/ab_path_grouped_forward_1296.json` (+ clean sandbox `data/ab_path_grouped_forward_sandbox.json`) |
| + edge-up/scatter tile | 290.77 | 224.1 | `README.md`; `data/ab_edge_up_scatter_1296.json` |
| **+ A1 UVU tile (main, gates PASS)** | **266.21** | **205.4** | `UVU_TILE_MAIN_1296.md`; `UVU_TILE_MAIN_GATES.md` |
| **+ N1 grouped FP64 UVU GEMM (generic, gates PASS)** | **189.30** | **146.1** | `bench/polar_uvu_gemm_engine_2026-10-21/ab/ab_1296_gemm.json`; `parity/parity_{128,1296}_gemm.json` |

The ladder is cumulative through the edge-up/scatter row. A1 is a new
1,296-atom step on the current main tree, not a 128-atom Plan B slice: baseline
`lmp_main_e0a961d` 290.196 ms (223.92 µs/atom) → candidate `lmp_uvu_tile`
266.214 ms (205.41 µs/atom), **−8.26%**, interleaved A/B 2×4 samples with
disjoint ranges. A1 patch `patches/uvu_tile_32x32.patch` (actual default tile
32×16); candidate sha256 `a9ce6481…`; source-matched gates module 5–8, 32/32,
1/2-rank parity all PASS.

The N1 row is measured on the S1.1+A1 tree; its same-run, same-binary generic
baseline is 255.35 ms (197.0 µs/atom), so the interleaved A/B ratio is
**0.7413 (−25.87 %)**. The materialized N1 arm is 201.91 ms (155.8 µs/atom,
0.7557×). Parity: 128 atoms dPE = 0 / max dF 1.8e-15; 1,296 atoms dPE = 0 /
max dF 5.8e-13 eV/Å (gate 1e-10); the 1,296 repeat-eval 1-ULP drift is the
pre-existing S1.1 behavior. Binary `lmp_gemm` sha256 `093982d5…7fa8a03`.
The A/B `peak_vram_mib` field is a whole-GPU `nvidia-smi` sample, not a
per-process value; do not use it.

5,184-atom N1 smoke (same binary, `run 0` + two `run 1`, 5 PE evaluations):
pre-N1 best `base_vec1` **986.28 ms / 190.25 µs/atom / 19.21 GiB**; N1 default
**731.46 ms / 141.10 µs/atom / 19.16 GiB**; N1 vec1 **730.49 ms / 140.91
µs/atom / 19.16 GiB**; PE drift 0 in all arms. N1 vs pre-N1 best = **0.7406×
(−25.94 %)**, and 2.12× faster than the old default (1,549 ms). Source:
`bench/polar_uvu_gemm_engine_2026-10-21/smoke5184/smoke_5184_gemm.json`.

**128-atom Plan B-micro headline (not a 1,296 row):** 30.33 ms/eval, binary
`7061ecb0…`, all gates + repeated-eval PASS; source `PLAN_B_MICRO_FINAL.md`.
The earlier 30.56 ms Plan B-30ms and 30.60 ms pre-fix micro screen are
historical.

The 230.5 and 224.1 rows use the stable candidate rounds; the baseline rounds
in those two A/B files include externally contended samples (see
`data/README.md`), and the clean baseline rounds agree with the preceding
ladder value (320.5 and 297.9 ms).

## 2. Per-kernel claims

| Claim | Value | Source |
|---|---:|---|
| message gather per launch | 10.26 → 3.50 ms (−66%) | `data/kernel_labels_1296.json`; traces `r3e_b4team1296`, `r3e_msgtile1296` |
| edge-feature write amplification | ~32× | `data/kernel_labels_1296.json`; geometry 32 lanes × 8 KB rows |
| edge-up gather | 7.81 → 1.70 ms/eval | `data/kernel_labels_1296.json` |
| scatter edge-up adjoint | 5.48 → 1.90 ms/eval | `data/kernel_labels_1296.json` |
| forward tensor product | 39.52 → 17.77 ms/eval (2.22×) | `data/profile_path_grouped_forward.json`; `data/ab_path_grouped_forward_sandbox.json` |
| polar-off backbone | 151.98 → 135.24 µs/atom | `data/ab_path_grouped_forward_sandbox.json` |
| remaining hotspots (pre-opt3) | polar UVU 105.83 ms/eval; MH-1 reverse 76.73; node/edge 91.87; cuBLAS 35.83; Ewald 9.56; host gap 8.65 | `data/kernel_labels_1296.json` (all /1.296 for µs/atom) |
| post-N1 GPU busy | 173.18 ms/eval; tensor 65.85, backbone 28.88, cuBLAS 25.11, polar 20.24, MLP 19.69, e3 13.16 | `bench/perf_decomp_1296_2026-10-21/analysis/n1_generic_1296_summary.json`; trace tag `n1_generic_1296` |
| N1 polar-pipeline effect | 87.95 → 20.24 ms/eval (−77 %) | pre/post traces; microbench predicted 22.03 |

The CUPTI trace covers LAMMPS setup plus one run, so per-launch values are
the stable comparison; per-eval values are taken from the A/B or the
per-eval stage profile.

## 3. Correctness gates

| Gate | Result | Source |
|---|---|---|
| LAMMPS 1-rank parity | dE 2.910e-11 eV, max dF 2.58e-14 eV/Å | `data/gate_1rank_main_b4team.json` |
| LAMMPS 2-rank exact | max dF 1.687e-6 eV/Å | `data/gate_2rank_wtfix.json` |
| 10-step NVE MD | max Δpos 9.8e-15 Å, max ΔF 1.4e-14 eV/Å | `data/gate_md_10step_128.json`; `MH1_TEAM_FP64.md` §"validation" |
| repeated evaluation (3 × run 0) | PE −49242.4564153149 three times; dF 1.0e-15 | `MH1_TEAM_FP64.md` §"Write-through bug" |
| module 5–8, 32/32 end-to-end | PASS | `PLAN_B_LITE_FINAL.md` §1 |

## 4. Comparison numbers

| Number | Derivation | Source |
|---|---:|---|
| PyTorch PolarMACE 734.4 µs/atom | 94.0 ms / 128 atoms × 1000 | `README.md` (PyTorch reference 94.0 ms) |
| our 128-atom 302.0 µs/atom | 38.66 ms / 128 × 1000 | `README.md` (Plan B-lite) |
| Plan B-full 244.8 µs/atom | 31.33 ms / 128 × 1000 | `PLAN_B_FULL_FINAL.md` |
| linear 9.69 µs/atom | 12.56 ms / 1,296 × 1000 | `PAPER_COMPARABILITY.md` (8.2/12.6 ms row) |
| linear 4.47–4.53 vs paper 4.50 | direct/JIT at 5,184 atoms | `PAPER_COMPARABILITY.md` |
| ML-IAP+cuEq 13.99 | paper baseline at 5,184 atoms | `PAPER_COMPARABILITY.md` |
| RuNNer 1.92 / 0.051 µs/atom | 0.03 / 1.12 ns/day at 1.5M atoms, 1 fs | Knoll et al., arXiv:2607.17978 (external, not our measurement) |

## 5. Negative results

| Experiment | Result | Source |
|---|---|---|
| 1-step SCF | dE 1.2705 eV, max dF 0.1013 eV/Å; FAIL | `data/scf1_probe.json` |
| team channel vec4 | 346.6 → 379.4 µs/atom | `data/ab_team_vec4_1296.json` |
| team size 32/64/128/256 | 267.9 / 267.8 / 270.3 / 279.8 µs/atom | `data/sweep_team_size_1296.json` |
| msg_vec=1, scalar/chunked cutoff | 294.3 / 284.6 / 295.3 µs/atom | `data/sweep_msg_vec_cutoff_1296.json` |
| polar-off team size | 153.0 / 152.0 / 155.3 / 163.6 µs/atom (32/64/128/256) | `data/sweep_team_size_polar_off_1296.json` |
| S2 receiver-owned generic forward | parity PASS (128/1,296; max dF 7.5e-14), but **2.36× slower**: 188.35 → 444.21 ms/eval (145.3 → 342.8 µs/atom); not promoted, opt-in stays off | `bench/polar_uvu_gemm_engine_2026-10-21/ab_s2/ab_1296_s2.json`; `parity_s2/parity_{128,1296}_s2.json` |
| Fused reverse MH-1 (one sparse pass) | parity PASS (128/1,296; max dF 5.7e-14), but **1.12× slower**: 188.54 → 211.10 ms/eval (145.5 → 162.9 µs/atom); per-thread partials + 16 team reductions outweigh the saved harmonics pass; not promoted, `MATOS_MH1_REV_FUSED` stays off | `bench/polar_uvu_gemm_engine_2026-10-21/ab_rev/ab_1296_rev_fused.json`; `parity_rev/parity_{128,1296}_rev_fused.json` |
| FP32 PolarMACE (precision study) | PyTorch reference, TF32 off: 128 atoms max dF 3.9e-5 eV/Å, 0.27 meV/atom, **1.09×** speed; 512 atoms max dF 8.1e-5, 1.52 meV/atom, **1.14×**. Full engine float32 needed a `MATOS_ALLOW_FLOAT32_GENERIC` test hook past the MH-1 fast-path guard; with it, engine FP32 accuracy is excellent (1,296: dE 3.3e-4 eV, max dF 5.1e-5 eV/Å) but **speed is 0.944×** (189.67 → 200.91 ms; 146.35 → 155.03 µs/atom) because the float path uses the legacy UVU (N1 grouped GEMM is FP64-only). FP32 is not a speed lever as implemented | `bench/fp32_sensitivity_2026-10-22/fp32_sensitivity.json`; `fp32_engine_sensitivity.json` |

## 6. Code provenance

| Change | Engine commit |
|---|---|
| FP64 team MH-1 + write-through fix + team-size override | `80c618a` |
| message-gather tile + team channel-vec knob | `9f97e84` |
| edge-feature tile | `46db0d1` |
| path-grouped MH-1 forward | `7167c26` |
| edge-up/scatter tile + cutoff copy fusion | `e0a961d` |
| N1 grouped FP64 path-GEMM UVU forward/reverse | `482537c` (patch `patches/polar_uvu_gemm_n1.patch`) |

The engine working tree is tracked in the `symmetrix-polar` patch
`patches/symmetrix-xl-polar-support.patch`; the engine fork push is blocked by
the remote fetch issue, so the patch is the durable artifact.

## 7. Reproduction environment

```
LD_LIBRARY_PATH=/workspace/caches/conda/envs/lammps-deepmd/lib:/opt/matos-tools/cuda/12.9/lib64:/opt/matos-tools/conda/envs/matos-compute/lib
CUDA_VISIBLE_DEVICES=0
SYMMETRIX_BACKEND=cuda12-sm80
SYMMETRIX_BACKEND_DESCRIPTOR_PATH=<models>/mace-polar/backend_cuda12_sm80.json
OMP_NUM_THREADS=1
MATOS_REPLAY_CACHE=1
MATOS_UVU_X2_MB_SMEM=0
MATOS_UVU_FWD_SWEEP=0
MATOS_CONV_CUTOFF_BACKEND=team
MATOS_MH1_TEAM_SIZE=64
MATOS_MH1_TEAM_CHANNEL_VEC=1
```

Input: `bench/paper_scale_2026-10-20/in.polar_water_1296` (1,296 atoms, four
measured `run 1`). The A/B runner is `bench/ab_opt2b_1296.py` (same protocol
for all rows). Parity uses `bench/opt3_parity_main.py` / `msgtile_parity.py`.

## 8. Remote queue004 probe and precision ceiling (2026-10-22)

Three queue004 tasks, all finished `Success`:

| Probe | Task | Result | Source |
|---|---|---|---|
| permission, no storage | `t-20261022130617-rpd2t` | A100-SXM4-80GB visible; echo + `nvidia-smi` only | logs |
| inline microbench, no storage | `t-20261022131227-gxsp4` | A100 visible, but `python3: command not found`; bare VASP runtime image has no system Python | logs |
| toolchain microbench | `t-20261022131550-rrzgp` | 58 s; Python 3.11.15 + torch 2.5.1+cu121, CUDA true | `data/q4_remote_microbench_2026-10-22.json`; `figures/Q4_REMOTE_MICROBENCH.png` |

The working recipe is the same one used by the successful 2026-10-19 probe
`t-20261019191724-4zbd7`: queue004, one `ml.pni2.3xlarge`, and the shared
vePFS `vepfs-cnbj2c98dea54433` mounted at
`/vepfs-mlp2/c20250207/lvdingyang` (`SubPath c20250207/lvdingyang`), which
provides `/vepfs-mlp2/c20250207/lvdingyang/matos-tools`. Without that mount
the image has no Python; with it, `matos-compute/bin/python` and
`matos-mlip-py311/bin/python` are the same 3.11.15 + torch 2.5.1 build.

Measured 2048³ GEMM on one A100-80GB (79.3 GiB, 108 SM, CC 8.0), 20 iters:

| Arm | ms | TFLOPS | A100 peak | Fraction |
|---|---:|---:|---:|---:|
| FP64 | 1.1653 | 14.742 | 19.5 (FP64 tensor core) | 75.6% |
| FP32 | 1.1835 | 14.516 | 19.5 (FP32 CUDA core) | 74.4% |

FP32 is **1.6% slower** than FP64 in wall time, because A100's FP32 CUDA-core
peak equals its FP64 tensor-core peak. This is the hardware explanation for
the engine-level FP32 result (0.944×, §5): FP32 alone is not a speed lever.
Only TF32/BF16 tensor cores (156/312 TFLOPS theoretical) change the ceiling,
and they require far more accuracy relaxation than FP32.

Access boundary confirmed: queue004 is one GPU and one replica per task;
`devinstance-queue3` rejects `ml_platform:CreateCustomTask`; the newer vePFS
`vepfs-cnbj4cc563ae78e0` rejects RDWR on both `250358` and `/`. Four
independent queue004 1-GPU tasks can run under the 4-GPU per-user cap, but a
single multi-GPU job is not currently submitable on this account. The
single-GPU FP64 floor of ~120–130 µs/atom therefore still stands.

**Historical note (2026-10-19).** The q4 MPI-scaling probe
`t-20261019191724-4zbd7` did **not** use a remote system LAMMPS. It ran a
locally built binary packaged into the workspace as `$WORKDIR/pkg/bin/lmp`
(195,981,328 bytes, sha256 `bc95c713…`, LAMMPS 30 Sep 2026, Kokkos 5.2.1,
`symmetrix/mace` pair style, RUNPATH to the local conda envs), together with
`pkg/lib` + `pkg/lib/mpich` and the 317 MB PolarMACE model
(`pkg/model/polar_symmetrix_probe.json`). The 128/432/1024-atom five-step
baselines were 0.6116 / 1.4717 / 3.0520 s, i.e. **955.6 / 681.3 / 596.1
µs/atom/step**. That is the pre-optimization materialized-edge build
(`streamed_edges='materialized'`), not the N1 build; it does not contradict
the current 146.1 µs/atom result. The packaging pattern is already proven; the
current `lmp_gemm` (sha256 `093982d5…`) has the same dynamic-dependency
structure (CUDA 12 runtime, MPI, OpenBLAS/OpenMP, glibc) and can replace
`pkg/bin/lmp` directly.

### q4 current-engine 1,296-atom speed (2026-10-22)

| Path | q4 median | q4 µs/atom | local median | local µs/atom | q4/local |
|---|---:|---:|---:|---:|---:|
| `streamed_edges generic` | 188.891 ms | 145.75 | 189.121 ms | 145.93 | 0.999 |
| `streamed_edges materialized` | 201.207 ms | 155.25 | 201.91 ms | 155.8 | 0.997 |

Tasks `t-20261022132315-b97jc` (materialized, 64 s) and
`t-20261022132500-94hhz` (generic, 69 s), both queue004 1×A100-80GB, binary
`lmp_gemm` sha256 `093982d5…`, model sha256 `0e98f0ae…`. PE
−895939.24763167975 eV in both, stable across all steps. LAMMPS wall time is
16 s in both runs, of which ~15 s is the 317 MB model load; the five measured
evals take ~0.95–1.0 s. Source:
`bench/q4_polar_1296_2026-10-22/`, `data/q4_polar_1296_2026-10-22.json`,
`figures/Q4_POLAR_1296_SPEED.png`.

The first q4 attempt omitted `streamed_edges generic` and therefore measured
the slower materialized path (155.25 µs/atom). The corrected generic run is
145.75 µs/atom, matching the local N1 median within 0.12%. queue004 does not
change single-GPU speed; it is an offload/parallelism route. Four independent
1-GPU tasks can run concurrently under the 4-GPU user cap.

### 10,368-atom capacity smoke (2026-10-22)

Task `t-20261022144341-pzc9x`, queue004 1×A100-80GB, `Success` in 77 s. Same
`lmp_gemm` (sha256 `093982d5…`) and model (`0e98f0ae…`) as the 1,296 runs;
`streamed_edges generic`; 10,368 water atoms (2×2×2 replication of the
1,296-atom box).

| atoms | median s/eval | µs/atom | peak GPU | PE |
|---:|---:|---:|---:|---:|
| 5,184 | 0.7305 | 140.91 | 19.16 GiB | stable |
| **10,368** | **1.54626** | **149.14** | **34,519 MiB (33.71 GiB)** | **−7,167,513.9810534371 eV stable** |

The 10,368-atom run is **2× the 5,184 capacity** at 1.76× the memory and
**+5.8% per-atom time**; no OOM on the 80 GB A100. The first attempt
`t-20261022144203-g67lk` failed at `run 0` because the ASE-written data file
omitted masses (`Not all per-type masses are set`); attempt-02 added
`mass 1 15.999` / `mass 2 1.008`. Source:
`bench/q4_polar_10368_2026-10-22/attempt-02/`.
