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

The ladder is cumulative through the edge-up/scatter row. A1 is a new
1,296-atom step on the current main tree, not a 128-atom Plan B slice: baseline
`lmp_main_e0a961d` 290.196 ms (223.92 µs/atom) → candidate `lmp_uvu_tile`
266.214 ms (205.41 µs/atom), **−8.26%**, interleaved A/B 2×4 samples with
disjoint ranges. A1 patch `patches/uvu_tile_32x32.patch` (actual default tile
32×16); candidate sha256 `a9ce6481…`; source-matched gates module 5–8, 32/32,
1/2-rank parity all PASS.

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

## 6. Code provenance

| Change | Engine commit |
|---|---|
| FP64 team MH-1 + write-through fix + team-size override | `80c618a` |
| message-gather tile + team channel-vec knob | `9f97e84` |
| edge-feature tile | `46db0d1` |
| path-grouped MH-1 forward | `7167c26` |
| edge-up/scatter tile + cutoff copy fusion | `e0a961d` |

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
