# Symmetrix-Polar

**GPU-accelerated PolarMACE** — nonlinear polar MACE with long-range
electrostatics — in the Symmetrix-XL C++/Kokkos/LAMMPS engine.

| Best 1,296 | Best 5,184 | Paper parity | Single-GPU floor |
|---:|---:|---:|---:|
| **133.04 µs/atom/step** | **128.99 µs/atom/step** | **4.47–4.53 vs 4.50 µs/atom/step** | **~120–130 µs/atom/step** |
| FP64 · exact Ewald · 2-step SCF · node-owned forward + fused2 reverse | FP64 · 5,184 atoms | linear MACE FP32, 5,184 atoms | projection |

**Definition:** `µs/atom/step = 10^6 × seconds per evaluation / atoms`.
All speed values in this README use **µs/atom/step**. Task wall times are
reported separately in seconds and are not used for speed comparisons.

*Paper parity is measured on the paper's linear MACE FP32 workload; full
PolarMACE is a different model.*

**Latest updates (2026-10-22):**

| Update | Result | Evidence |
|---|---|---|
| **10,368-atom capacity** | **149.14 µs/atom/step**, 33.71 GiB peak; 2× the 5,184 capacity on one A100-80GB | q4 task `t-20261022144341-pzc9x`; `docs/results/LATEST_SPEED_AND_PLAN_2026-10-22.md` |
| **R1 reverse MH-1** | exact, **−3.05%** same-binary; dPE 0, max dF 4.9e-16 eV/Å; projected 141.3 µs/atom/step at 1,296 | `docs/results/data/rev_r1_2026-10-22.json` |
| **Path B six-molecule teacher smoke** | H2O/NH3/C2H4/CH4/C2H6/CO2 labels; 8 configurations × 12 molecules each; no model change | `docs/results/PATH_B_SIX_MOLECULE_SMOKE_2026-10-22.md` |
| **Clean q4 A/B: node-owned + fused2** | **133.04 µs/atom/step** at 1,296, **−5.87 (4.2 %)** vs the 138.91 baseline; exact (PE identical, max dF 4.62e-14 eV/Å); both default on | q4 `t-20261022180640-dslc9`; `docs/results/NODE_OWNED_FORWARD_2026-10-22.md` |
| **5,184 production scale** | **128.99 µs/atom/step**, **−6.26 (4.6 %)** vs 135.25; PE identical; 7.74 days/ns at 1 fs | q4 `t-20261022182725-fgpwv`; `docs/results/data/q4_arch_ab_5184_2026-10-22.json` |
| **Node-owned forward alone** | 135.98 µs/atom/step, **−2.92 (2.1 %)**; `MATOS_MH1_FWD_NODE=0` opts out | same q4 run |
| **Fused reverse alone** | 136.08 µs/atom/step, **−2.83 (2.0 %)**; `MATOS_MH1_REV_FUSED2=0` opts out | `docs/results/FUSED_REVERSE_2026-10-22.md` |
| **Mathematics-first architecture** | exact-checkpoint route projects **97–115 µs/atom/step**; 90 not supported on one exact-FP64 GPU | `docs/design/MATH_ARCH_2026-10-22.md` |
| **Path B pruning audit** | zero-shot no-go beyond ~1.1×; rebuild + fine-tune targets 1.5–4× (C=384→320 first) | `docs/results/PATH_B_PRUNING_AUDIT_2026-10-22.md` |
| **Path B student128 trained** | 34 epochs on q4 `t-20261022182434-bk8kk`; val **9.5 meV/atom / 32.4 meV/Å** (GO gate 5 / 30 not met); 2.04M params, 7.5× smaller | `docs/results/PATH_B_STUDENT128_TRAIN_2026-10-22.md` |

[Latest speed & plan](docs/results/LATEST_SPEED_AND_PLAN_2026-10-22.md) ·
[Optimization ledger](docs/results/OPTIMIZATION_LEDGER.md) ·
[Node-owned forward](docs/results/NODE_OWNED_FORWARD_2026-10-22.md) ·
[Mathematics-first architecture](docs/design/MATH_ARCH_2026-10-22.md) ·
[Tutorial](docs/tutorial/SYMMETRIX_POLAR_TUTORIAL_2026-10-21.pdf) ·
[Technical report](docs/technical/SYMMETRIX_POLAR_TECHNICAL_REPORT.pdf) ·
[中文汇报](docs/report/SYMMETRIX_POLAR_REPORT_2026-10-21.pdf)

![Architecture evolution](docs/results/figures/ARCHITECTURE_EVOLUTION.png)

## Results

| System | Model | Precision | Atoms | Execution path | µs/atom/step | Peak GPU | Evidence |
|---|---|---|---:|---|---:|---:|---|
| Water | Full PolarMACE | FP64 | 1,296 | generic + node-owned forward + fused2 reverse | **133.04** | — | q4 `t-20261022180640-dslc9` |
| Water | Full PolarMACE | FP64 | 1,296 | generic streaming (N1 baseline) | 138.91 | — | same q4 run |
| Water | Full PolarMACE | FP64 | 1,296 | generic streaming (N1, older) | 145.75 | — | q4 `t-20261022132500-94hhz` |
| Water | Full PolarMACE | FP64 | 5,184 | generic + node-owned forward + fused2 reverse | **128.99** | — | q4 `t-20261022182725-fgpwv` |
| Water | Full PolarMACE | FP64 | 5,184 | generic streaming (N1, older) | 140.91 | 19.16 GiB | N1 smoke |
| Water | Full PolarMACE | FP64 | 10,368 | generic streaming | 149.14 | 33.71 GiB | q4 `t-20261022144341-pzc9x` (capacity) |
| OFF23 water | Linear MACE-OFF23 | FP32 | 5,184 | direct/JIT | 4.47–4.53 | — | paper-parity run |
| OFF23 water | Paper Symmetrix-XL | FP32 | 5,184 | standard streaming | 4.50 | — | paper |

All rows use **µs/atom/step**. The 10,368 row is a **capacity point**; the
speed headline is **133.04 at 1,296** and **128.99 at 5,184**. The 1,296
node/fused arms were measured in one q4 task (2 rounds × 4 arms, 5 steps per
arm); the 5,184 default row is from a separate q4 A/B (2 rounds × 2 arms).

### Clean q4 A/B at 1,296 atoms

Task `t-20261022180640-dslc9`, 1×A100-80GB, dedicated. All values are
**µs/atom/step**; medians of two rounds.

| Arm | µs/atom/step | Saving vs baseline |
|---|---:|---:|
| Baseline (N1 + R1) | 138.91 | — |
| Node-owned forward | **135.98** | −2.92 (2.1 %) |
| Fused2 reverse | **136.08** | −2.83 (2.0 %) |
| **Node-owned + fused2 (default)** | **133.04** | **−5.87 (4.2 %)** |

PE is identical to the baseline on the combined arm; max force difference is
**4.62e-14 eV/Å**. Both dataflow changes are default-on; set
`MATOS_MH1_FWD_NODE=0` / `MATOS_MH1_REV_FUSED2=0` to opt out.

![q4 clean A/B](docs/results/figures/Q4_ARCH_AB.png)

- **q4 deployment:** the 1,296 A/B ran on a dedicated q4 A100. Baseline
  138.91, node-owned forward 135.98, fused2 136.08, both **133.04**
  µs/atom/step. At 5,184: baseline 135.25, both **128.99**. Task wall time is
  dominated by model load and is not a speed metric.
- **Production runtime at 5,184 atoms** (derived from 128.99 µs/atom/step):
  **7.74 days per 1 ns** at 1 fs, **15.48 days** at 0.5 fs, **3.87 days** at
  2 fs.
- **Paper comparison is not like-for-like.** The paper's 4.50 is linear
  MACE-OFF23, FP32, no global SCF/Ewald. Full PolarMACE is **32.4×** that
  number. On the paper's own linear FP32 workload this engine matches
  **4.47–4.53 vs 4.50**.

## What it is

PolarMACE extends MACE with learned multipolar spin–charge densities,
GTO-Ewald k-space electrostatics, and a Fukui-balanced SCF fixed point. This
repository brings that layer into Symmetrix-XL's C++/Kokkos GPU engine and
wires it into LAMMPS as `pair_style symmetrix/mace/kk`, without retraining a
universal potential. The production path is **exact full-k Ewald**; Random
Batch Ewald is implemented but rejected by its production force gate.

## Architecture changes

The model weights, equations, FP64 precision, exact Ewald, and LAMMPS
interface are unchanged. All gains are execution changes:

- **S1.1** — node-contiguous streamed-edge tiles + fused reverse.
- **A1** — 32×16 UVU tile with chunked weight staging.
- **N1** — grouped FP64 path-GEMM: 6 UVU tables, 20 planes, Z cache, fused
  epilogue. This is the main win: **−25.87%** (197.03 → 146.06 µs/atom/step
  at 1,296).
- **B4 + team MH-1** — reverse gather, write-through fix, team size 64;
  together with the message/edge/path tiles: 462.6 → 224.1 µs/atom/step.
- **V3** — prepared path-plan cache; removes ~24 D2H syncs/eval and is the
  CUDA Graph prerequisite.
- **R1** — reverse MH-1 per-channel register accumulation and identity-mask
  elision; **−3.05 %** same-binary.
- **Node-owned forward** — receiver-node accumulation; drops the
  `[edges, message_width]` tensor and message gather; **−2.92 µs/atom/step**.
- **Fused2 reverse** — one team per edge, 16 `d input_2` accumulators in
  registers, one shared reduction; **−2.83 µs/atom/step**.
- **queue004** — packaged binary/model offload; same speed as local.

N1 binary `lmp_gemm` sha256 `093982d5…7fa8a03`. Patches:
[`polar_uvu_gemm_n1.patch`](patches/polar_uvu_gemm_n1.patch) ·
[`path_plan_cache_v3.patch`](patches/path_plan_cache_v3.patch) ·
[`symmetrix-xl-polar-support.patch`](patches/symmetrix-xl-polar-support.patch) ·
[`symmetrix-xl-polar-engine-2026-10-22.patch`](patches/symmetrix-xl-polar-engine-2026-10-22.patch).

## Quick start

```bash
git clone <symmetrix-xl> && cd symmetrix-xl
git apply /path/to/patches/symmetrix-xl-polar-support.patch
bash scripts/rebuild_symmetrix.sh v1
python harness/polar_engine_load_smoke.py
python harness/polar_module1_gate.py --live 2
python harness/polar_lammps_parity.py
```

The [tutorial](docs/tutorial/SYMMETRIX_POLAR_TUTORIAL_2026-10-21.pdf)
covers model export, fixtures, the GPU execution path, and the historical
measurements.

## Verification

- 128/1,296-atom parity: dPE = 0; max dF **1.8e-15 / 5.8e-13 eV/Å**
  (gate 1e-10).
- 5,184-atom smoke: **PE drift 0**, 19.16 GiB.
- LAMMPS 1/2-rank PASS; end-to-end vs PyTorch **32/32** (forces ≤1.6e-7 eV/Å).

Details: [theory & methods](docs/THEORY_AND_METHODS.md) ·
[technical report](docs/technical/SYMMETRIX_POLAR_TECHNICAL_REPORT.pdf).

## Limits

- **Single-GPU FP64 floor:** ~**120–130 µs/atom/step** after the
  tensor-product dataflow rewrite + CUDA Graph.
- **80 µs/atom/step:** needs multi-GPU or a precision/model-policy change.
- **40 µs/atom/step:** not reachable on one GPU with the current exact-FP64
  model.
- **5,184 atoms:** the old materialized path OOMs; the generic/streaming path
  uses 19.16 GiB.
- **q4:** one GPU per task; four independent 1-GPU tasks fit the 4-GPU cap.
  A single multi-GPU job is not currently available.

Negative results (S2, fused reverse, FP32, 1-step SCF, RBE) and every number's
source are in the [optimization ledger](docs/results/OPTIMIZATION_LEDGER.md).

## Documentation

- **Start here:** [latest speed & plan](docs/results/LATEST_SPEED_AND_PLAN_2026-10-22.md) ·
  [optimization ledger](docs/results/OPTIMIZATION_LEDGER.md) ·
  [progress to 80 µs/atom/step](docs/results/PROGRESS_TO_80US_2026-10-21.md) ·
  [theory-first execution plan](docs/design/THEORY_OPTIMAL_EXECUTION_PLAN.md) ·
  [two-path new architecture](docs/design/NEW_ARCHITECTURE_TWO_PATHS.md)
- **Model & verification:** [tutorial](docs/tutorial/SYMMETRIX_POLAR_TUTORIAL_2026-10-21.pdf) ·
  [technical report](docs/technical/SYMMETRIX_POLAR_TECHNICAL_REPORT.pdf) ·
  [theory & methods](docs/THEORY_AND_METHODS.md) ·
  [中文汇报](docs/report/SYMMETRIX_POLAR_REPORT_2026-10-21.pdf)
- **Context:** [V4 probes](docs/results/V4_PROBES_2026-10-22.md) ·
  [paper audit](docs/results/PAPER_COMPARABILITY.md) ·
  [MPI H1](docs/results/MPI_H1.md) ·
  [historical comparison](docs/COMPARISON_REPORT_2026-10-19.md)

## License

Tooling and documentation: MIT (see `LICENSE`). The engine patch is a
derivative work of Symmetrix-XL and follows its license. Upstreams:
[mace-torch](https://github.com/ACEsuit/mace), Symmetrix-XL (Zhu group),
graph_longrange (Baldwin et al.), and
[lammps-rbe](https://github.com/randbatch-md/lammps-rbe). See `NOTICE.md`.
