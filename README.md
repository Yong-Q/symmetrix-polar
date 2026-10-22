# Symmetrix-Polar

**GPU-accelerated PolarMACE** — nonlinear polar MACE with long-range
electrostatics — in the Symmetrix-XL C++/Kokkos/LAMMPS engine.

| Current best | 5,184 atoms | Paper parity | Single-GPU floor |
|---:|---:|---:|---:|
| **145.75 µs/atom** @ 1,296 | **140.91 µs/atom** | **4.47–4.53 vs 4.50** | **~120–130 µs/atom** |
| FP64 · exact Ewald · 2-step SCF | 730.49 ms/eval · 19.16 GiB | linear MACE FP32, 5,184 atoms | after tensor-product rewrite |

*Paper parity is measured on the paper's linear MACE FP32 workload; full
PolarMACE is a different model.*

[Latest speed & plan](docs/results/LATEST_SPEED_AND_PLAN_2026-10-22.md) ·
[Optimization ledger](docs/results/OPTIMIZATION_LEDGER.md) ·
[Tutorial](docs/tutorial/SYMMETRIX_POLAR_TUTORIAL_2026-10-21.pdf) ·
[Technical report](docs/technical/SYMMETRIX_POLAR_TECHNICAL_REPORT.pdf) ·
[中文汇报](docs/report/SYMMETRIX_POLAR_REPORT_2026-10-21.pdf)

![Architecture evolution](docs/results/figures/ARCHITECTURE_EVOLUTION.png)

## Results

| Workload | Precision | Atoms | Path | µs/atom/step | ms/eval |
|---|---|---:|---|---:|---:|
| **Full PolarMACE** | **FP64** | **1,296** | **generic streaming** | **145.75** | **188.89** |
| Full PolarMACE | FP64 | 5,184 | generic streaming | **140.91** | **730.49** |
| Linear MACE-OFF23 | FP32 | 5,184 | direct/JIT | 4.47–4.53 | — |
| Paper Symmetrix-XL | FP32 | 5,184 | standard streaming | 4.50 | — |

- **q4 validation:** same binary, 145.75 µs/atom vs local 146.06 (within
  0.2%); 69 s end-to-end, of which 16 s is LAMMPS (model load ~15 s).
- **1 ns at 5,184 atoms:** **8.45 days** at 1 fs, **16.9 days** at 0.5 fs,
  **4.23 days** at 2 fs.
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
  epilogue. This is the main win: **−25.87%** (255.35 → 189.30 ms at 1,296).
- **B4 + team MH-1** — reverse gather, write-through fix, team size 64;
  together with the message/edge/path tiles: 462.6 → 224.1 µs/atom.
- **V3** — prepared path-plan cache; removes ~24 D2H syncs/eval and is the
  CUDA Graph prerequisite.
- **queue004** — packaged binary/model offload; same speed as local.

N1 binary `lmp_gemm` sha256 `093982d5…7fa8a03`. Patches:
[`polar_uvu_gemm_n1.patch`](patches/polar_uvu_gemm_n1.patch) ·
[`path_plan_cache_v3.patch`](patches/path_plan_cache_v3.patch) ·
[`symmetrix-xl-polar-support.patch`](patches/symmetrix-xl-polar-support.patch).

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

- **Single-GPU FP64 floor:** ~**120–130 µs/atom** after the tensor-product
  dataflow rewrite + CUDA Graph.
- **80 µs/atom:** needs multi-GPU or a precision/model-policy change.
- **40 µs/atom:** not reachable on one GPU with the current exact-FP64 model.
- **5,184 atoms:** the old materialized path OOMs; the generic/streaming path
  uses 19.16 GiB.
- **q4:** one GPU per task; four independent 1-GPU tasks fit the 4-GPU cap.
  A single multi-GPU job is not currently available.

Negative results (S2, fused reverse, FP32, 1-step SCF, RBE) and every number's
source are in the [optimization ledger](docs/results/OPTIMIZATION_LEDGER.md).

## Documentation

- **Start here:** [latest speed & plan](docs/results/LATEST_SPEED_AND_PLAN_2026-10-22.md) ·
  [optimization ledger](docs/results/OPTIMIZATION_LEDGER.md) ·
  [progress to 80 µs/atom](docs/results/PROGRESS_TO_80US_2026-10-21.md) ·
  [theory-first optimal execution plan](docs/design/THEORY_OPTIMAL_EXECUTION_PLAN.md)
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
