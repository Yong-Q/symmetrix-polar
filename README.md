# Symmetrix-Polar

GPU-accelerated **PolarMACE** — nonlinear polar MACE with long-range
electrostatics — in the Symmetrix-XL C++/Kokkos/LAMMPS engine.

**Current best (2026-10-22): 145.75 µs/atom/step at 1,296 water atoms**
(188.89 ms/eval), full nonlinear PolarMACE, FP64, exact GTO-Ewald, one
A100-80GB. At 5,184 atoms: **140.91 µs/atom** (730.49 ms/eval, 19.16 GiB).
The same binary on queue004 matches local within 0.2%.

**中文入口：** [教程](docs/tutorial/SYMMETRIX_POLAR_TUTORIAL_2026-10-21.pdf) ·
[最新速度与方案](docs/results/LATEST_SPEED_AND_PLAN_2026-10-22.md) ·
[汇报 PDF](docs/report/SYMMETRIX_POLAR_REPORT_2026-10-21.pdf)

---

## Latest results

| Workload | Precision | Atoms | Path | µs/atom/step | ms/eval |
|---|---|---:|---|---:|---:|
| **Full PolarMACE** | **FP64** | **1,296** | **generic streaming** | **145.75** | **188.89** |
| Full PolarMACE | FP64 | 1,296 | materialized | 155.25 | 201.21 |
| **Full PolarMACE** | **FP64** | **5,184** | **generic streaming** | **140.91** | **730.49** |
| Linear MACE-OFF23 | FP32 | 5,184 | direct/JIT | 4.47–4.53 | — |
| Paper Symmetrix-XL | FP32 | 5,184 | standard streaming | 4.50 | — |
| Paper ML-IAP + cuEq | FP32 | 5,184 | baseline | 13.99 | — |

*Local N1 median: 146.06 µs/atom (189.30 ms/eval); the q4 row is 145.75
(188.89 ms/eval).*

- **q4 validation:** 1,296 atoms, 145.75 µs/atom (188.89 ms/eval), 69 s
  end-to-end (LAMMPS 16 s; ~15 s of that is the 317 MB model load); matches
  the local N1 median within 0.2%.
- **Targets normalized to 1,296 atoms:** 80 µs/atom = 103.68 ms/eval;
  40 µs/atom = 51.84 ms/eval. Current gap to 80: **1.83×**.
- **1 ns at 5,184 atoms** (140.91 µs/atom/step): **8.45 days** at a 1 fs
  timestep, **16.9 days** at 0.5 fs, **4.23 days** at 2 fs.
- **Paper comparison is not like-for-like.** The paper's 4.50 is linear
  MACE-OFF23, FP32, no global SCF/Ewald. Full PolarMACE is **32.4×** that
  number. On the paper's own linear FP32 workload this engine measures
  **4.47–4.53**, i.e. parity with 4.50.

The detailed numbers and their sources are in
[`docs/results/OPTIMIZATION_LEDGER.md`](docs/results/OPTIMIZATION_LEDGER.md);
the one-page summary is
[`docs/results/LATEST_SPEED_AND_PLAN_2026-10-22.md`](docs/results/LATEST_SPEED_AND_PLAN_2026-10-22.md).

## What this is

PolarMACE extends MACE with a self-consistent long-range electrostatic layer:

- learned multipolar spin–charge densities,
- GTO-Ewald k-space electrostatics,
- a Fukui-balanced SCF fixed point,
- full reverse-mode forces through the polar chain.

This repository brings that layer into the Symmetrix-XL C++/Kokkos GPU engine
and wires it into LAMMPS as `pair_style symmetrix/mace/kk`, without retraining
a universal potential. The production path is **exact full-k Ewald**; Random
Batch Ewald (RBE) is implemented and statistically validated, but its
production force gate is rejected.

## Quick start

```bash
# 1) apply the engine patch to a Symmetrix-XL checkout
git clone <symmetrix-xl> && cd symmetrix-xl
git apply /path/to/patches/symmetrix-xl-polar-support.patch

# 2) build (CUDA backend + Kokkos; validated configuration)
bash scripts/rebuild_symmetrix.sh v1

# 3) load smoke test (requires a PolarMACE checkpoint exported to Symmetrix JSON)
python harness/polar_engine_load_smoke.py

# 4) module gates (requires reference fixtures; see the tutorial)
python harness/polar_module1_gate.py --live 2
python harness/polar_module34_gate.py
python harness/polar_module5_gate.py

# 5) LAMMPS parity and MD gates
python harness/polar_lammps_parity.py
python harness/polar_lammps_md.py
```

The tutorial
([PDF](docs/tutorial/SYMMETRIX_POLAR_TUTORIAL_2026-10-21.pdf) ·
[HTML](docs/tutorial/index.html) ·
[Markdown](docs/tutorial/SYMMETRIX_POLAR_TUTORIAL_2026-10-21.md))
explains the model, the equations, the GPU execution path, and the historical
measurements.

## Architecture and optimization stack

![Architecture evolution](docs/results/figures/ARCHITECTURE_EVOLUTION.png)

*Baseline (paper/upstream) vs the current PolarMACE engine. The trained model,
equations, FP64, exact Ewald, and the LAMMPS interface are unchanged; all gains
are execution changes. Panel (d) is the measured 1,296-atom ladder.*

The main execution changes, in order:

- **S1.1** node-contiguous streamed-edge tiles + fused reverse.
- **A1** 32×16 UVU tile with chunked weight staging.
- **N1** grouped FP64 path-GEMM: 6 UVU tables, 20 planes, Z cache, fused epilogue.
- **B4** reverse gather + write-through fix; **team MH-1** (size 64).
- **V3** prepared path-plan cache; removes ~24 D2H syncs/eval and is the CUDA Graph prerequisite.
- **queue004** packaged binary/model offload; 145.75 µs/atom, matching local within 0.2%.

N1 binary `lmp_gemm` sha256 `093982d5…7fa8a03`; patches:
[`polar_uvu_gemm_n1.patch`](patches/polar_uvu_gemm_n1.patch),
[`path_plan_cache_v3.patch`](patches/path_plan_cache_v3.patch),
[`symmetrix-xl-polar-support.patch`](patches/symmetrix-xl-polar-support.patch).

### Where the 146 µs/atom goes (post-N1 CUPTI)

| Component | ms/eval | µs/atom | Share of GPU busy |
|---|---:|---:|---:|
| **tensor product (MH-1)** | **65.85** | **50.8** | **38.0%** |
| backbone nonlinear | 28.88 | 22.3 | 16.7% |
| cuBLAS / external | 25.11 | 19.4 | 14.5% |
| polar pipeline | 20.24 | 15.6 | 11.7% |
| MLP / dense | 19.69 | 15.2 | 11.4% |
| e3 linear | 13.16 | 10.2 | 7.6% |
| GPU busy | 173.18 | 133.6 | 100% |
| host gap | 16.12 | 12.4 | — |
| **total wall** | **189.30** | **146.1** | — |

N1 cut the polar pipeline **87.95 → 20.24 ms (−77%)**. The tensor product is
now the #1 item and is memory/launch-bound: 0.019 TFLOPS, 0.45 flops/byte,
0.1% of A100 FP64 peak. The next lever is a node-owned tensor-product dataflow
that removes edge materialization, not a FLOP-only Wigner-6j rewrite.

Every number, source, and gate for the ladder is in the
[optimization ledger](docs/results/OPTIMIZATION_LEDGER.md).

## Correctness and verification

| Gate | Result |
|---|---|
| 128-atom parity | dPE = 0; max dF **1.8e-15 eV/Å** |
| 1,296-atom parity | dPE = 0; max dF **5.8e-13 eV/Å** (gate 1e-10) |
| 5,184-atom smoke | PE drift 0; **19.16 GiB** |
| LAMMPS 1-rank / 2-rank | PASS; 2-rank max dF **1.7e-6 eV/Å** |
| End-to-end vs PyTorch | **32/32**; energy rel 4e-12–1.1e-11; forces ≤1.6e-7 eV/Å |
| Repeated evaluation | 1 ULP at 1,296 (pre-existing S1.1 behavior) |

The engine is validated module by module against frozen PyTorch fixtures and
live recomputation, including intermediate tensors at block boundaries. See
[`docs/THEORY_AND_METHODS.md`](docs/THEORY_AND_METHODS.md) and the
[technical report](docs/technical/SYMMETRIX_POLAR_TECHNICAL_REPORT.pdf).

## Negative results and current limits

| Experiment | Result |
|---|---|
| S2 receiver-owned generic forward | parity PASS, **2.36× slower** (188.35 → 444.21 ms); opt-in off |
| Fused reverse MH-1 | parity PASS, **1.12× slower** (188.54 → 211.10 ms); opt-in off |
| FP32 full engine | **0.944×** (189.67 → 200.91 ms); A100 FP32 CUDA peak = FP64 tensor-core peak |
| 1-step SCF | dE **1.27 eV**, max dF 0.10 eV/Å; FAIL |
| Random Batch Ewald | convergence PASS; production force gate REJECTED |
| 2-rank on one GPU | no speedup; true scaling needs one rank per GPU |

- **Single-GPU FP64 floor:** ~**120–130 µs/atom** after the tensor-product
  dataflow rewrite + CUDA Graph. This is the realistic single-GPU target.
- **80 µs/atom:** needs multi-GPU or a precision/model-policy change.
- **40 µs/atom:** not reachable on one GPU with the current exact-FP64 model.
- **5,184 atoms:** the old materialized path OOMs; the generic/streaming path
  uses **19.16 GiB** and is the current 5,184 configuration.
- **q4:** one GPU per task; up to four independent 1-GPU tasks under the 4-GPU
  user cap. A single multi-GPU job is not currently available on this account.

The next planned work is a native CUDA kernel for the reverse MH-1 pair using
node-owned accumulation, followed by CUDA Graph capture. The Triton probe
measured **4.12×/5.04×** and **−63% memory** in isolation, but the full S2 API
route failed; the kernel must be fused into the tensor-product dataflow. See
[`docs/results/V4_PROBES_2026-10-22.md`](docs/results/V4_PROBES_2026-10-22.md).

## Small-system and historical results

The 128-atom Plan B-micro headline is **30.33 ms/eval** (6.06× vs v29,
3.10× vs the PyTorch 94.0 ms reference, all gates PASS). The 432- and
1,024-atom rows (83.47 ms/step and 312.1 ms/eval) were not re-measured under
N1. The current main results are the 1,296/5,184 rows above. Earlier reports:
[Plan B-micro](docs/results/PLAN_B_MICRO_FINAL.md),
[round 3](docs/results/KERNEL_ROUND3_FINAL.md),
[progress to 80](docs/results/PROGRESS_TO_80US_2026-10-21.md).

## Project layout

```
patches/    full engine patch + N1/V3/A1 optimization patches
harness/    module gates, load smoke, LAMMPS parity/MD, diagnostics
scripts/    rebuild and build tooling
docs/       tutorial, technical report, results, design, report
```

## Documentation map

Start here:

- [Latest speed and plan (2026-10-22)](docs/results/LATEST_SPEED_AND_PLAN_2026-10-22.md)
- [Optimization ledger — every number and source](docs/results/OPTIMIZATION_LEDGER.md)
- [Progress to 80 µs/atom — N1 decomposition](docs/results/PROGRESS_TO_80US_2026-10-21.md)
- [V4 probes — next architecture decision](docs/results/V4_PROBES_2026-10-22.md)
- [Paper comparability audit](docs/results/PAPER_COMPARABILITY.md)

Model, theory, and verification:

- [Tutorial PDF](docs/tutorial/SYMMETRIX_POLAR_TUTORIAL_2026-10-21.pdf) ·
  [HTML](docs/tutorial/index.html) ·
  [Markdown](docs/tutorial/SYMMETRIX_POLAR_TUTORIAL_2026-10-21.md)
- [Technical report PDF](docs/technical/SYMMETRIX_POLAR_TECHNICAL_REPORT.pdf)
- [Theory and methods](docs/THEORY_AND_METHODS.md)
- [Chinese report PDF](docs/report/SYMMETRIX_POLAR_REPORT_2026-10-21.pdf)
- [MPI H1 design and results](docs/results/MPI_H1.md)
- [Historical comparison report (2026-10-19)](docs/COMPARISON_REPORT_2026-10-19.md)

Figures:

- [Current q4 vs local speed](docs/results/figures/Q4_POLAR_1296_SPEED.png)
- [Paper vs current](docs/results/figures/PAPER_VS_CURRENT.png)
- [Progress toward 80 µs/atom](docs/results/figures/PROGRESS_80US.png)
- [V4 probes](docs/results/figures/V4_PROBES.png)

## Reproduction environment

Validated local environment (adapt paths to your checkout):

```bash
export CUDA_VISIBLE_DEVICES=0
export SYMMETRIX_BACKEND=cuda12-sm80
export SYMMETRIX_BACKEND_DESCRIPTOR_PATH=<models>/mace-polar/backend_cuda12_sm80.json
export OMP_NUM_THREADS=1
export MATOS_REPLAY_CACHE=1
export MATOS_UVU_X2_MB_SMEM=0
export MATOS_UVU_FWD_SWEEP=0
export MATOS_CONV_CUTOFF_BACKEND=team
export MATOS_MH1_TEAM_SIZE=64
export MATOS_MH1_TEAM_CHANNEL_VEC=1
export MATOS_POLAR_UVU_GEMM=1
```

- Binary: `lmp_gemm`, sha256 `093982d5…7fa8a03`; model:
  `polar_symmetrix_probe.json`; input: 1,296 water atoms, four measured
  `run 1`.
- The same binary and model can be packaged for queue004 with a shared vePFS
  toolchain; the validated recipe and task IDs are in the
  [latest summary](docs/results/LATEST_SPEED_AND_PLAN_2026-10-22.md).

## Development workflow

Every optimization round is committed with its measured numbers, gates, and
binary hash. `patches/` is re-emitted from the working checkout after each
round; `docs/results/OPTIMIZATION_LEDGER.md` is the source of truth.

```bash
git add README.md docs patches harness
git commit -m "<round summary>"
git push origin master
```

## Upstreams and attribution

- [mace-torch](https://github.com/ACEsuit/mace) — PolarMACE model and PyTorch
  reference
- **Symmetrix-XL** (Zhu group) — the GPU engine this project extends
- **graph_longrange** (Baldwin et al.) — GTO electrostatics reference
- [lammps-rbe](https://github.com/randbatch-md/lammps-rbe) — Random Batch Ewald
  reference

See `NOTICE.md` for details. Upstream projects retain their own licenses.

## License

Tooling and documentation in this repository: MIT (see `LICENSE`). The engine
patch is a derivative work of Symmetrix-XL and follows its license.
