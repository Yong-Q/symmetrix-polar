# Latest speed and plan — 2026-10-22

**Scope:** full nonlinear PolarMACE, FP64, exact GTO-Ewald, one A100-SXM4-80GB,
one MPI rank. `µs/atom = 1000 × ms/eval / atoms`.

## TL;DR

- **Best full PolarMACE:** **145.75 µs/atom/step at 1,296 atoms**
  (188.891 ms/eval, `streamed_edges generic`) and **140.91 µs/atom at 5,184
  atoms** (730.49 ms/eval, 19.16 GiB).
- queue004 matches local within **0.12%**; a 1,296-atom task takes **69 s
  end-to-end** (LAMMPS 16 s, of which ~15 s is the 317 MB model load).
- On the Symmetrix-XL paper's own workload (linear MACE-OFF23, FP32,
  standard streaming, 5,184 atoms) this engine measures **4.47–4.53
  µs/atom**, i.e. parity with the paper's **4.50**. Full PolarMACE is a
  different model and is **32.4×** the paper's linear number; that gap is
  model/physics/precision, not engine implementation.
- **Single-GPU realistic floor:** **120–130 µs/atom** after the tensor-product
  dataflow rewrite + CUDA Graph. **80 µs/atom** needs multi-GPU or a
  precision/model-policy change; **40 µs/atom** is not reachable on one GPU
  with the current exact-FP64 model.

## Measured speeds

| Workload | Precision | Atoms | Path | µs/atom/step | ms/eval |
|---|---|---:|---|---:|---:|
| Full PolarMACE | FP64 | 1,296 | generic | **145.75** | 188.891 |
| Full PolarMACE | FP64 | 1,296 | materialized | 155.25 | 201.207 |
| Full PolarMACE | FP64 | 5,184 | generic | **140.91** | 730.49 |
| Linear MACE-OFF23 | FP32 | 5,184 | direct/JIT | 4.47–4.53 | — |
| Linear MACE-OFF23 | FP64 | 6,144 | materialized | 12.26 | 75.32 |
| Paper Symmetrix-XL | FP32 | 5,184 | standard streaming | 4.50 | — |
| Paper ML-IAP+cuEq | FP32 | 5,184 | baseline | 13.99 | — |

q4 tasks: `t-20261022132315-b97jc` (materialized, 64 s) and
`t-20261022132500-94hhz` (generic, 69 s), both `Success`. Binary `lmp_gemm`
sha256 `093982d5…`, model sha256 `0e98f0ae…`, PE
−895939.24763167975 eV in both.

## Where the 146 µs/atom goes (post-N1 CUPTI)

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

N1 cut the polar pipeline **87.95 → 20.24 ms (−77%)**. Tensor product is now
the #1 item at **50.8 µs/atom**, 77% of the remaining gap to 80.

## What changed

- S1.1 node-contiguous streamed-edge tiles; A1 UVU 32×16 tile.
- **N1 grouped FP64 path-GEMM** for the polar UVU: 6 tables, 20 path-GEMM
  planes, Z cache, fused epilogue; `MATOS_POLAR_UVU_GEMM=0` opts out.
- B4 reverse gather + write-through fix; team MH-1; message/edge-feature/
  edge-up tiles; path-grouped MH-1 forward.
- V3 prepared path-plan cache (removes ~24 D2H syncs/eval; prerequisite for
  CUDA Graph).
- q4 offload validated: one GPU per task, packaged binary/model in the shared
  vePFS workspace, same result as local.

## Negative results (do not repeat)

| Experiment | Result |
|---|---|
| S2 receiver-owned generic forward | parity PASS, **2.36× slower** (188.35 → 444.21 ms); opt-in off |
| Fused reverse MH-1 | parity PASS, **1.12× slower** (188.54 → 211.10 ms); off |
| FP32 full engine | dE 3.3e-4 eV / max dF 5.1e-5 eV/Å, but **0.944×** (189.67 → 200.91 ms) |
| 1-step SCF | dE 1.27 eV, max dF 0.101 eV/Å; FAIL |
| P1 k-space chunk | no measurable gain; not promoted |

## Plan (ordered by expected effect)

1. **Tensor-product node-owned dataflow.** Implement a native CUDA kernel for
   the reverse MH-1 channels + harmonics pair (39.4 ms / 30.4 µs/atom), using
   the Triton probe as the specification (4.12× at 1,296, 5.04× at 5,184,
   −63% memory, bitwise exact in isolation). The full S2 API route failed; do
   not repeat it. Target: tensor 65.85 → ~40 ms, total → ~126 µs/atom.
2. **CUDA Graph / megakernel.** Capture the host-free region with the V3
   path-plan cache; graph replay is 1.38×/launch at 100 kernels and ~7.4 ms
   saved over 1,762 launches. Target host gap 16 → ~8 ms, total → ~120 µs/atom.
3. **4-way q4 parallel A/B.** queue004 allows four independent 1-GPU tasks
   under the 4-GPU user cap. Run four kernel/tile variants concurrently; this
   is experiment throughput, not single-run speed.
4. **5,184 validation** of the winning variant; check PE/force parity and
   19.16 GiB memory. q4 1,296 already matches local.
5. **Only if 4.5 is a hard target:** make an explicit model/precision policy
   decision (TF32/BF16, PPPM/approximate Ewald, fewer SCF steps, or a trained
   surrogate). The nonlinear polar, 2-step SCF, exact Ewald, and FP64 are
   part of the trained/model contract; the engine cannot silently remove
   them. 1-step SCF already fails; FP32 alone gives no speedup on A100.

## 1 ns at 5,184 atoms

At 140.91 µs/atom/step and 5,184 atoms:

- one step = 140.91 × 5,184 = **730.49 ms**
- 1 fs timestep → 1,000,000 steps/ns → **730,490 s = 202.9 h = 8.45 days/ns**
- 0.5 fs production timestep → 2,000,000 steps/ns → **16.9 days/ns**
- 2 fs timestep → 500,000 steps/ns → **101.5 h = 4.23 days/ns**
- at the 80 µs/atom target → **115.2 h = 4.80 days/ns** (1 fs)
- at the 40 µs/atom target → **57.6 h = 2.40 days/ns** (1 fs)
- at the paper's linear 4.5 µs/atom (not PolarMACE) → **6.48 h/ns** (1 fs)

The 730.49 ms is the `run 1` kernel loop; dumps/thermo/I/O add overhead. A q4
task also has ~30–45 s of instance startup plus ~15 s model load; that is
negligible for a multi-day run, but ~8% for a short 1 ps run (12.2 min at
1 fs).

## Artifacts

- `docs/results/OPTIMIZATION_LEDGER.md` — every number and source.
- `docs/results/PROGRESS_TO_80US_2026-10-21.md` — N1 decomposition and
  projections.
- `docs/results/V4_PROBES_2026-10-22.md` — grouped GEMM, CUDA Graph,
  receiver-owned accumulation, roofline.
- `docs/results/PAPER_COMPARABILITY.md` — paper-vs-engine audit.
- `docs/results/data/q4_polar_1296_2026-10-22.json`
- `docs/results/data/paper_vs_current_speed_2026-10-22.json`
- `docs/results/figures/Q4_POLAR_1296_SPEED.png`
- `docs/results/figures/PAPER_VS_CURRENT.png`
- `docs/results/figures/PROGRESS_80US.png`
