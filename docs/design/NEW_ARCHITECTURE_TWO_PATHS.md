# Two paths to a new PolarMACE architecture

**Date:** 2026-10-22
**Status:** design only; no engine edit.
**Context:** the low-rank input truncation failed the E/F gate (dE 2–19 eV,
dF 0.20–0.27 eV/Å even at r=256). The current exact engine is 145.75 µs/atom
at 1,296 atoms; tensor product 65.85 ms, polar 20.24 ms, host gap 16.12 ms.

The literature points to two distinct routes. We pursue both.

## 1. Literature map

### Tensor product / equivariant execution

| Work | Mechanism | Reported effect |
|---|---|---|
| **E2Former** (NeurIPS 2025, DOI 10.52202/085713-0797) | Wigner-6j convolution; shifts tensor-product work from edges to nodes; `O(E) → O(V)` | 7–30× vs conventional SO(3) convolutions |
| **E2Former-V2** (arXiv:2601.16622) | SO(3)→SO(2) axis-aligned sparsification; on-the-fly equivariant attention; fused Triton; no materialized edge tensors | 20× TFLOPS; linear activation memory |
| **LunaTP** (arXiv:2610.18393) | Node-centric CGTP execution; path-group parallelism; multi-level reuse; higher-order derivatives | up to 250× kernel vs e3nn, 17.1× vs OpenEquivariance |
| **Gaunt Tensor Products** (arXiv:2401.10216) | CG coefficients → Gaunt coefficients; 2D Fourier basis; FFT | lowers full-tensor-product complexity |
| **SpinGTP** (arXiv:2607.01408) | Spin-weighted spherical harmonics; complete GTP including antisymmetric paths | GTP efficiency + full CGTP expressivity |
| **SPHNet** (arXiv:2502.01171) | Adaptive sparse gates; three-phase sparsity scheduler | up to 7× speedup |

### Long-range / polar / charge equilibration

| Work | Mechanism | Reported effect |
|---|---|---|
| **Latent Ewald Summation** (DOI 10.1038/s41524-025-01577-7) | Learn a hidden variable from local descriptors; apply Ewald to it; no explicit charges/QEq | ~2× the cost of a short-range MLIP |
| **Differentiable PME** (arXiv:2606.01598) | Learned charges + learned dipoles; differentiable particle-mesh Ewald | `O(N log N)`, production-MD scaling |
| **Polarizable atomic multipoles** (arXiv:2605.05746) | Semi-local latent monopoles/dipoles/quadrupoles; non-self-consistent linear response | avoids SCF; predicts BEC/polarizability/Raman |
| **CELLI** (DOI 10.1038/s41524-025-01790-4) | Charge-equilibration layer; model-agnostic | long-range with interpretable charges |
| **Soft-FQEq** (arXiv:2604.27910) | Differentiable fragment-constrained QEq | fragment-resolved charge conservation |
| **iQEq** (DOI 10.1063/5.0252566) | Iterative charge equilibration; quadratic scaling | replaces cubic direct QEq |
| **PME-QEq** (DOI 10.1021/acs.jctc.4c00334) | Quasi-linear particle-mesh QEq | avoids explicit Coulomb matrix |
| **Foundation polarizable long-range model** (DOI 10.1038/s41467-025-65496-3) | Polarizable charge equilibration + equivariant GNN | foundation model with long-range physics |
| **Self-consistent field NN** (DOI 10.1038/s41467-022-29243-2) | Explicit self-consistent long-range response | water with/without fields |
| **Design space of self-consistent electrostatic MLIPs** (arXiv:2603.14700) | Coarse-grained DFT view of electrostatic MLIPs | clarifies approximations/equivalences |

### Capacity / execution

| Work | Mechanism | Reported effect |
|---|---|---|
| **MoE for MLIPs** (DOI 10.1038/s44387-026-00144-9) | Element-wise sparse experts + shared experts | better accuracy/capacity; global routing unstable |
| **Multifidelity MoE Allegro** (arXiv:2604.26143) | Spatially partitioned experts with co-training | cheap bulk + expensive interface |
| **WPE** (arXiv:2610.17366) | Model/stage-specific GPU execution policy from GPUMD | 3.49× NEP, 3.61× qNEP, 10.5× PIMD |
| **Megakernel MLIP** (arXiv:2608.16329) | Aggressive kernel fusion; HBM footprint reduction | small-footprint multi-component MD |
| **DPA4C** (arXiv:2608.19041) | Architecture + compressed CUDA operators co-design | orders-of-magnitude throughput |
| **Symmetrix-XL** (arXiv:2610.01036) | Streamed-edge, model-specialized codegen, tiled execution | 3.1–5.0× vs ML-IAP+cuEq; 11.24M atoms |

## 2. Path A — same checkpoint, exact execution architecture

**Goal:** no retraining, same E/F, faster.

Components:

1. **Node-centric tensor-product dataflow** in the spirit of E2Former/LunaTP:
   - path-group parallelism, node-owned accumulation, no edge materialization;
   - Wigner-6j/Gaunt-style recoupling where the polar irrep layout permits it;
   - a custom CUDA/Triton kernel for the polar layout (the current MH-1 fast
     path rejects it).
2. **Persistent SCF + device-side control**:
   - SCF iteration as the execution unit;
   - `rho(k)`/Fukui device-side reductions; no host round-trips;
   - checkpoint-specialized codegen for the 20 UVU planes + nonlinear epilogue.
3. **Memory scheduling**:
   - one grow-only arena; node tiles; compute-and-release;
   - target 5,184 atoms <10 GiB (currently 19.16 GiB).
4. **Exact structural reductions only**:
   - CG/path sparsity and joint tensor decomposition if they are exact;
   - no input truncation.

**Projected speed:** tensor 65.85 → 30–35 ms; polar+host 36.36 → 24–26 ms;
total 145.75 → **~97–110 µs/atom (1.3–1.5×)**. 2× still needs multi-GPU.

**Prototypes:** node-owned tensor kernel at 1,296 (target ≤35 ms); persistent
SCF (target host+polar ≤25 ms).

## 3. Path B — new model architecture (retraining)

**Goal:** a genuinely new architecture, not a post-hoc approximation.

Components:

1. **Node-centric equivariant backbone**: Wigner-6j/Gaunt/SpinGTP tensor
   products; no edge tensors; `O(V)` tensor-product work.
2. **Non-self-consistent long-range**: latent Ewald (LES) or learned
   charges/dipoles + differentiable PME; one global solve, no 2-step SCF.
3. **Adaptive capacity**: element-wise MoE/shared experts; no global routing.
4. **Persistent node-centric execution**: fused Triton/CUDA, memory-bounded,
   WPE-style policy selection.

**Why it is new:** it changes the computational graph (edge→node), the
long-range solver (2-step SCF→latent/PME), and the capacity allocation (MoE),
and it is trained end-to-end. It is not the current checkpoint plus a kernel.

**Projected speed (speculative, [P]):** E2Former/LunaTP report 7–30× on tensor
products; LES is ~2× a short-range MLIP; PME is `O(N log N)`. A polar model at
2–3× the paper's linear FP32 4.5 µs/atom would be **9–15 µs/atom**, i.e.
roughly **10×** the current 145.75. This is a projection, not a measurement;
training cost and accuracy are the gates.

**Training/validation:** same data as PolarMACE; E/F vs DFT and vs the current
PolarMACE; document the new model contract. This is a new model, not an
inference approximation.

## 4. What the failed low-rank gate teaches

- Input SVD minimizes `||X-X_r||`, not `||J_X(X-X_r)||`. The force-relevant
  tail is not the high-variance head.
- Any low-rank compression must be **task-aware** (Jacobian-weighted) or
  **trained** (latent bottleneck), not post-hoc.
- Path B must train the low-dimensional latent representation; Path A should
  not truncate the input at all.

## 5. Decision matrix

| | Path A | Path B |
|---|---|---|
| Retraining | no | yes |
| Accuracy contract | exact same checkpoint | new model, revalidate |
| Expected single-GPU | 1.3–1.5× | potentially >2× (speculative) |
| Time | weeks | months |
| Risk | medium | high |
| Paper type | execution/co-design | new architecture/model |

## 6. Immediate experiments

1. Path A: node-centric tensor-product prototype at 1,296; go/no-go ≤35 ms.
2. Path A: persistent SCF prototype; go/no-go host+polar ≤25 ms.
3. Path A: CG/path sparsity and joint tensor decomposition audits.
4. Path B: choose LES vs PME vs polarizable multipoles; write the model/training
   spec and a small prototype.
5. Path B: verify E/F vs the current PolarMACE on a shared benchmark before
   scaling training.

## 7. Why this can be a paper

Path A is an **execution architecture** paper: node-centric tensor products +
persistent SCF + memory scheduling for a global fixed-point polar model.
Path B is a **new model architecture** paper: node-centric equivariant
backbone + non-self-consistent latent/PME long-range + adaptive compute.
The literature has the ingredients separately; no paper combines them for a
fourth-generation polar MLIP with a measured E/F contract. That is the gap.
