# Kernel Round 3 — final results (2026-10-20)

**中文摘要：** 最终 build 8 在 **128 原子（32 个 NH3 分子）** 的 full
PolarMACE（非线性 backbone + 极化 SCF/Ewald，FP64）上达到
**48.74 ms/eval**；同一二进制把全部 round-3 开关回退后是 53.54 ms，
即 **1.0985×**。相对 round-2 验收的 56.50 ms 是 **1.159×**，
相对 v29 原始 Symmetrix-XL Kokkos 引擎 183.8 ms 是 **3.77×**，
相对 v28 初始 polar 实现 6951.7 ms 是 **142.6×**，相对 PyTorch
`mace_polar` 参考 94.0 ms 是 **1.93×**。**1024 原子（256 个 NH3 分子）**
实测 **312 ms/eval**。RBE 的 P4 收敛门 PASS，但生产力/能量精度门
**REJECTED**：p=800 的 RMS dF 6.1e-3 eV/Å、max 3.0e-2 eV/Å、能量噪声
±0.05 eV，且 70.4 ms/step 比 exact Ewald 更慢；要降到 1e-4 需要
p≈8.5×10⁶（cos/sin 矩阵约 17 TB），因此 RBE 不是加速路径，生产继续用
exact full-k Ewald。优化内核完全支持非线性极化力场；本报告所有数字都是
full PolarMACE，不是 linear MACE。**module 5–8、32/32 end-to-end、
LAMMPS 1-rank parity 全部 PASS**（round-3 Python snapshot，rc 134 只是
`GATE PASS` 之后的 teardown abort）。

## 1. What is measured

| Item | Value |
|---|---|
| System | 128 atoms = **32 NH3 molecules**, 14.736 Å cubic, periodic |
| Model | `polar_symmetrix_probe.json`: `MACE_Nonlinear` + polar, 83 elements, r_c = 6 Å, FP64 |
| Workload | one energy + force evaluation per step |
| GPU | one NVIDIA A100-SXM4-80GB |
| Protocol | 1 warmup + 12 measured `run 1`, 2 interleaved rounds, median of per-round medians, GPU `flock` |
| Final binary | `build-kernel-r3/lmp`, sha256 `c7608f35845548ef037341833c138da87515484ce51c04a257a5920bb9424631` |

## 2. Final speed (128 atoms, full PolarMACE)

| Configuration | ms/eval |
|---|---:|
| Round-3 build 8 defaults | **48.74** |
| Same binary, all round-3 slices reverted | **53.54** |
| Saving | **4.80 ms (1.0985×)** |
| Round-2 accepted headline | 56.50 |
| Round-1 | 122.2 |
| v29 pre-round-1 Symmetrix-XL Kokkos build | 183.8 |
| v28 initial polar implementation | 6951.7 |
| PyTorch `mace_polar` reference | 94.0 |

Speedups of the final 48.74 ms:

| Baseline | Speedup |
|---|---:|
| Round-2 accepted (56.50) | **1.159×** |
| v29 pre-round-1 build (183.8) | **3.77×** |
| v28 initial polar (6951.7) | **142.6×** |
| PyTorch reference (94.0) | **1.93×** |

**Comparability note (2026-10-20):** every baseline in this table is an
internal or PolarMACE-specific reference (same model, same 128-atom box).
None of them is the Symmetrix-XL paper's ML-IAP + cuEquivariance comparison,
which uses MACE-OMAT-0/OFF23 in float32 at 5,000 atoms and the specialized
standard-streaming path. Do not read the 3.77× as the paper's 3.1–5.0×.
See [`PAPER_COMPARABILITY.md`](PAPER_COMPARABILITY.md) for the matched
linear-engine measurement (~12.3 µs/atom/step at 6,144 atoms, materialized
float64).

The same-binary fallback protocol is the attribution measurement: all round-3
slices are disabled with their documented env switches
(`MATOS_MH1_CHANNEL_VEC=1`, `MATOS_MH1_WRITE_THROUGH=0`,
`MATOS_MH1_HARMONIC_UNROLL=1`, `MATOS_MH1_WEIGHT_ZERO_ELIDE=0`,
`MATOS_UVU_REV_SWEEP=1`, `MATOS_UVU_FWD_SWEEP=1`,
`MATOS_AFFINE_LN_PARALLEL=0`, `MATOS_CONV_CUTOFF_BACKEND=scalar`), so the
ratio is not confounded by a different LAMMPS/Kokkos build.

## 3. System-size scaling

| System | Atoms | Molecules | ms/eval | Note |
|---|---:|---:|---:|---|
| NH3 1×1×1 | 128 | 32 | **48.74** | build 8, measured |
| NH3 2×2×2 | 1024 | **256** | **312.1** | build 8, measured, stable 311.6–312.5 ms |
| NH3 2×4×4 | 4096 | **1024** | ~1200–1300 (estimate) | run interrupted; linear extrapolation only |
| N2×216 | 432 | — | 126.6 | round-2 accepted; round 3 not re-measured |

2 ps MD estimates at the measured speeds (2000 steps at 1 fs, 4000 at 0.5 fs,
excluding model load and neighbor rebuilds):

| System | 1 fs | 0.5 fs |
|---|---:|---:|
| 128 atoms (32 NH3) | ~1 min 45 s | ~3 min 25 s |
| 1024 atoms (256 NH3) | ~10 min 24 s | ~20 min 48 s |
| ~4096 atoms (~1000 NH3) | ~40 min (estimate) | ~80 min (estimate) |

## 4. Kernel attribution

CUPTI (one warm eval, round-3 trace `r3c_full`; build 8 adds the phase-source
and polar LN-team slices on top):

| kernel | round 2 | round 3 | delta |
|---|---:|---:|---:|
| e3 tensor reverse mh1 fused channels | 4.288 | 2.662 | **−1.626** |
| e3 tensor reverse mh1 fused harmonics | 3.097 | 2.449 | **−0.649** |
| affine fused reverse layer norm silu | 1.539 | 0.490 | **−1.049** |
| affine fused layer norm silu | 1.262 | 0.392 | **−0.870** |
| polar desc ab reverse (+partial) | 1.043 | 0.077 | **−0.966** |
| polar desc proj phase reverse (+partial) | 0.846 | 0.221 | **−0.625** |
| polar reverse uvu mode0 x1 | 3.207 | 2.749 | −0.458 |
| polar uvu mode0 | 1.201 | 1.092 | −0.110 |
| e3 tensor product mh1 fused | 3.337 | 3.231 | −0.106 |
| polar reverse uvu mode0 x2 | 3.552 | 3.612 | +0.060 |
| **GPU-busy total** | **49.97** | **43.91** | **−6.06** |

Wall 48.74 vs GPU busy ≈43.9 leaves a ≈4.8–5.9 ms host/launch gap.

**Kept:** MH-1 reverse channels (4-channel register accumulation), MH-1
harmonics ILP, affine LN+SiLU team-per-row, UVU reverse x1 split, MH-1 forward
write-through, UVU forward split, k-space `desc ab` / `desc proj phase` splits,
phase-source split, polar LN+SiLU teams.

**Measured regressions, now off by default:** MH-1 forward 4-channel
(`MATOS_MH1_FWD_CHANNEL_VEC=4` opt-in; 4.04 vs 3.34 ms) and the two-stage
reverse convolution cutoff (`MATOS_CONV_CUTOFF_BACKEND=chunked` opt-in;
2.87 vs 1.53 ms). UVU reverse x2 split is neutral.

**Experimental:** the e3-linear block-plan merged unpack is implemented but
currently hits an illegal memory access on multi-contribution blocks; it is
opt-in (`MATOS_E3_LINEAR_MERGED_UNPACK=1`) and disabled by default.

## 5. Numerical validation

| Check | Result |
|---|---|
| LAMMPS 1-rank parity vs round-3 Python reference | **PASS**, dE **2.910e-11 eV** (≤5e-5), max dF **2.628e-14 eV/Å** (≤1e-4) |
| Stage-A build vs round-2 `lmp_r5` | PE identical, **max |dF| = 0.0** over all 384 components (bitwise) |
| Reordered k-space splits | deterministic; covered by the parity gate above |
| Module 5 (density/stage) | **PASS**, 12 density + 11 stage items; dens abs 6.689e-8, rel 4.218e-8 |
| Module 6 (Coulomb/SCF) | **PASS**, ρ̂ rel 1.27e-15, ΔE rel 1.02e-15, abs 7.77e-16 eV |
| Module 7 (local electron) | **PASS**, 16/16, out 4.55e-7 |
| Module 8 (forces) | **PASS**, FD 5.783e-4 (≤1e-3), parity 7.191e-7 (≤1e-3) |
| End-to-end 32 frames | **PASS**, 32/32 |
| Teardown | pre-existing `cudaErrorCudartUnloading` abort (rc 134) after `GATE PASS`, as in every prior round |

**The latest full-gate-suite-passing version is round 3 at 48.74 ms** on
128 atoms (32 NH3 molecules). The complete suite (module 5–8, 32/32
end-to-end, LAMMPS 1-rank parity) was rerun on a round-3 Python snapshot built
from the same source as the measured binary; all gates PASS. The rc-134 lines
in the gate logs are the pre-existing teardown abort after `GATE PASS`.

## 6. RBE final status

There are two different gates and they must not be conflated:

| Gate | Criterion | Result |
|---|---|---|
| P4 convergence/unbiasedness (`harness/polar_rbe_gate.py`) | slope ∈ [−0.8,−0.2], RMS drops ≥2×, energy mean within 3·SEM, same-seed reproducibility | **PASS** |
| Production force/energy accuracy (`docs/testing/GATES.md`) | RMS_F ≤ 1e-4 eV/Å, energy ~5e-6 eV | **REJECTED** |

Existing P4 p-scan (n=10, NH3 + CO2):

| p | RMS_F (eV/Å) | mean ΔE (eV) | max |ΔE| (eV) |
|---:|---:|---:|---:|
| 50 | 4.10e-2 | −0.051 ± 0.026 | 0.189 |
| 200 | 2.14e-2 | −0.023 ± 0.015 | 0.143 |
| 800 | 1.03e-2 | +0.0015 ± 0.0056 | 0.029 |

Quick independent check on this build (128-atom NH3, p=800, 6 evals,
`MATOS_RBE_SEED=777001`):

| Metric | Value |
|---|---:|
| RMS dF | **6.1e-3 eV/Å** |
| max |dF| component | **3.0e-2 eV/Å** |
| energy noise | **±0.05 eV** |
| 6-frame mean bias | max 8.8e-3, RMS 2.4e-3 eV/Å |
| speed | **70.4 ms/step** (exact ≈53–62 ms) |

Why it cannot pass / speed up:

1. **1/√p variance.** RMS_F ∝ p^−0.499. Reaching 1e-4 from 6.1e-3 at p=800
   needs p ≈ 8.5e6 (10,000× more samples).
2. **Memory.** The engine stores cos/sin(k·r) for every sampled k; p=8.5e6 ×
   128 atoms × 2 × 8 B ≈ **17 TB**.
3. **Systematic floor.** RBE samples the full k-space while the exact
   reference is hard-truncated at |k| ≤ 1.8467 Å⁻¹; the design doc estimates
   relative errors 8.9e-5 (l=0) to 1.5e-3 (l=1) and 5.3% sampling mass outside
   the cutoff. Even p→∞ may not reach 1e-4 without a truncated proposal.
4. **MH chain correlation** reduces the effective sample size below p.
5. **No speed benefit:** exact full-k Ewald is only ≈14 ms of the 48.74 ms
   eval, and RBE p=800 is already slower than exact.

**Decision:** exact full-k Ewald remains the production path. RBE stays an
opt-in experiment (`MATOS_RBE_P`) and is documented as a negative result. A
future attempt would need a truncated proposal plus a low-|k| exact /
high-|k| stochastic hybrid; the current evidence does not support it as a
speedup.

## 7. Nonlinear polarizable force field

The optimized kernels **do** support the nonlinear polarizable model. The
benchmarked system is `MACE_Nonlinear` + learned multipolar long-range
electrostatics (PolarMACE), i.e. a nonlinear backbone with a self-consistent
polar layer, run in FP64. The MH-1, UVU, affine MLP, polar SCF and Ewald
kernels are all on that path. The float-only CUDA team/direct-node paths are a
separate optimization and do not limit the double-precision nonlinear model.

## 8. Final vs historical

- **Final:** this document; `results/comparison_summary.csv` (round-2 rows are
  historical); `docs/examples/data/kernel_round3_summary.json`.
- **Historical:** `KERNEL_ROUND2.md`, `MPI_H1.md`,
  `COMPARISON_REPORT_2026-10-19.*`, `REPORT_2026-10-16.md`,
  `REPORT_2026-10-19.md`, and the round-1/2 figures. They remain as evidence
  but their headline numbers are superseded by this report.

## 9. Remaining gap to the 30 ms / 20–25 ms target

Round 3 reached **48.74 ms**, not 30 ms. The main blockers are:

- ≈5 ms host/launch gap (1194 launches; the e3-linear pack/unpack merge is the
  biggest single lever but its first implementation is not yet safe);
- MH-1 (≈8.3 ms), UVU (≈6.4 ms), gather (≈3.6 ms), cuBLAS (≈3.8 ms), and
  e3-linear pack/unpack (≈2.7 ms);
- the remaining low-coverage polar kernels;
- CUDA Graph is deferred (host↔device syncs in the polar reverse).

The round-3 plan's Stage C/D (launch elimination, CUDA Graph, tensor cores,
work removal) would be required for ≤30 ms; 20–25 ms additionally requires
work removal. None of these was accepted in this round.
