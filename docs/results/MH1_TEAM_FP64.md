# FP64 MH-1 team kernels: −32% at 1,296 atoms

**Status:** measured, parity-gated, and **correctness-fixed** on the main tree.
The original FP64 team enablement had a repeated-evaluation state bug (see
"Write-through bug" below); the fixed build gives 224.1 µs/atom at 1,296 atoms
with real-MD parity against the pre-team engine.

## What changed

The MH-1 forward/reverse kernels have two implementations:

- a **TeamPolicy** path (`e3 tensor product mh1 team channels`,
  `e3 tensor reverse mh1 team channels/harmonics`), previously restricted to
  `Precision=float` by `if constexpr`;
- the fused MDRange path used for FP64.

Removing the float restriction at the two team paths
(`libsymmetrix/source/e3nn_kokkos.cpp`) lets FP64 use the team kernels. The
team sizes are already set from `mh1_multiplicity_`.

## Measured result (1,296-atom H₂O, A100)

Environment: `MATOS_REPLAY_CACHE=1`, `MATOS_UVU_X2_MB_SMEM=0`,
`MATOS_UVU_FWD_SWEEP=0`, `MATOS_CONV_CUTOFF_BACKEND=team`.

| MH-1 path | median ms/eval | µs/atom/step |
|---|---:|---:|
| fused (previous) | 599.54 | 462.6 |
| **team FP64** | **407.75** | **314.6** |
| saving | **191.79 ms (32.0%)** | **148.0** |

Three clean rounds: fused 599.54 / 599.53 / 606.92; team 406.66 / 408.22 /
407.75. (A 128-atom check also improved 43.99 → 39.38 ms; the 1,296 water box
is the reference.)

## Parity gate

LAMMPS 1-rank parity with the team path and the candidate envs:
dE **2.91e-11 eV** (tol 5e-5), max dF **2.57e-14 eV/Å** (tol 1e-4) —
**PASS**. The FP64 team reductions preserve the force gate.

This gate is single-point only. It passed even with the write-through bug,
which is why the repeated-evaluation and real-MD gates below were added.

## CUPTI trace after the change (1,296 H₂O)

| family | fused trace | team trace |
|---|---:|---:|
| MH-1 tensor product | 274.96 ms (44.1%) | **76.10 ms (17.8%)** |
| backbone nonlinear | 112.49 (18.0%) | **112.53 (26.3%)** |
| polar pipeline | 111.15 (17.8%) | **111.30 (26.0%)** |
| long-range Ewald | 9.56 (1.67%) | 9.61 (2.57%) |
| GPU busy | 573.80 ms | **374.47 ms** |

New top single kernel: **`gather edge message adjoint` 59.38 ms (2 launches)** —
the Plan B4 (reverse-gather fusion) target, already present in the isolated B
tree. A B4 + team-MH-1 build is in progress; the expected next measurement is
at 1,296 atoms.

## Write-through bug in the FP64 team path (found 2026-10-21, fixed)

The float CUDA team path was protected by forcing
`mh1_output_write_through_=false`. Removing the float guard for FP64 exposed
that the **FP64 team forward kernel always used `+=`**, while `evaluate()`
skips the zero-fill when `mh1_output_write_through_` is true:

```cpp
if(!(mh1_fast_path&&mh1_output_write_through_))
    ordered_kokkos_deep_copy(output,Precision(0));
...
output(sample,output_index)+=scale*result*mask(output_index);  // team path
```

The first evaluation uses a freshly zeroed buffer and is correct; every later
evaluation accumulates into the previous result. This is **not** visible to a
single-point gate.

Measured drift at fixed positions (three `run 0` evaluations at 128 atoms):

| build | PE eval 0 | PE eval 1 | PE eval 2 | max dF eval 2 |
|---|---:|---:|---:|---:|
| original pre-team | −49242.4564153149 | −49242.4564153149 | −49242.4564153149 | 1.0e-15 |
| team build, before fix | −49242.4564153149 | −48812.106186406 | +47,277,036.78 | 3.3e8 |
| **team build, after fix** | **−49242.4564153149** | **−49242.4564153149** | **−49242.4564153149** | **1.0e-15** |

The fix branches on `write_through` in the team kernel, matching the vec4 and
scalar paths. No other numerics change.

**Validation after the fix**

- single point vs the accepted Python-engine gate: max dF **1.0e-15**;
- 10-step NVE MD (0.5 fs, 300 K initial velocities) vs the original pre-team
  binary: max Δpos **9.8e-15 Å**, max ΔF **1.4e-14 eV/Å**, PE identical at
  every printed step.

## Remaining distance to 4.5 µs/atom at 1,296 atoms

The reference target is normalized to the 1,296-atom H₂O system:

- target: 1,296 × **4.5 µs** = **5.832 ms/eval**;
- corrected B4+team + coalescing: **320.97 ms = 247.7 µs/atom**;
- gap: **55.0×** (320.97 / 5.832).

Even setting the whole MH-1 family to zero would leave >150 µs/atom, i.e.
>30× above the target. The 4.5 µs/atom number is the **linear MACE-OFF23 FP32
standard-streaming** result at 5,184 atoms; the FP64 PolarMACE materialized
path is not comparable and cannot reach it with kernel work alone. The measured
linear path on the same 1,296-atom box is 9.69 µs/atom (12.56 ms), which is
**2.15×** above 4.5.

## B4 + team (accepted, after the write-through fix)

The B4 source was cleanly merged into the stable main tree (output_gather
in `E3TensorProductKokkos::reverse`, gather skipped at the call site) and
rebuilt. At 1,296 atoms, fixed build:

| configuration | median ms/eval | µs/atom |
|---|---:|---:|
| main team, no B4 (unfixed binary) | 404.96 | 312.5 |
| **main B4 + team (write-through fixed)** | **348.97** | **269.2** |
| saving | **56.0 ms (13.8%)** | **43.3** |

Three clean rounds after the write-through fix, before the gather-tile change:
no-B4 404.15 / 404.96 / 406.18; B4+team 348.97 / 348.72 / 356.30 (one
489.56 ms contended outlier in round 2). The earlier 339.00 ms was measured
with the buggy kernel; the write-through branch costs ~3%, which is the price
of correctness.

**Gather-tile follow-up (2026-10-21):** the nonlinear message gather used the
default MDRange tiling (nodes in x), so every warp read a different edge row.
Tile `{1,128}` (feature block in x, one node per block) drops the kernel from
10.3 to 3.5 ms per launch (−66%) and the 1,296-atom wall time from
**346.91 → 333.17 ms (−13.7 ms, −4.0%)**. Forces match to 1e-15.
**Edge-feature coalescing (2026-10-21):** the edge-feature kernel launched one
thread per edge; each thread wrote 1026 consecutive doubles, so a warp wrote
32 rows 8 KB apart (~32× write amplification). Parallelizing over
(edge, feature) with tile {1,256} gives **333.7 → 321.0 ms (−12.8 ms,
−3.8%)**. Forces match to 1e-15 and three fixed-geometry evaluations are
identical. Cumulative from the fused-MH-1 + team-cutoff baseline:
599.54 → 290.4 ms (−51.6%).

The parity gate for the merged binary passes, and the repeated-evaluation gate
now holds.

## Path-grouped MH-1 forward (2026-10-21)

The FP64 team forward re-evaluated the sparse Wigner plan for every channel and
wrote each instruction block at stride `output_width`, spreading every warp
store across many sectors. The rewrite:

1. precomputes the channel-independent
   `g[a,c] = sum_b W[a,b,c] * input_2[b]` once per team in scratch;
2. runs a flat `channel*output_width + component` loop with coalesced stores;
3. hoists `input_1` into registers and the write-through branch out of the
   component loop.

| metric | before | after |
|---|---:|---:|
| forward tensor product per eval | 39.52 | **17.77** (2.22x) |
| full polar, 1,296 atoms | 247.7 µs/atom | **230.5 µs/atom** (−6.9%) |
| polar-off, 1,296 atoms | 152.0 µs/atom | **135.2 µs/atom** (−11.0%) |

Parity: dPE 0, max dF 1.0e-15; three fixed-geometry evaluations identical.

## 1,296-atom UVU knob check

Same stable main-team binary, only the env changes:

| variant | median ms/eval |
|---|---:|
| base (smem0, fwd sweep0) | **407.8–408.2** |
| `MATOS_UVU_X2_MB_SMEM=1` | 410.5 / 422.1 |
| `MATOS_UVU_FWD_SWEEP=1` | 415.2 |
| both on | 410.3 |

The 128-atom conclusion holds at 1,296: the defaults (smem off, sweep off)
are the best. No further runtime-knob gain is available; the stable best is
**407.75 ms / 314.6 µs/atom**.

## 1,296-atom MH-1 team-size check

Same stable main-team binary; `MATOS_MH1_TEAM_SIZE` overrides both channel
and harmonic team sizes (default 128 for multiplicity 512):

| team size | median ms/eval | µs/atom |
|---|---:|---:|
| default 128 | 408.06 | 315.0 |
| **64** | **403.97** | **311.3** |
| 256 | 418.31 | 322.8 |

Team size 64 is consistently ~4.1 ms faster. The stable best at 1,296 H₂O is
now **403.97 ms / 311.3 µs/atom** (main team, parity-gated MH-1 path).
