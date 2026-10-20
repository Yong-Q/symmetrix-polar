# FP64 MH-1 team kernels: −32% at 1,296 atoms

**Status:** measured and parity-gated on the main tree; not yet merged into the
coordinator's Plan B-lite build. The accepted headline is unchanged.

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

## Remaining distance to 4.6 µs/atom

1,296 × 4.6 µs = **5.96 ms/eval**. Current team result 407.75 ms is **68.4×**
away (310.0 µs/atom). Even setting the MH-1 family to zero would leave
>190 µs/atom. The 4.6 number is the linear MACE-OFF23 FP32 5,184-atom result;
it is not reachable for full PolarMACE with kernel work.

## B4 + team (provisional, not accepted)

An A/B of the coordinator's B4 binary vs a B4+team build measured
460.84 → 367.33 ms at 1,296 atoms (−20.4%). That binary is **not accepted**:
after the measurement the shared B tree was rebuilt by the concurrent
Plan B-core workstream, and the current B4+team binary aborts at the first
eval with `polar reverse: geometry cache missing or stale`. The B4 fused
final (460.8 ms) and the main-tree team build (407.75 ms) still run and pass
parity. The 367 ms number is withheld until B4 is merged into the stable main
tree and re-gated.

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
