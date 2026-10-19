# Symmetrix-Polar

**GPU-accelerated PolarMACE with long-range electrostatics — PolarMACE (MACE-POLAR)
support for the [Symmetrix-XL](https://github.com/) engine, with a full Ewald
electrostatic layer and Random Batch Ewald (RBE) k-space sampling in progress.**

PolarMACE extends MACE with a self-consistent, long-range electrostatic layer
(learned multipolar spin-charge densities, GTO-Ewald k-space electrostatics, a
Fukui-balanced SCF fixed point). This project brings that layer into the
Symmetrix-XL C++/Kokkos GPU engine so that polarizable long-range MLIPs run at
GPU speed in LAMMPS-class workflows, without retraining a universal potential.

## Status

| Layer | Component | Verification |
|---|---|---|
| Backbone | Short-range MACE (existing engine path) | 100-frame parity vs `mace-torch`: dE MAE **3.6e-7 eV** |
| P3 ① | `lr_source_maps` accumulation + e3nn basis conversion | 12 configs, max\|Δ\| **3.1e-8** (gate 5e-6) |
| P3 ② | Fukui initialization (gated readout head, charge/spin aware) | 12 configs, rel **6.3e-8**; corrected density **3.1e-8** |
| P3 ③ | k-grid (half-space enumeration) + GTO geometry precompute | bit-level: k **0.0**, cos/sin 6.9e-15, basis 4.4e-16 |
| P3 ④ | Potential features (`forward_dynamic`) | rel **6.8e-8** (gate 1e-6) |
| P3 ⑤ | SCF density updates (update blocks ×2, per-step Fukui) | density evolution 12/12, stage-level 11/11, rel **~5e-8** |
| P3 ⑥ | `coulomb_energy` (full-k Ewald baseline) | engine: ρ̂ rel **1.3e-15**, ΔE rel **8.4e-16** |
| P3 ⑦ | `local_electron_energy` | engine: 8/8 configs PASS (stage rel **≤1e-6**, out ≤4e-7) |
| **End-to-end** | Engine vs PyTorch PolarMACE (energy + forces) | **32/32 frames PASS: energy rel 4e-12–1.1e-11 (≤1.0e-6 eV); forces ≤1.6e-7 eV/Å** |
| P3 ⑧ | Forces (full-chain reverse mode) | **PASS (v27)**: 12/12 configs — FD self-consistency ≤5.8e-4 (gate 1e-3), force parity vs PyTorch ≤7.2e-7 (gate 1e-3; pre-⑧ baseline mean/max 1.24/4.15), ΣF ≈ 1e-14 (exact translation invariance). Root cause of the final gap: missing *project-phase adjoint* in the descriptor reverse (design §2 ⑧-3 (d3), second line) — located via checkpoint audit (`harness/polar_rev_stage_audit.py`), fixed in `descriptor_reverse_channel` (iii-b) |
| P4 | Random Batch Ewald k-space | **PASS (v28)**: RBE sampling (per-axis rounded-Gaussian + MH per lammps-rbe, counter-based SplitMix64 RNG, seed in JSON, exact discrete Z_q, full-space estimator with importance weights); p-scan RMS_F 4.1e-2 → 1.0e-2 eV/Å @ p=50→800, slope **−0.499** (≈ −1/√p), unbiased, same-seed bitwise-reproducible; see `harness/polar_rbe_gate.py` |
| P5 | LAMMPS pair wiring + performance | **PASS (v29, 2026-10-19)**: `pair_style symmetrix/mace/kk` now dispatches `MACE_Nonlinear`/PolarMACE (materialized), maps LAMMPS types, passes cell/node positions, and adds polar node forces + virial. Single-point parity vs the engine: **dE 2.9e-11 eV**, **max dF 2.6e-14 eV/Å**; 50-step NVT MD: no lost atoms, Etot span **0.21 eV**, 494 ms/step. Performance: **184 ms/eval** at 128 atoms (37.8× v28) and **1008 ms** at 1024 atoms (7.7× v28). Multi-rank is rejected until a distributed long-range density is implemented. See `docs/REPORT_2026-10-19.md` |

Every stage is accepted against **dual-source fixture gates** (frozen Python
reference dumps + live recomputation through the same PyTorch model), including
intermediate-tensor comparisons at each block boundary.

## Performance (v29, A100-80GB, f64, energy + forces)

| Configuration | v28 | v29 | Speedup |
|---|---|---|---|
| NH3×32, 128 atoms, deterministic | 6951.7 ms | **183.8 ms** | **37.8×** |
| NH3×32, 128 atoms, RBE p=200 | 6962.9 ms | **186.3 ms** | **37.4×** |
| NH3×32×2×2×2, 1024 atoms | 7795.6 ms | **1007.7 ms** | **7.7×** |

Root causes found by stage profiling (`MATOS_POLAR_PROFILE=1`):

1. **UVU tensor products were per-node serial**: the forward, reverse, and SCF
   replay kernels each used one GPU thread per node with an internal 512×512
   loop. Rewritten as `(node, u)` 2-D kernels (four sites: forward SCF, forward
   local-electron, reverse, replay), removing atomics in the reverse path.
2. **Python launch-tuning probe on non-JIT models**: `_apply_kernel_launch_tuning`
   queried the full device environment before checking whether a tunable JIT
   implementation existed, costing ~0.7 s/eval for materialized models. The
   implementation-identity check now runs first.

All numerical gates are unchanged after the optimization (module ⑤–⑧ PASS;
end-to-end 32/32 with energy rel ≤4.2e-12 and forces ≤1.6e-7 eV/Å; RBE gate
still slope −0.499 and bitwise seed-reproducible).

![v28 vs v29 performance](results/polar_perf_v28_vs_v29.png)

## Repository layout

```
patches/    symmetrix-xl-polar-support.patch — the complete engine patch
            (against symmetrix-xl upstream; includes a new standalone
             compilation unit for the polar layer)
harness/    verification gates, preflights, diagnostics, patch appliers,
            and the P5 LAMMPS parity/MD gates (run against frozen fixtures,
            the live reference model, or the patched LAMMPS build)
docs/       project plan (zh) + design documents:
              design/RBE_P4_DESIGN.md            — Random Batch Ewald plan
              design/POLAR_REVERSE_DESIGN.md     — force/adjoint chain design
              design/module{6,7,8}_patch_final.md — per-module patch specs
              design/polar_module5_audit.md      — serialization audit
scripts/    build tooling (full & fast incremental rebuild, status monitor)
results/    sample acceptance-gate plots
```

## Verification methodology (why “bit-near” is achievable)

The engine is validated **module by module**, and each module only opens after
its own numerical gate passes:

1. **Frozen fixtures**: reference activations are captured from the PyTorch
   model via forward hooks (initial densities, per-SCF-step states, every
   update-block tensor) — 40+ tensors per configuration.
2. **Stage gates**: the engine is compared tensor-by-tensor against those
   fixtures (e.g., the whole update-block chain matches at rel ≈ 3–7e-8,
   i.e. f64 accumulation-order noise).
3. **Live second source**: key gates additionally recompute the reference
   on freshly rattled configurations at gate time.
4. **Failure forensics**: every semantic convention (basis ordering, input
   normalization, f32 buffers, partial biases) was pinned numerically before
   implementation; `harness/polar_lint.py` encodes the historical error
   classes as pre-build static checks.

## Quick start

```bash
# apply the engine patch to a symmetrix-xl checkout
git clone <symmetrix-xl> && cd symmetrix-xl
git apply /path/to/patches/symmetrix-xl-polar-support.patch

# build (CUDA backend + Kokkos; see scripts/rebuild_symmetrix.sh for the
# validated configuration)
bash scripts/rebuild_symmetrix.sh v1

# load smoke test (expects a PolarMACE checkpoint exported to the
# Symmetrix JSON format)
python harness/polar_engine_load_smoke.py

# acceptance gates (need the reference fixtures regenerated via
# harness/polar_module3_refdump.py, polar_module5_hooks.py, ...)
python harness/polar_module1_gate.py --live 2
python harness/polar_module34_gate.py
python harness/polar_module5_gate.py

# P5 LAMMPS gates (after building the patched LAMMPS; edit LMP at the top of
# each script to point at the local build)
python harness/polar_lammps_parity.py
python harness/polar_lammps_md.py
```

## Sync discipline

Every change round is committed and pushed to this repository as an off-machine
backup (policy set 2026-10-16). The engine patch in `patches/` is re-emitted from
the working checkout after each round, and `harness/`, `docs/` are refreshed from
the working tree before committing:

```bash
git add harness docs patches README.md
git commit -m "<round summary>"
GIT_SSH_COMMAND="ssh -i /home/workspace/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  -o UserKnownHostsFile=/home/workspace/.ssh/known_hosts \
  -o StrictHostKeyChecking=accept-new" git push origin master
```

## Upstreams & attribution

- [mace-torch](https://github.com/ACEsuit/mace) — PolarMACE model & PyTorch reference
- [graph_longrange](https://github.com/) (Baldwin et al.) — GTO electrostatics reference
- [Symmetrix-XL](https://github.com/) (Zhu group) — the GPU engine this project extends
- [lammps-rbe](https://github.com/randbatch-md/lammps-rbe) — Random Batch Ewald reference

See `NOTICE.md` for details. This repository contains patches and tooling; the
upstream projects retain their own licenses.

## License

Tooling and documentation in this repository: MIT (see `LICENSE`).
The engine patch is a derivative work of Symmetrix-XL and follows its license.
