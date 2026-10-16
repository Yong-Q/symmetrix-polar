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
| **End-to-end** | Engine total energy vs PyTorch PolarMACE total | **32/32 frames PASS: rel 4e-12–1.1e-11 (≤1.0e-6 eV absolute)** |
| P3 ⑧ | Forces (full-chain reverse mode) | implementation landed (engine build **v23**, 2026-10-16); acceptance gate in progress — CH4 PASS (2.8–3.0e-4), NH3/N2/CO2 at 2.9e-3–1.2e-2 vs 1e-3 eV/Å target (down from O(0.1–4) pre-⑧); see `harness/polar_module8_gate.py` + `docs/design/POLAR_REVERSE_DESIGN.md` |
| P4 | Random Batch Ewald k-space | design complete (see `docs/design/RBE_P4_DESIGN.md`); implementation starts after ⑧ closes |

Every stage is accepted against **dual-source fixture gates** (frozen Python
reference dumps + live recomputation through the same PyTorch model), including
intermediate-tensor comparisons at each block boundary.

## Repository layout

```
patches/    symmetrix-xl-polar-support.patch — the complete engine patch
            (against symmetrix-xl upstream; includes a new standalone
             compilation unit for the polar layer)
harness/    verification gates, preflights, diagnostics, patch appliers
            (all run against the frozen fixtures or live reference model)
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
```

## Sync discipline

Every change round is committed and pushed to this repository as an off-machine
backup (policy set 2026-10-16). The engine patch in `patches/` is re-emitted from
the working checkout after each round, and `harness/`, `docs/` are refreshed from
the working tree before committing:

```bash
git add harness docs patches README.md
git commit -m "<round summary>"
GIT_SSH_COMMAND="ssh -i /home/workspace/.ssh/id_ed25519 -o IdentitiesOnly=yes" \
  git push origin master
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
