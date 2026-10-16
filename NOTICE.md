# Upstream Attribution

This repository contains a patch for Symmetrix-XL and verification tooling.
It builds on and references the following upstream projects:

- **mace / mace-torch** — PolarMACE model definition and PyTorch reference
  implementation (extensions.py / field_blocks.py / gate.py used as the
  mathematical ground truth for the port). MIT License.
- **graph_longrange** (Will Baldwin et al.) — reference implementation of the
  GTO-Ewald electrostatics (k-space features, energy, self-interaction).
- **Symmetrix-XL** (Zhu group) — the C++/Kokkos GPU engine extended by the
  patch in `patches/`. The patch is a derivative work; the upstream license
  applies to the patched files.
- **lammps-rbe** (randbatch-md) — Random Batch Ewald reference used for the
  P4 k-space sampling design (see `docs/design/RBE_P4_DESIGN.md`).
- **LAMMPS** — the simulation package the engine integrates with.

All upstream projects retain their own licenses; nothing here re-licenses them.
