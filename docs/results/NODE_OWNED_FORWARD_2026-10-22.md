# Node-owned forward dataflow — 2026-10-22

**Scope:** full PolarMACE, FP64, exact 2-step SCF, 1×A100-80GB (shared).
**Change:** forward MH-1 tensor product writes receiver-node messages directly;
the `[edges, message_width]` edge-message tensor and the separate message
gather are removed. Opt-in `MATOS_MH1_FWD_NODE=1`; default off until a clean
idle-GPU A/B confirms the gain.

## What changed

The streamed-edge forward previously ran, per tile and layer:

```text
edge_up gather -> edge tensor product ([samples,message_width]) -> message gather
```

The new kernel assigns one thread to `(receiver node, channel, instruction)`,
walks the node's incoming CSR segment in ascending edge order, and accumulates
the output element directly. The FP64 association is unchanged: per edge the
sparse plan entries are summed in plan order, then edges are added in CSR
order — the same order used by the edge path plus gather.

The implementation is `E3TensorProductKokkosT::try_evaluate_node_owned_forward`
(`e3nn_kokkos.cpp`) plus the streamed-path dispatch in
`mace_nonlinear_kokkos_impl.tpp`. It requires the MH-1 fast path with
write-through output and falls back to the edge path otherwise.

## Parity (1,296 water, one `run 0`)

| Quantity | Edge forward | Node-owned forward |
|---|---|---|
| Potential energy | −895939.24763167975 eV | −895939.24763167975 eV |
| max position difference | — | 0.0 Å |
| max force difference | — | **3.91e-14 eV/Å** |

The force gate is 1e-8 eV/Å, so the new dataflow is exact.

## Speed in the same contention window

The A100 was shared and busy during the measurement (`nvidia-smi` reported
100 % utilization and 1.6–1.9 GiB from other processes), so absolute values
are inflated. Two profiled runs were taken back to back in one window.
All speeds below are **µs/atom/step** at 1,296 atoms.

| Path | Profiled window | Projected clean |
|---|---:|---:|
| Edge forward | 280.9 | 145.75 |
| Node-owned forward | 271.2 | **~137.9** |
| Saving | **9.7** | **~7.9** |

The node-owned path removes the `[edges, message_width]` edge-message tensor
and the message gather. The measured saving is **7.9–9.7 µs/atom/step**, i.e.
**5–7 %**. The clean 137.9 projection is the 145.75 baseline minus the
7.9 µs/atom/step measured saving; it is a projection, not a clean measurement.

A formal interleaved wall-clock A/B could not resolve the ratio in this
window: per-round on/off ratios were 0.970 and 1.365 because the external GPU
load moved between arms. It must be rerun on an idle GPU before promotion.

## Why this is not 100 on its own

The change attacks the forward edge-message materialization, about
**7.9 µs/atom/step** on the clean baseline. The remaining budget to
100 µs/atom/step is dominated by:

- reverse MH-1 conv (~34.7 µs/atom/step in the same window),
- polar SCF/Ewald/LE chain plus host gap (~28 µs/atom/step clean),
- small FP64 GEMM/MLP/e3-linear launches.

The mathematics-first architecture note
(`docs/design/MATH_ARCH_2026-10-22.md`) projects **97–115 µs/atom/step** only
when the node-owned tensor product, persistent SCF/host execution (28 → 8.5–12.5)
and grouped small kernels are combined. This change is the first slice of that
program.

## Reproduction

```bash
# build
bash bench/polar_uvu_gemm_engine_2026-10-21/build_gemm.sh
# parity dumps
cd bench/arch_dataflow_2026-10-22
MATOS_MH1_FWD_NODE=0 lmp_gemm -k on g 1 -in in.parity ...
MATOS_MH1_FWD_NODE=1 lmp_gemm -k on g 1 -in in.parity ...
# A/B (needs an idle GPU)
python3 bench/arch_dataflow_2026-10-22/ab_fwd_node_1296.py --rounds 2
```

## Artifacts

- `bench/arch_dataflow_2026-10-22/ab_fwd_node_1296.py`
- `bench/arch_dataflow_2026-10-22/parity_off/`, `parity_on/`
- `bench/arch_dataflow_2026-10-22/on_prof.stdout`, `off_prof.stdout`
- `bench/arch_dataflow_2026-10-22/ab_1296_fwd_node.json`
