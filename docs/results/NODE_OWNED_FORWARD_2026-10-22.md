# Node-owned forward dataflow — 2026-10-22

**Scope:** full PolarMACE, FP64, exact 2-step SCF, 1×A100-80GB.
**Change:** forward MH-1 tensor product writes receiver-node messages directly;
the `[edges, message_width]` edge-message tensor and the separate message
gather are removed. **Default on** after the clean q4 A/B below;
`MATOS_MH1_FWD_NODE=0` opts out.

## Clean q4 A/B (decisive)

One q4 task (`t-20261022180640-dslc9`, 1×A100, 2 rounds × 4 arms, 5 steps per
arm) on 1,296 water. All speeds in **µs/atom/step**.

| Arm | q4 median | Saving vs baseline |
|---|---:|---:|
| Baseline (N1 + R1) | 138.91 | — |
| Node-owned forward | **135.98** | **2.92 (2.1 %)** |
| Fused2 reverse | 136.08 | 2.83 (2.0 %) |
| **Node-owned + fused2** | **133.04** | **5.87 (4.2 %)** |

PE is identical to the baseline on the baseline/node/both arms; the fused2-only
arm differs by 1.1e-10 eV, inside the 1e-9 eV gate. Force parity for the default
combination is **4.62e-14 eV/Å** (gate 1e-8). The clean q4 baseline is itself
lower than the old local 145.75 because R1 is on and the q4 GPU is dedicated.

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

## Local shared-GPU window (superseded)

Before the q4 run, a local shared-GPU window showed the forward conv+gather
phase at **280.9 → 271.2 µs/atom/step** and projected a 7.9 µs/atom/step
saving. The clean q4 A/B above is the authoritative number: **2.92
µs/atom/step** for the forward alone. The local projection was optimistic
because the shared-GPU window moved under the two arms.

## Why this is not 100 on its own

The change attacks the forward edge-message materialization, **2.92
µs/atom/step** on the clean baseline (5.87 together with fused2). The
remaining budget to 100 µs/atom/step is dominated by:

- reverse MH-1 conv,
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
