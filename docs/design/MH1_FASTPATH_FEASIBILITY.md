# MH-1 / Direct / JIT Fast-Path Feasibility for PolarMACE (MACE-POLAR-1-M)

Status: analysis-only, read-only for engine and model files. No engine code was
changed, no build was run, and no GPU benchmark was run for this report.

## 0. Executive summary

The PolarMACE checkpoint and its format-v3 JSON cannot use the current
MACE-MH-1 direct/JIT fast path. The blocker is architectural, not an export
defect:

- The engine's family descriptor rejects the model with
  `MH-1 node, edge, and hidden irreps are inconsistent.`
- The root causes are four differences from the engine's admitted MH-1 family:
  1. the second interaction uses `128x0e+128x1o` edge features instead of the
     first interaction's `512x0e+512x1o`;
  2. the second interaction keeps a vector component in `hidden_irreps`
     (`512x0e+512x1o` instead of the scalar-only `512x0e`);
  3. the second product returns `512x0e+512x1o` instead of a scalar-only
     `512x0e`;
  4. the second readout is `NonLinearBiasReadoutBlock` (with `linear_mid`)
     instead of `NonLinearReadoutBlock`.
- The original `MACE-POLAR-1-M.model` checkpoint has exactly the same
  architecture. Re-exporting it cannot make it MH-1-compatible without
  changing the learned function.
- A v3 `MH1_UVU` execution contract is embedded in the JSON, but it covers only
  the per-interaction UVU tensor products. It is not a v4 node contract and
  does not prove family compatibility. The v4 contract generator rejects this
  model first at `products[1] output irreps are outside the MH-1 support
  matrix`.
- Recommended route: staged route B. First extend the family analysis and pair
  conditioning and use the non-JIT generic streamed path as a correctness and
  FP32-enablement step (about 5-8 days; not a speed win). Only then invest in
  the v4 contract and generated host/device artifacts for direct/JIT (about
  15-25 additional days; target 1.5-3x on the 87.2 ms backbone in
  Python/native, roughly 1.2-1.5x on the full model while polar remains
  materialized). The 87.2/183.5 ms baselines are LAMMPS timings, so LAMMPS
  direct (route B4) is a separate project required to change them. Full polar
  direct/JIT is also a separate larger project.
- Route A (re-export) is infeasible. Route C (relaxing only the
  `mh1_fast_path` gate) is not correct: the generic streamed branch requires
  the MH-1 conditioned-MLP contribution tables and would either throw or
  compute garbage without them. It is also not a measured speed win: on the
  same 128-atom box the linear engine measured 6.6 ms materialized versus
  19.3 ms generic streamed.

## 1. Inputs and provenance

| Item | Value |
|---|---|
| Engine checkout | `/workspace/group/shared/bza/external-repos/symmetrix-xl` |
| Model JSON | `/workspace/group/shared/bza/task3_sampling_database/mlip/models/mace-polar/polar_symmetrix_probe.json` (316,958,257 bytes) |
| Checkpoint | `/workspace/group/shared/bza/task3_sampling_database/mlip/models/mace-polar/MACE-POLAR-1-M.model` (68,133,235 bytes) |
| Checkpoint archive entry | `mace-polar-spin-2L.model` |
| Checkpoint class | `mace.modules.extensions.PolarMACE` |
| Baseline timings | `/workspace/group/shared/bza/symmetrix-polar/results/engine_compare.json`; `/workspace/group/shared/bza/symmetrix-polar/docs/REPORT_2026-10-19.md` |

The 316 MB JSON is a single line, so its fields were extracted with a
read-only `mmap`-based parser rather than by loading the whole document. The
checkpoint was loaded CPU-only with:

```bash
PYTHONPATH=/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0 \
/opt/matos-tools/conda/envs/matos-compute/bin/python - <<'PY'
import torch
m = torch.load("MACE-POLAR-1-M.model", map_location="cpu", weights_only=False)
# inspect interactions/products/readouts
PY
```

No engine or model file was modified.

The 6.6 / 87.2 / 183.5 ms numbers used below are LAMMPS `Loop time`
measurements from `harness/polar_vs_linear_speed.py` (single A100, one MPI
rank, 128-atom NH3 box, `run 1`). They exercise the LAMMPS pair styles, not
the Python ASE calculator. Python/native direct/JIT gains therefore do not
appear in those numbers until the LAMMPS path is wired (route B4).

### 1.1 JSON top-level metadata

```text
"symmetrix_format_version": 3
"model_type": "MACE_Nonlinear"
"head": "Default"
"num_elements": 83
"r_cut": 6.0
"l_max": 3
"num_interactions": 2
"has_zbl": false
"polar_blocks": present
```

The v3 contract is tagged `symmetrix.execution.mh1_uvu/3`, version 3,
`interaction_count: 2`. Its per-interaction UVU irreps are exactly:

```text
interactions[0].irreps:
  irreps_in1 = "512x0e"
  irreps_in2 = "1x0e+1x1o+1x2e+1x3o"
  irreps_out = "512x0e+512x1o+512x2e+512x3o"
  channels   = 512
interactions[1].irreps:
  irreps_in1 = "128x0e+128x1o"
  irreps_in2 = "1x0e+1x1o+1x2e+1x3o"
  irreps_out = "128x0e+128x0e+128x1o+128x1o+128x1o+128x2e+128x2e+128x2e+128x3o+128x3o"
  channels   = 128
```

The radial basis has 8 Bessel weights (`"shape": [8]`). The contract has no
`products`, `readouts`, or `node_program` section; see Section 2.3.

## 2. Exact rejection chain

There is no class or function named `MH1Family` in this checkout. The exact
symbols are:

- `MaceMH1FamilyDescriptor analyze_mh1_family_architecture(const nlohmann::json&)`,
  declared in `libsymmetrix/source/mace_nonlinear_schema.hpp:17` and defined in
  `libsymmetrix/source/mace_nonlinear_schema.cpp:106`.
- The descriptor struct is `MaceMH1FamilyDescriptor` at
  `libsymmetrix/source/mace_nonlinear_schema.hpp:7-14`.
- It is called from the nonlinear constructor at
  `libsymmetrix/source/mace_nonlinear.cpp:372` and
  `libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp:3553`.

### 2.1 Actual irreps in this model

All strings below are copied verbatim from the JSON.

| Object | Field | Exact value |
|---|---|---|
| Node embedding | `irreps_in` | `83x0e` |
| Node embedding | `irreps_out` | `512x0e` |
| interaction[0] | `class` | `RealAgnosticResidualNonLinearInteractionBlock` |
| interaction[0] | `node_feats_irreps` | `512x0e` |
| interaction[0] | `edge_irreps` | `512x0e` |
| interaction[0] | `hidden_irreps` | `512x0e+512x1o` |
| interaction[0] | `target_irreps` | `512x0e+512x1o+512x2e+512x3o` |
| interaction[0] | `gate.irreps_in` | `2048x0e+512x1o+512x2e+512x3o` |
| interaction[0] | `gate.irreps_out` | `512x0e+512x1o+512x2e+512x3o` |
| interaction[0] | `conv_tp.irreps_in1` | `512x0e` |
| interaction[0] | `conv_tp.irreps_in2` | `1x0e+1x1o+1x2e+1x3o` |
| interaction[0] | `conv_tp.irreps_out` | `512x0e+512x1o+512x2e+512x3o` |
| interaction[1] | `class` | `RealAgnosticResidualNonLinearInteractionBlock` |
| interaction[1] | `node_feats_irreps` | `512x0e+512x1o` |
| interaction[1] | `edge_irreps` | `128x0e+128x1o` |
| interaction[1] | `hidden_irreps` | `512x0e+512x1o` |
| interaction[1] | `target_irreps` | `512x0e+512x1o+512x2e+512x3o` |
| interaction[1] | `gate.irreps_in` | `2048x0e+512x1o+512x2e+512x3o` |
| interaction[1] | `gate.irreps_out` | `512x0e+512x1o+512x2e+512x3o` |
| interaction[1] | `conv_tp.irreps_in1` | `128x0e+128x1o` |
| interaction[1] | `conv_tp.irreps_in2` | `1x0e+1x1o+1x2e+1x3o` |
| interaction[1] | `conv_tp.irreps_out` | `128x0e+128x0e+128x1o+128x1o+128x1o+128x2e+128x2e+128x2e+128x3o+128x3o` |
| products[0] | `node_feats_irreps` | `512x0e+512x1o` |
| products[0] | `target_irreps` | `512x0e+512x1o` |
| products[0] | `symmetric_contractions.irreps_in` | `512x0e+512x1o+512x2e+512x3o` |
| products[0] | `symmetric_contractions.irreps_out` | `512x0e+512x1o` |
| products[0] | contractions | 2, both correlation 3 |
| products[0] | `weights_max` shapes | `[1, 23, 512]`, `[1, 51, 512]` |
| products[1] | `node_feats_irreps` | `512x0e+512x1o` |
| products[1] | `target_irreps` | `512x0e+512x1o` |
| products[1] | `symmetric_contractions.irreps_in` | `512x0e+512x1o+512x2e+512x3o` |
| products[1] | `symmetric_contractions.irreps_out` | `512x0e+512x1o` |
| products[1] | contractions | 2, both correlation 3 |
| products[1] | `weights_max` shapes | `[1, 23, 512]`, `[1, 51, 512]` |
| readouts[0] | `class` | `LinearReadoutBlock` |
| readouts[0] | linear | `512x0e+512x1o` -> `1x0e` |
| readouts[1] | `class` | `NonLinearBiasReadoutBlock` |
| readouts[1] | `linear_1` | `512x0e+512x1o` -> `16x0e` |
| readouts[1] | `linear_mid` | `16x0e` -> `16x0e` |
| readouts[1] | `linear_2` | `16x0e` -> `1x0e` |

The original checkpoint reproduces these exactly:

```text
PolarMACE, r_max=6.0, num_interactions=2, heads=['Default']
interaction 0: node_feats 512x0e, edge 512x0e, hidden 512x0e+512x1o
interaction 1: node_feats 512x0e+512x1o, edge 128x0e+128x1o,
               hidden 512x0e+512x1o
product 0: sc out 512x0e+512x1o, 2 contractions, correlation [3,3]
product 1: sc out 512x0e+512x1o, 2 contractions, correlation [3,3]
readout 0: LinearReadoutBlock
readout 1: NonLinearBiasReadoutBlock (linear_1, linear_mid, linear_2)
```

### 2.2 Check-by-check result

`analyze_mh1_family_architecture` returns at the first failing check. The
first failing check is the combined node/edge/hidden check. The later failures
are latent and would surface if the earlier checks were relaxed.

| Order | Check (source lines in `mace_nonlinear_schema.cpp`) | This model | Result |
|---|---|---|---|
| 1 | exactly 2 interactions/products/readouts (114-118) | 2/2/2 | pass |
| 2 | `l_max` is 2 or 3 (119-121) | 3 | pass |
| 3 | non-empty radial basis (122-125) | 8 Bessel weights | pass |
| 4 | both interactions are `RealAgnosticResidualNonLinearInteractionBlock` (127-133) | both are | pass |
| 5 | first node and edge irreps are scalar (135-142) | `512x0e`, `512x0e` | pass |
| 6a | second `node_feats_irreps` equals `{512x0e, 512x1o}` (155-158) | `512x0e+512x1o` | pass |
| 6b | second `edge_irreps` equals `{512x0e, 512x1o}` (157-158; `edge_channels` taken from the first interaction at 143-144) | `128x0e+128x1o` | **fail** |
| 6c | first `hidden_irreps` equals `{512x0e, 512x1o}` (159-160) | `512x0e+512x1o` | pass |
| 6d | second `hidden_irreps` equals `{512x0e}` (161-162) | `512x0e+512x1o` | **fail** |
| 6 | combined rejection (163) | | `MH-1 node, edge, and hidden irreps are inconsistent.` |
| 7 | both `target_irreps` equal the full `l=0..3` set; gate in/out match (164-174) | all match | pass |
| 8 | both products use `use_sc`, `use_agnostic_product`, full contraction input, correct output, one contraction per output block, correlation 3, `weights_max.shape[0] == 1` (175-197) | product 0 passes; product 1 has output `512x0e+512x1o` (expected `512x0e`) and 2 contractions (expected 1) | **fail** |
| 8 | latent rejection (190) | | `MH-1 family execution requires agnostic products with skip connections.` |
| 9 | external weighted UVU tensor products using `edge_channels` for both layers (199-201; helper 72-102) | layer 0 passes at 512; layer 1 is `128x...` | **fail** |
| 9 | latent rejection (201) | | `MH-1 family execution requires external weighted uvu tensor products.` |
| 10 | readouts are exactly `LinearReadoutBlock` then `NonLinearReadoutBlock` (202-206) | second is `NonLinearBiasReadoutBlock` | **fail** |
| 10 | latent rejection (206) | | `MH-1 family execution requires linear then nonlinear readouts.` |

`has_external_uvu_tensor_layout` (lines 72-102) also requires the stored
tensor-product weight to be empty, at least one instruction,
`connection_mode == "uvu"`, `has_weight == true`, input-2 multiplicity 1, and
`path_shape == {multiplicity, 1}`. This model satisfies those conditions at its
own per-layer multiplicity; the failure is solely the hard-coded global
multiplicity of 512.

Two details matter for interpretation:

- The single `edge_channels` field in `MaceMH1FamilyDescriptor` is taken from
  the first interaction. Both the second edge check and the second external-UVU
  check reuse it. That is the direct cause of the 128-versus-512 rejection.
- The combined check also requires the second `hidden_irreps` to be
  scalar-only. This model's second hidden state is scalar plus vector, so the
  rejection is not only an edge-channel mismatch.

### 2.3 Why the embedded MH1_UVU contract is not sufficient

`execution_contracts.MH1_UVU` is generated by
`make_execution_mh1_contract` in
`symmetrix/source/symmetrix/execution_mh1_contract.py:1622-1650`. That function
builds only the two interaction-level UVU descriptors plus radial/layout/
derivative metadata. It does not call `_product_descriptor` or
`_readout_descriptor` and does not build a node program.

The v4 node contract is generated by `make_execution_mh1_v4_contract`
(`execution_mh1_contract.py:1653-1668`), which calls `_node_program_descriptor`
and therefore `_product_descriptor` and `_readout_descriptor`. For this model
it fails first at:

```text
products[1] output irreps are outside the MH-1 support matrix
```

from `execution_mh1_contract.py:529-535` (the code hard-codes a scalar-only
output for product index 1). If that were relaxed, the next v4 failure would be
in `_readout_descriptor` (`execution_mh1_contract.py:783-825`), which requires
`NonLinearReadoutBlock` exactly and has no `linear_mid` support.

## 3. What each fast-path gate requires, and which one fails

### 3.1 Kokkos constructor gate chain

In `MaceNonlinearKokkosT`:

```text
mh1_family            = analyze_mh1_family_architecture(data)
mh1_compiled_products = all products use a compiled plan
mh1_external_uvu      = all interactions' conv_tp uses the MH-1 tensor-product path
primitive_candidate   = mh1_family.compatible && compiled && external_uvu
mh1_pair_conditioning = primitive_candidate
                        && each interaction.prepare_pair_conditioning(...)
mh1_fast_path         = primitive_candidate && mh1_pair_conditioning
```

Source: `libsymmetrix/source/mace_nonlinear_kokkos_impl.tpp:3553-3573`.

| Gate | Exact requirement | This model | Consequence |
|---|---|---|---|
| `is_mh1_family()` | `mh1_family.compatible` from all analyzer checks | false, reason `MH-1 node, edge, and hidden irreps are inconsistent.` | all downstream gates short-circuit |
| `mh1_compiled_products` | both products `uses_compiled_plan()` | true | not the blocker |
| `mh1_external_uvu_tensors` | both `convolution.uses_mh1_fast_path()`; official layout and no internal weights | true | not the blocker; this is a tensor-product-level property, not family compatibility |
| `primitive_candidate` | family && compiled && external UVU | false | pair conditioning is never prepared |
| `mh1_pair_conditioning` | initialized to `primitive_candidate`; `prepare_pair_conditioning` only runs if the candidate is true | false, but **skipped** | this is not an independent architectural failure |
| `mh1_fast_path` | primitive_candidate && pair conditioning | false | `supports_streamed_edges()` is false |
| `supports_streamed_edges()` | returns `mh1_fast_path` | false | generic/direct cannot be selected |
| `supports_factorized()` | `mh1_fast_path` first; then per-interaction `convolution.supports_execution_uvu()` and `convolution_weights.supports_final_linear_factorization(num_bessel)` with matching dimensions | false because the first term is false | direct/receiver-factorized rejected |
| `set_streamed_edges("generic")` | `supports_streamed_edges()` | false | throws `Streamed edges require a compatible MACE-MH-1 family fast path.` (`mace_nonlinear.cpp:315-317`; Kokkos impl 2225-2227) |
| `set_streamed_edges("direct")` | `supports_streamed_edges()` and `supports_factorized()` | false | throws the same family error first; if the family gate were passed, it would require a factorable MH-1 edge tensor product (`impl:2231-2234`) |
| `set_streamed_edges("materialized")` | no streamed requirement | accepted | the only currently selectable mode |
| Float32 constructor | requires `mh1_fast_path` | false | throws `Float32 MACE_Nonlinear requires a compatible MACE-MH-1 family fast path.` (`impl:3584-3587`) |
| Default streamed mode | `direct` if `supports_streamed_edges()`, else `materialized` | materialized | matches the observed materialized execution |

Important nuance: `mh1_uses_pair_conditioning=False` in the Kokkos evaluator is
a consequence of the family short-circuit, not proof that this model's
conditioned MLPs are unusable. The structural preconditions look satisfiable:

- `source_embedding`: `83x0e -> 512x0e`; `target_embedding`: `83x0e -> 512x0e`.
- `conditioned_input_dimension = 8 + 512 + 512 = 1032`.
- `conv_tp_weights` and `density_fn` both start with a Linear layer of input
  width 1032.
- The final affine is 64 -> 2048 for layer 0 and 64 -> 1280 for layer 1.
- `supports_conditioned_input(8)` and
  `supports_final_linear_factorization(8)` therefore look true for both layers.

Thus the model fails exactly one primitive gate: the family architecture. The
other false flags (`supports_streamed_edges`, `supports_factorized`,
`mh1_pair_conditioning`) are short-circuit consequences, not independent
numerical failures. Pair conditioning is skipped, not proven impossible.

This should be confirmed at runtime as the first validation gate of route B.

### 3.2 Architecture vs export loss vs engine narrowness

| Question | Answer | Evidence |
|---|---|---|
| Is the model architecture incompatible with the engine's admitted MH-1 family? | Yes, on four axes: second edge channels, second hidden irreps, second product output, second readout class | Section 2; analyzer and v4 contract |
| Did the extractor/export lose information? | No | The original checkpoint has the same irreps and classes; `extract_mace_nonlinear.py` serializes `linear_mid`; `extract_polar_blocks.py` records the full polar block set |
| Does the engine implement only a narrower family? | Yes | The family analyzer uses one `edge_channels`; v4 product policy hard-codes a scalar-only second output; v4 readout policy requires `NonLinearReadoutBlock` and has no `linear_mid`; reference tests cover equal edge channels and the standard readout |
| Can a different exporter layout fix it? | No, without changing weights or the learned function | The 128-channel tensor-product weights, 128-block outputs, and `linear_mid` are part of the checkpoint |

## 4. Route assessment

### 4.0 Route comparison

| Route | Scope | Expected speedup | Effort | Risk | Key validation gates | Verdict |
|---|---|---|---|---|---|---|
| A | Re-export the checkpoint with an MH-1-compatible layout | None; not executable on the fast path | 0 days to prove infeasible; weeks-months to retrain/distill | High; changes the learned function | Checkpoint/JSON architecture equality | Reject |
| B1 | Family analysis + pair conditioning + generic streamed path (no JIT) | ~0.5-1.0x; FP32 may help; not a speed guarantee | 5-8 days | Medium | Family fixture; pair-conditioning parity; FP32/FP64 parity; polar SCF parity | Recommended first stage |
| B2 | B1 + v4 contract + JIT host/device artifacts for the backbone; polar materialized | 1.5-3x backbone; ~1.2-1.5x full model in Python/native | +15-25 days (20-33 total) | High | v4 contract; codegen; artifact ABI; numerical parity; performance/memory | Recommended if Python/native and backbone-bound |
| B3 | B2 + polar SCF/reverse under direct/JIT | Up to ~1.5-2x full model | +10-20 days | Very high | Polar observables; reverse parity; stress gate | Only after B2 parity |
| B4 | LAMMPS direct/JIT wiring | Required to change the measured LAMMPS timings | +10-15+ days | Very high | Pair-style parity; MPI qualification | Separate project |
| C | Relax only the gate, or add a new generic streaming kernel | <=1.0x; likely slower | 0.5-1 day (incorrect) or 5-10 days (correct kernel) | High | Numerical parity; not a speed route | Do not |

### Route A: re-export or convert the checkpoint with a compatible layout

Finding: infeasible for this checkpoint.

- The loaded `PolarMACE` has `edge_irreps = 128x0e+128x1o` and
  `NonLinearBiasReadoutBlock`; the JSON matches exactly.
- The second interaction's `conv_tp_weights` final affine is 64 -> 1280
  (10 paths times 128 channels), not 64 -> 2048. Reinterpreting it as 512
  channels would require padding, duplicating, or retraining weights and would
  change the model.
- The second product's `symmetric_contractions` has two correlation-3
  contractions producing a scalar and a vector. Dropping the vector output
  changes the second layer and the readout input.
- The readout has a learned 16x16 `linear_mid` stage between two SiLU
  activations. Relabeling it as `NonLinearReadoutBlock` would silently ignore
  that stage.
- The checkpoint has only one head (`Default`), so there is no alternate
  compatible head.

Expected speedup: none (not executable on the fast path). Effort: 0 days to
prove infeasibility (done here); weeks to months if the goal becomes training
or distilling a new MH-1-compatible PolarMACE. Risk: changing the learned
function; not acceptable for a performance-only task. Verdict: reject.

### Route B: extend the engine to the PolarMACE pattern

#### B1. Family analysis + pair conditioning + generic streamed path (no JIT)

What changes:

- `MaceMH1FamilyDescriptor` must carry per-layer edge channels and the
  second-layer product/readout policy instead of one global `edge_channels`.
- `analyze_mh1_family_architecture` must accept:
  - second edge `e2x0e+e2x1o` with `e2 != node_channels`;
  - second hidden `node_channels x0e + node_channels x1o` (the PolarMACE
    pattern; the existing scalar-only pattern can remain supported);
  - second product output `node_channels x0e + node_channels x1o` (the
    PolarMACE pattern; the existing scalar-only pattern can remain supported);
  - `NonLinearBiasReadoutBlock` (and the existing `NonLinearReadoutBlock`).
  It must keep the strict checks for `l_max`, first-layer scalars, gate shapes,
  correlation 3, `use_sc`, `use_agnostic_product`, path weights, output masks,
  and external UVU instructions.
- `has_external_uvu_tensor_layout` must take the per-layer multiplicity
  (512 for layer 0, 128 for layer 1).
- `prepare_pair_conditioning` itself is already per-interaction and
  channel-agnostic; it should pass once it is actually called. Add assertions
  and tests for layer 1.
- `supports_factorized` should then be re-checked; the underlying UVU path and
  final-linear factorization appear available for both layers.
- Generic streaming uses the existing `streams_layer` branch. That branch
  requires the conditioned-MLP contribution tables, which pair conditioning
  provides. The polar guard only rejects direct/prepared execution, so generic
  streaming can coexist with the polar pipeline in principle.
- FP32 becomes selectable because the constructor's `mh1_fast_path` gate is
  passed. This may be the largest immediate benefit of B1, because the model
  is currently float64-only.

Expected speedup: roughly 0.5-1.0x for generic streaming itself. The measured
linear-engine comparison on this box is 6.6 ms materialized versus 19.3 ms
generic streamed, so generic streaming is not a speed route. Its value is
FP32 enablement, memory reduction for large systems, and a correctness bridge.
Effort: 5-8 days. Risk: medium (family-policy errors, conditioning parity).
Verdict: do this first as a staged enabler, not as the final performance fix.

#### B2. Direct/JIT v4 for the backbone, polar kept materialized

What changes beyond B1:

- Extend the v4 contract generator:
  - `_product_descriptor` must allow the second product's scalar+vector
    output and the corresponding contraction count/dimensions
    (`execution_mh1_contract.py:500-546`).
  - `_readout_descriptor` must allow `NonLinearBiasReadoutBlock` and carry
    `linear_mid` through the descriptor, runtime parameter pack, semantic and
    structure payloads (`execution_mh1_contract.py:780-838`).
  - `_node_program_descriptor`, `_node_layer_descriptor`, and the normalizer
    must be updated consistently; the node arena layout already derives
    dimensions from the contract, but the contracted/readout dimensions change
    (the second product output is 2048, not 512, and the readout has an extra
    16x16 stage).
- Extend `mh1_jit_codegen.py`:
  - `_render_node_product_helpers` is mostly generic over components and should
    handle the scalar+vector second product once the contract admits it.
  - `_render_node_readout_helpers` currently emits `linear_1 -> SiLU ->
    linear_2`. Add the `linear_mid -> SiLU -> linear_2` form for the bias
    readout.
  - Update runtime parameter pack offsets, node arena sizing, launch policies,
    and artifact metadata for the 128-channel second interaction and the
    extra readout stage.
- Extend `jit_mh1_host_plugin.cpp` / `jit_mh1_cuda_plugin.cpp` validation for
  the new readout parameter count and per-layer dimensions. Bump the contract/
  artifact identity if the runtime layout changes.
- In `compute_node_energies_forces`, the `polar_enabled` guard
  (`impl:4315-4322`) must remain in force for direct/prepared execution unless
  the polar pipeline is ported. The current guard says:
  `polar_blocks requires the standard materialized evaluation graph.`
- The constructor's default `streamed_edges = direct` when
  `supports_streamed_edges()` is true will need a polar-aware policy: a
  compatible polar model must default to generic until the polar direct path
  exists. Note that a fast-path MH-1 model rejects `materialized` in
  `set_streamed_edges` (`impl:2219-2224`), so materialized is not an available
  fallback once the family gate is passed; generic is the only safe default.

Expected speedup: 1.5-3x on the backbone is a reasonable engineering target,
but this is an estimate, not a measurement. These estimates describe
Python/native evaluation; the measured 87.2/183.5 ms numbers are LAMMPS
timings, so B2 alone would not change them. Using the measured split
(87.2 ms backbone, about 96 ms polar, 183.5 ms full):

| Backbone speedup | Backbone (ms) | Full model with materialized polar (ms) | Overall speedup |
|---:|---:|---:|---:|
| 1.5x | 58.1 | 154.4 | 1.19x |
| 2.0x | 43.6 | 139.9 | 1.31x |
| 3.0x | 29.1 | 125.4 | 1.46x |

If the polar stage also benefits, the overall ceiling is higher, but that is a
separate route (B3). Effort: 15-25 days beyond B1 (20-33 days total). Risk:
high (numerical contracts, generated artifacts, fail-closed behavior).
Verdict: recommended only if the target is the Python/native evaluator and
profiling confirms the backbone is the dominant cost.

#### B3. Full polar direct/JIT

What changes:

- Port the polar SCF, per-layer source maps, mixer, field updates, readout,
  and reverse into the generated node-program execution model, or materialize
  the per-layer node features/adjoints that the polar stages require while
  keeping the edge path direct.
- Remove the polar guard only after parity is demonstrated.
- Implement the polar cell adjoint before claiming stress/virial support; the
  current code explicitly throws:
  `MH-1 stress reduction: polar cell adjoint (virial) is not implemented yet;
  disable virial/stress for polar models.` (`impl:7302-7310`).

Expected speedup: up to roughly 1.5-2x full model if the polar stage also
streams, but the polar stage is a coupled SCF and may not benefit as much as
the backbone. Effort: 10-20 days beyond B2. Risk: very high (coupled physics,
reverse-mode correctness, stress gap). Verdict: only after B2 parity and a
profile showing a worthwhile polar-stage cost.

#### B4. LAMMPS direct/JIT

Current state:

- The serial pair style rejects `MACE_Nonlinear` outright
  (`pair_symmetrix/pair_symmetrix_mace.cpp:163-164`).
- The Kokkos pair style loads `MACE_Nonlinear` only materialized. If
  `supports_streamed_edges()` is true it errors with
  `MACE_Nonlinear direct/JIT execution is not wired into this pair style;
  only materialized MACE_Nonlinear models (PolarMACE) are supported.`
  (`pair_symmetrix/pair_symmetrix_mace_kokkos.cpp:491-495`). It also requires
  `streamed_edges_mode() == "materialized"` (lines 506-509).
- The support matrix states MACE-MH-1 is not supported in LAMMPS
  (`docs/reference/execution_support_matrix.md:8`), and the pair-style README
  says format-v3 nonlinear models are Python/native only
  (`pair_symmetrix/README.md:44-46`). The README statement that both pair
  styles reject these models is stale relative to the current Kokkos
  materialized loader, but the direct/JIT gap is real.

The measured 87.2/183.5 ms baselines are LAMMPS timings, so this is the route
that would actually change them. Effort: 10-15+ days for the pair-style
wiring, H1 forward/reverse exchange, artifact loading, and multi-rank
qualification. Risk: very high. Verdict: a separate project; do not assume B2
automatically enables LAMMPS.

### Route C: relax only the `mh1_fast_path` gate and use generic streaming

This is not correct as a one-line change.

What the code requires for generic streaming:

- `set_streamed_edges("generic")` calls `supports_streamed_edges()` and throws
  unless `mh1_fast_path` is true (`impl:2225-2227`).
- When `streams_layer(layer)` is true, the forward branch
  (`impl:4454-5176`) uses:
  - `interaction.convolution_source_contributions`,
  - `interaction.convolution_target_contributions`,
  - `interaction.density_source_contributions`,
  - `interaction.density_target_contributions`,
  - `convolution_weights.evaluate_conditioned_prefix(...)` and
    `evaluate_conditioned(...)`.
- Those tables are prepared by `Interaction::prepare_pair_conditioning`, which
  is called only when `primitive_candidate` is true. They are empty otherwise.
- The correct generic materialized fallback (`source_embedding.evaluate(attrs)`
  and `convolution_weights.evaluate(edge_features)`) is in the `else` of
  `if(stream_layer)` (`impl:5202-5234`), so it is skipped in streaming mode.

Therefore, forcing `mh1_fast_path=true` without extending the family and
preparing pair conditioning would either hit zero-width contribution tables or
feed the conditioned MLP with the wrong number of columns. It would not be
numerically correct.

Additional reasons not to do this:

- The float32 constructor would start accepting the model with an unvalidated
  fast-path flag.
- The test suite explicitly asserts that a near-MH-1 model raises
  `set_streamed_edges("generic")` with
  `compatible MACE-MH-1 family` (`symmetrix/test/test_mh1.py:2648-2679`).
- The measured generic streamed path is slower than materialized on the same
  128-atom box (6.6 ms versus 19.3 ms for the linear engine). Even a correct
  generic implementation is not a speed win.

Effort: 0.5-1 day for an incorrect gate relaxation; 5-10 days for a correct
new generic streaming kernel that materializes source/target embeddings and
evaluates the MLP per edge. Expected speedup: <= 1.0x (memory reduction only).
Risk: high (silent numerical errors). Verdict: do not.

## 5. Can the linear engine run any PolarMACE layer?

No. The linear engine and the nonlinear engine are different model families,
not two backends for the same graph.

| Aspect | `MACEKokkos` (linear engine) | `MaceNonlinearKokkosT` (nonlinear engine) |
|---|---|---|
| Accepted `model_type` | `MACE`, `MACEField` | `MACE_Nonlinear` |
| JSON format | format-version 2 compact radial for this Kokkos loader (`uses_compact_radial = (format_version == 2)`) | format-version 3 nonlinear (`symmetrix_format_version == 3`) |
| Interaction | standard MACE interaction, M0/M1/R0/R1 contraction pipeline | `RealAgnosticResidualNonLinearInteractionBlock` with per-edge conditioned MLP weights, gate, correlation-3 product |
| Readout | standard linear MACE readout | `LinearReadoutBlock` then `NonLinearReadoutBlock` / `NonLinearBiasReadoutBlock` |
| Polar | `polar_blocks_enabled()` returns false (inert placeholder) | full polar block set and SCF pipeline |
| Direct gate | `supports_streamed_edges()` = `uses_compact_radial`; `supports_factorized()` = streamed plus `num_LM` limits | `supports_streamed_edges()` = `mh1_fast_path`; `supports_factorized()` = fast path plus UVU/factorization checks |
| H1 MPI | H1 is the first-interaction hidden node state (`num_LM x num_channels`) | `polar_mpi_message_passing` exchanges node features; materialized only |

Evidence:

- `MACEKokkos::load_from_json` rejects `MACE_Nonlinear` at
  `libsymmetrix/source/mace_kokkos_model.cpp:165-167`:
  `MACE_Nonlinear JSON must be loaded through the nonlinear MACE Kokkos
  evaluator, not legacy MACEKokkos.`
- `uses_compact_radial` is set at `mace_kokkos_model.cpp:201-205`; the
  compact-radial branch requires `radial_representation == "compact"` and the
  non-compact branch in this loader ends in an unsupported-format error at
  lines 440-442.
- `MACEKokkos::supports_streamed_edges()` returns `uses_compact_radial`
  (`mace_kokkos_runtime.cpp:592-595`); `supports_factorized()` adds the
  `num_LM` limits (lines 608-613).
- `MACEKokkos::polar_blocks_enabled()` returns false
  (`mace_kokkos.hpp:101-106`), with a comment that PolarMACE must dispatch to
  the nonlinear engine.

Consequently, the backbone cannot be delegated layer-by-layer to
`MACEKokkos`, and the linear engine's `mpi_message_passing` H1 path cannot be
reused for the nonlinear model. The naming is a collision: H1 in the linear
engine means the first-interaction hidden state, not the MACE-MH-1 model
family. The nonlinear pair style already has a `polar_mpi_message_passing`
path, but it is materialized. The 2026-10-19 report records a 3.4x 2-rank
slowdown for the upstream linear H1 message passing and 9.9x/28.1x slowdowns
for the ghost-based nonlinear variants; multi-rank is not a substitute for
the missing single-rank fast path.

## 6. Recommended route and implementation plan

Recommendation: route B, staged. Do B1 first. Only proceed to B2 if the target
is the Python/native evaluator (not LAMMPS) and a profile confirms the backbone
is the dominant remaining cost. Keep the polar pipeline materialized until B2
parity is established; treat full polar direct/JIT (B3) and LAMMPS direct (B4)
as separate projects.

### Phase 0: freeze the evidence (done)

- Record the exact rejection reason, all irreps, the checkpoint architecture,
  and the v3/v4 contract distinction.
- Confirm the checkpoint and JSON are the same architecture.

### Phase 1: family descriptor extension (2-3 days)

- Extend `MaceMH1FamilyDescriptor` with per-layer edge channels and a
  product/readout policy.
- Update `analyze_mh1_family_architecture` and
  `has_external_uvu_tensor_layout` to use the per-layer multiplicity.
- Update the Python bindings/getters if the descriptor layout changes; keep
  `mh1_edge_channels` backward-compatible (the current tests expect the
  first-layer value) or add per-layer getters.
- Add a synthetic PolarMACE-like fixture test that must pass, plus negative
  tests for parity, `l_max`, correlation, path weights, and output masks.
- Keep the exact rejection strings for genuinely unsupported models.

### Phase 2: pair conditioning and generic streaming (3-5 days)

- Confirm `prepare_pair_conditioning` returns true for both layers and that
  the contribution tables are finite and match the materialized edge features.
- Enable `streamed_edges="generic"` behind the extended family gate.
- Verify the polar guard: generic must not enter the direct/prepared guard, and
  the polar per-layer source maps must see the same node features as in
  materialized execution.
- Test FP32 and FP64 on CPU and the available GPU backend.
- Benchmark against FP64 materialized. Do not assume generic is faster.

### Phase 3: go/no-go for direct/JIT (1 day)

- Profile the backbone and the polar stage separately on the target graph.
- If the backbone is not the dominant cost, stop at B1 (generic/FP32) or keep
  the current materialized baseline.
- If the backbone is dominant and the deployment is Python/native, proceed.

### Phase 4: v4 contract extension (5-8 days)

- Allow the second product's scalar+vector output in `_product_descriptor`.
- Allow `NonLinearBiasReadoutBlock` and carry `linear_mid` through
  `_readout_descriptor`, runtime parameter packs, structure/semantic payloads,
  and the node program normalizer.
- Regenerate fingerprints and bump the contract/artifact identity if the
  runtime layout changes.
- Add exact negative tests for malformed products/readouts.

### Phase 5: JIT codegen and plugins (5-8 days)

- Add the `linear_mid` stage to `_render_node_readout_helpers`.
- Verify `_render_node_product_helpers` handles the scalar+vector second
  product, including component offsets and reverse accumulation.
- Update node arena sizing, runtime parameter pack offsets, launch policies,
  and host/CUDA plugin validation.
- Generate host and device artifacts; verify fail-closed loading and artifact
  identity.

### Phase 6: polar integration or explicit fallback (2-10 days)

- If polar remains materialized, add an explicit policy that a compatible
  polar model defaults to generic and direct is rejected before evaluation,
  not at the `polar_blocks` guard inside the hot path. Materialized is not an
  available selector once the MH-1 fast-path gate is passed.
- If polar direct is in scope, port the SCF/reverse and implement the polar
  cell adjoint before claiming stress support.

### Phase 7: numerical and performance gates (3-5 days)

Required gates:

1. Architecture gate: the synthetic PolarMACE fixture is admitted; malformed
   variants are rejected with the intended reason.
2. Pair-conditioning gate: both layers prepare; contribution tables match the
   materialized edge features.
3. Numerical parity gate: non-Kokkos float64 vs Kokkos materialized vs new
   generic/direct for energy, per-atom energy, forces, and (where supported)
   stress. Cover periodic and triclinic cells, cutoff-edge cases, ghost atoms,
   and multiple graph sizes.
4. Float32 gate: parity against float64 within an agreed tolerance; no NaNs;
   deterministic repeated runs.
5. Polar gate: charges, electrostatic potentials, Fukui source, local electron
   energy, and SCF convergence against the current materialized reference.
6. JIT gate: host and CUDA artifact parity, ABI/identity validation, and
   fail-closed behavior when the artifact is missing.
7. Performance gate: no regression versus materialized; record backbone and
   full-model speedups separately; record memory high-water marks.
8. LAMMPS gate (only if B4 is in scope): pair-style parity and MPI
   qualification.

## 7. Explicit "do not do this" items

- Do not force `mh1_fast_path=true`, bypass `analyze_mh1_family_architecture`,
  or delete the family check. The generic/direct branches would use unprepared
  conditioned-MLP tables and produce wrong results or throw.
- Do not pad, duplicate, truncate, or zero-extend the second interaction's
  128 edge channels to 512. That changes the learned tensor-product weights and
  the model function.
- Do not drop the second product's vector output or relabel
  `NonLinearBiasReadoutBlock` as `NonLinearReadoutBlock`. The `linear_mid`
  stage is learned and must be preserved.
- Do not silently ignore `linear_mid` in the v4 contract, codegen, runtime
  parameter pack, or plugin ABI.
- Do not enable direct/prepared execution for polar models by removing the
  `polar_blocks requires the standard materialized evaluation graph` guard
  without porting the polar SCF and reverse. The guard exists for correctness.
- Do not claim stress or virial parity for polar models. The polar cell adjoint
  is not implemented and the engine throws for polar stress.
- Do not present generic streaming as a speed fix. On the measured 128-atom
  box it is slower than materialized.
- Do not conflate the linear engine's H1 MPI state with the MACE-MH-1 family,
  and do not assume the linear engine can run any nonlinear layer.
- Do not assume LAMMPS support follows from Python/native direct support. The
  Kokkos pair style currently rejects direct/JIT for nonlinear models.
- Do not change weights, engine source, or model files as part of a
  performance-only task; do not commit or push this analysis.

## 8. Bottom line

| Question | Answer |
|---|---|
| Can the model use the MH-1/direct/JIT fast path today? | No |
| Exact first rejection | `MH-1 node, edge, and hidden irreps are inconsistent.` |
| Primary cause | Second interaction edge `128x0e+128x1o` versus the engine's single 512-channel MH-1 edge count; second hidden also keeps a vector component |
| Is this an export bug? | No; the checkpoint matches the JSON |
| Is a v3 `MH1_UVU` contract present? | Yes, but it is interaction-level only and does not imply family compatibility |
| Cheapest correct route | B1: extend the family analysis and pair conditioning, use generic streaming as a validation/FP32 bridge (~5-8 days; no speed guarantee) |
| Speed route | B2: v4 contract + JIT codegen + host/device artifacts for the backbone (~15-25 additional days; target 1.5-3x backbone, 1.2-1.5x full model with polar materialized) |
| Full polar direct/JIT | B3: separate 10-20 day effort; only after B2 parity |
| LAMMPS direct | B4: separate 10-15+ day project; currently explicitly unsupported |
| Route A (re-export) | Infeasible without retraining; do not pursue for performance |
| Route C (relax gate only) | Incorrect and likely slower; do not pursue |
