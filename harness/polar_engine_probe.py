#!/usr/bin/env python
"""Probe PolarMACE engine capabilities (load-only; no graph evaluation).

Reports which Symmetrix execution paths the polar model admits, so P5/performance
work targets the right path instead of guessing.
"""
import json
import os

import symmetrix

MODEL = os.environ.get(
    "POLAR_MODEL",
    "/workspace/group/shared/bza/task3_sampling_database/mlip/models/mace-polar/polar_symmetrix_probe.json",
)

if not symmetrix._kokkos_is_initialized():
    symmetrix._init_kokkos()

ev = symmetrix.MACENonlinearKokkos(MODEL, "")
print("engine_class =", type(ev).__name__)
print("r_cut =", ev.r_cut)
print("has_field_coupling =", ev.has_field_coupling)
print("selected_head =", ev.selected_head)
print("supports_streamed_edges =", ev.supports_streamed_edges)
print("supports_factorized =", ev.supports_factorized)
print("is_mh1_family =", ev.is_mh1_family)
print("mh1_family_rejection_reason =", ev.mh1_family_rejection_reason)
print("mh1_fast_path_rejection_reason =", ev.mh1_fast_path_rejection_reason)
print("streamed_edges_mode =", ev.streamed_edges_mode)
print("atomic_numbers =", list(ev.atomic_numbers))
for name in (
    "mh1_node_channels",
    "mh1_edge_channels",
    "mh1_radial_size",
    "mh1_l_max",
    "mh1_uses_compiled_products",
    "mh1_uses_pair_conditioning",
    "mh1_uses_external_uvu_tensors",
    "has_execution_mh1_contract",
    "execution_mh1_generation_fingerprint",
    "execution_mh1_semantic_fingerprint",
    "execution_mh1_structure_fingerprint",
):
    try:
        print(f"{name} =", getattr(ev, name))
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"{name} = <error {exc!r}>")
print("PROBE-DONE")
