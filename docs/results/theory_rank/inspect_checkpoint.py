#!/usr/bin/env python3
"""Inspect the PolarMACE checkpoint for polar UVU/table structure."""
import sys
import torch

MODEL = "/workspace/group/shared/bza/task3_sampling_database/mlip/models/mace-polar/MACE-POLAR-1-M.model"
sys.path.insert(0, "/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0")

m = torch.load(MODEL, map_location="cpu", weights_only=False)
print("TYPE", type(m))
print("ATTRS", [a for a in dir(m) if not a.startswith("_")][:80])
sd = m.state_dict()
print("NUM_STATE", len(sd))
print("=== polar/uvu-related keys ===")
for k, v in sd.items():
    kl = k.lower()
    if any(s in kl for s in ("polar", "field_dependent", "local_electron", "dot", "tp", "uvu", "charge", "fukui")):
        print(k, tuple(v.shape) if hasattr(v, "shape") else type(v), getattr(v, "dtype", ""))
print("=== 512x512 matrices ===")
for k, v in sd.items():
    if hasattr(v, "shape") and len(v.shape) == 2 and tuple(v.shape) == (512, 512):
        print(k, tuple(v.shape), v.dtype)
print("=== 512xN / Nx512 matrices (N != 512) ===")
for k, v in sd.items():
    if hasattr(v, "shape") and len(v.shape) == 2 and 512 in v.shape and tuple(v.shape) != (512, 512):
        print(k, tuple(v.shape), v.dtype)
