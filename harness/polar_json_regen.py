#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重生成 polar_symmetrix_probe.json(补 configs: num_recursion_steps /
field_feature_norms / from_ell_max_field_update; 2026-10-15)。

运行: PYTHONPATH=<symmetrix-xl>/symmetrix/source <polar venv python> polar_json_regen.py
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import json  # noqa: E402
import sys  # noqa: E402

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
POLAR = f"{ROOT}/models/mace-polar/MACE-POLAR-1-M.model"
J = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"


def main():
    from mace.calculators import mace_polar
    from symmetrix.extract_mace_nonlinear import extract_mace_nonlinear_data

    calc = mace_polar(model=POLAR, device="cpu", default_dtype="float64")
    model = calc.models[0]
    data = extract_mace_nonlinear_data(model)
    pb = data.get("polar_blocks")
    print("polar_blocks present:", pb is not None, flush=True)
    assert pb is not None
    print("configs:", json.dumps(pb["configs"], indent=1), flush=True)
    norms = pb["configs"]["field_feature_norms"]
    print("norms shape:", norms["shape"], "values:",
          [round(v, 4) for v in norms["values"]], flush=True)
    with open(J, "w") as fh:
        json.dump(data, fh)
    print(f"REGEN-OK -> {J}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
