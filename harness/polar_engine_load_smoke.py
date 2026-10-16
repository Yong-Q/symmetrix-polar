#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""polar_blocks 引擎解析冒烟(v12 后, 2026-10-15)。

正例: 含 polar_blocks 的 JSON 正常加载(解析 lr_source_maps×2 + fukui_source_map)。
负例: 把 lr_source_maps 截为 1 条 → 引擎应抛 "must match the interaction layer count"。

用法: CUDA_VISIBLE_DEVICES=<gpu> <polar venv python> polar_engine_load_smoke.py
环境: SYMMETRIX_BACKEND_DESCRIPTOR_PATH / SYMMETRIX_BACKEND / LD_LIBRARY_PATH / PATH shim
"""
import json
import os
import sys

ROOT = "/workspace/group/shared/bza/task3_sampling_database/mlip"
J = f"{ROOT}/models/mace-polar/polar_symmetrix_probe.json"
BAD = f"{ROOT}/bench/polar_bad_lr.json"


def main():
    from symmetrix.calculator import Symmetrix
    c = Symmetrix(J, dtype="float64")
    print("POS-OK: polar JSON loaded (polar_blocks parsed)", flush=True)

    data = json.load(open(J))
    assert "polar_blocks" in data, "fixture JSON lacks polar_blocks"
    data["polar_blocks"]["lr_source_maps"] = data["polar_blocks"]["lr_source_maps"][:1]
    json.dump(data, open(BAD, "w"))
    try:
        Symmetrix(BAD, dtype="float64")
        print("NEG-FAIL: no error raised on corrupted lr_source_maps", flush=True)
        rc = 1
    except Exception as e:
        ok = "lr_source_maps must match" in str(e)
        print(f"NEG-{'OK' if ok else 'UNEXPECTED'}: {str(e)[:160]}", flush=True)
        rc = 0 if ok else 1
    finally:
        try:
            os.remove(BAD)
        except OSError:
            pass
    print("LOAD-SMOKE-DONE", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
