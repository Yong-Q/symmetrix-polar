#!/usr/bin/env python3
"""Figure: queue004 remote A100 FP64/FP32 GEMM microbench (2026-10-22)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/workspace/group/shared/bza/symmetrix-polar")
DATA = ROOT / "docs/results/data/q4_remote_microbench_2026-10-22.json"
OUT = ROOT / "docs/results/figures"

d = json.loads(DATA.read_text())
m = d["tasks"]["toolchain_microbench"]
g = m["gemm"]
peaks = d["a100_sxm_reference_peaks_tflops"]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

# (a) measured FP64/FP32 GEMM throughput against the 19.5 TFLOPS ceiling
ax = axes[0]
labels = ["FP64\n(tensor cores)", "FP32\n(CUDA cores)"]
vals = [g["fp64_tflops"], g["fp32_tflops"]]
colors = ["#4c78a8", "#f58518"]
bars = ax.bar([0, 1], vals, color=colors, width=0.52)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(peaks["fp64_tensor_core"], color="#333333", ls="--", lw=1.1)
ax.text(0.02, peaks["fp64_tensor_core"] + 0.35,
        "A100 FP64-TC / FP32-CUDA peak = 19.5",
        ha="left", va="bottom", fontsize=8.3, color="#333333")
ax.set_xticks([0, 1])
ax.set_xticklabels(labels)
ax.set_ylim(0, 24.0)
ax.set_ylabel("2048$^3$ GEMM throughput (TFLOPS)")
ax.set_title("(a) Remote queue004 A100, measured", fontsize=10.5)
ax.grid(axis="y", alpha=0.25)

# (b) A100 precision ladder: measured vs theoretical peaks
ax = axes[1]
names = ["FP64 TC", "FP32 CUDA", "TF32 TC", "BF16 TC"]
x = np.arange(len(names))
theory = [peaks["fp64_tensor_core"], peaks["fp32_cuda_core"],
          peaks["tf32_tensor_core"], peaks["bf16_tensor_core"]]
ax.bar(x, theory, color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"],
       width=0.58, alpha=0.35, edgecolor="black", linewidth=0.6)
ax.bar([0], [g["fp64_tflops"]], color="#4c78a8", width=0.58)
ax.bar([1], [g["fp32_tflops"]], color="#f58518", width=0.58)
ax.plot([-0.35, 1.35], [peaks["fp64_tensor_core"]] * 2,
        color="#666666", ls="--", lw=0.9)
ax.text(0.5, peaks["fp64_tensor_core"] + 2.0,
        "FP64-TC / FP32-CUDA peak 19.5", ha="center", va="bottom",
        fontsize=7.8, color="#444444")
ax.text(0, 5.5, f"{g['fp64_tflops']:.2f}", ha="center", va="center",
        fontsize=8.2, color="white", fontweight="bold")
ax.text(1, 5.5, f"{g['fp32_tflops']:.2f}", ha="center", va="center",
        fontsize=8.2, color="white", fontweight="bold")
for xi, t in zip(x[2:], theory[2:]):
    ax.text(xi, t + 5, f"peak {t:g}", ha="center", va="bottom", fontsize=8.3)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("A100-SXM4-80GB throughput (TFLOPS)")
ax.set_title("(b) Precision ladder: FP32 alone is not a lever", fontsize=10.5)
ax.set_ylim(0, 355)
ax.grid(axis="y", alpha=0.25)

fig.suptitle("queue004 remote A100: Python/torch microbench "
             f"(task {m['taskId']}, {m['elapsed_s']} s)", fontsize=11.5)
fig.text(0.5, 0.015,
         "torch 2.5.1+cu121, TF32 off (default), 2048$^3$ GEMM, 20 iters; "
         f"FP32 is {1.0 / g['fp64_over_fp32_speed_ratio']:.2f}x the FP64 wall time "
         "(no speedup)",
         ha="center", va="bottom", fontsize=8.3, color="#444444")
fig.text(0.5, -0.015,
         "Filled bars = measured FP64/FP32; faint bars = theoretical A100-SXM4-80GB peaks. "
         "TF32/BF16 require far more precision relaxation than FP32.",
         ha="center", va="bottom", fontsize=8.0, color="#666666")
fig.tight_layout(rect=[0, 0.07, 1, 0.93])
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"Q4_REMOTE_MICROBENCH.{ext}", dpi=150, bbox_inches="tight")
print("wrote", OUT / "Q4_REMOTE_MICROBENCH.png")
