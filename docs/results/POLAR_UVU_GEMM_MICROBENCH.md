# Polar UVU 分组 FP64 GEMM 微基准（N=1,296 / 5,184）

- 运行日期：2026-10-21（容器时钟）
- GPU：NVIDIA A100-SXM4-80GB（SM 1410 MHz，80 GB），全程持有
  `flock /workspace/group/shared/bza/.tmp/gpu.lock`，同一时刻只有一个 GPU 作业
- 软件：PyTorch 2.8.0+cu128，CUDA 12.8，FP64（TF32 关闭），
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- 模型：`/workspace/group/shared/bza/task3_sampling_database/mlip/models/mace-polar/polar_symmetrix_probe.json`
  （MD5 `7dec0a9a6d9480c2439a9ac7b0e9acc6`，真实 checkpoint 权重）
- 基准脚本 SHA256：`0b88412e0632ac5c2e1b25d584a3f3c96cbf58232217856827068cb88d0108f6`
- **未修改引擎源码，未构建引擎，未 commit/push。**

---

## 0. 中文摘要（结论）

**结论：GO。** 分组 FP64 GEMM 完全满足 2026 年最大的 CUDA 加速项 N1 的前置性能门槛。

- **14-GEMM 组**（文档口径）：1,296 原子 **0.942 ms**（逐调用 loop，10.10 TFLOPS）
  或 **0.599 ms**（单次 batched GEMM，15.89 TFLOPS）；5,184 原子 2.259 ms
  （16.85 TFLOPS）/ 2.143 ms（17.75 TFLOPS）。
- **实际 checkpoint 是 20 个 path-GEMM**，不是 14 个：checkpoint 有 **6 张 UVU 表**
  （两个 `field_dependent_charges_maps`，各含 `dot_products` + `tp_out`，再加 LE q/v），
  前向 20 个 `512 x N x 512` GEMM，共 13.59 GFLOP。CUPTI 的发射数
  （6× `polar uvu mode0`、2× `polar uvu mode1`、4× `polar le uvu mode0`）
  与 6 张表完全吻合。文档 §3.1 的 14-GEMM / 9.5-GFLOP 估计少算了一个 fdc block；
  本报告同时给出 14 和 20 两组，所有投影采用更保守的 20 组。
- **前向组合**（4 平面打包 + 20 GEMM + epilogue）：1,296 原子 **1.885 ms**，
  5,184 原子 **5.632 ms**。
- **反向组合**（两个 X1/X2 adjoint pass）：复用前向 Z 时 1,296 原子 **2.960 ms**，
  5,184 原子 **10.421 ms**；不保存 Z、重算前向 Z 时为 **4.237 / 13.561 ms**。
- **前向+反向合计**：1,296 原子 **4.845 ms**（保存 Z）或 **6.122 ms**（重算 Z）。
  当前 CUPTI 的 polar UVU 是 **70.31 ms**（polar pipeline 87.95 ms），
  即 **UVU 降低 93.1%**；即使按最保守的重算方案也降低 91.3%。
- **投影到整条 polar pipeline**：87.95 → **22.03 ms**（相对任务口径 87.5 ms 为
  -74.8%；相对 trace 的 87.954 ms 为 -74.95%）；
  **端到端 eval**：253.84 → **188.37 ms**（**-25.8%**，节省 65.47 ms）。
  相对 pre-A1 UVU 105.83 ms，UVU 层面节省 **100.99 ms**。
- **数值**：与显式 per-node pair-matrix 参考实现相比，N=64 时最大绝对误差
  **3.91e-14**（相对输出幅值 **5.9e-16**）；5 次重复求值逐位一致（max diff = 0）。
- **判定**：GO 门槛是“比当前 polar UVU 估计快 ≥20%”；实测 UVU 组合为
  4.85 ms，仅为 87.5 ms 的 **5.5%**（或为 70.31 ms UVU 分量的 6.9%），
  远超门槛，无性能风险。建议进入引擎实现（P0 阶段仍需 128/1,296 全量 parity gate）。

---

## 1. 基准代码与运行协议

### 1.1 代码路径

| 内容 | 绝对路径 |
|---|---|
| 基准主脚本 | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/bench_polar_uvu_gemm.py` |
| 结果 JSON | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/results_polar_uvu_gemm.json` |
| 首次完整运行 JSON | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/results_run1_2026-10-21.json` |
| 完整日志 | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/run_full2.log` |
| N=128 smoke | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/smoke_128.json` |
| 反向校验脚本 | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/validate_reverse.py` |
| 反向校验日志 | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/validate_reverse.log` |
| 绘图脚本 | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/make_figure.py` |
| 图（PNG/PDF） | `/workspace/group/shared/bza/symmetrix-polar/docs/results/figures/POLAR_UVU_GEMM_MICROBENCH.{png,pdf}` |

运行命令（完整 GPU 作业，单进程、单锁）：

```bash
flock -w 10800 /workspace/group/shared/bza/.tmp/gpu.lock -c \
  "CUBLAS_WORKSPACE_CONFIG=:4096:8 \
   /workspace/caches/conda/envs/polar/bin/python bench_polar_uvu_gemm.py \
     --sizes 1296 5184 --iters 100 --warmup 20 --base-iters 5 \
     --large-gemm 4096 --out results_polar_uvu_gemm.json"
```

### 1.2 计时方法

- 每个变体先 warm-up，再用 CUDA event 包住一次完整调用，`synchronize()` 后读取
  elapsed time；100 次采样取中位数，同时记录 mean/min/p95 和 CPU enqueue 时间。
- GEMM 输入、W、Z workspace 全部预分配；bmm 变体把 W/X2 预堆叠（堆叠成本单独测量）。
- `torch.mm(..., out=)` / `torch.bmm(..., out=)` 避免计时区间内的分配。
- 所有张量为 `torch.float64`；`allow_tf32=False`。4096³ DGEMM 实测 **18.03 TFLOPS**，
  高于 A100 FP64 DFMA 峰值 9.7 TFLOPS，证明 cuBLAS 走的是 **DMMA FP64 tensor core**，
  不是 FP32/TF32，也不是 SIMT DFMA。

### 1.3 当前实现的基线（CUPTI，非本脚本测量）

来源：`bench/perf_decomp_1296_2026-10-21/analysis/1296_materialized_labels.csv`
与 `1296_materialized_summary.json`（combined `6582bc6` materialized trace，
eval window 286.11 ms，GPU busy 238.09 ms，polar pipeline 87.954 ms）：

| 标签 | 时间 (ms) | 发射数 |
|---|---:|---:|
| `polar uvu mode0` | 5.732 | 6 |
| `polar uvu mode1` | 5.048 | 2 |
| `polar le uvu mode0` | 4.583 | 4 |
| **前向 UVU 小计** | **15.363** | 12 |
| `polar reverse uvu mode0 x2` | 21.978 | 10 |
| `polar reverse uvu mode0 x1` | 20.882 | 10 |
| `polar reverse uvu mode1 x1` | 5.134 | 2 |
| `polar reverse uvu mode1 x2` | 5.024 | 2 |
| `polar reverse uvu mode0 x2 reduce` | 0.968 | 10 |
| `polar reverse uvu mode0 x1 reduce` | 0.962 | 10 |
| **反向 UVU 小计** | **54.949** | 44 |
| **UVU 总计** | **70.312** | 56 |
| **polar pipeline 总计** | **87.954** | 231 |

`polar uvu zero` / `mask` / `mode0 reduce` 等关联 kernel 混在其它 raw-kernel 标签中，
未计入 70.312 ms，因此用 70.312 ms 做减法是**保守**的。任务给出的当前最佳
1,296 原子 eval 为 **253.84 ms**、polar **87.5 ms**、pre-A1 UVU **105.83 ms**；
本报告在投影时同时给出 87.5（取整）和 87.954（精确）两种口径。

---

## 2. 精确工作量与形状

### 2.1 checkpoint 实际路径表

`x1, x2` 为 node-major `[N, 2048] = [512x0e | 512x1o]`，1o 分量按 `u*3+c` 交错；
`out` 为 `[N, 2048]`。每个 path 的 W 是独立的 `512x512` FP64 行主序矩阵
（`W[u,v]`），共 **12 条 path / 12 个 W 矩阵 = 24 MiB**（每个 table 2 条 path；
文档按 4 表算的 16 MiB 少算了第二个 fdc block）。
**所有权重 nnz 比例 = 1.000（完全稠密），所有 output_mask 全为 1.0。**

| table | path | mode | d1 | GEMM 数 | X2 平面 | out 块 | pw |
|---|---|---:|---:|---:|---|---|---:|
| `fdc0/dot_products` | 0 | 0 | 1 | 1 | `x2[:,0:512]` | `out[:,0:512]` | 0.03125 |
| `fdc0/dot_products` | 1 | 0 | 3 | 3 | `x2[:,512+c:2048:3]`, c=0..2 | `out[:,0:512]` | 0.03125 |
| `fdc0/tp_out` | 0 | 0 | 1 | 1 | `x2[:,0:512]` | `out[:,0:512]` | 0.044194 |
| `fdc0/tp_out` | 1 | 1 | 3 | 1 | `x2[:,0:512]` | `out[:,512:2048]` (u*3+c) | 0.076547 |
| `fdc1/dot_products` | 0 | 0 | 1 | 1 | `x2[:,0:512]` | `out[:,0:512]` | 0.03125 |
| `fdc1/dot_products` | 1 | 0 | 3 | 3 | `x2[:,512+c:2048:3]` | `out[:,0:512]` | 0.03125 |
| `fdc1/tp_out` | 0 | 0 | 1 | 1 | `x2[:,0:512]` | `out[:,0:512]` | 0.044194 |
| `fdc1/tp_out` | 1 | 1 | 3 | 1 | `x2[:,0:512]` | `out[:,512:2048]` | 0.076547 |
| `le/dot_products_q` | 0 | 0 | 1 | 1 | `x2[:,0:512]` | `out[:,0:512]` | 0.03125 |
| `le/dot_products_q` | 1 | 0 | 3 | 3 | `x2[:,512+c:2048:3]` | `out[:,0:512]` | 0.03125 |
| `le/dot_products_v` | 0 | 0 | 1 | 1 | `x2[:,0:512]` | `out[:,0:512]` | 0.03125 |
| `le/dot_products_v` | 1 | 0 | 3 | 3 | `x2[:,512+c:2048:3]` | `out[:,0:512]` | 0.03125 |

- **14-GEMM 口径**（文档）：`fdc0/dot_products` + `fdc0/tp_out` + LE q + LE v
  = 4+2+4+4 = **14**，9.513 GFLOP @ N=1,296，38.051 GFLOP @ N=5,184。
- **实际 20-GEMM 口径**：再加入 `fdc1/dot_products` + `fdc1/tp_out`
  = **20**，13.590 GFLOP @ N=1,296，54.358 GFLOP @ N=5,184。
- 每个 GEMM 的形状完全相同：**M=512, N=N_nodes, K=512**，FP64。
  `Z = X2_plane @ W^T`，即 `Z[n,u] = sum_v x2[v] W[u,v]`。
- 反向：两个 adjoint pass；dX1 pass 在保存前向 Z 时是逐元素 `mbar_u * Z_u`，
  dX2 pass 是 `P @ W`（`P_u = mbar_u * x1_u`）。因此
  **reverse = 20 GEMM（保存 Z）或 40 GEMM（重算 Z）**。

### 2.2 为什么是 6 张表而不是 4 张

CUPTI 一次 eval 的发射数为：6× `polar uvu mode0` + 2× `polar uvu mode1`
+ 4× `polar le uvu mode0`。一个 fdc block 的 `dot_products` 有 2 条 mode0 path，
`tp_out` 有 1 条 mode0 + 1 条 mode1，即每个 block 贡献 3 条 mode0 + 1 条 mode1；
两个 block 正好 6 + 2。LE q/v 各 2 条 mode0，合计 4。
这与 checkpoint 中两个 `field_dependent_charges_maps` 的权重不相同
（`dot_products` 两组权重最大差 52.86，`tp_out` 21.78）一致，
说明两个 block 都是每次 eval 实际执行的。文档的 “4 UVU tables” 只数了类型，
漏掉了第二个 fdc block 实例。

---

## 3. 结果

### 3.1 14-GEMM 组（任务要求的口径）

| N | 变体 | ms（中位数） | GFLOP | 有效 TFLOPS |
|---|---|---:|---:|---:|
| 1,296 | 14× 单独 `torch.mm`（loop） | **0.942** | 9.513 | **10.10** |
| 1,296 | 1× `torch.bmm`（batch=14, NT） | **0.599** | 9.513 | **15.89** |
| 5,184 | 14× 单独 `torch.mm`（loop） | **2.259** | 38.051 | **16.85** |
| 5,184 | 1× `torch.bmm`（batch=14, NT） | **2.143** | 38.051 | **17.75** |

对照文档 §3.1.2 的悲观估计 4.0 ms（2.4 TFLOPS），实测 1,296 的
0.942/0.599 ms 比悲观估计还快 4.2×/6.7×；文档估计偏保守。

### 3.2 实际 20-GEMM 组（checkpoint 全量）

| N | 变体 | ms（中位数） | GFLOP | 有效 TFLOPS | host enqueue (ms) |
|---|---|---:|---:|---:|---:|
| 1,296 | 20× 单独 `torch.mm`（loop） | **1.311** | 13.590 | **10.37** | 0.905 |
| 1,296 | 1× `torch.bmm`（batch=20, NT） | **0.869** | 13.590 | **15.64** | — |
| 1,296 | 1× `torch.bmm`（batch=20, W@X2ᵀ） | 0.893 | 13.590 | 15.21 | — |
| 5,184 | 20× 单独 `torch.mm`（loop） | **3.198** | 54.358 | **17.00** | 0.898 |
| 5,184 | 1× `torch.bmm`（batch=20, NT） | **3.032** | 54.358 | **17.93** | — |
| 5,184 | 1× `torch.bmm`（batch=20, W@X2ᵀ） | 3.095 | 54.358 | 17.56 | — |
| — | 单个 `512xNx512`（N=1,296） | 0.115 | 0.680 | 5.89 | — |
| — | 单个 `512xNx512`（N=5,184） | 0.203 | 2.718 | 13.39 | — |
| — | 参考大 GEMM 4096³ | 7.622 | 137.44 | **18.03**（峰值 19.5） | — |

**要点**：1,296 原子时单个 DGEMM 只有 5.89 TFLOPS，分组 batched 后达到
15.64–15.89 TFLOPS；5,184 原子时单个也有 13.39 TFLOPS，分组后 17.9 TFLOPS，
达到 A100 FP64 DMMA 峰值的 92%。1,296 的收益主要来自减少 launch gap，
5,184 的收益来自 GEMM 本身已接近峰值。

### 3.3 前向组合、反向组合与分解

前向组合 = 4 个连续平面打包 + 20 GEMM + epilogue；
反向组合 = `mbar`/`P` 准备 + 20 或 40 GEMM + dX1/dX2 scatter。

| N | 变体/分量 | ms（中位数） | 说明 |
|---|---|---:|---|
| 1,296 | 打包 4 个 X2 平面 | 0.047 | stride-3 的 1o 三平面 + 0e 平面 |
| 1,296 | 20 GEMM（loop） | 1.311 | 10.37 TFLOPS |
| 1,296 | 前向 epilogue | 0.551 | 20 个 Z tile（mode1 的 Z 广播到 3 个分量）+ mask + 路径累加 |
| 1,296 | **前向组合（loop）** | **1.885** | 1.885 ≈ 0.047+1.311+0.551 |
| 1,296 | 前向组合（bmm，含 stack 0.204） | 1.572 | 分组 bmm 更快 |
| 1,296 | 反向 20 GEMM（保存 Z） | 1.166 | 11.66 TFLOPS |
| 1,296 | **反向组合（保存 Z）** | **2.960** | 含 prep + GEMM + scatter |
| 1,296 | 反向 40 GEMM（重算 Z） | 2.462 | 11.04 TFLOPS |
| 1,296 | **反向组合（重算 Z）** | **4.237** | 备选方案 |
| 1,296 | **前向+反向（保存 Z）** | **4.845** | 推荐方案 |
| 1,296 | **前向+反向（重算 Z）** | **6.122** | 保守方案 |
| 5,184 | 打包 4 个 X2 平面 | 0.124 | |
| 5,184 | 20 GEMM（loop） | 3.198 | 17.00 TFLOPS |
| 5,184 | 前向 epilogue | 2.392 | 占前向组合 42.5% |
| 5,184 | **前向组合（loop）** | **5.632** | |
| 5,184 | 前向组合（bmm，含 stack 0.617） | 5.963 | stack 成本抵消了 launch 收益 |
| 5,184 | 反向 20 GEMM（保存 Z） | 3.179 | 17.10 TFLOPS |
| 5,184 | **反向组合（保存 Z）** | **10.421** | |
| 5,184 | 反向 40 GEMM（重算 Z） | 6.313 | 17.22 TFLOPS |
| 5,184 | **反向组合（重算 Z）** | **13.561** | |
| 5,184 | **前向+反向（保存 Z）** | **16.053** | |
| 5,184 | **前向+反向（重算 Z）** | **19.193** | |

**host/launch 分解（PyTorch 上界）**：

| 序列 | N=1,296 enqueue (ms) | N=5,184 enqueue (ms) |
|---|---:|---:|
| 20× `torch.mm` | 0.905 | 0.898 |
| 前向 epilogue（~50 个小 kernel） | 0.449 | 0.457 |
| 4 平面打包 | 0.020 | 0.028 |
| 前向组合整体 | 1.415 | 1.425 |
| 反向 GEMM 20 次 | 0.797 | 0.872 |
| 反向组合整体 | 2.291 | 2.346 |

这些是 **PyTorch Python 逐 op 调度**的上界；C++ 引擎直接调 20 次
`cublasDgemm` 或一次 `cublasGemmGroupedBatchedEx` 会低得多（约 0.05–0.1 ms 量级）。
1,296 时前向组合的 enqueue（1.415 ms）仍小于 GPU 时间（1.885 ms），
说明序列是 GPU-bound 的；bmm 相比 loop 省下的 ~0.44 ms 与“20 次 launch gap”一致。
反向组合中 GEMM 只占 1.166/3.179 ms，其余 **1.794/7.242 ms** 是 PyTorch 的
`mbar`/`P` 准备、dX1/dX2 逐元素 scatter（多个小 kernel）；引擎应把它融成
1–2 个 kernel，因此 2.960/10.421 ms 也是反向的保守上界。

### 3.4 per-node / GEMV 式基线

按“每个 node 物化 512×512 pair 矩阵，再与 W 逐元素乘并沿 v 求和”的
PyTorch 实现（TN=32 的 node tile），仅前向 20 组：

| N | ms（中位数） | 近似 pair FLOPs | 有效 TFLOP/s | 每 tile 形状 |
|---|---:|---:|---:|---|
| 1,296 | **81.628** | 20.39 GFLOP | 0.250 | `[32,512,512]` |
| 5,184 | **325.123** | 81.56 GFLOP | 0.251 | `[32,512,512]` |

这个基线比实际引擎的前向 UVU（15.363 ms）还慢 5.3×，因为 PyTorch 物化了
pair 矩阵并产生数千次 PyTorch kernel launch；它只用于说明“pair/GEMV 类实现”在
1,296 原子时就被带宽和 launch 支配，**不能当作引擎当前性能的替代测量**。
权威的当前性能仍是 CUPTI 的 87.5/70.31 ms。

### 3.5 投影

下表把当前 polar pipeline 拆成“UVU 70.312 ms + 非 UVU 17.642 ms”，
用实测的分组 GEMM + epilogue 替换 UVU 部分：

| 量 | N=1,296 | N=5,184（4× 线性缩放，估计） |
|---|---:|---:|
| 当前 UVU（CUPTI） | 70.312 ms | 281.248 ms |
| 分组前向+反向（保存 Z） | **4.845 ms** | **16.053 ms** |
| 分组前向+反向（重算 Z） | 6.122 ms | 19.193 ms |
| UVU 节省（保存 Z） | **65.467 ms（-93.1%）** | **265.195 ms（-94.3%）** |
| UVU 节省（重算 Z） | 64.191 ms（-91.3%） | 262.055 ms（-93.2%） |
| 当前 polar pipeline | 87.954 ms（任务口径 87.5） | 350.0 ms（4×） |
| 投影 polar（保存 Z） | **22.033 ms（-74.8%）** | **84.805 ms（-75.8%）** |
| 投影 polar（重算 Z） | 23.309 ms（-73.4%） | 87.945 ms（-74.9%） |
| 当前 eval | 253.84 ms | — |
| **投影 eval（保存 Z）** | **188.373 ms（-25.8%）** | — |
| 投影 eval（重算 Z） | 189.650 ms（-25.3%） | — |
| 相对 pre-A1 UVU 105.83 | UVU 节省 **100.99 ms** | 423.32 → 16.05 ms |

> N=5,184 没有新的 CUPTI 全量 trace；表中的 281.248 / 350.0 / 423.32 ms
> 是把 1,296 的 UVU 分量按 N 线性放大 4× 得到的估计，仅用于量级判断。
> N=1,296 的 87.5 ms 与 253.84 ms 来自不同运行（87.954 是 materialized trace，
> 253.84 是最佳交错 A/B），所以端到端投影是估计值，不是同一次运行的差。
> polar 百分比按任务口径 87.5 ms 计算；按 trace 的 87.954 ms 计算为 -74.95%。

---

## 4. 数值校验与确定性

| 项目 | 结果 |
|---|---|
| 校验规模 | N=64，真实 checkpoint 权重，随机 FP64 特征 |
| 参考实现 | 显式 per-node pair 矩阵 + W 加权求和（mode0），逐 path 累加 |
| 最大绝对误差 | **3.908e-14**（与 1,296 使用同一 checkpoint 权重） |
| 最大绝对误差 / 输出幅值 | **5.91e-16** |
| RMS 误差 | 2.98e-15 |
| 5 次重复 forward 组合 | **max abs diff = 0.000e+00**（逐位一致） |
| 反向 dX1/dX2 索引与方向校验 | N=16，12 paths，20 个 dX2 GEMM；显式公式 vs 基准向量化路径 **max|A-B| = 0**（逐位一致） |
| FP64 证据 | 输入/权重/Z 全 `float64`；4096³ DGEMM 18.03 TFLOPS > 9.7 DFMA 峰值 |
| 权重稀疏性 | nnz fraction = 1.000（12 个 W 全稠密） |
| output_mask | 全部为 1.0（仍按公式应用） |

误差来源仅为 FP64 求和顺序改变（GEMM 的 v 归约顺序 vs 逐线程顺序），
与 `CUDA_SPEEDUP_PAPERS_2026.md` §3.1.4 的预期一致；门槛
`dE<=1e-9, max dF<=1e-8` 比实测误差宽 4–6 个数量级。

---

## 5. 注意事项与风险

1. **小 batch DGEMM 效率**：单个 512×1,296×512 只有 5.89 TFLOPS；
   必须把 14/20 个 GEMM 分组/批处理，才能到 15.6–15.9 TFLOPS。
   1,296 时 loop 与 bmm 差 0.44 ms（launch gap），5,184 时 loop 反而略快
   （bmm 需要 0.617 ms 的堆叠）。引擎实现应优先用
   `cublasGemmGroupedBatchedEx`（指针数组，无需堆叠），或 20 次
   `cublasDgemm`；两者都应在 1,296 上接近 bmm 的 0.87–1.31 ms。
2. **内存布局**：`d1=3` 的 X2 三平面在原张量中是 stride-3
   （`x2[:,512+c:2048:3]`），BLAS 无法用 `ldb` 表达，必须打包；
   实测打包 0.047 ms（1,296）/0.124 ms（5,184）。`0e` 平面可以直接用
   `ldb=2048` 的列主序视图，无需拷贝（本基准为统一口径也复制了它）。
   bmm 需要把 20 个平面堆叠，成本 0.204/0.617 ms，这是 5,184 时 loop 胜出的原因。
3. **Epilogue**：本基准用 PyTorch 多 kernel 实现（~50 个小 kernel），
   0.551 ms（1,296）/2.392 ms（5,184），在 5,184 时占前向组合的 42.5%。
   这是**上限**；引擎应实现单遍 fused epilogue（或 GEMM epilogue），
   按 `u_tile x n_tile` 读 Z/X1/mask 一次写 out，带宽下界约 0.7 ms（5,184）。
   epilogue 必须保持 path 升序、mask 只乘一次、`(a+b)*mask` 的顺序。
4. **Launch 开销**：PyTorch 的 20×`mm` enqueue 是 0.905 ms（1,296），
   前向组合 1.415 ms；这是 Python 调度上界，不应写进引擎预算。
   C++ 的 20 次 cuBLAS 调用约 0.05–0.1 ms；一次 grouped/batched 调用更低。
5. **确定性**：cuBLAS DGEMM 在固定形状/架构下是确定性的，5 次重复逐位一致。
   实现时应 `cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED)`、
   固定算法、避免 split-K；反向的多 path 累加顺序必须固定，dX1/dX2 的
   overlapping block 用确定性 reduce，不要用 atomicAdd。
6. **保存 Z 的内存代价**：前向 20 个 Z = **106 MB（1,296）/ 425 MB（5,184）**
   （每个 Z 为 `N x 512` FP64：5.31 MB / 21.23 MB；文档 §3.1.3 的 74/297 MB
   是按 14 个 Z 算的）。当前峰值 8.55 GiB，完全放得下；如内存紧张则重算 Z，
   代价 +1.28 ms（1,296）/+3.14 ms（5,184），仍然远低于门槛。
7. **反向的两个 adjoint pass**：保存 Z 时 dX1 是逐元素 `mbar*Z`（无 GEMM），
   dX2 是 `P@W`（20 GEMM）；重算 Z 时再加 20 个前向 GEMM。
   本报告两种都实测，投影用保存 Z 的 4.845 ms，并给出重算的 6.122 ms 兜底。
8. **14 vs 20 的口径**：文档的 14-GEMM/9.5-GFLOP 估计少算一个 fdc block。
   本报告的实际 20-GEMM/13.59-GFLOP 投影更保守；若引擎只实现 14 组，
   前向会略快，但会漏掉 `fdc1` 的输出，属于正确性问题，不是优化选择。
9. **PyTorch pair 基线不是引擎**：它物化 `[TN,512,512]` 的 pair 矩阵，
   1,296 时 81.6 ms，比真实前向 UVU 15.36 ms 慢 5.3×。不要把 81.6 ms
   当作“当前实现”来夸大收益；收益的权威基准是 CUPTI 的 70.31/87.5 ms。
10. **算法/驱动依赖**：cuBLAS 的 heuristic 可能随 workspace、驱动或
    CUDA 版本变化；引擎实现应固定 handle 配置并做 5× repeat 校验。
    本基准使用默认 heuristic + `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
11. **N=5,184 的投影是 4× 缩放**：没有新的 5,184 CUPTI 全量 trace；
    5,184 的 GEMM/epilogue 是实测，当前 UVU 的 281 ms 是缩放估计。
12. **时钟/功耗**：A100-SXM4-80GB，SM 1410 MHz；4096³ DGEMM 18.03 TFLOPS
    说明运行期间没有明显降频。所有 GPU 作业串行持有 `gpu.lock`。

---

## 6. GO / NO-GO 判定

**门槛**：分组 GEMM + epilogue 至少比当前 polar UVU 估计快 20%
（即 ≤ 0.8 × 87.5 = 70.0 ms）。

| 口径 | 实测/投影 | 相对 87.5 ms | 结论 |
|---|---:|---:|---|
| 14-GEMM loop（1,296） | 0.942 ms | -98.9% | 通过 |
| 14-GEMM bmm（1,296） | 0.599 ms | -99.3% | 通过 |
| 前向 20-GEMM 组合（1,296） | 1.885 ms | -97.8% | 通过 |
| 前向+反向 保存 Z（1,296） | **4.845 ms** | **-94.5%** | **通过** |
| 前向+反向 重算 Z（1,296） | 6.122 ms | -93.0% | 通过 |
| 前向+反向 保存 Z（5,184） | 16.053 ms | **-94.3%**（vs 4×70.31=281.25 UVU） | 通过 |

**判定：GO。** 分组 FP64 GEMM 在 1,296 上把 polar UVU 从 70.31 ms 压到
4.85 ms（-93.1%），整条 polar pipeline 87.95 → 22.03 ms（-74.8%），
端到端 253.84 → 188.37 ms（-25.8%）。即使采用不保存 Z 的最保守方案，
UVU 仍为 6.12 ms（-91.3%），结论不变。性能风险已消除；剩余风险是
数值 parity、epilogue 融合和 cuBLAS 确定性，均属于实现问题，不影响 GO。

**建议的引擎实现顺序**：
1. 用 20/14 个 `cublasDgemm`（或 `cublasGemmGroupedBatchedEx`）替换 UVU
   主循环，保留 `MATOS_POLAR_UVU_GEMM=0` 回退。
2. d1=3 三平面打包；0e 用 `ldb=2048` 列主序视图；Z workspace `[N,512]` 复用。
3. 单遍 fused epilogue（路径升序、mask 一次、out 一次写）。
4. 前向 Z 保存到 replay cache（106/425 MB，20 个 Z）；反向 dX1 逐元素、dX2 GEMM。
5. 128/1,296 parity gate：`dE<=1e-9`、`max dF<=1e-8`、5× repeat、
   并对比 `MATOS_POLAR_UVU_GEMM=0/1` 的 bitwise/容差结果。

---

## 7. 图

![Polar UVU grouped FP64 GEMM microbenchmark](/workspace/group/shared/bza/symmetrix-polar/docs/results/figures/POLAR_UVU_GEMM_MICROBENCH.png)

- (a) N=1,296 当前 UVU/前向/反向与各分组变体的时间（log 轴）。
- (b) 各 GEMM 变体的有效 TFLOPS，对比 A100 FP64 DMMA 峰值 19.5 与实测
  4096³ DGEMM 18.03 TFLOPS。
- (c) N=1,296 的 polar pipeline 与端到端 eval 投影（保存 Z 方案）。

---

## 8. 产物清单

| 产物 | 路径 |
|---|---|
| 基准脚本 | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/bench_polar_uvu_gemm.py` |
| 结果 JSON（完整） | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/results_polar_uvu_gemm.json` |
| 首次运行 JSON | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/results_run1_2026-10-21.json` |
| 运行日志 | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/run_full2.log` |
| N=128 smoke | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/smoke_128.json` |
| 反向校验脚本 | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/validate_reverse.py` |
| 反向校验日志 | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/validate_reverse.log` |
| 绘图脚本 | `/workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21/make_figure.py` |
| 图 PNG | `/workspace/group/shared/bza/symmetrix-polar/docs/results/figures/POLAR_UVU_GEMM_MICROBENCH.png` |
| 图 PDF | `/workspace/group/shared/bza/symmetrix-polar/docs/results/figures/POLAR_UVU_GEMM_MICROBENCH.pdf` |
| CUPTI 标签来源 | `/workspace/group/shared/bza/symmetrix-polar/bench/perf_decomp_1296_2026-10-21/analysis/1296_materialized_labels.csv` |
| CUPTI 汇总来源 | `/workspace/group/shared/bza/symmetrix-polar/bench/perf_decomp_1296_2026-10-21/analysis/1296_materialized_summary.json` |
| 本报告 | `/workspace/group/shared/bza/symmetrix-polar/docs/results/POLAR_UVU_GEMM_MICROBENCH.md` |

### 复现

```bash
cd /workspace/group/shared/bza/symmetrix-polar/bench/polar_uvu_gemm_2026-10-21
flock -w 10800 /workspace/group/shared/bza/.tmp/gpu.lock -c \
  "CUBLAS_WORKSPACE_CONFIG=:4096:8 \
   /workspace/caches/conda/envs/polar/bin/python bench_polar_uvu_gemm.py \
     --sizes 1296 5184 --iters 100 --warmup 20 --base-iters 5 \
     --large-gemm 4096 --out results_polar_uvu_gemm.json"
/workspace/caches/conda/envs/polar/bin/python make_figure.py
```

`results_polar_uvu_gemm.json` 中包含：模型 MD5、每条 path 的 mode/d1/in-offset/
out-offset/pw/W-absmax/mask-nnz、每个变体的 median/mean/min/p95/wall/enqueue、
N=64 校验误差、5× 确定性结果、4096³ DGEMM 参考、以及 1,296/5,184 的投影块。
