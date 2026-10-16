# MACE-POLAR-1-M 环境配置与正确性核验(2026-10-14)

## 版本配对(关键,踩坑记录)

**官方文档引脚是滞后的**:MACE 文档(`mace-docs.../guide/polar_mace.html`,footer 实为
mace 0.3.13 版文档)写 "mace-torch ≥0.3.16 + graph_electrostatics@v0.4.4",
但该组合在代码层面**不兼容**:

| 组件 | 事实 | 证据 |
|---|---|---|
| mace-torch 0.3.16 (2026-05-10) | 两个调用点都传 `force_pbc_evaluator` | tag v0.3.16 的 `mace/modules/extensions.py:729,923`(与容器内文件逐字节一致) |
| graph v0.4.3 (2026-05-18)+ | **废弃** `force_pbc_evaluator`,改为 `pbc_handling` 选择器 | `precompute_geometry(self,...,pbc)` 无该参数;需要实例属性 `_precompute_geometry_impl` |
| graph v0.4.0 (2026-03-12) | `precompute_geometry(..., force_pbc_evaluator: bool = False)` ✓ | features.py:463-473、energy.py:81 |
| MACE-POLAR-1-M checkpoint | pickle 状态无 `pbc_handling`/`_precompute_geometry_impl` → 对应 v0.4.0 时代类 | 反序列化后 `electric_potential_descriptor.__dict__` |

**结论:本 checkpoint + mace-torch 0.3.16 的正确配对 = `graph_electrostatics@v0.4.0`**。
v0.4.4 的 `external_source_features.py` 里还有一句注释
"Older host models may still pass this retired graph_longrange option"(pop 掉
`force_pbc_evaluator`)——说明上游知道旧宿主会传,但基础类已不接受。

## 工作环境(已验证,无任何补丁/shim)

- Python: `/workspace/caches/conda/envs/polar/bin/python`
  (venv,--system-site-packages 指向 `/opt/matos-tools/conda/envs/matos-mlip-py312`)
- mace-torch 0.3.16: 系统 env(与官方 tag v0.3.16 逐字节一致)
- graph_longrange **0.4.0**: `pip install --no-deps ./external-repos/graph_electrostatics-0.4.0`
  (源码树:`/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0`)
- torch 2.8.0+cu128;`mace_polar(model=<local .model>, device="cuda", default_dtype="float32")`
- **import 顺序纪律**:任何脚本必须先 `import mace`(其 `__init__` 设
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`),再 import graph_longrange/e3nn;
  否则 e3nn 的 constants.pt 在 torch≥2.6 下加载失败。或手动设该环境变量。
- **不要传 `pbc_handling` 给 mace_polar**:0.3.16 的计算器没有这个参数(那是 0.3.17-dev 文档)。
  0.3.16 中 Ewald/realspace 选择由 `data["pbc"]`(即 atoms.pbc)自动决定。

## 冒烟结果(`logs/polar_smoke_v040.log`,clean run,exit 0)

生产同款盒子 NH3/CH4/N2/CO2 × 32 分子(ρ=0.01,seed=7705xx),GPU0(A100-80G),
float32,缓存破除协议(每步微扰,rattle 1e-4;预热 3,计时 10 取中位):

| 体系 | MACE-OFF23 | MACE-POLAR-1-M | 用时比 | POLAR ms/call |
|---|---|---|---|---|
| NH3-32 (128 原子) | 919.7 µs/atom | 3790 µs/atom | 4.12× | 485.2 |
| CH4-32 (160 原子) | 781.5 µs/atom | 3102 µs/atom | 3.97× | 496.3 |
| N2-32 (64 原子) | 2342.5 µs/atom | 8147 µs/atom | 3.48× | 521.4 |
| CO2-32 (96 原子) | 1159.5 µs/atom | 4846.8 µs/atom | 4.18× | 465.3 |

- 逐位能量差 dE(OFF23 vs POLAR)= 40–73 eV 属正常(两模型 E0 参考不同);力尺度同量级。
- POLAR 单次调用时间几乎恒定(~465–520 ms),SCF+k 网格开销主导,弱依赖体系尺寸。

## torch_sim 批处理支持(给生产替换用)

**mace-torch 0.3.16 自带官方 TorchSim 集成**:
`mace/calculators/mace_torchsim.py::MaceTorchSimModel`
- 构建 PolarMACE 所需全部附加键: `rcell = 2π·inv(cell^T)`(按 pbc mask)、`volume`、
  `external_field`(缺省 zeros)、`fermi_level`(zeros)、`density_coefficients`(zeros);
- 带原子/边/体系预算的 padding 机制(编译友好),`total_charge`/`total_spin` 补齐;
- 用法:`MaceTorchSimModel(model=<path|nn.Module>, device=..., dtype=...)`。

torch_sim 0.6.0 自带的 `torch_sim.models.MaceModel` **不喂**这些键,直接对 PolarMACE 会挂;
生产批处理替换应包装 `MaceTorchSimModel`(我们的外场 COM 包装器同构迁移)。

## 核验门(四关)— 全部通过 ✅(2026-10-14)

脚本:`bench/mace_polar_correct.py` → 日志 `logs/polar_correct_v040b.log`(exit 0);
float64 精校 `bench/polar_fd_f64.py` → `logs/polar_fd_f64.log`;
批处理核验 `bench/polar_prep_check.py` → `logs/polar_prep_check2.log`(exit 0)。

1. **有限差分力**:float32 h=1e-4 下 max|F_an−F_num| = 0.22 eV/Å(能量量化噪声主导);
   **float64 精校:4.96e-7 eV/Å**(3 原子×2 轴)且 h 收敛(1e-3:4.7e-5 / 1e-4:5.0e-7 /
   1e-5:2.8e-7)→ 力实现精确。
2. **NVE 1000 步(0.5 ps)**:dE_tot = **5.1e-3 eV**(3NkBT 的 1.2e-3);轨迹
   2.7→4.1→3.7→6.1→5.1 mEv **有界徘徊、非系统性加热**。
3. **尺度对照**:POLAR PE=−49242 eV / fmax 0.90 vs OFF23 −49282 eV / 1.10 —— 同量级。
4. **长程确认**:CO2 二聚体扫描(相互作用能,eV)——POLAR: 4Å −0.0127, 5Å +0.0010,
   6Å +0.0049, 8/10/12Å 各 +0.001~0.002(截断外持续非零);OFF23: 5Å 及以外**严格 0.0000**。

## 批处理集成核验(prep check,全部通过)

- **A1 site 等价**:重构版 vs 冻存版 dE=0.000e+00、dFmax=1.2e-6 eV/Å(逐位一致)。
- **A2 com 等价**:重构版(live-state 映射,无需传参)vs 冻存版(显式 system_idx)
  **dE=0.000e+00、d(field_energy)=0.000e+00** —— 与事故修复模板逐位一致;
  A2b 生产同款调用的事故签名同前([0.1432,0,…,0])。
- **B POLAR minigate**:5 步 Langevin 全部有限;T=506.5 K(目标 500);PE≈−49243 eV。
  过程中定位并修复一个上游地雷:**mace 的 `data["positions"].requires_grad_(True)`
  是原地操作且不还原**(mace/modules/utils.py:632),叠加 MaceTorchSimModel 的
  `state.clone()` + 积分器把非规范输出写进 SimState extras → 带 grad 张量被
  `copy.deepcopy` 拒绝(仅当状态先经过 MaceModel 系包装器时触发)。
  修复 = 包装器输出 **detach 卫生处理**(`PerSystemSiteFieldMaceTorchSimModel.forward`
  末尾整体 detach;数值不变)。
- **C 吞吐(本地 GPU0,机器同时有 LAMMPS 编译负载)**:POLAR 0.73 s/批步(16 流)
  vs OFF23 0.16 s/批步 → **单步耗时比 4.68×**(与单调用 3.5–4.2× 一致)。
  批步即全流各 1 步 → POLAR ≈ 1.37 步/s·流(本机负载下;OFF23 6.25)。
  投影远程 A100(不计本地负载):POLAR ≈ 8.3/4.68 ≈ **1.8 步/s·流** →
  16 流 1 ns 波次 ETA ≈ **13–14 天**(OFF23 同波 ~2.8 天)。
- 修复版 runner 端到端冒烟通过(`logs/runner_smoke_fix.log`:MINIGATE OK、
  300 batch-steps、no-NaN)。

## 复现命令

```bash
PV=/workspace/caches/conda/envs/polar/bin/python   # mace 0.3.16(系统) + graph v0.4.0
cd /workspace/group/shared/bza/task3_sampling_database
CUDA_VISIBLE_DEVICES=0 $PV mlip/bench/mace_polar_smoke.py      # 冒烟/计时
CUDA_VISIBLE_DEVICES=0 $PV mlip/bench/mace_polar_correct.py    # 四关
CUDA_VISIBLE_DEVICES=0 $PV mlip/bench/polar_fd_f64.py          # float64 FD
CUDA_VISIBLE_DEVICES=0 $PV mlip/bench/polar_prep_check.py      # 批处理核验 + 吞吐
```

