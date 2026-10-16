# RBE (P4) 设计文档 —— 随机批 Ewald k 求和移植设计（2026-10-15）

**定位**:本文 = `SYMMETRIX_POLAR_RBE_PLAN.md` §8 账本中「RBE k 向量采样」一行的设计展开
（[PLAN] L111;P4 阶段定义与验证门见 [PLAN] L38-39）。任务 = 把 lammps-rbe 的
`kspace_style rbe` 算法逐点吃透,并映射到本项目 P3 的 k 求和结构上,产出一份可直接
指导引擎侧实施的对照文档。只写本文档,不改任何代码。

## 0. 依据、读取方式与边界声明

- **lammps-rbe 仓库状态(重要)**: `/workspace/group/shared/bza/external-repos/lammps-rbe`
  在本次读取开始时**工作区为空**(索引中全部文件为 staged deleted),且为**部分克隆**
  (promisor 远端 `github.com/randbatch-md/lammps-rbe` 当前不可达,读取部分 blob 时曾报
  连接失败);撰写期间检出正被**外部流程**部分恢复(截至落稿 `src/` 仍未恢复),与本报告
  无关、本报告未改动其中任何内容。本文所有 RBE 源码引用均通过 `git show HEAD:<path>`
  从对象库读出(与工作区恢复与否无关),HEAD =
  `905943b4864e33d294e7b51d20ae1141b2a8f55e`(单提交 "1.0.0",2024-03-05,huan.liu)。
  行号以该提交内容为准;读取期间 git 触发了其例行的后台 "Auto packing",未改动
  索引内容。
- 引擎侧引用: `libsymmetrix/source/`(含已入码的 polar 模块①②) 与
  `symmetrix/source/`(提取器/calculator),路径前缀
  `/workspace/group/shared/bza/external-repos/symmetrix-xl/`。
- P3 参考侧: `graph_longrange v0.4.0` 本地副本
  (`/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0`);
  mace 0.3.16 安装副本
  (`/opt/matos-tools/conda/envs/matos-mlip-py312/lib/python3.12/site-packages/mace/`),
  `extensions.py` 行号与该副本核对一致。
- 约定:标 **[自算]** 的数值/推导 = 本报告自行转写公式复算的结果,不是仓库输出;
  [PLAN] = 本目录 `SYMMETRIX_POLAR_RBE_PLAN.md`。

---

## 1. RBE 算法逐点(lammps-rbe `kspace_style rbe`)

引用简写(均为 `lammps-rbe@905943b` 内路径):
`[RBE]` = `src/KSPACE/rbe.cpp`;`[RBE-H]` = `src/KSPACE/rbe.h`;
`[RBE-E]` = `src/ELECTRODE/rbe_electrode.cpp`;`[EX]` = `RBE_Example/in.spce-bulk-nvt`。

### 1.1 类结构、注册与生命周期

- 样式注册:`KSpaceStyle(rbe,Rbe)` 宏([RBE-H] L1-3);类 `Rbe : public KSpace`
  ([RBE-H] L17);参数读取在 `settings()`([RBE] L207-235):
  `alpha=arg[0]`(L214, Ewald 分裂参数)、`P=int(arg[1])`(L215, 批量大小)、
  `RankID=int(arg[2])`(L216,**无条件读 arg[2]**,但 narg 检查允许 2 个参数——narg==2
  时属越界读取,源码级观察到的一个隐患;示例始终给 3 个参数)。
- 生命周期:`init()`([RBE] L142-203) 里 `g_ewald = alpha_sqrt = sqrt(alpha)`(L195/L222,
  即 **alpha 是标准 Ewald 的 α²**),末行调用 `qsum_qsq(); setup();`(L199-200);
  `setup()`([RBE] L239-253) 会被 LAMMPS 在 run 开始时再次调用。
- 兼容性约束:必须与能提供 `cut_coul` 的 pair style 搭配([RBE] L167-171,否则
  "KSpace style is incompatible with Pair style");拒绝 2d(L146-147)与非周期边界
  (L151-152);支持 slabflag + `slabcorr()`(L153-160, L599, L618-669)。
- **与实空间部分的耦合方式**:倒空间(本类)与实空间 erfc 部分(由 pair style 负责,
  如示例的 `pair_style lj/cut/coul/long 10.0 10.0`, [EX] L26)是标准的
  LAMMPS Ewald 分解;RBE 只承担 **reciprocal 求和 + 自能/背景项**;单位换算用
  `force->qqrd2e`(L368);`dielectric` 读入后未使用(L369,死变量,已核实全文件仅此一处)。

### 1.2 k 向量如何采样(随机批)

- **格点结构:整数 m ∈ Z³,无 k 壳**:每个分量独立采样后取整;
  k = 2πm/L 逐维换算(L390 `pxyz = {2π/xprd, 2π/yprd, 2π/zprd}`;L398-403 `Kx = K*pxyz`)。
  非立方盒逐维各向异性(每维用各自的 L_i)。
- **提议分布(逐轴)**:`m_new = round( N(0, σ_i) )`,其中
  `σ_i = sqrt(alpha·L_i²/(2π²))`(L283-285 首次生成;L290-339 循环体 L292-294/307-309/322-324)。
  首次采样拒绝全零向量 (0,0,0)(L286 `while` 条件;L337-338 链内产生 (0,0,0) 时
  `i=i-1` 重试该槽位)。
- **批内是短马尔可夫链,不是 iid**:第 i+1 个成员以第 i 个成员为当前状态
  (`mold = mx[i]`,L293/L308/L323)。链首(L283-285)为纯 iid 提议。
- **Box-Muller 高斯发生器**:`randn_box_muller_linear_congruential()`
  ([RBE] L62-75):两组 `u1,u2 = rand()*(1.0/RAND_MAX)`(L69-70,`do…while(u1<=1.17549e-38)`
  L71 防 log(0));`z0 = sqrt(-2 ln u1)·cos(2π u2)`,`z1 = …·sin(2π u2)`(L72-73);
  **只返回 z0,z1 丢弃**(L74,L66 的 static 变量实为缓存但未实现缓存逻辑 —— 每个正态数
  消耗 2 个均匀数)。全文件**无 `srand`**(已核实:grep srand 仅命中 rand() 调用点),
  即用 C 库默认种子。
- **批量生成量级**:每批 3P 个正态数(坐标三轴各 P)+ 3(P−1) 个 MH 均匀数
  (L300/315/330 的 `rand()%10000/1e4`),即约 6P 次 rand() 调用/批。
- **输出缓冲**:整数三元组存 `K_Sample[P][3]`(float,L340-345;句柄声明 [RBE-H] L43);
  每步把根 rank 的 `K_Sample` 拷入 `K[P][3]` 并广播(L348-358)。

### 1.3 Metropolis 接受/拒绝步

- **"目标" p(m)**:`Prob(m,alpha,L,S1) = exp(-k²/(4α))/S1`,k=2πm/L([RBE] L77-81)。
- **提议质量 q(m)**:`MH_D`([RBE] L83-95):
  m=0 时 `erf(0.5/sqrt(αL²/π²))`(L88),m≠0 时
  `0.5[erf((|m|+0.5)/sqrt(αL²/π²)) − erf((|m|−0.5)/sqrt(αL²/π²))]`(L92)。
  该式 = 对 N(0, sqrt(αL²/(2π²))) 取整到整数 m 的概率质量(区间积分)——与提议
  分布严格对应(分母 √(αL²/π²) = σ·√2,即区间宽度换算,已逐项核对)。
- **接受率公式**:`acce = min(1, p(m_new)q(m_old)/(p(m_old)q(m_new)))`([RBE] L299),
  即独立提议型 MH 比(含 q 比项);判定 `y = (rand()%10000)/1e4`,`y<acce` 接受、
  否则保留旧值(L300-306)。y 轴、z 轴同式(各维目标为乘积分布的一维因子)。
- **数学注记(本报告分析)**:p(m) ∝ exp(−m²/(2σ²)) 与 q(m)(取整高斯质量)同尺度同形状,
  所以接受率在精确算术下理论上趋近 1,**Metropolis 步在功能上接近"恒等"**
  ——它主要修正离散质量与高斯密度的细微差别(尾部分),而非做大量拒绝采样。
  **[自算] 数值佐证**(按 L77-95 公式转写,CPU 复算 4000 步/组):α=0.07、L=81.6 Å →
  拒收率 0.2%;α=0.07、L=29.4 Å → 0.8%;α=0.5/1.0、L=14.7 Å → 0.2%/0.6%。
  ⇒ 批内样本近似 iid(链结构不引入显著自相关),但**论文级论断未核实**。
- **无 burn-in、无 thinning、无跨步复用**(每步整批重生成);接受与否只写入
  下一个批槽(L301-306)。

### 1.4 估计量公式与误差控制

- **结构因子**(未归一,直接对原子坐标求和):
  `Rho[i] = (Σ_j q_j cos(k_i·r_j), Σ_j q_j sin(k_i·r_j))`([RBE] L434-457,
  AVX-512 `_mm512_sincos_ps` L451;`MPI_Allreduce` 汇总 L465)。
- **能量估计量**(全局,eflag_global,L533-546):
  `E += Σ_i (2π S/(P·V)) · |ρ(k_i)|²/|k_i|²`(L541),再减去自能+背景:
  `− g_ewald·qsqsum/√π − (π/2)·qsum²/(g_ewald²·V)`(L543-544,MY_PIS=√π、MY_PI2=π/2),
  最后 `× qscale = qqrd2e`(L545,L531)。
  其中 **S = S1_X·S1_Y·S1_Z − 1**(L193/L252)是采样分布(见下)的离散归一化。
- **估计量与标准 Ewald 的对应(本报告推导,逐项核对)**:采样分布
  `q(m) = exp(−k²/(4α))/S`(m≠0);则
  `E_q[(2πS/V)·|ρ|²/k²] = (2π/V)·Σ_{m≠0} exp(−k²/(4α))·|ρ(k)|²/k²`
  —— 正是标准 Ewald 倒空间和(summand 含高斯因子)。即:lammps-rbe 把
  **1/k² 留在估计量里、把高斯包络 exp(−k²/4α) 当选样分布**做重要性抽样
  (1/k² 在 k→0 不可积,故不能直接作为采样密度)。
- **S 的两种实现(源码内部不一致,以 setup 为准)**:`init()` 用暴力求和
  Σ_{i=-10000}^{10000} exp(−π²i²/(αL_i²))([RBE] L181-193),随后被 `setup()` 的
  Poisson 对偶近似覆盖:`S1_X = sqrt(α·xprd²/π)(1+2e^{−α·xprd²})` 等([RBE] L249-251)。
  注意 L250-251 中 S1_Y/S1_Z 的**前因子错用 xprd**(指数里才是 yprd/zprd)
  —— 立方盒无影响,非立方盒属源码级 slip(已核实行文)。
- **误差控制**:**源码中无任何误差估计/方差控制/自适应批大小/收敛判据**。
  唯一的内建"修正"是 S 归一与 m=0 排除(L286/L337-338);S1 的 Poisson 近似
  (只保留 n=0,±1 项)是额外的精度来源,未做误差记账。收敛性只能靠外部
  p 扫描评估(P4 门,见 [PLAN] L39)。
- **力**:`F_i = q_i·Σ_j Imag·midterm`([RBE] L467 MIDTERM=−4πS·qqrd2e/(P·V);
  L479-508 向量化;L522-527 写回 atom->f),其中 `Imag =
  (cos(k·r_i)·Im ρ − sin(k·r_i)·Re ρ)·MIDTERM`,`midterm = k/|k|²`(L405-410)。
  **[自算] 已解析验证 F = −∂E_est/∂r**(对 L541 的 E 求导后与 L492-506 完全一致)
  ⇒ **能量与力用同一批 K 计算(L487 与 L535 同批),步内能量-力严格自洽**,
  这对 RBE 噪声下的 MD/检验(力有限差分)很重要。
- 维里 L549-573(`coef2 = 1/(4g_ewald²)+1/k²`),per-atom 自能 L577-597,
  slabcorr L618-669 —— 本设计只关心能量/力/维里三者的公式结构,数值细节从略。

### 1.5 更新频率与 RankID 轮转

- `Step` 每次 `compute()` 自增([RBE] L360;Step 在 init 归零 L197),即**每 MD 步
  一次 kspace 求值,每步重采样**(见下)。
- 采样条件:`(Step % RankID)==0 && me < RankID`([RBE] L280);广播 root =
  `this_rank = Step % RankID`([RBE] L278, L348-358)。
- 语义推演(源码级):设 RankID=R。Step=0 时 ranks 0..R−1 各自生成一批;之后
  Step=1..R−1 分别以 rank 1..R−1 已生成的批为广播 root(R 步一轮);Step=R 时
  全部重新生成。⇒ **仿真每步看到的批都更换(轮转),每条"链流"每 R 步刷新一次**。
- 默认值:`RankID` 初始被强制置 1([RBE] L211,覆盖了刚读入的
  `MPI_Comm_size` 结果——原意或是"采样轮转度"而非真实进程数),narg==3 时取 arg[2]。
  示例:`kspace_style rbe 0.07 200 4`([EX] L28;SPC/E 水盒,53367 原子
  ——`RBE_Example/RBE.lammpstrj` 头部 ITEM: NUMBER OF ATOMS=53367)。
- `This_Index = Step*P`([RBE] L276)计算后未使用(疑似为 K 列表缓存预留,死代码)。

### 1.6 复现性与随机数种子

- **无显式种子**:全仓库 `src/KSPACE/rbe.cpp` 无 `srand`;RNG 为 C 库全局
  `rand()`(线性同余,glibc 实现有已知低位周期性问题,此处仅用高位做除法,
  `rand()%10000` 的模偏置量级 ~10⁻⁶ 可忽略)。
- ⇒ 跨运行可复现的前提 = 同一 libc/同一机器;跨平台/跨编译器**不可复现**;
  无用户种子接口;无法与 LAMMPS 其他随机源(如 fix langevin)通过种子联动。
- MPI 侧:批生成只在 `me<RankID` 且 `Step%RankID==0` 的 rank 上进行,只有 root 的
  `K_Sample` 进入广播(L348-358);因此**总进程数不改变采样序列**(只要 rank 0 存在),
  但每 rank 各持一份 K_Sample(非 root 的副本为死数据)。
- `z1` 丢弃(L73)使单次调用产出的两个独立正态只用其一,随机数消耗翻倍。

### 1.7 移植前的工程化注记(必读)

- **精度流水线**:k 采样/结构因子/力全部 float32(K_Sample float,L229-233;
  X/F/Q float,L373-388;Rho float,L391;force 写回 double,L522-527);只有
  能量归约与自能常数段为 double(L531-545)。与引擎 P2/P3 的 f64 门
  (亚微瓦级对账,见 [PLAN] L117-130)不同卡尺,移植须决策精度策略。
- **平台绑定**:`#include <mathimf.h>`(Intel 数学库,[RBE] L3)、`<mkl.h>`(L18)、
  AVX-512 intrinsics(`_mm512_sincos_ps` 等,L436-457)、`#pragma simd`(L553-556)
  —— 只能作为算法参照,不可逐行照搬到 CUDA/Kokkos。
- **非标准写法**:`float K[P][3]`(L274)为 VLA(C++ 扩展);`Find`(L103-133)、
  `cmul/conjugate`(L36-54)、`Time`(L202)、`K_All`(L227)、`pi_sqrt`(L224)、
  `lx/ly/lz`([RBE-H] L36)、`infile/pointer`([RBE-H] L34-35)均为死代码/死字段
  (已逐项 grep 核实)。
- 错误消息仍写 "Ewald"([RBE] L147/L152)—— 复制自 ewald.cpp,无功能影响。
- **electrode 变体(不在本次移植范围,但须知悉)**:`rbe/electrode`([RBE-E];
  注册 `KSpaceStyle(rbe/electrode,RbeElectrode)` 于 `src/ELECTRODE/rbe_electrode.h` L6。
  注意与常数电势电极耦合,`compute_vector`/`compute_matrix` 等
  [RBE-E] L742-858)。其采样逻辑被复制进 `prepareSampleK()`([RBE-E] L313-405),
  **源码级作用域疑点(未运行验证)**:L321 写入的是函数**局部** `K[P][3]`,而
  `compute()` L418 声明、L423 广播的是**另一个**未初始化 K(`compute_vector()`
  L757/L760 同病)——即抽样结果无法经此数据流回到调用方;作为对照,单类版本
  [RBE] L348-358 的写入目标就是广播缓冲,无此问题。移植时**只以 [RBE] 为准**。

---

## 2. 参数清单

| 参数 | 含义 | 源码 | 默认/示例 | 备注 |
|---|---|---|---|---|
| `alpha` | Ewald 分裂参数(α = g_ewald²) | [RBE] L214 | 示例 0.07(real 单位) | 决定采样高斯宽度 √(2α) 与自能 |
| `P` | 随机批大小(每批 k 向量个数) | [RBE] L215 | 示例 200 | 无自适应 |
| `RankID` | 批轮转度(每 R 步全员刷新) | [RBE] L216, L211 | 默认 1;示例 4 | 与 MPI 进程数无关 |
| `Step` | 更新时基(每 compute+1) | [RBE] L360 | — | 每 MD 步重采样 |
| `S1_X/Y/Z`,`S` | 采样分布离散归一化(每维和/3D 和−1) | [RBE] L249-252 | 内算 | Poisson 近似;L250-251 有 xprd/yprd 混用 slip |
| `sigma_i` | 每维提议高斯 σ = √(αL_i²/2π²) | [RBE] L283-285 | 内算 | 与目标分布同尺度 |
| **收敛/误差参数** | —— | —— | **不存在** | 源码无误差估计/自适应/收敛判据(§1.4) |
| 种子 | —— | —— | **不存在** | 无 srand/无接口(§1.6) |

---

## 3. 映射到我们的 P3 结构

### 3.1 我方 P3 现状(锚点)

引用简写(均为 `/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0/`):
`[GL-K]` = `graph_longrange/kspace.py`;`[GL-F]` = `graph_longrange/features.py`;
`[GL-E]` = `graph_longrange/energy.py`;`[GL-G]` = `graph_longrange/gto_utils.py`;
`[EXT]` = mace 0.3.16 `mace/modules/extensions.py`;`[CFG]` = `external-repos/mace-polar/config-mace-polar-1.yaml`;
`[REF]` = `mlip/bench/polar_module3_refdump.py`。

- **k 网格(确定性)**:`compute_k_vectors_flat`([GL-K] L11-127):
  半空间枚举 = origin + 半线 + 半面 + 半球([GL-K] L56-82),逐图滤波
  |k|²≤cutoff²(L94),flatten 后按图分组(L103-127);k0_mask 标记每图首项
  (原点,L108-109);调用点 [EXT] L716-718,cutoff=1.8467166423797607
  (probe JSON;= [GL-G] L181-183 启发式 `0.75·(1/σ_min)·(max_l+1)^0.3·3` 取
  σ=1.5、max_l=1 的值,已数值核对)。
- **GTO 预计算 cache(P3 模块③的目标)**:`_pbc_precompute_geometry`
  ([GL-F] L568-615):`cosines/sines=cos/sin(k·r)·mask`(L579-583),
  `density_basis_fs`(L585)、`feature_basis_fs`(L586)、`volume_per_k`(L588)、
  `k_factor_coulomb`(k=0→0,否则 1/k²,L590-591)、`k_factor_proj`(k=0→0.5,
  否则 1,L592-593)。基表含 **GTO 高斯增宽权重** exp(−k²σ²/2)
  ([GL-G] L33-48;密度 σ=1.5,特征 σ∈{1.5,3.0},[CFG] L26/L28,[REF] L108-114)。
- **势特征动态求值**:`_pbc_forward_dynamic`([GL-F] L617-658):
  ρ(k) 装配([GL-F] L28-67,含 (2π)³/V 归一 L66)→ ×(1/k²)×FIELD_CONSTANT
  ([GL-F] L70-83)→ 投影回特征([GL-F] L86-118,含 k_factor_proj 与 ×2 半空间因子
  L115)→ 通道重排(L637-638)。
- **能量**:`GTOElectrostaticEnergy._pbc_energy_batch`([GL-E] L113-194):
  cos/sin(L125-129)→ ρ(L131-139)→ φ=ρ/k²(L141-148)→
  `E=0.5·V·Σ_half 2Re(ρφ*)/(2π)⁶`([GL-E] L17-30)。**[自算] 常数核对**:
  上式 + L66 的 (2π)³/V 归一 ⇒ E_k = (2π·qqrd2e/V)·Σ_{k≠0}(GTO 形式因子)²/|k|²,
  即标准 Ewald 倒空间和,高斯因子由 GTO 形式因子自带(无 erfc 分裂)。
- **确定性配角(不参与 k 采样)**:自相互作用(`GTOSelfInteractionBlock`,
  [GL-E] L55-62/L156-166;PolarMACE 生产配置
  `include_electrostatic_self_interaction: True`([CFG] L32)⇒ L157 的减法分支
  不激活,自相互作用作为学习模型的一部分保留)、PBC 修正
  (molecule/slab,[GL-E] L168-192;PBC 模式下 features 侧分支不触发
  [GL-F] L640-657)、非周期实空间分支([GL-E] L101-111,[GL-F] L546-565)。
- **帧内调用图(A 线索的"每帧多步")**:[EXT] L710-730(k 网格+缓存,每帧 1 次)
  → L777-792(SCF 循环 `num_recursion_steps` × 2 自旋通道,每步调用
  `forward_dynamic`)→ L913-924(`coulomb_energy` 每帧 1 次,复用同一组 k 向量)。
  `num_recursion_steps: 2`([CFG] L23)⇒ **每帧 k 相关求值 = 2×2 次势特征 + 1 次能量
  = 5 次 k 求和**;缓存(L716-730)在全部 SCF 步间复用。
- 夹具实测规模:`geo3/index.json`:立方盒 L=14.7361 Å、cutoff=1.84672 ⇒
  **Nk=171**(4 个分子盒同规模;[REF] 生成)。

### 3.2 RBE 替换哪些部分、哪些保留

| P3 组件 | RBE 下的去向 | 依据 |
|---|---|---|
| k 集合生成(半空间枚举 + \|k\|²≤cutoff² 滤波) | **替换**为随机批采样(§1.2 的取整高斯 + MH;见 3.3 的分布重推) | [GL-K] L56-94 vs [RBE] L283-339 |
| k=0 特殊化(k0_mask、coulomb=0、proj=0.5、ESP=0.5) | **删除**(RBE 采样空间 = m≠0,S 已含排除;\|k\|²=0 不可能) | [GL-K] L108-109; [GL-F] L589-593/L131-133; [GL-E] L141-143 vs [RBE] L286/L337-338 |
| 半空间 ×2 因子(±k 对偶) | **改为全空间对称采样,去掉 ×2**(或"半空间采样+2×";二者期望相同,§3.3) | [GL-F] L115/L128-129; [GL-E] L24 vs [RBE] 全空间 q(m) |
| (1/P) 平均 + 采样归一化 S | **新增**(确定性"全和"→蒙特卡洛"平均×S") | [RBE] L541 模式 |
| 结构因子 cos(k·r)/sin(k·r) | **保留**(在采样 k 上现算;批每帧固定则每帧一次) | [GL-F] L579-583 |
| GTO 基表(含高斯增宽权重 exp(−k²σ²/2)) | **保留**,是"被估计的物理量"而非采样分布 | [GL-G] L33-48/L152-176 |
| 1/k² 库仑核 + FIELD_CONSTANT | **保留在估计量内**(k→0 不可积,不能进采样分布——与 lammps-rbe 同构) | [GL-F] L70-83; [RBE] L541 |
| (2π)³/V 与 (2π)⁶ 归一约定 | **保留**(估计量常数与 [GL-F] L66/L117、[GL-E] L30 一致) | — |
| 自相互作用(GTO self block) | **保留,确定性** | [GL-E] L55-62/L156-166 |
| PBC 修正(molecule/slab) | **保留,确定性** | [GL-E] L168-192 |
| 非周期实空间分支 | **不受影响** | [GL-E] L101-111 |
| 能量-力同批自洽 | **保留该原则**(同帧同一批进特征与能量) | [RBE] L487/L541 同批; [EXT] L783-792/L913-924 同缓存 |

### 3.3 采样分布与权重的重新推导(不能照抄点电荷公式)

lammps-rbe 的 q ∝ exp(−k²/4α) 来自**点电荷 erfc 分裂**;PolarMACE 是 GTO 多极
体系,其倒空间 summand 自带形式因子 exp(−k²σ²/2)([GL-G] L35)与 k^l 多项式
([GL-G] L41-48)。因此 P4 的采样分布应改成**与形式因子匹配的包络**
(建议 q(k) ∝ exp(−k²σ_q²/2),先取 σ_q = σ_min = 1.5,即密度基的宽度;
特征基 σ=3.0 通道的因子由重要性权重承担),而 1/k²、k^l、cl_scale 等全部留在
估计量里 —— 与 lammps-rbe 的"高斯选样 + 1/k² 计权"架构同构,但常数逐项需重推:

```
L1:   E_k(全和) = (FIELD_const/V) · Σ_{k≠0, 全空间} |ρ̃_GTO(k)|² / |k|²
      // ρ̃_GTO(k) = Σ_j [GTO 密度基项(k)]·e^{ik·r_j}  (未归一;exp(−k²σ²/2) 与 k^l 在该项内)
      // 常数由 [自算] 核对:V/(2π)⁶·Σ_half 2Re(ρφ*) 与 [GL-F] L66 的 (2π)³/V 归一合并即得此式
L2:   无偏估计量  Ê = (1/P) Σ_{i=1..P} f(k_i)/q(k_i),
      // f(k) = 采样空间上的目标项:全空间采样取 f=(FIELD_const/(2V))·|ρ̃|²/k²;
      // 半空间采样取 f=(FIELD_const/V)·|ρ̃|²/k²(即含 ±k 对偶的 ×2)
L3:   q(k) = exp(−k²σ_q²/2) / Z_q,  Z_q = Σ_{k∈采样空间} exp(−k²σ_q²/2)  (离散归一化,类比 [RBE] 的 S)
```
**[自算] 截断口径差异提醒**:确定性网格在 |k|≤1.8467 硬截断([GL-K] L94),
而 RBE 的采样分布无支撑上限(或需自定义截断)。就 σ=1.5 的形式因子而言,
截止之外未被计入的 k 求和占比 ≈ erfc(k_cut·σ)= **8.9×10⁻⁵**(l=0 通道)、
**1.5×10⁻³**(l=1 通道,径向 k² 权重更重);同一 q 下 |k|>k_cut 的采样质量占
**5.3%**(上述均为自算)。⇒ "目标=截断网格" 与 "目标=全支撑分布" 之间存在
~10⁻⁴~10⁻³ 相对量级的系统性差异,**必须在 P4 中显式选择口径**(未决问题 #2);
若目标是复现 P3 的截断值(例如对齐训练口径),采样分布应采取"截断高斯+重归一
(S 相应改为截断空间的和)"。

### 3.4 RBE 的"随机批"如何与"每 SCF 步都要 k 求和"交互

- 事实:一帧内 k 网格/缓存构建 1 次([EXT] L716-730),`forward_dynamic` 被调用
  2(SCF 步)×2(自旋通道) 次([EXT] L783-792),能量再 1 次([EXT] L913-924);
  三者共享同一缓存与同一批 k。
- **方案 A(推荐基线)= 批按帧固定**:每帧为抽样批重建缓存(cos/sin/基表尺寸
  从 Nk 变 P),帧内 SCF 保持"给定批的确定性映射";帧间重采样 ⇒ 轨迹上叠加
  无偏噪声(与 lammps-rbe "每 MD 步重采样"最接近:帧 ≈ MD 步,[RBE] L360)。
  该方案对引擎改动最小(缓存生命周期=帧,与 P3 现存结构一致)。
- **方案 B = 批跨 N 帧复用**:象 lammps-rbe 的 RankID 轮转的"多次刷新"思想,
  但注意其轮转是**每步换一条新鲜批**([RBE] L278-280 语义推演),并非"批用 N 步";
  B 可换取更小噪声但引入批间相关性,属折衷设计(未决问题 #4)。
- **方案 C(不推荐)= SCF 迭代内重采样**:每步的势特征来自不同的随机批 ⇒
  SCF 变成随机映射,不动点/收敛语义丢失;密度更新链式依赖还会放大噪声
  (势特征经 `field_feature_norms`([CFG] L29)缩放后进 charges_maps,
  [EXT] L811-836)。
- **一致性约束**:同一帧内 `potential_features`(SCF 出口,[EXT] L818-820)与
  `coulomb_energy` 必须用同一批,否则能量与势/密度不自洽、力不再等于能量梯度。
- 噪声传导视角:4 次势特征求值(含 SCF)× 1 次能量若用同一批则误差相关,
  若逐次重采样则近似独立;势特征是 charges_maps 的输入 ⇒ **RBE 噪声会进入
  SCF 迭代**(两轮),其放大/阻尼行为需 P4 数值实验(未决问题 #6)。

---

## 4. 引擎集成点清单(当前引擎侧 k 求和尚未实现;按实施顺序)

现状核实:引擎全库无 ewald/kspace/k 空间模块(与 [PLAN] L22 一致),
`libsymmetrix/source/` 无任何随机数设施(grep rand/randn/curand/mt19937/Philox
无命中);polar 模块①②已入码(`mace_nonlinear_kokkos_impl.tpp` L5380-5445),
模块③(k 网格/GTO precompute)是 [PLAN] §12 表的下一项(门 = 与 Python 逐项对账)。

| # | 模块 | 现状与挂点 | 改动 | 注意事项 |
|---|---|---|---|---|
| 1 | **k 网格构造/采样器** | 新增;帧入口 `MaceNonlinearKokkosT::compute_node_energies_forces`(`mace_nonlinear_kokkos_impl.tpp` L3733,声明 `mace_nonlinear_kokkos.hpp` L353);cell/inverse_cell 已有来源:`prepare_factorized_geometry`(`mace_nonlinear_kokkos.hpp` L172-180)、`reduce_stress(volume,…)`(同文件 L191-192) | P3:移植 [GL-K] L11-127 的确定性枚举;P4:按 [RBE] L283-339 模式实现批采样(整数 m、逐轴高斯、MH 接受) | **GPU 并行随机数**:引擎无 RNG,需引入 counter-based RNG(如 Philox/curand),种子入 JSON/绑定;无 MPI(单 GPU)⇒ lammps-rbe 的"rank0 采样+广播+RankID 轮转"([RBE] L278-358)**不需要也不可能**直搬;批采样建议设备端一次生成全批。**[PLAN] L74-77 纪律**:RNG 选型需补上游依据入账 |
| 2 | **结构因子 + GTO 基表(模块③)** | 新增;对应 [GL-F] L568-615 产物清单 | 在采样 k 上现算 cos/sin、density_basis_fs、feature_basis_fs、volume_per_k、核/投影因子(去掉 k0 特殊化) | 精度:lammps-rbe 全 f32([RBE] §1.7),我方门是 f64 亚微瓦级([PLAN] L117-130)⇒ 建议 f64 或受控混合;**批每帧固定后缓存只建一次**(与 P3 相同寿命);批图(batch 多盒)时每图各自的采样与 S |
| 3 | **coulomb 能量(模块⑥)** | 新增;对应 [GL-E] L113-194 | 端口径"帧 1 次";估计量结构 (1/P)Σ·S([RBE] L541 模式) | **S/归一化**的数值实现要用精确离散和(建议弃 [RBE] L249-251 的 Poisson 近似,或做误差审计);能量-力-维里同批([RBE] L487/L541 自洽原则);与 [PLAN] 门(gold 2×10⁻⁵ eV)的偏差预算需含 S 误差与 §3.3 截断口径 |
| 4 | **反向传播(模块⑧)** | 现有原语:`Gate::reverse`(tpp L510)、`Readout::reverse`(tpp L1245);模块②门已验证 fukui 读出头反向([PLAN] L299-308);主链式伴随在 tpp 内 | 新增 k 空间伴随:`∂cos/∂r`、`∂sin/∂r`([GL-F] L579-583 的伴随)、`∂ρ/∂c`(基表线性)、同批自洽([RBE] F=−∂E/∂r 已验证) | **k 对坐标无依赖**(∂k/∂r=0),RNG 无梯度,不参与反向;SCF 展开 2 步的伴随与现模块⑤(charges_maps)反向耦合;virial/stress 需 ∂k/∂cell 与 ∂S/∂L(若 P4 含维里则必须;见未决 #7) |
| 5 | **SCF 链接线(模块④⑤)** | ①②已在 tpp L5380-5445;③④⑤待建,hook 点在 L5445 之后;`field_independent_density` 已产出(L5445) | RBE 使缓存以"批"为键;把批表示与生命周期(帧)纳入 polar 执行规格 | 批间方差进入 SCF(§3.4);`field_feature_norms` 缩放([CFG] L29)会改变噪声传递比例;批大小 P 建议先设为可配置(默认对齐 P3 行为需 rbe_enabled=false) |
| 6 | **绑定/配置** | `calculator.py` 调用点 L3403(已含 total_charge/total_spin 直传;与 [PLAN] L258-260 的口径一致,行号为其后修订版);只读访问器做法见 [PLAN] L257 | 新增配置字段:`rbe_enabled/alpha(P 参数)/P/seed`;访问器暴露批与 S 供门比对 | 复现性:种子接口 + 单机确定序列(与 [RBE] 的无种子现状相反,这是**我们要做的增量**,须入依据账本) |
| 7 | **JIT(可选/后置)** | `jit_nvrtc.cpp` 等(libsymmetrix/source/jit_*);P2/P3 先走非 JIT Kokkos 路径([PLAN] L48) | k 空间内核若成热点再 JIT 化 | 批大小固定时内核形状稳定,适合 JIT;但 P4 阶段不必先做 |

**横切注意(批间方差等)**:
- 能量/力噪声对 MD 能量守恒的影响属本轮 RBE 的固有特性(lammps-rbe 示例用
  BAOAB + SHAKE 跑 NVT 并以 RDF/MSD/VACF 为观测量,[EX] L38-70——说明其接受该噪声;
  但其仓库内**无**收敛/误差数字,**未核实**其精度水平);
- 随机项(倒空间)与确定性项(自能、PBC 修正、GTO self)混合时,只有倒空间项携带
  批间方差;验收表(P4 门,力 RMS vs p)应只对倒空间口径回归;
- **成本模型**:确定性成本 ~ Nk×N 结构因子 + 枚举;RBE 成本 ~ P×N 结构因子 + 采样。
  当前夹具 Nk=171(P=200 时 RBE 无优势甚至更贵);RBE 的收益在大盒
  (Nk ∝ V;L 加倍 ⇒ Nk≈8×)与低精度需求场景 —— 基准方案需用大盒(lammps-rbe
  示例 5.3 万原子、P=200)对齐口径。

---

## 5. 未决问题

1. **PolarMACE 的采样包络与权重**:q(k) 的 σ_q 选择(单独 1.5?1.5/3.0 混合?
   逐通道分解?)、k^l 与 cl_scale 留在估计量的具体形式。已核实:GTO 径向公式
   ([GL-G] L33-48)与归一化 ([GL-G] L117-120);未核实:RBE-GTO 的既有文献做法。
2. **目标口径:截断网格 vs 全支撑**。自算差量级:l=0 通道 8.9e-5、l=1 通道
   1.5e-3(σ=1.5、cutoff=1.8467)。需要 P4 数值实验定夺,并写进验收偏离预算。
3. **P 的建议值与成本模型**:当前夹具 Nk=171(已核实);大盒基准、P 扫描表
   (P4 门)尚未做。
4. **更新频率**:方案 A(每帧) vs B(跨 N 帧) vs lammps-rbe 的 RankID 轮转语义
   (每步换批、每 R 步刷新链流)——单 GPU 引擎侧的对应物未定。
5. **种子/可复现性接口**:lammps-rbe **无**此设施(已核实);引擎需要自建
   (种子入配置、RNG 算法选型),且按 [PLAN] L74-77 纪律必须补上游依据。
6. **误差控制与验收判据**:源码无方差控制(已核实);引擎侧是否需要
   批内误差估计(如多批平均的方差代理)?SCF 对势特征噪声的放大系数未知。
7. **virial/stress**:若 P4 范围含维里,需 ∂k/∂cell 与 ∂S/∂L 的完整链
   (lammps-rbe 有解析维里 [RBE] L549-573 但为点电荷口径);P3 当前只承诺能量/力,
   维里可后置但接口需预留。
8. **electrode 变体**:`prepareSampleK` 的 K 作用域疑点([RBE-E] L321 vs L418/L423、
   L757/L760)仅源码级分析,未运行验证;我们的 pair style 是否需要电极耦合
   (常数电势)决定是否参考该文件。
9. **精度策略**:lammps-rbe f32 流水线 vs 引擎 f64 门([PLAN] L117-130)的取舍
   (速度/显存 vs 门限),需在模块③设计时定案。
10. **批图(batched graphs)语义**:多盒同帧时"每图各自 P 个采样 + 各自 S"
    与 **跨图共享** 的取舍;确定性路径用统一超集 + 逐图滤波([GL-K] L50-53/L94),
    RBE 无此机制,需新设计。

---

## 附录 A:关键行号索引(全部已核实)

- lammps-rbe(kSpace rbe):RNG L62-75;Prob/MH_D L77-95;settings L207-235;
  init(L181-203);setup(L239-253);compute L257-600(采样 L280-339/拷贝广播 L348-358/
  步进 L360;Kx/midterm L390-410;ρ L434-465;力 L467-527;能量 L533-546;
  维里 L549-573;per-atom L577-597);slabcorr L618-669;注册 [RBE-H] L1-3。
- 示例:`kspace_style rbe 0.07 200 4`([EX] L28);53367 原子轨迹
  `RBE_Example/RBE.lammpstrj`(头部)。
- graph_longrange:k 网格 [GL-K] L11-127;势特征缓存 [GL-F] L568-615;
  动态求值 [GL-F] L617-658;装配/核/投影 [GL-F] L28-118;能量 [GL-E] L17-30/L113-194。
- mace 0.3.16:`compute_k_vectors_flat` 调用 [EXT] L716-718;precompute_geometry
  [EXT] L720-730;SCF [EXT] L777-792;能量装配 [EXT] L913-929。
- 引擎:`compute_node_energies_forces` tpp L3733(hpp L353);polar 模块①②
  tpp L5380-5445;reverse 原语 tpp L510/L1245;cell 输入 hpp L172-180;
  calculator 调用点 `symmetrix/source/symmetrix/calculator.py` L3403。

## 附录 B:本报告自算(非仓库输出)清单

1. MH 接受率数值复算(§1.3;按 [RBE] L77-95 公式转写,4000 步/组);
2. F = −∂E_est/∂r 的解析验证(§1.4,按 [RBE] L467-508/L541);
3. GTO k 和常数核对(§3.1,E_k = (2π·qqrd2e/V)·Σ_{k≠0} form²/k²);
4. 截断尾部估计 erfc(k_cut·σ) 与采样质量 5.3%(§3.3);
5. kspace_cutoff 启发式数值核对(§3.1)。

## 附录 C:已核实 vs 未核实边界

- **已核实(源码/数据在案)**:上表全部行号;rbe 无 srand、无误差控制、死代码清单;
  f32 流水线与 AVX-512/Intel 依赖;electrode 变体 K 作用域疑点(源码级);
  `in.spce-bulk-nvt` 参数;Nk=171(夹具 index.json);num_recursion_steps=2
  ([CFG] L23);polar 模块①②入码位置与门结果([PLAN] L291-308)。
- **未核实/需进一步确认**:lammps-rbe **实际运行行为**(仓库工作区为空、示例
  `RBE.out`/`RBE_BAOBAO.log` 记录的是失败运行[缺 MOLECULE 包],无成功运行日志可
  对照——算法描述完全基于源码静态阅读 + 自算);MH 接受率的论文级结论;
  RankID 轮转的确切设计意图(源码存在 `RankID=1` 覆盖 [RBE] L211 与
  arg[2] 无条件读取 L216 两处可疑写法);rbe/electrode 变体的可运行性;
  引擎侧批图采样、JIT 化、性能基准均未开始。
