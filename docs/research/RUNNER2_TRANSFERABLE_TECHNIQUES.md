# RuNNer 2.0: What Symmetrix-Polar Can Learn, and What It Must Not Copy

**Prepared:** 2026-10-20  
**Primary source:** Knoll et al., *RuNNer 2.0: A Software Suite for High-Dimensional Neural Network Potentials*, arXiv:2607.17978v2; J. Chem. Phys. **165**, 152502 (2026) (journal citation as supplied; see §8).  
**Scope:** CPU/web reading only. No RuNNer build or run, no GPU jobs, no engine source changes. All Symmetrix-Polar numbers below are from the existing project reports, not from new measurements.

**One-line verdict.** The transferable asset in RuNNer 2.0 is its *memory-first, matrix-free, local/global-separated* long-range architecture; its QEq plane-wave solver is **not** a drop-in replacement for our Fukui-balanced SCF fixed point, and its CPU throughput numbers are not a like-for-like speed baseline for an A100 FP64 PolarMACE engine.

**Current Symmetrix-Polar reference.** 30.56 ms/eval at 128 NH3 atoms (238.8 us/atom); 224.1 us/atom at 1,296 water atoms (290.4 ms/eval); the engine OOMs at 2,592 atoms; 5,184 atoms is the next capacity target. At 1,296 atoms the exact long-range Ewald is ~1.67% of GPU busy time; the bottlenecks are polar UVU, MH-1 tensor products, and host/launch gaps.

---

## 0. Chinese executive summary (6 lines)

- RuNNer 2.0 最值得迁移的是“内存优先”的长程设计：显式内存预算、分块/缓存、避免形成 N×N 矩阵；其 QEq 平面波求解器本身不适用于我们的 Fukui/SCF 不动点。
- 在 1,296 原子单卡上，Ewald 仅占 GPU busy 的 1.67%，抄 QEq/PPPM 不会带来显著加速；瓶颈仍是 UVU、MH-1 与 host/launch gap。
- 5,184 原子的首要障碍是显存而非算力：2,592 原子已在 80 GB 卡上 OOM，必须做分块/流式 workspace 或分布式全局相位。
- 可借鉴的并行思想是局域-全局两阶段分离和“一次构图、共享中间量”；不可照搬其 root-rank 全局 QEq 汇聚，后者会成为多卡瓶颈。
- 其 CPU ns/day 数字与我们的 A100 FP64 PolarMACE 不是同口径，不能作为速度基线；训练/委员会缓存策略只适用于训练或未来 UQ。
- 建议按“显存分块 → launch/host gap → 多卡分布式全局求解 → SCF 加速 → 委员会共享”排序投入，并以现有数值门作为硬验收。

---

## 1. Bottom line before the details

RuNNer 2.0 is a careful CPU/Fortran engineering effort for HDNNPs. The parts that matter to us are not the raw timings. They are:

1. **A memory budget is part of the algorithm.** RuNNer estimates the RAM for six long-range-training components and chunks the workload to fit. Our 5,184-atom failure is exactly the absence of such a budget at the scale of a *single* graph.
2. **Never form the dense Coulomb operator if a matrix-free application is enough.** RuNNer's QEq solver replaces an O(N^3) dense solve and O(N^2) matvec with an FFT-based O(N log N) matvec (empirically O(N log^2 N) overall). Our exact GTO-Ewald is already reciprocal-space and does not form a dense matrix, so the *pattern* is familiar; the *specific solver* is not ours.
3. **Separate local work from global work.** RuNNer domain-decomposes local features but gathers charges for a global QEq/electrostatics step on the root rank. That is acceptable for their 4G benchmark sizes (up to 2,592 atoms) but is a warning, not a recipe, for our multi-GPU 5,184+ target.
4. **Share expensive intermediates across consumers.** Their committee implementation computes neighbor lists and feature maps once for N_comm networks. Our analogue is sharing graph/features across SCF steps, readouts, and (if ever needed) ensemble members; the replay cache and path-grouped MH-1 are already steps in that direction.
5. **Do not confuse CPU ns/day with GPU ms/eval.** RuNNer's 0.03 ns/day 4G number is 96,000 atoms on 192 CPU threads with a different model and physics. It is not a target for our 128-atom or 1,296-atom A100 engine.

---

## 2. Paper facts, with exact numbers and verbatim equations

This section reports what the paper says. Equations are reproduced as the paper's LaTeX alttext (verbatim), with equation numbers from the paper. They are not translated or re-derived.

### 2.1 2G, 3G, and 4G HDNNPs

**2G-HDNNP.** The total energy is a sum of local atomic energies, each predicted by an element-specific feed-forward network from an atomic feature vector:

```latex
E^{\mathrm{2G}}=\sum_{i=1}^{N}E_{i}\left(\mathbf{G}_{i}\left(\mathbf{R}\right)\right)\,,
```
(paper Eq. 1, verbatim)

Forces are obtained by chain-rule differentiation through the feature map:

```latex
\mathbf{F}_{j}=-\nabla_{\mathbf{R}_{j}}E^{\mathrm{2G}}=-\sum_{i=1}^{N}\sum_{\mu=1}^{M_{i}}\frac{\partial E_{i}}{\partial G_{i,\mu}}\frac{\partial G_{i,\mu}}{\partial\mathbf{R}_{j}}\,,
```
(paper Eq. 2, verbatim)

There is one network evaluation per atom, so the cost is linear in system size. The paper calls 2G the original 2007 HDNNP generation and notes that 2G is local: long-range interactions are included only implicitly in an averaged way.

**3G-HDNNP.** 3G adds explicit long-range terms computed from environment-dependent atomic properties predicted by additional networks:

```latex
E^{\mathrm{3G}}=\sum_{i=1}^{N}E_{i}\left(\mathbf{G}_{i}\right)+E_{\mathrm{long}}\left(\mathbf{P}(\mathbf{G})\right)\,,
```
(paper Eq. 3, verbatim)

and, in the concrete electrostatics + dispersion form used by RuNNer 2.0:

```latex
E^{\mathrm{3G}}=\sum_{i=1}^{N}E_{i}\left(\mathbf{G}_{i}\right)+E_{\mathrm{elec}}\left(\mathbf{q}\left(\mathbf{G}\right)\right)+E_{\mathrm{vdW}}\left(\mathbf{V}\left(\mathbf{G}\right)\right)\,,
```
(paper Eq. 4, verbatim)

The charges `q` and Hirshfeld volumes `V` are still *local* quantities predicted from each atom's environment. The resulting Coulomb and dispersion interactions can have infinite or much increased range, but the charges themselves do not respond to a distant perturbation. An optional total-charge constraint may be applied during inference. 3G training is typically sequential: train the property models first, subtract the long-range contributions from the reference energies/forces, then train the short-range model on the residual.

**4G-HDNNP.** 4G introduces globally equilibrated charges via a QEq step. The atomic electronegativities `chi_i` and hardnesses `J_i` are learned, and QEq distributes the total charge `q_tot` by minimizing an electrostatic energy functional. The energy is:

```latex
\displaystyle E^{\mathrm{4G}}=
\displaystyle\sum_{i=1}^{N}E_{i}\left(\mathbf{G}_{i},\,q_{i}\left(\mathbf{G},q_{\mathrm{tot}}\right)\right)
\displaystyle+E_{\mathrm{elec}}\left(\mathbf{q}\left(\mathbf{G},q_{\mathrm{tot}}\right)\right)+E_{\mathrm{vdW}}\left(\mathbf{V}\left(\mathbf{G}\right)\right)\,.
```
(paper Eq. 5, verbatim alttext lines)

The resulting charges serve two purposes: they enter the long-range electrostatic energy, and they are appended to the atomic feature vectors so that the local energy network sees global information. This is what allows 4G to describe *non-local charge transfer* (charged defects, adsorbates on doped oxides, protonation states). The electrostatically embedded variant, ee4G, additionally appends element-specific electrostatic potentials `Gamma_i^z` at each site as extra features. 4G is also trained in two steps.

### 2.2 Coulomb kernel and QEq

The electrostatic energy uses Gaussian charge distributions of element-specific width:

```latex
E_{\text{elec}}=\frac{1}{2}\mathbf{q}^{\mathrm{T}}\mathbf{E}\mathbf{q}\,,
```
(paper Eq. 11, verbatim)

```latex
E_{ij}=\begin{cases}\dfrac{1}{\sigma_{i}\sqrt{\pi}}\,,&\text{if }i=j\\[10.0pt]
\dfrac{\text{erf}\left(\dfrac{r_{ij}}{\sqrt{2}\gamma_{ij}}\right)}{r_{ij}}\,,&\text{otherwise}\end{cases}
```
(paper Eq. 12, verbatim)

with `gamma_ij = sqrt(sigma_i^2 + sigma_j^2)`. The Gaussian smearing preserves the correct `1/r` behavior at large separation and removes the short-range divergence; an optional screening function can damp the electrostatic energy inside the local environments.

For 4G QEq, the paper minimizes:

```latex
E_{\text{QEq}}=\frac{1}{2}\mathbf{q}^{\mathrm{T}}\mathbf{A}\mathbf{q}+\mathbf{q}^{\mathrm{T}}\bm{\upchi}\,,
```
(paper Eq. 13, verbatim)

with:

```latex
\mathbf{A}=\mathbf{E}+\mathbf{J}^{\mathrm{T}}\mathbf{I}\,.
```
(paper Eq. 14, verbatim)

Setting `dE_QEq/dq = 0` gives the constrained linear system:

```latex
\left(\begin{array}[]{c|c}\mathbf{A}&\begin{matrix}1\\
\vdots\\
1\end{matrix}\\
\hline\cr\begin{matrix}1&\dots&1\end{matrix}&0\end{array}\right)\left(\begin{array}[]{c}q_{1}\\
\vdots\\
q_{N}\\
\hline\cr\lambda\end{array}\right)=\left(\begin{array}[]{c}-\chi_{1}\\
\vdots\\
-\chi_{N}\\
\hline\cr q_{\text{tot}}\end{array}\right)\,,
```
(paper Eq. 15, verbatim)

or, in compact form:

```latex
\mathbf{A^{\prime}}\mathbf{q^{\prime}}=-\bm{\upchi^{\prime}}\,,
```
(paper Eq. 16, verbatim)

The Lagrange multiplier `lambda` enforces `sum_i q_i = q_tot`. The paper notes that `J > 0` must be enforced to keep `A` positive definite, and that `chi` may be negative (physically reasonable for electropositive species). Initializing `chi` and `J` between one and twenty times standard Pearson electronegativity/hardness values is reported to give optimal convergence to physically meaningful parameters.

**Why the QEq solve is expensive, and what RuNNer changes.** A direct LU factorization of the `(N+1) x (N+1)` system scales as `O(N^3)` and is feasible only for small/medium systems. Iterative conjugate gradient (CG) reduces the per-step cost, but a naive matrix-vector product `A v` still scales as `O(N^2)` because the Coulomb kernel couples every pair. RuNNer therefore uses the quasi-linear-scaling particle-mesh QEq scheme of Gubler et al. (paper ref. 62). The key identity is:

```latex
(\mathbf{A}\cdot\mathbf{q})_{i}=\frac{\partial E_{\text{elec}}}{\partial q_{i}}=\int\rho_{i}(|\mathbf{r}-\mathbf{r}_{i}|)\,V(\mathbf{r})\,\mathrm{d}\mathbf{r}\,,
```
(paper Eq. 18, verbatim)

The charge density is projected onto a uniform real-space grid, transformed by FFT, multiplied by the reciprocal-space kernel `4 pi / G^2`, and back-transformed; the potential is integrated against the localized Gaussian density of atom `i`. The matrix-vector product is thus evaluated at `O(N log N)` total cost, dominated by the FFT, **without ever forming the N x N Coulomb matrix**. The same transformed potential yields analytical forces and stress. In practice, the number of CG iterations grows with system size; Gubler et al. found empirically that the overall method scales as `O(N log^2 N)`. This is the origin of the abstract's claim:

> "reducing QEq computational complexity from `O(N^3)` to `O(N log^2 N)`"

The abstract's complexity is therefore an *empirical overall* scaling; the per-matvec cost is `O(N log N)`. The abstract summarizes the consequence as "linear or quasi-linear scaling ... across all HDNNP generations."

**Force trick.** Differentiating Eq. 13 naively requires `3N` linear systems for the charge derivatives. RuNNer instead solves one adjoint system per structure, with Lagrange multipliers `lambda` satisfying:

```latex
(\mathbf{A}+\mathbf{J})^{T}\bm{\lambda}=-\partial E/\partial\mathbf{q}
```
(paper SI text immediately before Eq. 34; verbatim)

and assembles the force as:

```latex
\mathbf{F}_{k}=-\frac{\partial E_{\mathrm{short}}}{\partial\mathbf{R}_{k}}-\left.\frac{\partial E_{\mathrm{elec}}}{\partial\mathbf{R}_{k}}\right|_{q}-\bm{\lambda}^{T}\frac{\partial\mathbf{A}}{\partial\mathbf{R}_{k}}\,\mathbf{q}-\sum_{i}\left(\frac{\partial E}{\partial q_{i}}+\lambda_{i}\right)\frac{\partial(J_{i}q_{i}+\chi_{i})}{\partial\mathbf{R}_{k}}\,.
```
(paper SI Eq. 34, verbatim)

This avoids differentiating through the QEq linear solver and costs one additional linear system of the same size as Eq. 33.

### 2.3 Ewald and PPPM

For periodic systems, RuNNer evaluates the Coulomb energy by direct Ewald summation or by two mesh-based schemes (plane-wave solver and PPPM):

```latex
E_{\mathrm{elec}}=E_{\mathrm{real}}+E_{\mathrm{recip}}+E_{\mathrm{self}}\,,
```
(paper SI Eq. 35, verbatim)

The paper states the scalings explicitly:

- **Direct Ewald:** `O(N^{3/2})`.
- **PPPM and plane-wave solver:** both achieve `O(N log N)` by evaluating the Coulomb kernel in reciprocal space via FFT.
- **Non-periodic systems:** direct pairwise Coulomb sum.

The Ewald splitting parameter `eta` is chosen automatically to equalize the cost of the real-space and reciprocal-space sums at a given target precision. The real-space sum decays as `erfc(eta r)/r`; the reciprocal-space sum uses a Gaussian filter `exp(-G^2/(4 eta^2))`. Gaussian charges with width `sigma_i` modify the self-energy and the short-range real-space term.

For PPPM, the SI gives concrete defaults: `eta = 2.2 Bohr^-1`, or the maximum Gaussian width `max_i sigma_i` when Gaussian charges are used; the FFT grid is chosen so that grid points per Gaussian standard deviation satisfy a precision criterion, and grid dimensions are restricted to multiples of 4 for efficiency. For QEq, a separate FFT-based Lagrange solver imposes the total-charge constraint directly in Fourier space.

### 2.4 Hybrid MPI/OpenMP

RuNNer uses a two-level strategy: MPI distributes work across compute nodes (or processes within a node), and OpenMP exploits thread-level parallelism within each process.

- **Training:** structures are distributed across MPI ranks by round-robin; neighbor lists are generated serially for each structure; feature evaluation over atoms and the unrolled matrix multiplications for chain rules are OpenMP-parallelized; the linear-algebra kernels are offloaded to a threaded BLAS/LAPACK. Gradients are reduced to the root, the optimizer step is collective, and updated parameters are broadcast.
- **Prediction over a dataset:** structures are divided into memory-adapted chunks distributed across ranks; after the forward pass, no inter-rank communication is needed until the final results are collected on the master.
- **MD in LAMMPS:** LAMMPS domain decomposition distributes atoms and neighbor lists. For 3G/4G, the electrostatic and QEq steps require global knowledge; the interface uses LAMMPS' forward/reverse communication to assemble the full local-plus-ghost atom list on the root rank, performs a single global QEq/electrostatics calculation there, and distributes per-atom quantities back.
- **Best thread count:** the paper reports that optimal performance is reached with **eight OpenMP threads per MPI task**.

This is a CPU-node design. Its transferable lesson for us is the *phase separation*: local feature work is distributed, while global charge/electrostatics work is centralized. Its root-rank gather is precisely the part we must not copy at 5,184 atoms.

### 2.5 Memory management for long-range training (and prediction)

Before training, RuNNer estimates the RAM required by **six major components**:

1. raw structure data,
2. neighbor lists,
3. precomputed feature vectors,
4. their nuclear-coordinate derivatives,
5. for 4G, the global charge features,
6. for 4G, the associated Coulomb matrix `A`.

The dataset is partitioned into the smallest number of equal-sized chunks that fit a memory budget (set explicitly or queried from the OS). If the entire dataset fits in one chunk, the code precomputes and caches all features and derivatives before the epoch loop, removing feature evaluation from the inner loop. If multiple chunks are required, features are computed on the fly and released immediately after the optimizer step. The two approaches can be mixed at fine granularity (e.g., cache features but recompute their derivatives).

The paper is explicit about the long-range memory scaling:

- Storage of environment-dependent features and their derivatives scales **linearly** with the number of atoms in the training dataset.
- For 4G and ee4G force training, the additional global-dependency features and derivatives are also cached.
- The dense matrix `partial q / partial r` scales as **`O(N^2)`** with the number of atoms in the system.
- For 4G charge training, the Coulomb matrix and its inverse are precomputed and stored; the required memory also scales **quadratically** with the number of atoms per structure.

In prediction mode, the number of chunks is adjusted so that all feature values and derivatives fit for a single chunk; chunks are distributed over MPI ranks without cross-structure communication. In MD, each MPI task manages only the part of the box it owns: the first step creates a buffered neighbor list and static feature storage; if the owned atom/neighbor count changes, these structures are resized with a preallocated safety margin. Because such reallocations are rare, this is substantially faster than dynamic reallocation.

**Training speedups.** With full RAM caching, 4G force training completes each epoch **25 times faster** than RuNNer 1.3; 4G charge training is **5-10 times** faster; 2G force training is an **order of magnitude** faster at identical thread counts. The conclusions summarize the caching strategies as reducing per-epoch wall time by a factor of **5-25** relative to RuNNer 1.3. The abstract frames the goal as "allowing 4G-HDNNPs to be trained with the same efficiency as their local counterparts."

### 2.6 Committee computation sharing

A committee consists of `N_comm` independently initialized/trained networks evaluated jointly. At evaluation time RuNNer computes neighbor lists and feature maps **once per call**; each of the `N_comm` networks performs its own forward pass over the shared feature vectors. The paper's conclusion:

> "The marginal cost of adding a committee member is therefore approximately equal to the cost of a single neural-network forward pass for each member and atom, which is typically small compared to feature evaluation and neighbor-list construction for realistic system sizes."

All per-member outputs (energies, forces, stresses, charges, Hirshfeld volumes) are stored separately. Mean values are used for MD propagation; the standard deviation is the uncertainty estimate for active learning. The paper reports that **committees of four to eight members are computationally affordable even for 96,000 atoms**. For committee training with on-the-fly features, evaluating the features once also reduces the per-member training cost substantially; otherwise the per-member training cost is essentially that of a single model.

### 2.7 Benchmarks and scale

**Hardware (OpenMP/MPI/ASE benchmarks).** Otus cluster at PC2: each node has two AMD EPYC 9655 processors, **96 physical cores / 192 hardware threads**, up to 4.5 GHz, **768 GiB RAM**, Nvidia NDR InfiniBand at 200 Gbps per port. RuNNer 2.0 version 2.0.0-alpha.11 (commit ef43e5f0bf), GCC 14.3, MKL 2025.2.0, OpenMPI 5.0.8, `-O3` with link-time optimization and `-march=native`.

**96,000-atom OpenMP scaling.** For a 96-atom water unit cell replicated to 96,000 atoms:

- 2G is dominated by feature evaluation; near-ideal speedup is maintained up to **64 threads**, beyond which synchronization limits further gains.
- 3G adds a long-range solver contribution of the same order as feature evaluation.
- 4G is dominated by QEq and electrostatic force evaluation via the force trick. FFT operations scale near-ideally up to **96 threads**, where memory access across the two CPUs becomes a bottleneck; the code still benefits up to **192 threads**.
- Maximum 4G throughput: approximately **0.03 ns/day** at a 1 fs timestep (96,000 atoms, one 192-thread node).

**MPI strong scaling (2G).** Ideal up to **36 compute nodes (6,912 cores)** across all investigated sizes from **12,000 to approximately 12 million atoms**, with no communication bottleneck until the per-node workload becomes too small.

- **1.5 million atoms:** **0.03 ns/day** on one node -> **1.12 ns/day** on all 36 nodes (1 fs timestep). Note: 1.12/0.03 = 37.3, slightly above 36, consistent with rounding of the reported numbers.
- **12 million atoms:** the single out-of-memory point occurs when using only one node (768 GiB total RAM), confirming that **memory, not performance, is the practical constraint** at that scale.

**Comparison against n2p2 and PANNA.** On a 96,000-atom water system, 2G RuNNer 2.0 is "on par" with n2p2 and PANNA under pure-MPI and hybrid MPI/OpenMP; RuNNer reaches lower absolute wall times across the tested configurations, with the gap widening at low MPI task counts. For 4G, RuNNer outperforms both codes at **96, 768, and 2,592 atoms**; the paper reports a gap of **up to two orders of magnitude**. The stated algorithmic reason: n2p2's iterative QEq requires one full `O(N^2)` matrix-vector product with the dense Coulomb matrix per CG iteration, whereas RuNNer's plane-wave decomposition reduces each such product to `O(N log N)`. The paper also notes caveats: PANNA uses a slightly different ACSF implementation and predicts electronegativity, hardness, and short-range energy with a single multi-output network; the comparison is not perfectly like-for-like.

**Comparison hardware.** The n2p2/PANNA comparison was run on Ganymede at Ruhr-Universität Bochum: two Intel Xeon Gold 6430 processors per node, 32 cores each (**64 cores**), up to 3.4 GHz, **500 GiB RAM** per node; Intel oneAPI 2025.3 toolchain, with a GNU-toolchain cross-check.

**At-a-glance numbers (all from the paper):**

| Benchmark | System / hardware | Reported value |
|---|---|---|
| 4G maximum throughput | 96,000 water atoms, one node, 192 threads | **~0.03 ns/day** at 1 fs |
| 2G MPI strong scaling | 12,000 to ~12 million atoms, up to 36 nodes / 6,912 cores | ideal up to 36 nodes |
| 2G 1.5 million atoms | 1 node -> 36 nodes | **0.03 -> 1.12 ns/day** at 1 fs |
| 2G 12 million atoms | one node, 768 GiB RAM | **OOM** (memory-limited) |
| 2G 96,000 atoms vs n2p2/PANNA | pure MPI and hybrid MPI/OpenMP | on par; RuNNer lower absolute wall times |
| 4G 96 / 768 / 2,592 atoms vs n2p2/PANNA | optimal scheme per code | up to **two orders of magnitude** faster |
| Committee size 4-8 | 96,000 atoms | computationally affordable |
| 4G force training | full RAM cache vs RuNNer 1.3 | **25x** faster per epoch |
| 4G charge training | vs RuNNer 1.3 | **5-10x** faster |
| 2G force training | vs RuNNer 1.3 at same threads | **order of magnitude** faster |

### 2.8 Training-only performance details

- **Atom-centric vs neighbor-centric feature derivatives.** The atom-centric route computes all derivatives of a central atom's feature vector with respect to its neighbors in one pass and caches the Jacobian for the whole structure batch; it is mandatory for 4G because the global charge feature creates a dense dependency between every atom's energy and every other atom's position. The neighbor-centric route evaluates only the derivatives of neighbors' feature vectors with respect to one target atom and stores no batch-wide Jacobian; it reduces peak memory when only a small fraction of force components is used per batch. The paper states these strategies only optimize performance, not final training results (up to summation-order differences).
- **Caching the QEq matrix.** From `q' = -A'^{-1} chi'` (paper Eq. 19, verbatim: `\mathbf{q^{\prime}}=-\mathbf{A^{\prime-1}}\bm{\upchi}^{\prime}\,,`), the Coulomb matrix `A` and its inverse can be cached, reducing each charge update to a matrix-vector product. If hardness is fixed, the matrices for all structures can be precomputed and inverted once before training. If hardness is learned, `A` is stored without the hardness diagonal, updated with the new hardness values after each forward pass, and re-inverted; this avoids rebuilding `A` from scratch.
- **Kalman filters.** The multistream variant of Singraber et al. gives each MPI rank one data stream and performs a single collective Kalman update, avoiding a sequential gradient loop; the paper reports this scales nearly as efficiently as gradient descent.
- **Optimizers:** GD, SGD, Adam, and standard/fading-memory extended Kalman filters.

---

## 3. What we can learn

1. **Budget memory before optimizing time.** RuNNer's six-component estimate and chunked loader are a template for our 5,184-atom work. Our failure mode is a single-graph workspace that exceeds 80 GiB, not a lack of arithmetic throughput. A hard cap, a per-workspace ledger, and a tile/stream policy should be designed before more kernel tuning.
2. **Keep the Coulomb operator matrix-free.** RuNNer's central algorithmic win is applying `A` through FFT rather than forming `A`. Our exact GTO-Ewald already avoids a dense matrix, but any future preconditioner, linearized SCF step, or training loss that needs a Coulomb matvec should use the same pattern.
3. **Separate local and global phases explicitly.** Their training/prediction/MD decomposition is a clean example of local feature work plus a global charge/electrostatics phase. For our multi-GPU plan, this argues for a distributed global phase (distributed k-space or all-reduce) rather than a root-rank gather.
4. **Compute shared intermediates once.** The committee result is a quantitative demonstration that sharing neighbor lists/features makes K models much cheaper than K independent models. Our analogues are shared graph/features across SCF steps, readouts, and potential ensemble/UQ members.
5. **Preallocate static buffers with a safety margin.** Their MD prediction mode avoids dynamic reallocation by resizing only when the owned atom/neighbor count changes, with a preallocated margin. Our materialized path should do the same for the 5,184 workspace; allocator churn is also a hidden source of host gaps.
6. **Report per-routine time and memory, including OOM.** The paper decomposes per-step time by feature evaluation, force calculation, long-range solver, and QEq. Our CUPTI traces already do this; the additional lesson is to report the memory ceiling and the exact failing allocation, as our MPI OOM report does.
7. **Use committee UQ design only if UQ is a requirement.** If we add uncertainty quantification, a single call that shares graph/features and stores per-member outputs is the right interface; mean for propagation, standard deviation for active learning.
8. **Training pipeline lessons are real but out of scope.** Chunked data loading, feature/derivative caching, `A`/`A^{-1}` caching, atom-centric Jacobians, and Kalman multistream are valuable if we build an in-engine trainer. They do not improve inference.
9. **Benchmark discipline.** Compare at the same system size, model, precision, and gate status; report hardware and thread/rank counts; do not compare a CPU 4G water benchmark to a GPU FP64 PolarMACE benchmark.

---

## 4. What we should not copy

1. **Do not port the QEq plane-wave solver into our SCF/Fukui fixed point.** This is the most important boundary in this memo.

   > RuNNer's QEq solver computes the matrix-vector product of the Coulomb matrix `A` for the linear system `A' q' = -chi'` (Eq. 16), i.e. it solves a *quadratic* charge-equilibration problem. Our long-range layer is a *nonlinear fixed point* `rho^{(t+1)} = (1-beta) rho^{(t)} + beta F(rho^{(t)})` with a Fukui-balanced update block (our model, not RuNNer's). The unknowns (density/multipoles vs. scalar charges), the nonlinearity, the convergence criterion, and the force adjoint all differ. The QEq solver is not a faster Ewald and cannot be dropped in without changing the model.

2. **Do not copy the root-rank global QEq/electrostatics gather for multi-GPU.** RuNNer assembles the full local-plus-ghost list on the root for a single global solve. At 5,184 atoms and multiple GPUs this centralizes memory and serializes the global phase. It is acceptable in their 4G benchmarks up to 2,592 atoms; it is not a scaling recipe for our target.

3. **Do not use RuNNer's CPU ns/day as a like-for-like speed baseline.** Their 4G 0.03 ns/day is 96,000 atoms on 192 CPU threads with a different model and physics; our 30.56 ms/eval is 128 NH3 on one A100 with a 270 MB nonlinear PolarMACE model in FP64. The two numbers refer to different system sizes, models, and hardware, so their ratio says nothing about kernel quality.

4. **Do not apply training caches at inference scale.** Caching all features and derivatives is a *training* strategy to trade RAM for repeated-epoch speed. At 5,184 atoms, inference is already RAM-limited; caching more is counterproductive. Use on-the-fly/tiled execution there.

5. **Do not introduce dense `A` or `A^{-1}` storage.** RuNNer's `O(N^2)` per-structure matrices are for 4G charge training and are paired with a linear QEq solve. Our inference path should remain matrix-free; an `N x N` matrix at 5,184 atoms is exactly the wrong direction.

6. **Do not copy ACSF-specific batching literally.** RuNNer groups ACSFs by shared parameters and reuses pairwise distances/angles/cutoff values. Our features are MACE tensor products; the general principle (group by shared parameters, reuse intermediates, SIMD over parameter arrays) is already embodied in our path-grouped MH-1 and team kernels. Porting ACSF code would be a category error.

7. **Do not treat committee averaging as an accuracy fix.** Committees provide uncertainty estimates and stabilize predictions through averaging; they do not make a single model more accurate in the sense of fixing a missing physical term. If we need UQ, add it deliberately; do not add K models to chase speed or accuracy.

8. **Do not generalize RuNNer's 2G 36-node scaling to 4G or to our SCF.** Their ideal 36-node strong scaling is for 2G local HDNNPs from 12,000 to ~12 million atoms. The 4G benchmarks stop at 2,592 atoms, and the global QEq phase is root-centered. There is no evidence in the paper that the 4G/global-charge path scales like the 2G path.

9. **Do not replace exact full-k Ewald with PPPM solely for asymptotic scaling at 1,296 atoms.** The exact Ewald is already only 1.67% of GPU busy (9.6 ms of 573.8 ms pre-team; 2.57% after team), and our gate is exact-force parity. The crossover where a mesh method wins is far above our current sizes; the approximation risk is not justified by the speed budget.

10. **Do not use FP32/TF32 or mixed precision because RuNNer is CPU double precision.** Our production contract is FP64 everywhere; the paper does not provide a GPU mixed-precision result that would satisfy our force gate.

---

## 5. Mapping to Symmetrix-Polar

The tables below separate the four questions the memo must answer: (a) 1,296-atom single-GPU speed, (b) 5,184+ capacity/throughput, (c) multi-GPU/MPI scale-out, and (d) training-only. "Applies?" is about *direct technical applicability*, not whether the idea is interesting.

### 5.1 (a) 1,296-atom single-GPU speed

Current reference: **224.1 us/atom = 290.4 ms/eval** for full PolarMACE on one A100 (B4 + team MH-1, main tree). The dominant GPU-busy families in the post-team, pre-B4 trace are MH-1 76.10 ms (17.8%), backbone nonlinear 112.53 ms (26.3%), polar pipeline 111.30 ms (26.0%), and long-range Ewald 9.61 ms (2.57%); GPU busy is 374.47 ms. The final wall time is 290.4 ms after B4/team-coalescing, so older trace percentages are indicative, not exact post-B4 fractions.

| RuNNer 2.0 technique | Applies to Symmetrix-Polar? | Expected effect at 1,296 atoms | Risk |
|---|---|---|---|
| **Plane-wave/FFT QEq solver** (`A q` without forming `A`, `O(N log^2 N)` overall) | **No, not directly.** It solves the 4G QEq linear system for scalar charges; our long-range layer is a Fukui-balanced nonlinear SCF fixed point. | **0% direct.** The 0-3% is the absolute ceiling if it were a drop-in Ewald replacement: Ewald is 9.56 ms / 573.80 ms GPU busy (1.67%) pre-team and 9.61 ms / 374.47 ms (2.57%) post-team, or 9.6 ms / 290.4 ms wall (3.3%). Even a 10x faster Ewald saves at most ~8.6 ms (~3%). | **High.** Changes the model and exactness; QEq math does not match SCF. |
| **PPPM or plane-wave Ewald instead of exact full-k Ewald** | **Partial/approximate only.** Our exact full-k Ewald is a gate; a mesh method would be an approximation. | **0-3% at 1,296** for the same reason; it may help memory at 5,184 only if k-space tables are the OOM driver, which is not established. | **High.** Interpolation error and force-gate risk (1e-4 eV/A); not justified by the time budget. |
| **Force trick** (one adjoint linear solve instead of `3N`) | **Already conceptually present.** Our reverse-mode adjoint differentiates the whole SCF chain; there is no separate 3N-solve loop. | **0% incremental.** | **Low.** Verify no redundant adjoint solves remain. |
| **Compute neighbor lists/features once; share across consumers** | **Partly already.** Replay cache, path-grouped MH-1, cached k-grid/GTO/cos-sin. | **Already captured.** The remaining time is kernel work and launch gaps, not repeated feature construction. | **Low.** |
| **Committee sharing** (one graph/features, `N_comm` forward passes) | **No production committee.** Only relevant if we add UQ/ensembles. | **0% for the single model.** If added, marginal cost ~one NN forward per member/atom (paper). | **Memory at 5,184; not on the critical path.** |
| **Single-call/batched execution to reduce launch and host gaps** (our interpretation of their shared-work design; the paper has no GPU graph technique) | **Yes, as our own engineering direction.** | **4.8-5.9 ms** host/launch gap at 128 atoms (16-19% of 30.56 ms); at 1,296 the pre-B4 trace implies **~33 ms** (8.2% of 407.75 ms), not re-measured post-B4. A 50% gap cut is ~15 ms (~5%) at 1,296 and ~2.4-3.0 ms (~8-10%) at 128. | **High.** Host-visible Fukui/fbar sums; CUDA Graph capture is not safe; a prior merged-unpack attempt was slower. |
| **Hybrid MPI/OpenMP thread tuning** | **No.** Our engine is GPU/MPI; OpenMP thread counts and MKL pinning do not apply. | **0%.** | **N/A; could mislead tuning.** |

### 5.2 (b) 5,184+ capacity/throughput

Current reference: **2,592 atoms OOM** on a 2.0 GiB allocation after filling an A100 80 GB; **5,184 atoms OOM** on a 0.567 GiB allocation. A naive linear extrapolation of the 2,592 peak (>78 GiB) gives >156 GiB at 5,184, so fitting one 80 GB GPU requires at least a **~2x peak-memory reduction**. This extrapolation is ours, not measured.

| RuNNer 2.0 technique | Applies to Symmetrix-Polar? | Expected effect at 5,184+ | Risk |
|---|---|---|---|
| **Explicit six-component memory budget** | **Yes, by analogy.** Our OOM is a memory-budget failure for a single graph. | Enables a hard cap and a per-workspace ledger; prerequisite for 2,592/5,184 rather than a speed number. | **Low-medium.** Requires allocator instrumentation and discipline. |
| **Chunked data loading / tiling** | **Yes, as the closest analogue.** Their chunks are over training structures; ours must be over edges/nodes of one structure. | Target **2,592 <= 75 GiB** and **5,184 <= 78 GiB** peak. If per-edge buffers dominate, edge tiling can cap memory; the first acceptance gate is fit, not speed. | **Medium-high.** Extra launches, recomputation, summation-order changes; enable only above an N threshold. |
| **Cache if it fits; on-the-fly otherwise** | **Methodology yes; caching at 5,184 is counterproductive.** | Use on-the-fly/streamed execution at 5,184; keep caching for <=1,296 where it fits. | **Medium.** Memory-vs-recompute trade; cache invalidation. |
| **Static buffered neighbor list/feature storage with safety margin (MD)** | **Yes.** | Small speed gain and fewer reallocation spikes; **not** a capacity cure by itself. | **Low.** |
| **Avoid dense `N x N` Coulomb matrix** | **Yes, already matrix-free.** Verify no dense Coulomb workspace remains. | Prevents `O(N^2)` memory; potentially large at 5,184 if any dense matrix is hiding in the SCF/Ewald path. | **Low if already absent.** |
| **Particle-mesh Ewald for k-space memory** | **Speculative.** Only if k-space tables (not edges) are the OOM driver. | Unknown; could reduce memory, but the Ewald time fraction is only 1.67%, so speed is not the goal. | **High.** Model/gate changes; not established as the memory driver. |
| **Committee sharing for ensemble throughput** | **Future UQ only.** | K=4-8 members could be affordable if graph/features are shared; marginal NN forward per member. | **Memory at 5,184; defer.** |

### 5.3 (c) Multi-GPU/MPI scale-out

Current reference: 2 ranks on one shared A100 give no speedup (both ranks share 108 SMs); the 2-GPU test has no engine blocker but queue capacity prevented the run. H1/default 2-rank parity passes; per-rank footprints at 128 atoms are ~31 GiB (default exact 2-hop) and ~3 GiB (H1), but those are small-system numbers and do not establish 5,184 memory.

| RuNNer 2.0 technique | Applies to Symmetrix-Polar? | Expected effect at multi-GPU | Risk |
|---|---|---|---|
| **Hybrid MPI/OpenMP two-level decomposition** | **Analogy only.** Ours is MPI + GPU, not MPI + OpenMP. | Architectural guidance: keep local feature work distributed and make the global phase explicit. No direct speed number. | **Low if translated as local/global phase separation.** |
| **Root-rank global QEq/electrostatics gather** | **No; do not copy.** | **Negative at 5,184.** Centralizes charges and the full local+ghost list; a serial global solve and memory hotspot. | **High.** Scaling and memory bottleneck. |
| **Structure-level MPI parallelism for dataset prediction** | **Only for throughput over independent structures**, not MD. | Near-linear for dataset evaluation; irrelevant to a single-structure MD step. | **Low, but scope mismatch.** |
| **LAMMPS domain decomposition + forward/reverse ghost communication** | **Already used; H1 path exists.** | Keep and optimize overlap/communication volume. On one shared GPU, 2 ranks cannot win by construction. | **Medium.** Ghost replication; H1 currently runs the full node pipeline on ghost rows. |
| **One rank per GPU** | **Yes, planned.** The paper's CPU node scaling is not evidence for this. | Target **>=1.6x** 2-GPU speedup at 5,184, per-rank peak <=40 GiB. | **High.** Requires a distributed global SCF/Ewald phase; depends on (b) fitting per rank. |
| **36-node 2G strong scaling as a model for 4G** | **No.** | Not transferable: 4G benchmarks stop at 2,592 atoms; the global QEq phase is root-centered. | **Misleading if used as a baseline.** |

### 5.4 (d) Training only (not applicable to the inference engine)

Our production engine is inference-only; training is done in PyTorch/external tooling. The following RuNNer techniques are real but do not improve `/eval` latency or single-GPU capacity. They are listed so they do not get misclassified as inference optimizations.

| RuNNer 2.0 technique | Applies to Symmetrix-Polar? | Expected effect on engine runtime | Risk / note |
|---|---|---|---|
| **Chunked dataset loader; RAM cache vs on-the-fly vs disk cache** | **No.** | **0%** on inference; the reported **5-25x** is RuNNer training vs RuNNer 1.3. | Scope creep if imported into the engine. |
| **Cache `A` and `A^{-1}`; store `A` without the hardness diagonal and re-invert** | **No.** QEq training only; `O(N^2)` memory. | **0%** on inference. | Memory blow-up at 5,184; not applicable to nonlinear SCF. |
| **Atom-centric vs neighbor-centric Jacobian; periodic-image packing** | **No.** Training force-derivative strategy. | **0%** on inference. | N/A. |
| **Kalman multistream; committee training with shared features** | **No.** | **0%** on inference. | N/A. |
| **Two-step 3G/4G training (properties first, short-range residual)** | **No.** Model-training workflow. | **0%** on inference. | N/A. |

---

## 6. Concrete next experiments, ranked by expected value

The ranking is by expected value for the stated goals (1,296-atom single-GPU speed, 5,184+ capacity/throughput, multi-GPU scale-out). It is not a commitment to run GPU jobs now; these are proposals with hard acceptance gates. Each experiment must use the existing gate suite and must not trade accuracy for speed.

### E1 (rank 1) — Bound the polar workspace and tile the edge dimension

**Category:** (b) 5,184+ capacity/throughput.  
**Why:** RuNNer's memory-budget/chunking discipline is the closest match to our OOM failure. 2,592 atoms already fails a 2.0 GiB allocation after filling 80 GB; 5,184 fails a 0.567 GiB allocation. The first problem is fit, not FLOPs.  
**Method:**

1. Re-run the existing `cudaMalloc` tracer at 1,296 and 2,592 atoms; record every allocation >0.5 GiB and its live range. Classify per-edge, per-node, k-space, model, and SCF-state workspaces.
2. Build a six-component-style budget table (analogous to RuNNer's) and set a hard cap (e.g., 72 GiB) for the single-GPU path.
3. Tile the edge dimension (start with 4,096-16,384 edges per tile) and fuse forward/reverse passes per tile; reuse the forward edge-message storage for the adjoint (the MPI fix already does this); avoid keeping both raw and weighted edges live; stream SCF states rather than materializing the full chain.
4. Pre-size all workspaces with a safety margin; add an N-threshold switch so <=1,296 atoms keep the current fast path.
5. If edge tiling alone is insufficient, tile node-feature blocks as a second dimension; do not introduce a dense Coulomb matrix.

**Acceptance criteria:**

- **2,592 atoms:** `run 0` returns rc=0; peak <=75 GiB; dE <=5e-5 eV; max dF <=1e-4 eV/A against the established reference; three repeated evaluations at fixed positions agree to <=1e-15; 10-step NVE stable (no lost atoms).
- **5,184 atoms:** rc=0; peak <=78 GiB; same dE/dF/repeat/NVE gates; per-atom time <=1.5x the accepted 1,296 value (<=336 us/atom, <=1.74 s/eval) as the first capacity gate. Tighten only after the fit is proven.
- **No regression at 1,296:** <=290.4 ms/eval; if tiling is slower there, it must be disabled below a measured threshold.
- If the peak cannot be reduced below 80 GiB, report the exact failing allocation and stop; do not drop physics or precision to fit.

**Risk:** medium-high. Tiling adds launches and may reorder reductions; use bitwise-identical tiles where possible and re-run the full gate suite. The `2,592 -> 5,184` linear-extrapolation assumption (>156 GiB) is ours and may be pessimistic if edges/atoms scale sublinearly, or optimistic if they do not.

### E2 (rank 2) — Close the launch/host gap and reduce launch count

**Category:** (a) 1,296-atom single-GPU speed.  
**Why:** Our measured host/launch gap is 4.8-5.9 ms at 128 atoms (16-19% of 30.56 ms), with 1,228 launches (916 Kokkos + 312 cuBLAS/external). At 1,296 the pre-B4 trace implies ~33 ms (8.2% of 407.75 ms), though the post-B4 gap was not re-measured. RuNNer's relevant lesson is "one call, shared intermediates," not a GPU graph recipe.  
**Method:**

1. Inventory every host-visible scalar sum and `create_mirror_view_and_copy` in the forward/reverse path (Fukui/fbar sums, global reductions, `dbar`/sources, SCF update-block host loops).
2. Pre-size all workspaces; remove `ensure_view`/`Kokkos::realloc` from the hot path.
3. Fuse e3-linear pack/unpack where the layout allows (408 launches, 2.86 ms at 128); group tiny cuBLAS calls (312 external launches, 3.91 ms at 128); finish safe affine/LN epilogue fusions.
4. Only after the host-sync inventory is empty, evaluate CUDA Graph capture for the deterministic subset. Do not capture host-visible Fukui/fbar sums.
5. Do not repeat a merged-unpack design that already measured slower; require a layout redesign and an A/B.

**Acceptance criteria:**

- 128 atoms: median <=28.4 ms (>=7% below 30.56 ms), <=900 launches, full gate suite + repeated-eval PASS.
- 1,296 atoms: median <=275 ms (>=5% below 290.4 ms), no regression >2% in any gate.
- Each accepted slice must save >=0.2 ms at 128 atoms (the project's existing threshold); sub-threshold slices are rejected.
- If a host-sync removal changes forces beyond 1e-4 eV/A or breaks repeated-eval determinism, revert.

**Risk:** high. CUDA Graph is blocked by host-visible sums; prior merged-unpack work was negative; bitwise/determinism gates are strict. The expected gain is real but bounded by the 4.8-5.9 ms gap at 128 and the ~33 ms pre-B4 gap at 1,296.

### E3 (rank 3) — One rank per GPU with a distributed global SCF/Ewald phase

**Category:** (c) multi-GPU/MPI scale-out.  
**Why:** RuNNer separates local and global phases but gathers the global QEq/electrostatics on the root. That is a warning. Our 2-GPU test has no engine blocker, but 2 ranks on one A100 cannot speed up; true scaling needs one rank per GPU.  
**Method:**

1. Run 2x A100, one rank per GPU, at 2,592 and 5,184 atoms (after E1 makes them fit), in both default exact 2-hop and H1 modes.
2. Measure ghost ratio, communication fraction, per-rank peak, and per-rank GPU utilization.
3. Implement or verify a distributed global phase: distribute the k-space/charge reduction across ranks (all-reduce or distributed FFT) rather than gathering the full system on rank 0. If a replicated global solve is unavoidable, keep it off the critical path and overlap it with local work.
4. Do not port RuNNer's root-rank gather; use it only as a checklist of what to avoid.

**Acceptance criteria:**

- 5,184 atoms completes on 2 GPUs; speedup >=1.6x vs 1 GPU at the same atom count; per-rank peak <=40 GiB.
- Cross-rank dE = 0 (or <=5e-5 eV), max dF <=1e-4 eV/A; 20-step NVT stable; no lost atoms.
- Communication <=25% of wall time; per-rank utilization not dominated by one rank.
- If 2-GPU speedup is <1.6x, report the communication/global-phase breakdown and stop; do not claim scale-out.

**Risk:** high. H1 currently runs the full node pipeline on ghost rows; global reductions serialize; E1 is a prerequisite for per-rank memory. The paper's 36-node 2G result is not evidence for this path.

### E4 (rank 4) — Probe SCF fixed-point acceleration without changing the converged density

**Category:** (a) 1,296-atom single-GPU speed, with possible (b) benefit.  
**Why:** The polar pipeline is 111.3 ms (26.0% of GPU busy) at 1,296, and the SCF has two update blocks. RuNNer's QEq CG is not applicable, but the general idea of avoiding repeated global solves and using a better preconditioner is worth a bounded probe.  
**Method:**

1. Instrument the cost per SCF iteration and per k-space pass.
2. Test Anderson acceleration or a preconditioned fixed-point iteration while keeping the same Fukui update and mixing factor; compare the converged density, not just the final energy.
3. Alternatively, fuse SCF/k-space passes to reduce launches (the project already flagged this as a global-reduction-barrier problem; measure before committing).
4. **Do not** replace the fixed point with a QEq linear solve. That is a different model.

**Acceptance criteria:**

- Converged density relative error <=5e-8 (existing module 5 gate); end-to-end 32/32; max dF <=1e-4 eV/A vs the reference.
- >=10% reduction in the polar pipeline at 1,296 (>=11 ms) with no regression at 128; otherwise stop.
- Repeated-eval determinism <=1e-15; if the iteration path breaks it, reject.

**Risk:** high. The fixed point is only ~26% of GPU busy, so the maximum upside is bounded; changing the iteration path can break bitwise gates. This is lower expected value than E1-E3.

### E5 (rank 5) — Committee/UQ sharing, only if UQ becomes a requirement

**Category:** (b) throughput for ensembles / future UQ; not on the current critical path.  
**Why:** RuNNer computes neighbor lists/features once for `N_comm` members; the marginal cost is ~one NN forward per member/atom. If we need uncertainty quantification, this is the right design.  
**Method:**

1. Implement a single call that evaluates K members over shared graph/features and shared k-space geometry.
2. Store per-member outputs separately; use the mean for propagation and the standard deviation for UQ.
3. Measure K=2,4,8 at 128, 1,296, and (after E1) 5,184.

**Acceptance criteria:**

- K=4 total <=1.5x the single-model time; marginal per-member <=1.2x a single NN forward; memory <=1.5x; all gates PASS.
- If memory at 5,184 exceeds the E1 cap, defer.

**Risk:** medium. Memory is the limiter at 5,184; without a UQ requirement this is pure scope expansion.

### Ranking summary

| Rank | Experiment | Category | Expected value | Effort | Risk |
|---:|---|---|---|---|---|
| 1 | E1 workspace budget + edge tiling | (b) capacity | Unlocks 2,592/5,184; no speed claim at 1,296 | Medium-high | Medium-high |
| 2 | E2 launch/host-gap reduction | (a) 1,296 speed | ~5% at 1,296; ~7-10% at 128 if the gap is cut in half | Medium | High |
| 3 | E3 one rank/GPU + distributed global phase | (c) scale-out | >=1.6x 2-GPU at 5,184, if E1 fits | High | High |
| 4 | E4 SCF fixed-point acceleration probe | (a)/(b) | >=10% of the 26% polar pipeline, if gates hold | Medium | High |
| 5 | E5 committee/UQ sharing | (b) ensembles | K=4 <=1.5x single model, if UQ is required | Medium | Medium |

**Explicit non-experiment:** do not spend a cycle porting RuNNer's QEq plane-wave solver into the SCF. The Ewald fraction at 1,296 is 1.67% of GPU busy; the math is for a different problem. If a future preconditioner needs a Coulomb matvec, reuse the *pattern* (FFT/matrix-free), not the QEq solver.

---

## 7. Citations and source URLs

**Primary paper**

- Knoll, A. L. M.; Schäfer, M. R.; Lausch, K. N.; Gubler, M.; Wang, H.; Springborn, R.; El Haouari, R.; Liebetrau, A. M.; Finkler, J. A.; Kocer, E.; Eckhoff, M.; Goedecker, S.; Schmitz, G.; Behler, J. *RuNNer 2.0: A Software Suite for High-Dimensional Neural Network Potentials.* arXiv:2607.17978v2 (2026). https://arxiv.org/abs/2607.17978
- HTML v2: https://arxiv.org/html/2607.17978v2
- PDF v2: https://arxiv.org/pdf/2607.17978v2
- arXiv DOI: https://doi.org/10.48550/arXiv.2607.17978
- Journal citation as supplied by the requester: *J. Chem. Phys.* **165**, 152502 (2026). The arXiv abstract page does not list a journal reference; see §8.
- Local downloaded HTML: `/workspace/group/shared/bza/symmetrix-polar/.tmp_refs/runner2.html`

**Paper references used for context**

- Gubler et al., quasi-linear-scaling particle-mesh charge equilibration, *J. Chem. Theory Comput.* **20**, 7264 (2024) (paper ref. 62).
- Rappe et al., QEq (paper ref. 24).
- Singraber et al., multistream Kalman filter (paper ref. 59).
- n2p2 (paper ref. 30); PANNA 2.0 (paper ref. 32); LAMMPS (paper ref. 64); ASE (paper ref. 58).

**RuNNer software and data**

- RuNNer 2.0 source (GPL-v3): https://gitlab.com/runner-suite/runner2
- runnerase: https://gitlab.com/runner-suite/runnerase-2.0
- Paper data/scripts: https://gitlab.com/runner-suite/runner2-paper-data
- Conda packages: https://anaconda.org/conda-forge/runner

**Symmetrix-Polar project sources used for the mapping**

- `/workspace/group/shared/bza/symmetrix-polar/README.md` (headline performance, capacity, MPI status)
- `/workspace/group/shared/bza/symmetrix-polar/docs/results/MH1_TEAM_FP64.md` (1,296-atom CUPTI family fractions, 224.1 us/atom, 290.4 ms/eval)
- `/workspace/group/shared/bza/symmetrix-polar/docs/results/PLAN_B_30MS_FINAL.md` (128-atom blocking kernels, host/launch gap context)
- `/workspace/group/shared/bza/symmetrix-polar/docs/design/PLAN_B_AUDIT.md` (1,228 launches, 4.8-5.9 ms gap, static floor 22.93 ms)
- `/workspace/group/shared/bza/symmetrix-polar/docs/results/PAPER_COMPARABILITY.md` (1,296 vs 5,184 capacity, Ewald fraction table)
- `/workspace/group/shared/bza/symmetrix-polar/docs/results/MPI_H1.md` and `docs/results/MPI_2RANK_FIX_AGENT_REPORT.md` (MPI memory/parity status)
- `/workspace/group/shared/bza/symmetrix-polar/docs/theory/ENERGY_MODEL.md` (SCF fixed point, GTO-Ewald, RBE)
- `/workspace/group/shared/bza/symmetrix-polar/docs/theory/PERFORMANCE_MODEL.md` (GPU busy vs SM coverage, launch overhead)
- `/workspace/group/shared/bza/symmetrix-polar/docs/THEORY_AND_METHODS.md` (notation, gate methodology)

---

## 8. Claims I did not verify

The following are explicitly outside what this CPU/web reading established. They should not be quoted as independently verified facts.

1. **Journal metadata.** The citation *J. Chem. Phys.* **165**, 152502 (2026) was supplied by the requester. The arXiv abstract page (fetched 2026-10-20) has no journal-reference field; I did not query the publisher or DOI for the JCP version.
2. **No RuNNer execution.** I did not build, install, or run RuNNer 2.0, its LAMMPS pair style, or its test suite. All RuNNer numbers are from the paper text/SI.
3. **Figure values.** I read the text and SI prose, but did not digitize Figs. 5-7. Exact per-configuration wall times in those figures (beyond the numbers quoted in the text) are not independently extracted.
4. **Raw benchmark artifacts.** I did not download or inspect the `runner2-paper-data` repository, the raw timing logs, or the SI tables behind the scaling curves.
5. **1.5 million atoms / 12 million atoms.** The 0.03 -> 1.12 ns/day and the 12-million-atom OOM are paper statements for 2G-HDNNPs on the stated hardware; I did not reproduce them or check the underlying data.
6. **96,000-atom comparison.** The "on par" and "up to two orders of magnitude" statements are the paper's; I did not independently benchmark n2p2 or PANNA, and the paper itself notes methodological caveats (different ACSF forms, PANNA multi-output network).
7. **`O(N log^2 N)`.** This is the paper's empirical overall scaling (CG iterations grow with N); the per-matvec cost is `O(N log N)`. I did not derive or measure a worst-case bound.
8. **4G scaling beyond 2,592 atoms.** The paper's 4G comparison covers 96, 768, and 2,592 atoms. I did not find a 4G multi-node strong-scaling demonstration comparable to the 2G 36-node result; my statement that the root-rank global solve is a scaling warning is an inference, not a paper claim.
9. **Our 5,184 memory extrapolation.** The ">156 GiB" figure is a naive linear extrapolation from the 2,592 OOM (>78 GiB). The actual peak may scale differently with edges/atoms; E1 must measure it.
10. **Our 1,296 host/launch gap.** The ~33 ms (8.2%) figure is inferred from the pre-B4 trace (wall 407.75 ms, GPU busy 374.47 ms). The post-B4 wall is 290.4 ms, and the post-B4 gap was not directly measured.
11. **Particle-mesh benefit for our memory.** The suggestion that a mesh Ewald could reduce our 5,184 memory is speculative; I did not establish whether k-space tables or per-edge buffers dominate the OOM.
12. **SCF vs QEq distinction.** The mathematical distinction is based on the project's `docs/theory/ENERGY_MODEL.md` (nonlinear fixed point with Fukui update) and the paper's Eq. 16 (linear QEq system). I did not perform a formal equivalence proof; the models are different by construction.
13. **RuNNer source behavior.** I did not inspect the RuNNer source code for implementation details beyond the paper/SI description (e.g., exact chunk-size policy, FFT backend, root-rank communication implementation).
14. **Hardware details.** CPU model, clock, RAM, and interconnect are from the SI; I did not verify them against the cluster documentation.
15. **Paper's physical claims.** I report the paper's claim that message-passing MLPs cannot capture the Au/MgO dopant charge transfer because a slab can be thicker than the message-passing range; I did not independently validate that scientific claim.

---

*End of memo.*
