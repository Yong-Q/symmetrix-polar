# Module 4 — MPI, Ghost Atoms, and the H1 Feature Exchange

**中文摘要：** H1 交换是“第一层隐藏态换 ghost”，数学上精确；1-rank H1 模式
曾因把通信截断设到 8 Å（大于 7.37 Å 半盒）造出 1042 个周期 ghost，模型跑在
1170 个节点上，造成 4.87× 开销。该缺陷已修复：1-rank 图折叠到 local atoms，
H1 1-rank **418.64 → 50.20 ms**，与默认路径 50.84 ms 持平（0.99×）；
1-rank parity dE=0、max dF 1.41e-15 eV/Å。2-rank 目前被引擎侧 workspace
分配阻塞，需要先修复才能做 2/4 卡 scaling。单卡多 rank 共享 SM，不会增加算力；
真正的 scaling 需要一 rank 一卡。

This module explains the multi-rank design, the exactness condition, and the
measured one-GPU behaviour. It records the 1042-ghost diagnosis and the accepted
one-rank fold, then the engine-side two-rank blocker.

## 4.1 Domain decomposition vocabulary

| Term | Meaning |
|---|---|
| rank | one MPI process; it owns a subset of atoms |
| local atom | owned by this rank; its energy/force is counted here |
| ghost atom | a periodic or remote copy needed as a **source** for local edges |
| \(N_l\) | number of local atoms |
| \(N_g\) | number of ghost atoms |
| ghost ratio | \(g=N_g/N_l\) |
| H1 | the first-interaction hidden node state that is exchanged between ranks |
| exact 2-hop | each rank builds the full local+ghost graph, including ghosts' own neighbours; numerically exact, expensive |
| 1-hop | only first-hop ghosts are supplied; **approximate**, rejected by the force gate |

The production single-rank path uses no ghosts and no message passing. The
multi-rank path must supply, for every local receiver, the features of every
source within \(r_c\), including sources owned by other ranks.

## 4.2 The H1 exchange

A naive exact multi-rank evaluation builds a 2-hop ghost graph: every rank
needs the neighbours of its ghosts to compute the ghosts' messages. That is
correct but the ghost graph is much larger than the local graph.

The H1 scheme avoids the second hop by exchanging the first-interaction hidden
state:

1. each rank computes layer 0 for its local atoms;
2. **forward exchange:** send \(H_1\in\mathbb R^{N\times C}\) for the ghost
   rows (from the owner to every rank that has that atom as a ghost);
3. layer 1 uses local receivers + ghost **sources** with the exchanged \(H_1\);
4. **reverse exchange:** after the layer-1 adjoint, sum
   \(\bar H_1\in\mathbb R^{N\times C}\) from ghost copies back to the owner;
5. readout and polar seeds are masked on ghosts, so a ghost's energy
   derivative is not double-counted.

![H1 exchange schematic](../figures/18_mpi_h1_schematic.png)

The exchange is exact for the same reason the 2-hop graph is exact: the ghost
row is exactly the owner-computed \(H_1\), and the owner receives the summed
adjoint. The numerical gate (dE 2.91e-11 eV, max dF 1.41e-15 eV/Å) confirms it.

## 4.3 Communication volume

Per exchange,

\[
\boxed{
V_{\rm comm}=N_g\,C\,b,\qquad b=8\ \text{bytes (FP64)}
}
\]

![communication volume](../figures/eq_comm.png)

For the 128-atom NH3 box, the 1-rank H1 diagnostic reported
\(N_g=1042\), \(C=512\):

\[
V_{\rm comm}=1042\times512\times8\approx4.27\ \text{MB per exchange}.
\]

With two forward and two reverse exchanges per eval, the total is ~17 MB. At a
conservative 10 GB/s host/PCIe path this is ~1.7 ms — **not** the 700+ ms
overhead observed. The overhead is dominated by **computing the model on the
ghost-inflated node set**, not by the bytes moved.

## 4.4 The 1042-ghost diagnosis (1 rank)

The H1 1-rank timing input deliberately keeps the H1 mode on one rank to test
the plumbing. The diagnostic log reports:

```
Nlocal:  128
Nghost:  1042
```

So the model runs on \(N=1170\) nodes instead of 128, a factor

\[
\frac{N}{N_l}=\frac{1170}{128}=9.14.
\]

The edge-based stages still only use local receivers, but node-based stages
(embedding, linear-up, products, readout, polar) scale with \(N\). The observed
1-rank H1 slowdown is **4.87×** (894.6 ms vs 183.6 ms), consistent with a
node-heavy, not edge-heavy, inflation.

**Cause:** the H1 mode sets the LAMMPS communication cutoff to
\(r_c+\text{skin}=6+2=8\) Å. The 14.7 Å box has a half-length of 7.37 Å, so
LAMMPS creates periodic ghost images even on one rank. The correct 1-rank path
should either use the production local graph or set the comm cutoff to
\(r_c=6\) Å (< 7.37 Å) and bypass the H1 callbacks.

**Accepted fix (2026-10-20):** the one-rank graph is folded onto local atoms
through the atom map (same trick as `no_domain_decomposition`), the comm-cutoff
extension is skipped on one rank, and the exchange hooks remain installed but
are zero-sized when there are no ghost rows. LAMMPS performs periodic
*self*-swaps even on one rank; the zero-sized hooks avoid writing ghost rows
that no longer exist. The H1 exchange itself uses device-side Kokkos
pack/unpack with `reverse_comm_device = 1`.

| metric (128-atom NH3, 1 rank) | before | after |
|---|---:|---:|
| default `no_domain_decomposition` | 122.80 ms | 50.84 ms |
| H1 `polar_mpi_message_passing` | 418.64 ms | **50.20 ms** |
| **H1 / default** | **3.41x slower** | **0.99x** |
| same-binary A/B of the fold | 176.77 ms | 54.62 ms (**3.24x**) |

The default-path gain (122.8 → 50.8 ms) is the concurrent kernel round; the
MPI change is the 3.41x → 0.99x ratio and the same-binary 3.24x. Gates: H1
1-rank vs default dE = 0.0 eV, max dF = 1.41e-15 eV/Å; 20-step NVT stable
(no lost atoms, drift 4.74e-14 Å / 4.70e-14 eV/Å). See
[`../results/MPI_H1.md`](../results/MPI_H1.md).

**Two-rank fix (accepted):** the default 2-hop OOM was caused by two full-size
reverse buffers (`edge_message_adj`, `raw_weight_adj`) whose live ranges do not
overlap their forward counterparts; they now reuse the forward storage /
alias `weight_adj`. Peak **80,966 MiB (OOM) → 63,088 MiB (~30.8 GiB/rank)**;
parity dE=0, max dF **1.687e-6 eV/Å**. The H1 SIGSEGV was a communication-width
mismatch: `node_feature_width()` returned 512, while layer-1 exchanges 2048;
it now returns the product-block width and adds extent guards. H1 2-rank peak
**6,120 MiB (~3 GiB/rank)**, parity dE=0, max dF **1.01e-14 eV/Å**. Both
2-rank paths pass 20-step NVT; the 2-GPU test has no engine blocker.

## 4.5 One GPU with multiple ranks vs one GPU per rank

Let \(W_l\) be the local work and \(W_g\) the ghost work per rank. On **one**
GPU with \(r\) ranks, the GPU time-slices the same SMs:

\[
T_{1\rm GPU}(r)\approx r\,(W_l+W_g)+W_{\rm comm}.
\]

On **\(r\)** GPUs, each rank gets its own SMs:

\[
T_{r\rm GPU}(r)\approx \max_{\rm rank}(W_l+W_g)+W_{\rm comm}
+W_{\rm global},
\]

where \(W_{\rm global}\) is the global reduction (polar k-space, SCF
allreduce). The speedup over one GPU is therefore approximately

\[
S_{r\rm GPU}\approx
\frac{r\,W_l}{W_l+W_g+W_{\rm comm}+W_{\rm global}}.
\]

This is why:

- **one GPU, two ranks is slower**, even with a perfect MPI implementation —
  the same SMs are shared and the ghost work is duplicated;
- **two GPUs, one rank each** can be faster, but the gain is reduced by
  \(W_g\) and communication;
- the ghost ratio \(g=N_g/N_l\) is large for a 14.7 Å box and shrinks as the box
  grows. The 128-atom case is the worst case for domain decomposition; 432 and
  1024 atoms are the meaningful scaling targets.

## 4.6 Comparison with the upstream linear engine

| Metric | Upstream linear MACE-OFF23 | PolarMACE (pre-fix, pre-round-2) | PolarMACE (final 1-rank) |
|---|---:|---:|---:|
| materialized 1-rank | 6.6 ms | 183.6 ms | 50.84 ms |
| H1 1-rank | 13.586 ms | 894.6 ms | **50.20 ms** |
| H1 1-rank / materialized | 2.06x | **4.87x** | **0.99x** |
| H1 2-rank (one GPU) | 45.878 ms | 1600.5 ms | blocked (engine workspace) |
| H1 2-rank / H1 1-rank | 3.38x | **1.79x** | n/a until the 2-rank fix |

The historical H1 2-rank/H1 1-rank ratio was 1.79x, better than the mature
linear engine's 3.38x. The old 8.7x headline was dominated by the 1-rank H1
baseline (4.87x), which the one-rank fold has now removed (0.99x). The 2-rank
path is currently blocked by engine workspace sizing, not by the pair-style
exchange.

## 4.7 What to measure next

1. Fix the engine-side 2-rank workspace sizing (owned/local rows and local
   edges) and re-run 2-rank default + H1 parity on one GPU. This is a
   prerequisite for any multi-GPU test.
2. On 2 GPUs (one rank per GPU), measure wall vs 1 GPU on 432 and 1024 atoms;
   the expected speedup is below \(r\) because of \(W_g\) and \(W_{\rm comm}\).
3. On 4 GPUs, repeat; watch \(W_{\rm global}\) (polar allreduce) and load
   imbalance.
4. Keep the force gate as the hard acceptance criterion; 1-hop remains
   rejected.

## 4.8 Operational constraint

The A100 has 108 SMs and 80 GB. The 80 GB is **capacity**, not compute: it lets
several ranks, models, and ghost workspaces fit, but it does not add SMs or
memory bandwidth. Multi-rank on one GPU is a correctness/portability test; a
single-simulation speedup requires multiple GPUs, one rank per GPU.
