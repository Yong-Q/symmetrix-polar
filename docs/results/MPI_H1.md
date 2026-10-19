# Results — MPI H1 Status and Diagnosis (2026-10-19)

**中文摘要：** H1 2-rank 数值精确（dE 2.9e-11 eV、max dF 1.4e-15 eV/Å），
但同构建下比 1-rank 慢 8.7×；拆开看，H1 2-rank/H1 1-rank 只有 1.79×
（优于上游线性 MACE 的 3.38×），主要缺陷是 1-rank H1 造出 1042 个周期 ghost
（1170 节点 vs 128，4.87×）。第一版修复 gate 失败（dE 756.6 eV），正在返工；
在重新通过 parity 前不报新的 MPI 速度。

This module records the MPI state **as measured**, including a failed fix
attempt. It is deliberately separate from the accepted single-rank results.
No MPI speedup is claimed on one GPU.

## 1. Status summary

| Item | State |
|---|---|
| H1 exactness (2 ranks, pre-fix binary) | **PASS**: dE 2.91e-11 eV, max dF 1.41e-15 eV/Å |
| H1 2-rank wall (one A100) | 1600.5 ms/eval |
| Exact 2-hop 2-rank (one A100) | 5155.5 ms/eval |
| 1-rank production (pre-round-2) | 183.6 ms/eval |
| 1-rank production (round 2) | **56.50 ms/eval** |
| H1 1-rank diagnostic | 894.6 / 990.6 ms |
| Current H1 fix | **not accepted**: first attempt failed the gate |
| Multi-GPU scaling | planned; queue3 has no free 2/4-GPU capacity at the time of writing |

![H1 diagnosis](../figures/17_mpi_h1_diagnosis.png)

## 2. The numbers, apples to apples

The **same-build** comparison is:

\[
\frac{T_{\rm H1,2rank}}{T_{\rm production,1rank}}
=\frac{1600.5}{183.6}=8.72.
\]

Do **not** divide 1600.5 ms by the round-2 56.5 ms: the H1 measurement was
taken before the round-2 kernel changes, so that ratio (≈28×) mixes builds.
The 8.72× number is the honest pre-fix figure.

Decompose it further:

\[
\frac{T_{\rm H1,1rank}}{T_{\rm production,1rank}}=\frac{894.6}{183.6}=4.87,
\qquad
\frac{T_{\rm H1,2rank}}{T_{\rm H1,1rank}}=\frac{1600.5}{894.6}=1.79.
\]

So the 2-rank exchange itself scales at **1.79×**, which is better than the
mature upstream linear engine's **3.38×** (13.586 → 45.878 ms). The defect is
the **H1 1-rank baseline**: 4.87× instead of the upstream's 2.06×.

## 3. Root cause: 1042 periodic ghosts on one rank

Diagnostic log from the H1 1-rank path:

```
Nlocal:  128
Nghost:  1042
```

The model runs on \(N=1170\) nodes instead of 128:

\[
\frac{N}{N_l}=\frac{1170}{128}=9.14.
\]

The H1 mode sets the LAMMPS communication cutoff to
\(r_c+\text{skin}=6+2=8\) Å. The 14.7 Å box half-length is 7.37 Å, so LAMMPS
creates periodic ghost images even without MPI. Node-based stages scale with
\(N\); edge-based stages still use local receivers. The observed 4.87× is
consistent with a node-heavy inflation, not with communication.

The communication volume itself is small:

\[
V_{\rm comm}=N_g C b=1042\times512\times8\approx4.27\ \text{MB per exchange},
\]

about 17 MB for two forward + two reverse exchanges, ~1.7 ms at 10 GB/s. The
700+ ms overhead is **compute on ghost nodes**, not bytes on the wire.

![H1 schematic](../figures/18_mpi_h1_schematic.png)

## 4. The attempted fix and why it is not accepted

The first fix attempted to:

1. skip the comm-cutoff extension / ghost graph on one rank;
2. replace the full host-mirror H1 exchange with device-side staging.

Gate result of `lmp_mpi_fix1`:

| Config | Result |
|---|---|
| default 1-rank | PASS (reference energy −49242.4564 eV) |
| H1 1-rank | **FAIL**: dE 756.6 eV, max dF 2.79e4 eV/Å |
| H1 2-rank | **rc=11** (abort) |
| exact 2-hop 1-rank | PASS (dE 0.0, max dF 1.07e-15) |
| exact 2-hop 2-rank | **OOM** (1.343 GiB allocation) while the GPU was shared with the kernel agent |

The `H1 1-rank` diagnostic with the ghost graph kept (`keepghost`) gives the
correct energy (−49242.4564 eV) and still shows \(N_g=1042\); the zero-ghost
variant is wrong. So the correct fix is not merely “remove ghosts”: the 1-rank
path must bypass the H1 callbacks and use the production local path, or set a
comm cutoff below the half-box and still construct the correct local graph.
This is the current work item. **No new MPI number is accepted until the H1
2-rank parity gate passes again.**

Part of the failure was cross-agent build interference: the kernel agent was
editing `mace_nonlinear_kokkos_impl.tpp` / `mace_polar_impl.tpp` in the same
tree while the MPI agent linked its binary. The MPI work is being isolated
before the next measurement.

## 5. One GPU vs multiple GPUs

On one GPU, \(r\) ranks share the same 108 SMs:

\[
T_{1\rm GPU}(r)\approx r\,(W_l+W_g)+W_{\rm comm}.
\]

On \(r\) GPUs, each rank gets its own SMs:

\[
T_{r\rm GPU}(r)\approx \max(W_l+W_g)+W_{\rm comm}+W_{\rm global}.
\]

The ghost ratio \(g=N_g/N_l\) is large in the 14.7 Å box (worst case) and
shrinks for 432/1024 atoms. Therefore:

- 128 atoms: multi-GPU is not expected to be a large win; it is the
  correctness/worst-case test.
- 432 atoms (33 Å) and 1024 atoms (29.5 Å): the meaningful scaling targets.
- 80 GB is capacity, not compute; it lets more ranks/models fit but does not
  add SMs.

## 6. Planned multi-GPU test (2 GPUs first)

- Queue: `devinstance-queue3`, flavor `ml.pni2.4xlarge` (2×A100-80G, 16C).
- Systems: 128 (worst case), 432, 1024 atoms.
- Modes: H1 exact, exact 2-hop control; 1/2 ranks; parity + 20-step NVT.
- Metrics: ms/eval or ms/step, speedup vs 1 rank, ghost ratio, communication
  fraction, per-rank utilization.
- Acceptance: dE ≤5e-5 eV, max dF ≤1e-4 eV/Å, no lost atoms; then report the
  measured speedup (or the honest absence of one).

## 7. Operational state (2026-10-19 ~11:20 UTC)

- A cheap 1-GPU environment probe/baseline task was submitted to `queue004`:
  `t-20261019191724-4zbd7` (task `polar-mpi-scaling-probe-1gpu`), phase
  `queued` at the last check. It validates the self-contained remote package
  and measures 1-rank baselines for 128/432/1024 atoms.
- The 2-GPU submit to `devinstance-queue3` returned `SubmitOutcomeUnknown`
  (Volc CLI exit 255); the local reservation is
  `pending-7dea5a64…` and holds 2 GPUs. It must be reconciled by task
  name/workdir before a resubmit; the structured status tool cannot resolve a
  pending reservation without a remote task id.
- Queue3 reported `freeGpuCount=0`; queue004 has 12 free GPUs but a hard
  1-GPU/task policy and no 2-GPU flavor.

## 8. Next actions

1. Fix the 1-rank H1 path and re-run the H1 2-rank parity gate.
2. Re-measure H1 on the round-2 kernel build (the 1600 ms number is pre-round-2).
3. Reconcile/free the 2-GPU reservation, then run the 2-GPU test on queue3.
4. Keep the 1-hop mode rejected; do not trade the force gate for speed.
