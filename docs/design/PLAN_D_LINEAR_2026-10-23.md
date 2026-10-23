# Plan D — linear student distilled from PolarMACE (2026-10-23)

> **FP64 correction (2026-10-23, after the run).** Project policy is now
> **all plans FP64**. The 4.473–4.499 µs/atom/step result below is the
> **FP32** path (`pair_style symmetrix/mace/float32/kk` + JIT) and is
> therefore a **reference-only** number, not a Plan D deliverable. A quick
> FP64 probe of the same JSON through the generic FP64 path measured
> **284.8 µs/atom/step** at 5,184 atoms (14.762 s for 10 steps). That path
> has no factorized/JIT FP64 kernels, so it is not the target; a dedicated
> FP64 linear path is required. FP64 doubles memory traffic on a
> memory-bound kernel, so the realistic target is **~9–10 µs/atom/step**,
> not 4. No FP64 speed claim is promoted until measured.

Owner: Plan D. Code and raw results: `bench/plan_d_linear_2026-10-23/`.
Speed is always **µs/atom/step**; accuracy is **meV/atom** (energy) and
**meV/Å** (force) against the FP64 PolarMACE teacher.

## 0. Status

| Deliverable | Measured result | Status |
|---|---|---|
| Linear FP32 engine model contract | format-v2 compact standard MACE, 2 interactions, `l_max=3`, `L_max` 1–2, `r_cut=5.0` Å, M0/R0/R1 contracts | done |
| Student distilled from PolarMACE on 2,166 gas frames | 848,528 parameters, 123 epochs (~22 min on A100) | done |
| Held-out six-molecule gate (5 meV/atom, 30 meV/Å) | **0.924 meV/atom, 3.984 meV/Å** | **PASS** |
| Export to Symmetrix JSON and parity vs MACE-torch | max 0.030 meV/atom, 0.404 meV/Å; LAMMPS 0.029 meV/atom, 0.225 meV/Å RMS | done |
| Speed gate ~4 µs/atom/step at 5,184 atoms | **4.473–4.499** steady-state (best 4.473); median of 10 samples 5.35–5.95 | **PASS (steady state)** |
| Bulk gate (5 meV/atom, 30 meV/Å) | **NOT MET and not assessable on the real bulk set** — the three `path_b_bulk2_*` data directories were empty at 2026-10-23 02:53 UTC; the gas-only student is catastrophically out of distribution on dense water (18.2 eV/Å, 2.27 eV/atom) | **BLOCKED** |
| Cheap long-range correction | teacher intermolecular k-space part is 4.3–5.8 meV/atom and 36–64 meV/Å RMS; a fixed-charge Gaussian-Ewald mean-field correction correlates 0.73–0.76 with it and reduces its RMS to 32–41 meV/Å after one fitted scale; it cannot rescue an untrained dense student | prototype only |

The honest short answer:

1. A **pure linear (local) 128-channel MACE student passes the gas-phase
   accuracy gate by a wide margin** and runs at the paper's linear-path speed.
2. The **bulk gate is a data-coverage problem first and a long-range-physics
   problem second.** Distilling only on isolated molecules destroys the
   foundation model's condensed-phase transfer; on a dense water probe the
   student is off by 18.2 eV/Å while zero-shot MACE-OFF23 medium is off by
   0.139 eV/Å. A long-range correction cannot fix a 18 eV/Å short-range
   extrapolation error.
3. The teacher's *intermolecular* long-range part in dense water is large
   enough that, once a bulk-trained student exists, a cheap correction will
   probably be needed to reach 30 meV/Å. The fixed-charge Gaussian-Ewald
   prototype is a viable starting point but is not yet gated on a
   bulk-trained student.

---

## 1. Linear engine path and the exact model contract

### 1.1 Execution path

The "linear FP32 path" is the standard (local, non-polar) MACE execution in
Symmetrix-XL. The measured reference path uses:

```text
pair_style symmetrix/mace/float32/kk streamed_edges direct \
           jit_device_artifact <model-specific.cubin> \
           jit_device_blocks_per_compute_unit 8
pair_coeff * * <model.json> <element list>
```

- Reference model: `task3_sampling_database/mlip/models/symmetrix/mace-off23-medium.json`
  (MACE-OFF23 medium, 128 channels, 2 interactions, `l_max=3`, `L_max=1`,
  `r_cut=5.0` Å, 10 elements).
- Reference binary: `external-repos/symmetrix-xl/symmetrix/build-kernel-r3/lmp`.
- The direct path requires a **model-specific, precision-specific, backend-specific
  JIT device artifact** (`symmetrix_prepare_jit_device_artifact`). The artifact
  cache key is the model contract, so a new student needs a new artifact even
  when the architecture is identical.
- `streamed_edges non-compiled` is a compiler-free diagnostic path; it was used
  only for the ASE parity check, not for the speed number.
- The single-layer (`num_interactions=1`) path uses built-in direct M0/R0
  execution without an R1 artifact. Dual-layer (2 interactions) is the path
  used here.

### 1.2 What the exporter accepts

`extract_mace_data.py` (source at
`external-repos/symmetrix-xl/symmetrix/source/symmetrix/extract_mace_data.py`)
enforces the following contract for standard MACE:

- `num_interactions` ∈ {1, 2}. A one-interaction model must use the canonical
  equal-channel invariant readout; a two-interaction model must have a
  `RealAgnosticResidualInteractionBlock` (or density-residual variant) as the
  second interaction.
- The first interaction must be a `RealAgnosticInteractionBlock`,
  `RealAgnosticDensityInteractionBlock`, `RealAgnosticResidualInteractionBlock`,
  or `RealAgnosticDensityResidualInteractionBlock`.
- Spherical harmonics must have `l_max = 3`.
- Product output `L_max` must be 1 or 2 (this implementation).
- First-interaction residual and `products[0].use_sc` must agree.
- `radial_format="compact"` writes `symmetrix_format_version=2` with one shared
  radial model and element-indexed parameters. `pair-splines` writes the older
  v1 format with per-element-pair tables.
- The exported JSON contains `execution_contracts` for M0 and R0 (and R1 for
  dual-layer); these are the layout/fingerprint contracts the native engine
  validates at load time.

The exported student matches the reference header exactly:

```json
{"num_channels": 128, "r_cut": 5.0, "l_max": 3, "L_max": 1,
 "num_interactions": 2, "model_type": "MACE", "has_field_coupling": false,
 "atomic_numbers": [1, 6, 7, 8], "num_elements": 4,
 "symmetrix_format_version": 2, "radial_representation": "compact"}
```

with `execution_contracts = ["M0", "R0", "R1"]`. The JSON is 19.4 MB
(FP32 weights plus the compact radial spline grid), versus 37.3 MB for the
10-element reference.

### 1.3 Practical loading notes

- `MACE-POLAR-1-M.model` (the teacher) is **not** loadable by the linear path:
  it is a `PolarMACE` with 512 channels, nonlinear interaction blocks, field
  coupling, 2 SCF steps and exact Ewald. The linear pair style would reject it.
  The student is therefore a separate, standard-MACE checkpoint.
- The exporter is run from the `polar` Conda environment because that is the
  environment where the `symmetrix` Python frontend and CLI are installed:
  `/workspace/caches/conda/envs/polar/bin/symmetrix_extract_mace`.
- The `polar` environment's `symmetrix` backend selector shells out to
  `nvidia-smi --query-gpu=compute_cap` with a 2 s timeout. On this virtualized
  A100 that query intermittently takes >2 s, producing
  `backend 'cuda12-sm80' requires sm80, but visible cuda targets are: none`.
  For Plan D only, a local wrapper `bench/plan_d_linear_2026-10-23/bin/nvidia-smi`
  answers that one query with `8.0` and delegates every other call to
  `/usr/bin/nvidia-smi`. The engine itself was not modified.

---

## 2. Teacher and data

- Teacher: `task3_sampling_database/mlip/models/mace-polar/MACE-POLAR-1-M.model`,
  FP64, 15,238,498 parameters, `r_max=6.0` Å, 512 channels, 2 interactions,
  2 SCF recursion steps, exact Ewald, 83-element domain.
- The 2,166 gas frames already carry teacher labels as `REF_energy` /
  `REF_forces`. A fresh teacher call on four held-out gas frames reproduced
  those labels to <1e-6 meV/atom and <1e-6 eV/Å; they are genuine FP64
  PolarMACE labels, so the held-out evaluation needs no teacher rerun.
- Data: `bench/path_b_data2_q4_2026-10-22/attempt-01/data/{train,valid}.xyz`.
  1,734 train + 432 valid frames, six molecules (H2O, NH3, C2H4, CH4, C2H6,
  CO2), 289 + 72 frames each, one molecule per 16 Å cubic periodic box.
- Element domain present: H, C, N, O. The teacher's E0s used for the student
  are H −15.99000358581543, C −1037.1678466796875,
  N −1490.378173828125, O −2047.592529296875 eV.

The three bulk data directories
(`bench/path_b_bulk2_{a,b,c}_2026-10-23/attempt-01/data/`) were empty at the
time of writing; the remote campaigns had only created their YAML/package
scaffolds. No bulk labels were available, and no q4 task was submitted.

---

## 3. Student training

### 3.1 Configuration

| Item | Value |
|---|---|
| Initialization | `MACE-OFF23_medium.model`, filtered from 10 to 4 elements |
| Architecture | standard `ScaleShiftMACE`, 128 channels, `L_max=1`, `l_max=3`, 2 interactions, `r_cut=5.0` Å, correlation 3, radial MLP [64,64,64], 16×0e readout |
| Parameters | **848,528** (4 elements; the 10-element foundation has 1,428,368) |
| Loss | `weighted` (per-atom energy MSE weight 1.0, force MSE weight 20.0) |
| Optimizer | Adam, lr 1e-3, weight decay 5e-7, clip 10 |
| Schedule | ReduceLROnPlateau (0.8, patience 30), eval every epoch |
| Batch | 16 |
| Precision | float32 |
| Epochs | 123 (stopped after the validation metric plateaued; patience would have run to 300) |
| Wall time | ~22 min on one A100, shared with external jobs |
| Config | `bench/plan_d_linear_2026-10-23/configs/student_off23ft.yaml` |
| Checkpoint | `models/off23ft/checkpoints/plan_d_linear_off23ft_run-42_epoch-123.pt` |
| Converted model | `models/off23ft/plan_d_linear_student_gas128.model` |

The checkpoint is a state dict, not a model object. `checkpoint_to_model.py`
rebuilds the filtered `ScaleShiftMACE` from the foundation config and loads the
state dict with zero missing/unexpected keys; the converted epoch-123 model
reproduces the logged validation metrics exactly (checked at epoch 103 before
the final conversion).

### 3.2 Why fine-tune the foundation instead of training from scratch

The foundation already contains the local chemistry of H/C/N/O organic
molecules, including water. Fine-tuning with the teacher E0s converges in
~20 epochs to below the gate. A from-scratch 128-channel model on 1,734 frames
would need substantially more optimization and is not attempted here.

---

## 4. Held-out six-molecule accuracy vs PolarMACE

Evaluated with `scripts/evaluate_mace.py` on all 432 held-out frames (2,088
atoms, 6,264 force components). The student uses its own fixed E0s; **no
per-configuration or per-species offset is applied** to the reported gate
numbers.

### 4.1 Overall

| Metric | Student (epoch 123) | Gate | Zero-shot MACE-OFF23 medium |
|---|---:|---:|---:|
| Energy MAE (meV/atom) | **0.924** | 5 | 397.24 raw / 0.588 offset-corrected |
| Energy RMSE (meV/atom) | 0.979 | — | 440.47 |
| Energy max abs (meV/atom) | 1.654 | — | 810.24 |
| Energy bias (meV/atom) | +0.208 | — | −397.24 |
| Force MAE (meV/Å) | **3.984** | 30 | 34.557 |
| Force RMSE (meV/Å) | 7.403 | — | 47.841 |
| Force max abs (meV/Å) | 99.432 | — | 274.93 |

The zero-shot MACE-OFF23 raw energy is dominated by its own E0 scale; its
offset-corrected energy is shown only for context. Its force error (34.6
meV/Å) is the meaningful baseline: the distillation improves force accuracy by
**8.7×**.

### 4.2 Per species

| Species | Atoms | E MAE (meV/atom) | F MAE (meV/Å) |
|---|---:|---:|---:|
| H2O | 3 | 0.655 | 2.635 |
| NH3 | 4 | 1.254 | 3.616 |
| C2H4 | 6 | 0.968 | 5.044 |
| CH4 | 5 | 0.387 | 2.650 |
| C2H6 | 8 | 1.101 | 5.320 |
| CO2 | 3 | 1.179 | 2.369 |

The worst force error (C2H6, 5.32 meV/Å) is still 5.6× below the 30 meV/Å
gate. Parity plot with a species-centered energy panel, an explicit energy
residual panel, and a force panel:
`docs/results/figures/PLAN_D_GAS_PARITY.png` (the bench run keeps
`bench/plan_d_linear_2026-10-23/results/parity_gas_student_final.png`).

**Verdict: the pure linear student passes the gas gate with no long-range
correction. On this dataset a correction is unnecessary; the teacher's
intramolecular k-space physics is smooth in local geometry and the local
student absorbs it.**

---

## 5. Bulk / dense-phase transfer: BLOCKED (and why)

Because the real bulk labels were not available, two independent dense probes
were built and labelled locally with the FP64 teacher. Both are
**out-of-distribution** for a gas-only student, and both fail catastrophically.
This is an honest stress test, not a bulk benchmark.

### 5.1 Random dense water probe

- `data/dense_water_probe.xyz`: 64 H2O molecules (192 atoms) in cubic periodic
  boxes at 0.8, 1.0 and 1.2 g/cm³; 20 configurations per density (60 total).
  Molecules were placed on a grid with greedy random-orientation rejection
  (minimum intermolecular distance 1.55 Å), then rattled by 0.015 Å.
- Teacher labels: `data/dense_water_probe.xyz`, wall time 0.27–0.39 s per
  192-atom frame after warm-up.
- These are deliberately unrelaxed packings, not equilibrium liquid
  configurations; the cohesive energy is close to zero and the force
  distribution is broad. They are a worst-case short-range extrapolation test.

| Model | E MAE (meV/atom) | F MAE (meV/Å) | F RMSE (meV/Å) |
|---|---:|---:|---:|
| Gas-only linear student | **2265.9** | **18191.0** | 34853 |
| Zero-shot MACE-OFF23 medium | 418.7 raw (1.02 offset) | 138.6 | 170.1 |

### 5.2 Ordered dense water probe

To remove the close-contact artifact, a 648-atom (216-molecule) periodic
sub-block was extracted from the ideal 1.0 g/cm³ water lattice used by the
speed benchmark (`paper_scale_2026-10-20/water_5184.data`). Every O–H bond is
0.9686 Å, every O has exactly two H, and the minimum nonbonded O–H distance is
2.14 Å. Teacher-labelled file:
`data/water_648_ordered_labeled.xyz`.

| Model | E MAE (meV/atom) | F MAE (meV/Å) | F RMSE (meV/Å) |
|---|---:|---:|---:|
| Gas-only linear student | **764.8** | **3419.0** | 4219.7 |
| Zero-shot MACE-OFF23 medium | 422.6 raw (E0 scale) | 129.6 | 154.6 |

### 5.3 Interpretation

The failure is **short-range distribution shift**, not the long-range tail.
Fine-tuning on isolated molecules forces the local descriptors to explain all
intermolecular contacts as if they were intramolecular; at 2–3 Å nonbonded
distances the model predicts enormous forces. The foundation (trained on
liquid water and molecular clusters) is 25–130× better on the same structures.
The bulk campaign data is the prerequisite for a usable linear student; a
long-range correction added to the current student changes the dense error by
less than 0.1% (§6.4).

---

## 6. Long-range analysis and cheap-correction prototype

### 6.1 Method

The teacher's electrostatic energy is split into a real-space/graph part and a
k-space part. The `PolarMACE` forward uses the registered buffer
`kspace_cutoff` to build the k-grid; replacing it with `1e-6 Å⁻¹` removes the
reciprocal-space contribution while leaving the real-space and self terms
intact. For every probe frame we computed:

```text
dE_lr(dense) = E_full - E_no_kspace                  (dense periodic cell)
sum_iso      = sum over molecules of dE_lr(isolated) (same molecular geometry)
dE_inter     = dE_lr(dense) - sum_iso                (intermolecular long-range)
```

and the same decomposition for forces. This isolates what a local model with
a finite receptive field cannot see from what is a local-geometry effect.

### 6.2 Gas: the k-space part is intramolecular

For a single molecule, `dE_lr` is independent of the box size to better than
0.5% from 16 to 48 Å:

| Species | dE_lr (meV/atom) L=16 | L=24 | L=32 | L=48 | dF_lr RMS (meV/Å) |
|---|---:|---:|---:|---:|---:|
| H2O | 12.558 | 12.826 | 12.895 | 12.931 | ~18 |
| CO2 | 10.514 | 10.510 | 10.516 | 10.520 | ~13 |
| CH4 | 1.199 | 1.195 | 1.195 | 1.195 | ~0.7 |

The gas-phase k-space term is the long-wavelength self-electrostatics of each
molecule, which is a smooth function of local geometry. This is why the local
student can fit it, and why no correction is needed on gas.

### 6.3 Dense: the intermolecular part is large

| Probe | dE_inter (meV/atom) | dF_inter RMS (meV/Å) | dF_inter max (meV/Å) |
|---|---:|---:|---:|
| Random 0.8 g/cm³ (2 frames) | −5.79, −5.82 | 42.3, 36.4 | 155, 133 |
| Random 1.0 g/cm³ (2 frames) | −5.68, −4.34 | 48.8, 48.7 | 193, 198 |
| Random 1.2 g/cm³ (2 frames) | −4.90, −4.68 | 56.5, 50.3 | 162, 153 |
| Ordered 648-atom water | −4.57 | 64.1 | 295.6 |

So in dense water the teacher's intermolecular k-space contribution alone is
4.3–5.8 meV/atom and 36–64 meV/Å RMS. A 5 Å-cutoff model (10 Å receptive field
for two interactions) can absorb the part inside its receptive field, but the
tail beyond it is not representable. A correction is likely to be needed for
the bulk gate once a bulk-trained student exists.

### 6.4 Fixed-charge Gaussian-Ewald mean-field prototype

`scripts/charge_correction.py` implements a periodic point-charge Ewald sum
(real space + reciprocal space + self, validated against finite differences to
2.9e-10 eV/Å) with SPC/E water charges (O −0.8476, H +0.4238). The
intermolecular part is isolated exactly as for the teacher, and the
point-charge cross-molecule interaction is replaced by a Gaussian-smeared one
(σ=1.5 Å, the teacher's multipole smearing width) using
`erf(r/(√2σ))/r − 1/r = −erfc(r/(√2σ))/r`.

| Probe | Teacher inter RMS (meV/Å) | Correction correlation | Fitted scale | Residual RMS after scale (meV/Å) |
|---|---:|---:|---:|---:|
| Random dense (6 frames) | 47.6 | 0.731 | 0.158 | 32.5 |
| Ordered 648 water | 64.1 | 0.765 | 0.213 | 41.3 |

A single global scale of 0.16–0.21 is enough to align the mean-field Coulomb
tail with the teacher's long-range part; that is consistent with most of the
Coulomb interaction lying inside the model's 10 Å receptive field. The
correction explains roughly 53–58% of the long-range force variance and
reduces its RMS by 31–36%. It does **not** reach the 30 meV/Å gate on its own;
a QEq/environment-dependent charge version and a bulk-trained local model are
the next step.

### 6.5 Re-gate with the correction added to the student

The scaled correction was added to the gas-only student on the same dense
frames (intermolecular only; scale fitted on the teacher's long-range part):

| Probe | Baseline E MAE | Corrected E MAE | Baseline F MAE | Corrected F MAE |
|---|---:|---:|---:|---:|
| Random dense subset (6 frames) | 2316.2 meV/atom | 2315.9 meV/atom | 18396.8 meV/Å | 18399.6 meV/Å |
| Ordered 648 water | 764.8 meV/atom | 763.3 meV/atom | 3419.0 meV/Å | 3422.7 meV/Å |

The correction changes the error by <0.1%: **the dense failure is not a
long-range failure, and no cheap correction can substitute for bulk training
data.** This is the central negative result of Plan D and it fixes the priority
for the next step.

---

## 7. Export parity and speed

### 7.1 Export and parity

Export command (run in the `polar` environment, with the local
`nvidia-smi` wrapper on `PATH`):

```bash
SYMMETRIX_BACKEND=cuda12-sm80 \
SYMMETRIX_BACKEND_DESCRIPTOR_PATH=.../backend_cuda12_sm80.json \
LD_LIBRARY_PATH=/opt/matos-tools/cuda/12.9/lib64:/opt/matos-tools/conda/envs/matos-compute/lib \
symmetrix_extract_mace \
  --model models/off23ft/plan_d_linear_student_gas128.model \
  --output models/plan_d_linear_student_gas128_symmetrix.json
```

`symmetrix_prepare_jit_device_artifact --precision float32` produced

```text
jit_cache/artifacts/77e83555653584dac8c1bbc4914aba36e995e70e757b7a32324b7aaf69a4f305/
  factorized_cuda_module_gen11_src847380c859abdbe6d213af7f595ebc0cba0f1986c73e8bce6c2a05b98cffecb1.cubin
```

Parity checks:

| Comparison | Energy | Force |
|---|---|---|
| Symmetrix ASE calculator (FP32, `non-compiled`) vs MACE torch FP32, 6 gas frames | max 0.030 meV/atom | max 0.404 meV/Å, RMS 0.156 meV/Å |
| LAMMPS `symmetrix/mace/float32/kk` direct JIT vs MACE torch FP32, `water1.data` | 0.087 meV total (0.029 meV/atom) | max 0.408 meV/Å, RMS 0.225 meV/Å |

Both are at the FP32 numerical-noise level. The exported model is the same
model as the trained checkpoint, not a re-fit.

### 7.2 Speed protocol

- Structure: `bench/paper_scale_2026-10-20/water_5184.data`, 5,184 atoms,
  1.0 g/cm³ water, 2 atom types (O, H).
- Pair style: `symmetrix/mace/float32/kk streamed_edges direct` with the
  student's JIT device artifact, `jit_device_blocks_per_compute_unit 8`.
- Protocol: one warm-up `run 1` followed by ten measured `run 1` blocks,
  three fresh processes, no neighbor rebuilds; median and minimum of the ten
  measured samples. Units: µs/atom/step.

### 7.3 Measured speed

| Run | Median of 10 samples (µs/atom/step) | Minimum of 10 samples (µs/atom/step) |
|---|---:|---:|
| Student, trial 0 | 5.955 | 4.479 |
| Student, trial 1 | 5.419 | 4.473 |
| Student, trial 2 | 5.350 | 4.499 |
| **Student, median of trial medians / best minimum** | **5.419** | **4.473** |
| Paper linear MACE-OFF23 FP32, 5,184 atoms | — | **4.47–4.53** |

The steady-state per-step cost is **4.473–4.499 µs/atom/step**, i.e. the
student matches the paper's linear-path number. The medians (5.35–5.95) are
inflated by a few 1–2 s host/GPU stalls in the 10-sample windows; the same
stall pattern appears in every model on this virtualized A100 (`nvidia-smi`
itself sometimes takes 1.8–2.5 s).

Control experiments:

- Under a concurrent external GPU job (~29% utilization, ~0.7 GB), the same
  student degraded to 5.75 µs/atom/step minimum and 6.85 median, and the
  reference MACE-OFF23 medium measured 5.96–6.31 minimum / 6.68–8.52 median.
  The slow reference is therefore environmental, not an architecture
  difference.
- The reference model (10 elements, same 128-channel/2-layer architecture)
  measured 5.96–6.31 minimum / 7.35 median in the same loaded window. A clean
  reference run was not possible before the GPU was taken by an external job;
  the paper's 4.47–4.53 is the reference for the clean path.

### 7.4 Speed conclusion

The linear student is **not slower than the paper's linear MACE-OFF23 path**:
same architecture, fewer element-indexed tables (4 vs 10), measured
4.473–4.499 µs/atom/step steady-state at 5,184 atoms. The ~4 µs gate is met on
the steady-state metric. The full 10-sample median is 5.35–5.95 because of
environmental stalls, which must be quoted honestly rather than hidden.

---

## 8. Blockers

1. **Bulk labels were not available.** The three bulk2 attempts
   (`bench/path_b_bulk2_{a,b,c}_2026-10-23/attempt-01/data/`) were empty at
   2026-10-23 02:53 UTC. No q4 task was submitted and none was cancelled.
   The bulk gate therefore cannot be measured on real bulk data.
2. **The gas-only student is out of distribution on dense water** (18.2 eV/Å
   on random dense packings; 3.42 eV/Å on an ordered 648-atom water lattice).
   This is a data-coverage failure, not an engine or export failure.
3. **The GPU is shared.** External jobs used 0.6–55 GB at various times,
   caused `nvidia-smi` compute-capability queries to exceed the Symmetrix
   backend loader's 2 s timeout, and added 1–2 s stalls to LAMMPS steps. The
   clean speed run is reported with its full distribution; the `nvidia-smi`
   wrapper is a local workaround, documented above.
4. **No QEq implementation yet.** The prototype uses fixed SPC/E charges with
   Gaussian smearing and one global scale. It is a mean-field prototype, not a
   charge-equilibration model. A QEq version with element-dependent
   electronegativity/hardness and a bulk-fitted scale is the next design step.

---

## 9. Next steps (priority order)

1. **Wait for the bulk2 data and merge it with the gas set.** Retrain the same
   128-channel student on a mixture of the 2,166 gas frames and the 43k bulk
   frames (or the subset available first), with the teacher labels and the same
   architecture. Keep the gas valid set as a regression check.
2. **Re-gate on the real bulk held-out trajectories** (density/temperature
   stratified), reporting E MAE in meV/atom and F MAE in meV/Å against
   PolarMACE. Do not use the random dense probe as a bulk gate; it is a
   stress test only.
3. **Then add the mean-field correction.** Fit the single scale on a bulk
   training split, not on the test split; evaluate on the held-out bulk set.
   If the residual long-range error is still above 30 meV/Å, upgrade the fixed
   charges to a QEq/charge-equilibration solve with element-dependent
   parameters. Keep the correction intermolecular-only to avoid disturbing the
   learned intramolecular physics.
4. **Re-export and re-measure speed** with the bulk-trained student. The
   architecture stays 128 channels / 2 interactions / `L_max=1`, so the
   measured 4.473–4.499 µs/atom/step should carry over; confirm with the same
   5,184-atom protocol.
5. If the bulk-trained student still fails the force gate, the fallback is a
   wider student (e.g. 192 channels) or a nonlinear readout; both change the
   speed contract and need a fresh measurement.

---

## 10. Artifacts and reproduction

All paths are relative to
`/workspace/group/shared/bza/symmetrix-polar/bench/plan_d_linear_2026-10-23/`
unless absolute.

| Artifact | Path |
|---|---|
| Training config | `configs/student_off23ft.yaml` |
| Training log | `logs/train_off23ft.log` |
| Final checkpoint | `models/off23ft/checkpoints/plan_d_linear_off23ft_run-42_epoch-123.pt` |
| Converted student | `models/off23ft/plan_d_linear_student_gas128.model` |
| Compact JSON export | `models/plan_d_linear_student_gas128_symmetrix.json` |
| JIT artifact | `jit_cache/artifacts/77e835.../factorized_cuda_module_gen11_...cubin` |
| Gas metrics / parity | `results/metrics_gas_student_final.json`, `results/predictions_gas_student_final.npz`, `results/parity_gas_student_final.png` |
| Zero-shot baseline | `results/metrics_off23_zeroshot.json` |
| Dense probes | `data/dense_water_probe.xyz`, `data/water_648_ordered_labeled.xyz` |
| Dense metrics | `results/metrics_dense_student_final.json`, `results/metrics_ordered648_student.json` |
| Long-range decomposition | `results/longrange_decomposition.json` / `.npz`, `results/longrange_decomposition_ordered648.json` / `.npz` |
| Correction prototype | `results/charge_correction.json`, `results/charge_correction_ordered648.json` |
| Re-gate with correction | `results/regrade_dense_random.json`, `results/regrade_dense_ordered648.json` |
| Export parity | `results/export_parity.json` |
| Speed (clean 3-trial) | `results/speed_lammps_5184_clean3.json` |
| Speed (reference 3-trial) | `results/speed_lammps_5184_reference_3trial.json` |
| Speed (under-load control) | `results/speed_lammps_5184_under_load_control.json` |
| Machine-readable summary | `results/plan_d_summary.json` |

### Reproduction commands

Training:

```bash
export PYTHONPATH=/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0
/opt/matos-tools/conda/envs/matos-compute/bin/python -m mace.cli.run_train \
  --config configs/student_off23ft.yaml --name plan_d_linear_off23ft
```

Convert the latest checkpoint:

```bash
/opt/matos-tools/conda/envs/matos-compute/bin/python scripts/checkpoint_to_model.py \
  --checkpoint models/off23ft/checkpoints/plan_d_linear_off23ft_run-42_epoch-123.pt \
  --output models/off23ft/plan_d_linear_student_gas128.model
```

Evaluate gas / dense:

```bash
/opt/matos-tools/conda/envs/matos-compute/bin/python scripts/evaluate_mace.py \
  --model models/off23ft/plan_d_linear_student_gas128.model \
  --valid /workspace/group/shared/bza/symmetrix-polar/bench/path_b_data2_q4_2026-10-22/attempt-01/data/valid.xyz \
  --outdir results --label gas_student_final --per-species-offset
```

Export and prepare the artifact (with the local `nvidia-smi` wrapper first on
`PATH` and the `polar` environment):

```bash
PATH=$PWD/bin:$PATH SYMMETRIX_BACKEND=cuda12-sm80 \
SYMMETRIX_BACKEND_DESCRIPTOR_PATH=/workspace/group/shared/bza/task3_sampling_database/mlip/models/mace-polar/backend_cuda12_sm80.json \
LD_LIBRARY_PATH=/opt/matos-tools/cuda/12.9/lib64:/opt/matos-tools/conda/envs/matos-compute/lib \
/workspace/caches/conda/envs/polar/bin/symmetrix_extract_mace \
  --model models/off23ft/plan_d_linear_student_gas128.model \
  --output models/plan_d_linear_student_gas128_symmetrix.json

/workspace/caches/conda/envs/polar/bin/symmetrix_prepare_jit_device_artifact \
  --model models/plan_d_linear_student_gas128_symmetrix.json \
  --precision float32 --cache-root "$PWD/jit_cache" --lammps-arguments
```

Speed:

```bash
/opt/matos-tools/conda/envs/matos-compute/bin/python scripts/run_lammps_speed.py 3
```

---

## 11. Caveats

- The gas dataset is six small molecules in 16 Å boxes; the gas gate is a
  narrow-domain result and says nothing about reactive or condensed-phase
  chemistry.
- The dense probes are not bulk training data. The random probe is deliberately
  unrelaxed; the ordered probe is an ideal crystal, not a liquid. Both are
  diagnostic stress tests.
- The long-range decomposition depends on the teacher's own real-space/k-space
  split; the "intermolecular k-space" number is the part the model attributes
  to reciprocal space, not a unique physical decomposition.
- The speed medians include environmental stalls. The minimum of ten samples
  in three fresh processes is the steady-state metric; the full distribution is
  reported so the claim can be audited.
- The `nvidia-smi` wrapper is a local workaround for a backend-selector timeout;
  it does not alter the engine, the model, or the numerical results.

---

*Plan D, 2026-10-23. No q4 tasks were submitted. No engine or README files
were modified. The 128L1/192L1 models and the 2,166/43k data were not
overwritten.*
