# 4G-HDNNP distillation status (2026-10-25)

**Question:** can a 4G-HDNNP student distilled from PolarMACE reach the
5 meV/atom · 30 meV/Å accuracy gate while running at 4–20 µs/atom/step?

**Answer so far: speed yes, accuracy no.** Measured FP64 speed is
**13.45 µs/atom/step at 5,184 atoms** (inside the target band). The best bulk
accuracy is **43 meV/atom / 87 meV/Å**, i.e. 8.6× and 2.9× over the gate. The
error breakdown below explains why, and what has to change.

## 1. Provenance: this is not the RUNNER- repository

`git@github.com:Yong-Q/RUNNER-.git` was cloned to `bza/runner-repo` and
inspected. It is a Slurm/VASP committee workflow, not a 4G model:

> "训练准备：MACE / PolarMACE 多种子任务；RuNNer **2G** 数据转换、训练及原生
> 委员会组装。**4G 电荷标签和两阶段训练尚未接通。**"

It has 7 real VASP labels, 3 MACE-POLAR fine-tuned members, 3 RuNNer **2G**
members, and states that the weights do not meet production requirements.

The model here is an independent PyTorch implementation of the 4G-HDNNP
construction in RuNNer 2.0 (J. Chem. Phys. **165**, 152502 (2026);
arXiv:2607.17978): local descriptors → per-atom χ and J networks → periodic
QEq/PME charge solve → `E = E_local + ½ qᵀE_coul q + Σε`, with forces from one
adjoint solve (the construction of RuNNer2 SI Eq. 34). **Documented deviation:**
Adam instead of the multistream Kalman filter.

`bza/runner-repo` remains useful as a *method reference* (committee +
disagreement selection + teacher labelling), and that is what the active
learning loop here follows.

## 2. Speed (measured, FP64, A100)

| quantity | value |
|---|---:|
| atoms | 5,184 (1,728 H₂O) |
| density | 0.0334 atoms/Å³ |
| full energy+force, median | **13.45 µs/atom/step** |
| energy only | 8.47 µs/atom/step |
| neighbour rebuild | 3.64 µs/atom |
| descriptors | 0.49 µs/atom |
| QEq CG iterations | 60 (residual 8.8e-7) |
| train step | 31.8 µs/atom |
| parameters | 72,087 |
| peak memory | 1.06 GB |

At 88 CG iterations (residual 6.6e-10) the same model takes 19.26 µs/atom/step.
Both ends of the 4–20 µs target are therefore measured, not extrapolated.

The earlier 0.72 µs/atom figure was a random-descriptor prototype that skipped
real descriptor evaluation and the force trick; it is retracted.

## 3. Accuracy: the gate is not met

All runs use the decorrelated subset `data_20261025_subset` (1,731 bulk + 990
gas frames; see `DATASET_REBUILD_2026-10-25.md`), FP64, hidden 160, cg 40,
batch 8, 30 epochs, seeds 20261023 / 43 / 44.

| member | bulk E | bulk F | gas E | gas F |
|---|---:|---:|---:|---:|
| c1 | **43.0** | **86.9** | 54.6 | 16.5 |
| c2 | 46.7 | 88.3 | 56.3 | 17.5 |
| c3 | 45.3 | 87.8 | 55.6 | 17.5 |

Units meV/atom and meV/Å, on 253 held-out bulk frames / 168 held-out gas
frames. Gate: 5 / 30.

The three seeds agree to ~4 meV/atom, so this is a systematic limit of the
model, not seed noise.

## 4. Error breakdown: two separate problems

`plan_c_4g/error_breakdown.py` splits the validation error by density and by
species. The split matters because the two halves need different fixes.

### 4.1 Force error explodes with density

| ρ (molecules/Å³) | n | E (meV/atom) | F (meV/Å) |
|---|---:|---:|---:|
| 1e-4 | 48 | 54.4 | 54.3 |
| 5e-4 | 54 | 54.5 | 34.4 |
| 2e-3 | 48 | 52.7 | 57.2 |
| 5e-3 | 47 | 42.3 | 98.1 |
| 1e-2 | 32 | **8.4** | 105.0 |
| 2e-2 | 24 | 23.0 | **338.8** |

Force error roughly quadruples between 5e-3 and 2e-2. The teacher's
intermolecular long-range term in dense water is 4.3–5.8 meV/atom and
36–64 meV/Å RMS (measured in the Plan D note), and that is the scale we are
failing to reproduce. Our electrostatics are **charge-monopole only**; the
teacher has l=1 atomic multipoles plus exact Ewald. Adding layers cannot create
that missing physics.

### 4.2 Energy error is species-specific

| species | n | E (meV/atom) | F (meV/Å) |
|---|---:|---:|---:|
| H2O | 51 | **7.4** | 70.5 |
| NH3 | 49 | **8.7** | 62.2 |
| CO2 | 50 | 10.8 | 177.2 |
| CH4 | 42 | 22.7 | 49.8 |
| C2H6 | 27 | **125.7** | 70.0 |
| C2H4 | 34 | **153.2** | 112.4 |

Water and ammonia are already at 7–9 meV/atom, so the network, the training
loop and the E0 fit are all working. The 15–20× larger error on C2H4/C2H6
points at the **angular descriptor**, which has only 8 G4-like basis functions
over 10 element-pair channels and no explicit many-body/three-body
element-resolved terms beyond that. C–C containing molecules need angular
resolution that H₂O does not.

## 5. Capacity escalation (in progress)

`ModelConfig` now exposes `n_trunk_layers` and `n_property_layers` (default 1,
which reproduces the original architecture bit-for-bit). Two larger students
are training on the same decorrelated data:

| tag | width | layers | params (approx) |
|---|---|---:|---:|
| `4g_d3_h192` | 192 | 3 trunk + 2 property | ~200 k |
| `4g_d2_h256` | 256 | 2 trunk + 2 property | ~300 k |

Parameter count vs the original 58 k: 3.5× and 5×. Speed cost is expected to
stay well inside 20 µs/atom/step because the QEq solve, not the MLP, dominates.

## 6. Active learning (implemented, smoke-tested)

`plan_c_4g/active.py` fixes the two defects of the 2026-10-22 round
(`bench/path_b_active_q4_2026-10-22/attempt-01/pkg/active_learning.py`):

| defect | old behaviour | new behaviour |
|---|---|---|
| diversity | `spacing=3` on candidate **frame indices** | refractory window `tau_mult × tau_int` **per trajectory**, tau from the ACF |
| candidates | single molecule in a 16 Å box only | one MD per `(species, ρ, T)` grid point, so dense states are actually proposed |
| selection | top force variance with the spacing hack | force-variance rank → ACF filter → FPS on force signatures |
| labels | teacher, merged into `train.xyz` | teacher FP64, written to a per-round directory; merge is an explicit separate stage |

Measured `tau_int` on the proposal MD is ~6 frames at `dump_every=5`, which is
consistent with the ~15-frame bulk value at `dump_every=10` after accounting for
the sampling interval.

Smoke test (6 states, H₂O at ρ ∈ {5e-3, 2e-2, 3.5e-2}, T ∈ {300, 600} K,
16 molecules, 40 steps): 54 candidates → 40 after the ACF filter → 40 selected
→ 40 teacher-labelled. End-to-end pipeline works.

## 7. Honest assessment

1. **Speed target is met** (13.45–19.26 µs/atom/step, FP64, measured).
2. **Accuracy target is not met and cannot be reached by width alone.** The
   density trend says long-range physics is missing; the species trend says the
   angular descriptor is too thin.
3. The realistic paths to the gate, in order of expected payoff:
   - **Add multipole electrostatics** (l=1 on each atom, as in the teacher) to
     attack the 339 meV/Å dense-state force error. This is the single largest
     term.
   - **Widen the angular basis** and make it element-resolved (more η / ζ
     pairs, separate C–C channels) for the C2H4/C2H6 energy error.
   - Only then spend more layers/width, and only then run active learning —
     the loop selects frames, it cannot invent missing functional forms.
4. **Fallback that is already validated:** the 128L1 PolarMACE student reaches
   **3.8 meV/atom / 6.1 meV/Å** at **91.6 µs/atom/step**. If accuracy is the
   non-negotiable requirement and 90 µs is acceptable, that model already
   passes both gates. 4G should be positioned as the fast end of a two-model
   portfolio, not the sole deliverable.

## 8. Artifacts

- `plan_c_4g/` (tracked copy of the package)
- `bench/plan_c_4g_2026-10-23/results_al/plan_c_4g_4g_c{1,2,3}.pt`
- `bench/plan_c_4g_2026-10-23/results_al/breakdown_c1.json`
- `bench/plan_c_4g_2026-10-23/results/{speed_fp64_5184_final,speed_fp32_5184_final}.json`
- `bench/plan_c_4g_2026-10-23/active/smoke_r1/`
- `data_20261025_subset/`
