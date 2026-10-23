# Model and software interfaces — 2026-10-23

One repository, four interfaces around the same model artifact. The engine
stays an external worktree/patch; models are exported to a JSON contract and
consumed by LAMMPS, ASE/Python, and the benchmark harness.

```
MACE/PolarMACE checkpoint
        │  scripts/export_symmetrix_model.py
        ▼
Symmetrix JSON  ──────────────► LAMMPS pair_style symmetrix/mace/kk
        │                        scripts/make_lammps_case.py
        │
        └──────────────────────► ASE calculator
                                 scripts/symmetrix_ase_calculator.py
```

## 1. Model export (checkpoint -> engine JSON)

```bash
python scripts/export_symmetrix_model.py \
  --checkpoint path/to/student.model \
  --output bench/student/student_symmetrix.json
```

- Uses the MACE `PolarMACE` calculator as the authoritative checkpoint loader.
- Serializes with the Symmetrix extractor and patches
  `GatedEquivariantBlock` activation constants (`silu` 1.6791767924,
  `sigmoid` 1.8467055342) that the generic extractor omits.
- Rejects node irreps with l>1, which the exact engine cannot run.
- Prints the local-electron channel width C (512 teacher, 192/128 students).

## 2. LAMMPS interface

```bash
python scripts/make_lammps_case.py \
  --structure frame.xyz \
  --model-json bench/student/student_symmetrix.json \
  --outdir bench/student/case_h2o \
  --species O H --steps 0 --dump
```

The generated `in.lammps` uses the production pair style:

```text
pair_style symmetrix/mace/kk charge 0.0 spin 1.0 streamed_edges generic
pair_coeff * * model.json O H
```

Run with the MatOS engine binary and the usual CUDA/Kokkos environment
(`RUN.txt` prints the command). `scripts/make_lammps_case.py` writes an
orthogonal-cell data file; for triclinic cells use ASE's `lammps-data`
writer or an orthogonal supercell.

## 3. ASE / Python interface

```python
from symmetrix_ase_calculator import SymmetrixPolarCalculator
calc = SymmetrixPolarCalculator(
    "bench/student/student_symmetrix.json",
    "bench/polar_uvu_gemm_engine_2026-10-21/bin/lmp_gemm")
atoms.calc = calc
e = atoms.get_potential_energy()
f = atoms.get_forces()
```

The calculator writes a one-point LAMMPS case under
`bench/interfaces_test/ase_runs/` and parses energy/forces. It is a
correctness/integration interface, not a fast MD path; use the native pair
style for production MD.

## 4. Model manifest (software contract)

`schemas/model_manifest.schema.json` defines the artifact a downstream tool
should accept:

- `model_type` (`PolarMACE`, `4G-HDNNP`, `LinearMACE`, ...), precision,
  species, cutoff, node irreps, SCF steps;
- checkpoint and exported JSON paths + sha256;
- engine binary, pair style, backend;
- held-out accuracy (`MAE_E_meV_per_atom`, `MAE_F_meV_per_A`, sample count);
- measured speed entries (`atoms`, `us_per_atom_step`, `task_id`).

Example: `docs/results/data/model_manifest_128L1.json`.

**No silent model swap.** A new architecture (Plan C 4G or Plan D linear)
must publish a new manifest with its own accuracy/speed and a distinct
`model_type`; the exact PolarMACE engine manifest stays unchanged.

## Compatibility matrix

| Model | Node irreps | Engine path | Status |
|---|---|---|---|
| 512 teacher | 512x0e+512x1o | exact polar, FP64 | 133.04 / 128.99 |
| 192L1 student | 192x0e+192x1o | exact polar, FP64 | 2.1 / 6.7; 116.84 / 121.16 |
| 128L1 student | 128x0e+128x1o | exact polar, FP64 | 3.8 / 6.1; 91.64 / 95.11 |
| 4G-HDNNP (Plan C) | local MLP + QEq | new lightweight path | prototype |
| Linear MACE (Plan D) | linear | linear FP32 path | 4.47–4.53 (paper parity) |

## Verification

- H2O single point: engine vs PyTorch MACE parity 1.1e-8 eV / 1.08e-8 eV/Å
  for the 128L1 student.
- LAMMPS case generator reproduces the same energy as the direct engine run.
- `scripts/check_repo.py` validates the manifest JSON and docs links.
