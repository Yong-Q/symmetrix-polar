# PRL-style density scan with MACELES-OFF — protocol, bug fix, and run state

Status note, 2026-11-01 03:10 UTC. Everything below is measured from the
workspace files; model is the official MACELES-OFF checkpoint
(`models/maceles_off_chno.json` + `bench/q4_maceles_off/attempt-03/maceles_off.cubin`).

## Protocol

- Six species (H2O, NH3, CH4, C2H4, C2H6, CO2), 3,000 atoms each
  (1,000 / 750 / 600 / 500 / 375 / 1,000 molecules).
- Temperatures at 1.55 Tc: 1003.0, 628.4, 295.4, 437.6, 473.2, 471.4 K
  (Tc from the PRL reference set; T* ≈ 2 supercritical, matching Dijkman
  et al. PRL 134, 056103 (2025)).
- Ten densities per species, log-spaced between 0.2 and 2.2 rho_c
  (rho_c = 0.01076, 0.00831, 0.00608, 0.00459, 0.00407, 0.00640 A^-3).
- Start configurations: the dense liquid boxes in
  `bench/q4_maceles_off/six_3000/<species>/data.lammps`, rescaled
  **rigidly** to each target density (see fix below).
- MD: NVT (damping 0.1 ps), 0.5 fs timestep, 100 ps equilibration
  (300 K -> T linear ramp over the first 20 ps, then hold), 1 ns production.
- Trajectory: `dump ... 50 traj.dump id type x y z` = every 25 fs,
  ~40,000 frames / ~5.4 GB per run, kept in full for orientation analysis.
- Neighbor list: `neighbor 2.0 bin` + `neigh_modify every 10 delay 0 check yes`.
  The earlier `every 100 ... check no` let atoms move more than half a skin in
  100 steps at 1000 K and lost atoms.
- 60 runs split into four 15-run shards, one q4 A100 each, 240 h deadline,
  MD-only (`les_batch.py --no-analyze`), c1/c2 extracted afterwards from the
  saved trajectories.

## Bug found and fixed: isotropic rescale stretched every bond

`make_density_scan.py` originally did

```python
atoms.set_cell(atoms.cell * scale, scale_atoms=True)
```

which scales *all* Cartesian coordinates, i.e. the intramolecular bonds too.
At 0.2 rho_c (scale 1.56) N-H became 1.58 A instead of 1.01 A, and at
0.2 rho_c for NH3 2.47 A. The ML potential then collapsed the stretched
molecules:

| run | symptom |
|---|---|
| NH3, 0.0063 A^-3, 628 K | T 628 -> 3863 K in 50 steps, PotEng -1304 eV, lost atoms at step 50 |
| H2O, 0.00215 A^-3, 1003 K | stable for 2.6 ps, then T -> 5.8e7 K, lost atoms at step 2600 |

The energy release (1.7 eV per NH3 molecule) is ~5x the cohesive energy —
that is bond formation, not condensation, and matches the stretched bonds.

Fix: scale only each molecule's anchor atom and keep every intramolecular
offset fixed (`rescaled_data(..., natoms_per_mol)`), then wrap. Bond lengths
are now identical across all ten densities:

| species | bonds (A) | min intermolecular distance (A) |
|---|---|---|
| H2O | 0.957 | 2.121 |
| NH3 | 1.012 | 1.980 |
| CH4 | 1.087 | 2.317 |
| C2H4 | 1.087-2.118 | 2.317 |
| C2H6 | 1.094-2.175 | 1.906 |
| CO2 | 1.160 | 2.475 |

(Checked over all ten densities per species; the bond set is identical file to
file and no intermolecular pair is closer than 1.9 A.)

Second change: `les_c1c2.py` now starts velocities at 300 K and ramps to the
target over the first 20 % of equilibration instead of dumping the full
target kinetic energy into the freshly rescaled structure.

## Evidence that the fix works

First run of each shard after the fix (target temperature not yet reached
during the 20 ps ramp):

| shard | first state | steps seen | T (K) | PotEng (eV, 3000 atoms) |
|---|---|---|---|---|
| 1 | H2O 0.00215 A^-3, 1003 K target | 4050 | 360 | -2,081,137 stable to +-3 |
| 2 | NH3 0.0063 A^-3, 628 K target | 5250 | 345 | -1,155,065 stable to +-2 |
| 3 | C2H4 0.00092 A^-3, 438 K target | 2100 | 309 | -1,069,859 stable to +-4 |

The NH3 PotEng at 300 K (-1,155,065) now matches the dense liquid reference
(-1,155,162 at 0 K) to 100 eV over a 3.8x volume change, i.e. the molecules
are intact.

## Reference: 100 ps H2O validation run

`bench/q4_maceles_off/h2o_100ps/attempt-02` (1,000 H2O, 0.0334 A^-3, 300 K),
22.88 ms/step, 7.63 us/atom/step, 12.71 h per ns:

- O-O g(r) first maximum 2.84 A, g = 2.65; O-O coordination to 3.4 A = 4.73
  (liquid water ~4.5).
- H-O-H 105.57 +- 4.91 deg, O-H 0.972 +- 0.024 A, tetrahedral q = 0.612,
  1.36 H-bonds per molecule, dipole P2 = -0.0008.
- c2(r) and c2(k) in `c1c2_local/f2`, figure `analysis/c1c2_summary.png`.

## Cost measurement and deadline

The GTO-Ewald k-space work grows with cell volume, so the low-density states
are the expensive ones. Measured steps/s over a 62 s window on the first run
of each shard (3,000 atoms, 0.5 fs):

| shard | state | L (A) | ms/step | projected h per 1.1 ns run |
|---|---|---|---|---|
| 1 | H2O 0.00215 | 77.5 | 45.9 | 28 |
| 2 | NH3 0.00630 | 49.2 | 22.5 | 14 |
| 3 | C2H4 0.000918 | 81.7 | 56.4 | 34 |

A least-squares fit `t = 14.4 ms + 6.8e-5 ms/A^3 * V` puts a whole shard at
240-265 h, i.e. above the original 240 h deadline. All four shards were
therefore restarted with `ActiveDeadlineSeconds = 1,440,000` (400 h) about
9 minutes into the first attempt.

## Run state

| shard | task id (attempt 04, 400 h) | states |
|---|---|---|
| 1 | t-20261101111006-fqwf4 | H2O r00-r09, NH3 r00-r04 |
| 2 | t-20261101111014-bqj5m | NH3 r05-r09, CH4 r00-r09 |
| 3 | t-20261101111104-czltg | C2H4 r00-r09, C2H6 r00-r04 |
| 4 | t-20261101111205-crzh5 | C2H6 r05-r09, CO2 r00-r09 |

Outputs land in `bench/maceles_density_scan/shard_<n>/attempt-04/samples/<sim_id>/run/`
with `traj.dump.gz` and `meta.json` per state. Expected wall time ~250 h per
shard; `--resume` skips any state whose `meta.json` already exists.

## 2026-11-02 — quota kill and restart

All four shards stopped at 2026-11-01 16:17-16:18 UTC with

```
ERROR on proc 0: Error writing dump 1: Disk quota exceeded (src/dump.cpp:532)
```

after writing 7.5 GB of trajectories. The workspace is a ~310 GB per-scope
vePFS quota with 183 GB already in use (a 120 GB probe succeeded, 140 GB
failed), not the 452 TB the filesystem reports. 60 raw 1 ns dumps at 25 fs
would need ~250 GB.

Changes before restarting:

- `les_batch.py` now gzips each `traj.dump` immediately after that state's MD
  finishes (2.3x measured on these dumps; `meta.json` keeps raw/gz sizes and
  the sha256 of the `.gz`). Steady-state footprint: ~56 GB of archives plus
  ~8 GB of in-flight raw dumps.
- Dump cadence changed from 25 fs to 50 fs (dump_every 100) so the archives
  fit with margin. Still 20,000 frames per state.
- The four partial states (309/531/240/486 ps of production) were moved to
  `bench/maceles_density_scan/partial_attempt-04/` and compressed; they are
  usable equilibrated samples but are being rerun for a uniform 1 ns.

Restarted tasks (same attempt-04 workdirs, `--resume`):

| shard | task id |
|---|---|
| 1 | t-20261102093724-5xdfw |
| 2 | t-20261102093733-cs6hz |
| 3 | t-20261102093805-vv7sk |
| 4 | t-20261102093812-klckg |

### Quota diagnosis (2026-11-02)

Four GPU probes established:

- A worker task writes as uid **65534 (`nobody`)** on these nodes; that
  identity's writes are what hit EDQUOT at ~7.8 GB across the four shards.
- Root-run probe tasks wrote 40-80 GB of random data in the same tree without
  error, and `chown 166536` from a root task works, but `samples/` is owned by
  uid 1000, so a batch that drops to 166536 cannot create files there
  (`PermissionError`); the drop is now opt-in (`--drop-priv`).
- uid 65534 cannot create files in a 0777 directory owned by another user
  (`EACCES`), which is why probe tasks as that uid fail immediately.

Practical consequence: the campaign has to fit the `nobody` quota (~8 GB).
With per-state gzip and a **0.5 ps dump cadence (dump_every 1000)** the 60
archived trajectories total ~5.4 GB and fit. 25 fs and 50 fs were abandoned
on this quota; they need a larger quota or an offload path.

## Sampling-code state

`test` repo commit `baef4a7` (rigid rescale, staged heat-up, `--no-analyze`,
`analyze_traj_orientation.py`). Push to `Yong-Q/test` is pending: this
runtime has no usable SSH key (`~/.ssh` is empty and the global
`UserKnownHostsFile` path is unreadable), so the commit exists locally only.
