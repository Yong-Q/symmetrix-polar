# Measured data behind the optimization ledger

These JSON files are the compact, committed summaries of the A/B, sweep, gate
and CUPTI measurements used in
[`../OPTIMIZATION_LEDGER.md`](../OPTIMIZATION_LEDGER.md) and the report. Raw
logs, dumps and traces stay under the gitignored `bench/` tree because of
size; every value below was copied from that raw output without changing the
numbers.

## A/B files

| File | Experiment | Validity note |
|---|---|---|
| `ab_gather_tile_1296.json` | message-gather tile {1,64,128,256} vs old default | clean 2 rounds |
| `ab_edge_features_1296.json` | edge-feature tile vs old | clean 2 rounds |
| `ab_path_grouped_forward_1296.json` | path-grouped forward vs previous main | baseline rounds 0 and 2 were externally contended (699/999 ms); round 1 is clean (320.5 ms). Candidate is stable 298.4–299.3 ms across 3 rounds. Use candidate vs the clean baseline; the sandbox file below is a clean 2-round cross-check. |
| `ab_path_grouped_forward_sandbox.json` | same forward change vs main, sandbox | clean 2 rounds; full 247.75 → 230.10 µs/atom, polar-off 151.98 → 135.24 µs/atom |
| `ab_edge_up_scatter_1296.json` | edge-up/scatter tile + cutoff copy fusion | baseline rounds 0 and 1 were externally contended (749/799 ms); round 2 is clean (297.9 ms). Candidate is stable 290.0–290.8 ms. |
| `ab_team_vec4_1296.json` | team channel vec4 vs vec1 | clean 3 rounds; negative |
| `sweep_team_size_1296.json` | MH-1 team size 32/64/128/256 | clean 2 rounds |
| `sweep_msg_vec_cutoff_1296.json` | msg_vec=1, scalar/chunked cutoff | clean 2 rounds; all negative |
| `sweep_team_size_polar_off_1296.json` | polar-off backbone team size | clean 2 rounds |

## Traces and gates

| File | Content |
|---|---|
| `cupti_b4team_1296_summary.json` | CUPTI family/label summary, full polar 1,296 atoms, pre-opt3 |
| `kernel_labels_1296.json` | Extracted per-launch/per-eval kernel values used in the report |
| `profile_path_grouped_forward.json` | Stage profile for the path-grouped forward prototype |
| `gate_1rank_main_b4team.json` | 1-rank LAMMPS parity (dE, max dF, pass) |
| `gate_2rank_wtfix.json` | 2-rank exact 2-hop parity |
| `gate_md_10step_128.json` | 10-step NVE MD comparison against the original engine |
| `parity_path_grouped_forward_128.json` | Single-point parity for the path-grouped build |
| `scf1_probe.json` | 1-step vs 2-step SCF probe (negative) |

## Base environment

All timing rows use: 1,296-atom H₂O, full PolarMACE, FP64, one A100-80GB,
`MATOS_REPLAY_CACHE=1`, `MATOS_UVU_X2_MB_SMEM=0`, `MATOS_UVU_FWD_SWEEP=0`,
`MATOS_CONV_CUTOFF_BACKEND=team`, `MATOS_MH1_TEAM_SIZE=64`,
`MATOS_MH1_TEAM_CHANNEL_VEC=1`, `flock` on the shared GPU lock. The input is
`bench/paper_scale_2026-10-20/in.polar_water_1296`.

The baseline rounds marked "contended" above show 2× or more than the clean
round because another process used the GPU without the lock; they are kept in
the JSON for honesty but the reported medians use the stable candidate rounds
and the clean baseline.
