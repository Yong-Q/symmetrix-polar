# Parallel task board — 2026-10-23

Protocol: `skills/parallel-task-executor` (pulled from
https://hub.cocoloop.cn/skills/1559). P0 urgent, P1 high, P2 normal, P3
optional. A task is only `completed` with its own artifact/log evidence.
Concurrency is bounded by real resources: 4 remote A100 admission slots, 1
local A100, and CPU for code work.

## Resource allocation

| Resource | Count | Owner | Status |
|---|---:|---|---|
| q4 A100 (remote) | 4 / 4 admission cap | Plan B bulk sampling (pilot + shards A/B/C) | running |
| Local A100 | 1 | Plan C prototype + shared test slot | running / intermittent |
| CPU agents | 4 | Plans A, B, C, D owners | running |

The Volc queue itself reports 15 free GPUs, but MatOS per-user admission caps
active veMLP tasks at **4 GPUs**. `devinstance-queue3` has 2 free but is not
enabled for submission. So the usable ceiling is 4 remote + 1 local = 5 cards.
Keep one slot (the local card, then the freed pilot slot) available for
testing rather than filling every card permanently.

## Board

| ID | P | Task | Owner | Depends on | Status | Evidence |
|---|---|---|---|---|---|---|
| A1 | P0 | AffineMLP bias-epilogue fusion | Plan A agent | — | running | engine commit + parity/speed report |
| A2 | P0 | AffineMLP tape-copy aliasing | Plan A agent | A1 | queued (same agent) | parity + speed |
| A3 | P1 | Fresh CUPTI trace of current default | Plan A | A1/A2 | pending | trace summary |
| B1 | P0 | 43k-frame bulk campaign (pilot + A/B/C shards) | q4 tasks | — | running | `bench/path_b_bulk*/.../data/*.xyz` |
| B2 | P0 | Bulk dataset merge + state-level split + ACF | Plan B agent | pilot data | running | `data/split_manifest.json` |
| B3 | P0 | 128L1 bulk fine-tune | Plan B | B1, B2 | queued | run dir + eval report |
| B4 | P1 | Stratified eval (species x density x T) | Plan B agent | B2 | running | `scripts/eval_student_bulk.py` |
| C1 | P1 | 4G-HDNNP speed prototype + training design | Plan C agent | — | running | `bench/plan_c_4g_2026-10-23/` |
| C2 | P2 | 4G bulk fine-tune smoke | Plan C agent | pilot data | running | train log |
| D1 | P1 | Linear MACE distillation + long-range correction | Plan D agent | — | running | `bench/plan_d_linear_2026-10-23/` |
| W1 | P2 | Sampling v2: batch 8-16 boxes per forward | unassigned | — | pending | throughput benchmark |
| G1 | P2 | Git workflow: pre-push + CI + interfaces | main | — | done | `docs/GIT_WORKFLOW.md`, `.github/` |
| G2 | P3 | README / FOUR_PLANS reconciliation | main | A1/B3 | pending | docs diff |

## Notes

- Shards A/B/C print only at species completion; their first H2O/CO2/NH3/C2H4
  files have not appeared yet, so ETA is unknown until the first file lands.
  The pilot has 4/6 species done (H2O, NH3, C2H4, CH4) and is on C2H6.
- The local A100 is shared with an intermittent external process; Plan A
  benchmarks must check `nvidia-smi` and report variance.
- Plan D's latest linear fit is not usable (energy MAE ~2.3 eV/atom); the
  current bottleneck is model form (long-range electrostatics), not GPU.
