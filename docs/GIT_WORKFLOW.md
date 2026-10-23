# Git workflow

This repository mixes code, reproducible benchmark evidence, and long-form
reports. The workflow keeps `master` stable, makes every speed claim traceable
to a task id, and keeps large artifacts out of Git.

## Branch model

| Branch | Purpose | Rules |
|---|---|---|
| `master` | stable, tagged milestones | only merges from `dev`; no direct feature work |
| `dev` | integration | all feature work merges here first |
| `plan/a`, `plan/b`, `plan/c`, `plan/d` | long-lived plan branches | one plan per branch; merge into `dev` at milestones |
| `feat/<topic>` | short-lived feature/fix | branch from `dev`; PR into `dev`; delete after merge |
| `results/<date>` | optional benchmark batch | evidence-only PR; merge into `dev` |

Feature branches are small and short-lived. A plan branch may live for weeks
but must be rebased/merged from `dev` before a milestone PR.

## Commit format

```text
type(scope): summary

Why: one or two sentences.
Evidence: task id / atoms / unit / parity numbers.
Risk: exactness or accuracy caveat.
```

Types: `feat`, `fix`, `perf`, `docs`, `results`, `test`, `build`, `chore`.
Scopes: `plan-a`, `plan-b`, `plan-c`, `plan-d`, `engine`, `data`, `docs`,
`q4`.

Examples:

```text
perf(plan-a): fuse AffineMLP input tape aliasing
results(plan-b): 128L1 91.64/95.11 us/atom/step at 1,296/5,184
docs(plan-c): 4G speed prototype and accuracy gate
```

## Evidence contract

- Public speed values are **µs/atom/step** only. Per-evaluation durations may
  appear in arithmetic when explicitly marked as durations; they are not
  speed claims.
- Every benchmark result records: task id (q4/local), atom count, queue/GPU,
  number of rounds/steps, and the exact binary/model sha256 when promoted.
- Raw benchmark artifacts stay under `bench/` (ignored). Committed summaries
  and small JSON live under `docs/results/data/`.
- Exactness: dE ≤ 1e-9 eV, max dF ≤ 1e-8 eV/Å. Student accuracy gates are
  stated per plan in `docs/design/FOUR_PLANS_2026-10-23.md`.
- Never commit model checkpoints, extxyz datasets, or q4 packages larger than
  a few MB; reference their workspace paths instead.

## Pull requests

1. Branch from `dev`: `git switch -c feat/<topic> dev`.
2. Run `scripts/git_prepush.sh`.
3. Open a PR into `dev`; fill `.github/PULL_REQUEST_TEMPLATE.md`.
4. Squash merge; delete the branch.
5. `dev` -> `master` only at a milestone, with a tag and release notes.

## Tags and releases

- `milestone-YYYYMMDD` — measured milestone (e.g. first <100 µs student).
- `paper-YYYYMMDD` — report/figure freeze.
- Annotated tags only: `git tag -a milestone-20261023 -m "..."`.
- Release notes summarize: speed, accuracy, evidence task ids, known limits.

## Pre-push checks

```bash
bash scripts/git_prepush.sh
```

It runs `git diff --check`, validates JSON under `docs/results/data`, checks
relative Markdown links, compiles the repository Python scripts, and warns on
public `ms/eval`/`ms/atom` speed text. CI runs the same checker on every push
and PR (`.github/workflows/docs-check.yml`).

## Remote and credentials

- Remote: `git@github.com:Yong-Q/symmetrix-polar.git`.
- SSH key: `/workspace/group/shared/.ssh/matos_github_ed25519` (fingerprint
  `SHA256:U2rY9RkMRJEYhss+1NZZNKt0eAc/2aklnlKA4MSYXDI`).
- Never force-push `master`; never commit secrets or `.volc` credentials.
- Push at least once per work session; benchmark/plan docs must be pushed
  together with the code or data summary they describe.
