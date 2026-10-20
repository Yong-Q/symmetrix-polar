# Plan B — separate audit of the 20–25 ms rewrite

**Status:** audit only. No engine change was made for this document.
**Date:** 2026-10-20
**Question:** under the existing FP64, exact-physics, full-gate contract, what
would it take to reach 20–25 ms/eval on 128 atoms (32 NH3 molecules), and is
that credible?

## 0. Executive answer

- **30–35 ms is credible** with a targeted rewrite: cache the forward SCF
  states instead of replaying them (−3.5 ms), remove the host/launch gap with a
  graph/persistent path (−3–4.5 ms), fix and enable the e3-linear merged
  unpack (−1.5–2.2 ms), fuse the reverse gather (−0.5–1.5 ms), and finish the
  small polar kernels (−1–2 ms). Expected **33–38 ms**.
- **25–30 ms is aggressive**: it additionally requires MH-1/UVU work removal
  and a fused SCF/k-space loop, with real gate and determinism risk.
- **20–25 ms is not credible under the current contract.** The round-2 static
  SM-weighted estimate for the *current work* is **22.93 ms**; that is already
  at/below the 25 ms target, and it assumes perfect scaling that the bandwidth
  and dependency structure do not deliver (UVU x2 proved this). Reaching 20 ms
  requires **removing work**, not only making kernels faster; that changes the
  scientific/numerical contract and must be re-gated.
- Recommendation: do **Plan A** (2–4 ms structural slices) first; then a
  **Plan B-lite** (replay cache + merged unpack + gather fusion + graph) to
  target **36–40 ms**. Do not promise 20–25 ms.

## 1. Current budget (build 5/8 trace, 128 atoms)

| Item | ms | Note |
|---|---:|---|
| Wall (build 8) | **48.74** | measured, full gates PASS |
| GPU busy | **43.91** | CUPTI kernel sum |
| Host/launch gap | **4.8–5.9** | wall − GPU busy |
| Launches | **1228** | 916 Kokkos + 312 cuBLAS/external |
| Static SM-weighted (current work) | **22.93** | estimate, not a hard floor |

Family roll-up (r3c trace):

| Family | ms | launches |
|---|---:|---:|
| polar pipeline | 19.13 | 272 |
| tensor product (MH-1) | 8.43 | 12 |
| backbone nonlinear | 6.38 | 44 |
| cuBLAS/external | 3.80 | 312 |
| e3nn linear | 2.69 | 488 |
| MLP/dense | 1.85 | 65 |
| other | 1.60 | 19 |

Replay alone is **3.51 ms / 46 launches** (forward SCF states are recomputed
in reverse).

## 2. Candidate B changes

| # | Change | Expected saving | Effort | Risk | Gate/determinism |
|---|---|---:|---|---|---|
| B1 | **Cache forward SCF states for reverse** (remove replay) | **3.5–3.7 ms** | M | Low–M | bitwise if exact copies; memory +10–30 MB |
| B2 | **CUDA Graph / persistent forward path** (remove host gap) | **3–4.5 ms** | L | High | requires removing host syncs; must reproduce bitwise |
| B3 | **Fix + enable e3-linear merged unpack** | **1.5–2.2 ms** | M | Medium | reordered FP64 sums; full gates |
| B4 | **Fuse reverse gather into MH-1 reverse** | **0.5–1.5 ms** | M–H | Medium | bitwise if same read values |
| B5 | Finish small polar kernels (readout/dense/LN) | **1–2 ms** | M | Low | deterministic; near-bitwise |
| B6 | UVU x2 shared-memory `mixed_bar` (already built) | 0.3–0.5 ms | S | Low | **bitwise confirmed** |
| B7 | MH-1 work removal (skip redundant entry/plan loads) | 2–4 ms | H | High | may reorder sums; gates |
| B8 | Tensor cores beyond current DMMA | 0–2 ms | H | High | FP64 DMMA already used for medium GEMMs; MH-1 contractions are too small |
| B9 | Fuse SCF steps / k-space passes into persistent kernels | 1–3 ms | H | High | changes launch/reduction order; gates |

## 3. Scenarios

| Scenario | Changes | Expected wall | Comment |
|---|---|---:|---|
| B-lite | B1+B3+B4+B5+B6 | **40–43 ms** | realistic, mostly work removal |
| B-core | B-lite + B2 | **36–39 ms** | graph must be proven bitwise |
| B-full | B-core + B7+B9 | **31–35 ms** | high effort/risk |
| 20–25 ms | B-full + B8 + further work removal | **not credible** | static floor 22.93 ms for current work; would need to remove physics/work |

## 4. Host-sync inventory (the CUDA Graph blocker)

The reverse pass contains host↔device scalar round trips and host-side
decisions, e.g.:

- `polar_global_sum_host` calls in the polar pipeline and reverse;
- `create_mirror_view_and_copy` of `dbar`/sources and host-computed Fukui
  coefficients (`mace_polar_impl.tpp` ~2036–2054);
- SCF/update-block host loops (`polar_recursion_steps = 2`);
- workspace `ensure_view`/`Kokkos::realloc` which must be pre-sized and
  frozen before capture.

A graph is only credible for the **forward/backbone** region first; the reverse
needs the host scalar path moved to device or a fixed-shape two-pass design.

## 5. Work-removal inventory (the only route below 30 ms)

1. **Replay caching (B1)** — the forward already computes the two SCF states;
   store them and feed the reverse instead of replaying. 3.5 ms, memory cost.
2. **SCF loop fusion (B9)** — the two SCF steps are fixed; fuse their
   per-step kernels to avoid intermediate global writes/launches.
3. **k-space reuse** — geometry/cos/sin/basis are already cached; the density
   changes per SCF step, so the density-dependent sums cannot be cached. Any
   “reuse” must be a mathematical reformulation, not a free win.
4. **MH-1 entry/plan loads (B7)** — the kernels are instruction/store-bound;
   pre-scale Wigner values by `path_weight`, elide the identity-mask multiply,
   and hoist the plan/indices per channel block. This removes instructions,
   not FLOPs.
5. **Tensor cores (B8)** — the e3-linear medium GEMMs already use FP64 DMMA.
   The MH-1 per-output contractions have K≤21, C≤7, M≤512 and arithmetic
   intensity ≈0.8–1.8 FLOP/B, 5–12× below the DMMA ridge; tensor cores would
   idle. Do not budget them as a general 2×.

## 6. Effort and risk estimate

| Scope | Engineering | Validation | Probability of landing |
|---|---:|---|---|
| B-lite | 3–5 days | 1–2 days | high (~80%) |
| B-core (+graph) | +1–2 weeks | +3–5 days | medium (~50%) |
| B-full | +3–5 weeks | +1–2 weeks | low–medium (~25%) |
| 20–25 ms | research project | full re-gate | low (<10%) |

## 7. Recommendation and kill criteria

**Recommendation:** implement **Plan A** first (e3 merged unpack fix, gather
fusion, launch reduction). Then, if the target remains <40 ms, do **B-lite**
(B1 replay cache + B3 + B4 + B5). Only attempt B2 (graph) after the host-sync
inventory is empty and the forward path is deterministic. Treat 20–25 ms as a
research question, not an engineering target.

**Kill criteria for B:**

- If replay caching cannot be made bitwise (or within the force gate) → drop.
- If the forward graph changes any gate output or cannot be captured because
  of a remaining host sync → keep the non-graph path and accept ~40 ms.
- If MH-1 work removal changes forces beyond the gate → revert; do not trade
  accuracy for the 30 ms number.
- If the measured static floor with the reduced work is still >28 ms → stop;
  report the floor rather than a synthetic number.

## 8. Bottom line

- **A** (structural slices): realistic **40–44 ms**.
- **B-lite**: **40–43 ms**; **B-core**: **36–39 ms**; **B-full**: **31–35 ms**.
- **20–25 ms**: requires work removal below the current static floor and a
  model/algorithm change; not achievable under the current FP64, exact-physics,
  full-gate contract with kernel engineering alone.
