---
id: L001
target_slug: mm-vmscan-c-shrink-node
mechanism: hoist-invariant
scope: "mm/vmscan.c :: shrink_node"
status: landed
verdicted_by: os-opt-manager
verdicted_at: "pipeline Gate D, run r_2026_0426_mm1"
iteration: 1
landed_on: 2026-04-26T11:00:00Z
delta_pct: -0.8
compare_level: function
validation_path: "local/runs/r_2026_0426_mm1/bench/mm_reclaim__iter1_validation.md"
rationale: "Hoisting the loop-invariant sc->priority read out of the per-page scan removed one redundant load per iteration; function-level instruction count down 0.8%, confirmed by a paired A/B."
related_ids: []
---

# L001 — hoist sc->priority invariant out of shrink_node

Verdict ledger entry for the `hoist-invariant` mechanism applied to
`mm/vmscan.c :: shrink_node`. Landed in iteration 1 of run
`r_2026_0426_mm1` with a measured -0.8% function-level instruction-count delta.

Cross-reference: the curated fact is `F001` (same target); the generalizing
heuristic is `H001`.
