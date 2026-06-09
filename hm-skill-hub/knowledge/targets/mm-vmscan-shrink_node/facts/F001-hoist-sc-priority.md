---
id: F001
type: pattern
title: "Hoist the repeated sc->priority read out of shrink_node's per-page loop"
scope:
  level: function
  subsystem: mm-reclaim
  target_slug: mm-vmscan-shrink_node
applies_when: "shrink_node re-reads sc->priority on every page iteration while priority is invariant within the scan"
source:
  - {kind: bench, ref: bench/mm_reclaim__iter1_validation.md}
  - {kind: commit, ref: "mm/vmscan.c@r_2026_0426_mm1"}
evidence:
  delta_pct: -0.8
  compare_level: function
  confirmations: 2
maturity: L2
status: active
subsumed_by: [H001]
invalidation: "rebase that re-numbers struct scan_control fields, or a refactor that makes priority mutate inside the scan"
contributor: os-opt-manager
created_at: 2026-04-26T10:00:00Z
---

# F001 — hoist sc->priority read out of the shrink_node reclaim loop

In `shrink_node`, the per-page reclaim loop re-reads `sc->priority` every
iteration even though it is invariant for the duration of the scan. Hoisting
the read into a local above the loop removes one redundant load per page.

Measured function-level instruction count fell 0.8% on the paired A/B in
`bench/mm_reclaim__iter1_validation.md`. This is a concrete instance of the
general "hoist loop-invariant reads" heuristic (H001), which subsumes it.
