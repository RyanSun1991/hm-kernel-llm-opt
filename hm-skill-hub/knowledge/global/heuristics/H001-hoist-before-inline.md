---
id: H001
lesson: "When a hot path has both a loop-invariant computation and an inline candidate, hoist first — inlining first hides the invariant."
kind: heuristic
applies_when: "hot path contains nested loops with cross-call invariants"
do_or_dont: "do: hoist loop-invariant work to the outermost stable scope before considering inline, then re-measure"
tags: [instruction-count, hot-loop, ordering]
evidence:
  - {kind: bench, ref: bench/sysmgr_pwrmgr_validation.md}
  - {kind: bench, ref: bench/mm_reclaim__iter1_validation.md}
confidence: observed
added_on: 2026-04-26
added_by: os-opt-manager (run r_2026_0426_mm1)
status: active
subsumes: [F001]
---

# H001 — prefer hoisting before inlining on hot paths

When a hot path has both a loop-invariant computation and an inline candidate,
hoist the invariant first. Inlining first folds the callee into the loop body
and can obscure the invariant, preventing the hoist that would have been the
larger win.

Sequence: hoist loop-invariant work to the outermost stable scope, re-measure,
then evaluate inlining on the reduced body.
