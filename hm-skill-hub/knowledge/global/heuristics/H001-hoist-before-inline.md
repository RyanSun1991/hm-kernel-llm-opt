# Heuristics

Reusable rules of thumb that tend to pay off. Stable id prefix `H`.

---

### H001 — prefer hoisting before inlining on hot paths
- **lesson**: When a hot path has both a loop-invariant computation and an inline
    candidate, hoist first; inlining first hides the invariant and prevents it.
- **kind**: heuristic
- **applies_when**: hot path contains nested loops with cross-call invariants
- **do_or_dont**: "do: hoist loop-invariant work to the outermost stable scope before
    considering inline; re-measure"
- **tags**: [instruction-count, hot-loop, ordering]
- **evidence**:
    - {kind: bench, ref: bench/sysmgr_pwrmgr_validation.md}
    - {kind: bench, ref: bench/mm_reclaim__iter1_validation.md}
- **confidence**: observed
- **added_on**: 2026-04-26
- **added_by**: os-opt-manager (run r_2026_0426_mm1)
- **status**: active
