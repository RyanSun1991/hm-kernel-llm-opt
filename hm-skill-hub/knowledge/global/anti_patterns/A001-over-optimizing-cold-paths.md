# Anti-Patterns

Meta-level patterns to avoid (broader than per-mechanism bad_plans). Stable id prefix `A`.

---

### A001 — over-optimizing cold paths
- **lesson**: Agents gravitate toward easy-looking cold-path optimizations that
    do not move the primary metric. Burns iteration budget.
- **kind**: anti_pattern
- **applies_when**: candidate touches code with measured share < 1% of hot path
- **do_or_dont**: "don't: spend an iteration on changes whose code share is below
    the funnel's noise floor; require a hot-path delta thesis"
- **tags**: [instruction-count, planning, scope]
- **evidence**:
    - {kind: review, ref: reviews/wq_threadpool__iter1_plan_review.md}
- **confidence**: tentative
- **added_on**: 2026-04-20
- **added_by**: kernel-plan-reviewer
- **status**: active
