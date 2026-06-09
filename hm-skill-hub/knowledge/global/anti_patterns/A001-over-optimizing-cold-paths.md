---
id: A001
lesson: "Agents chase easy cold-path optimizations that don't move the primary metric — burns iteration budget."
kind: anti_pattern
applies_when: "candidate touches code with measured share < 1% of the hot path"
do_or_dont: "don't: spend an iteration on changes whose code share is below the funnel noise floor; require a hot-path delta thesis"
tags: [instruction-count, planning, scope]
evidence:
  - {kind: review, ref: reviews/wq_threadpool__iter1_plan_review.md}
confidence: tentative
added_on: 2026-04-20
added_by: kernel-plan-reviewer
status: active
---

# A001 — over-optimizing cold paths

Agents gravitate toward easy-looking cold-path optimizations that do not move
the primary metric. Burns iteration budget on changes whose code share sits
below the funnel's noise floor.

Require a hot-path delta thesis before spending an iteration: if the touched
code is < 1% of the measured hot path, the expected primary-metric delta is
within measurement noise and the change should be deferred or rejected.
