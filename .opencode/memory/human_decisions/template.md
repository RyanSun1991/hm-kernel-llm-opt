# Human Decision Log — <target>

Target: `<target>`
Target slug: `<target_slug>`
Owning primary agent(s): `<kernel-research | kernel-plan | kernel-function-research>`
First session: `<YYYY-MM-DD>`

Chronological log of every significant human review turn.  Append-only.  Each turn creates two blocks: one when the agent posts the review request (`Awaiting Review`), and one when the human replies (`Human Verdict`).

---

<!--
Example blocks:

## Turn 1 — Awaiting Review — 2026-04-24 15:28 UTC

**Agent**: kernel-research  |  **Target slug**: sysmgr_pwrmgr

### In scope
- .opencode/docs/sysmgr_pwrmgr_design.md

### Agent summary posted to human
Hot path is sched_tick_load_update → sched_ind_notify_load_change (80% of measured instructions). Design doc now covers subsystem boundary, entry points, 3 key structs, hot/cold split, concurrency model, and 4 instruction-count waste hotspots. Open questions: whether the refcount on struct load_change is really necessary under the current CPU-affinity invariant.

### Agent recommendation
Design is coherent; the open question around the refcount is the main thing worth a second pair of eyes.

---

## Turn 1 — Human Verdict — 2026-04-24 15:34 UTC

**Agent**: kernel-research  |  **Target slug**: sysmgr_pwrmgr

### Human verdict
- **Structured**: needs-more-research
- **Parse confidence**: high
- **Rationale points**:
  - refcount question is a real concern — trace back to commit that added it
  - also: verify the CPU-affinity invariant under hot-plug
  - > "don't trust the comment on line 142; that function was rewritten twice"
- **Scope additions**:
  - history review of struct load_change refcount
  - CPU hotplug interaction
- **Next action**: research iteration 2; append Research Iteration 2 — Questions / Findings to design doc

---

## Turn 2 — Awaiting Review — 2026-04-24 16:05 UTC
(...)

## Turn 2 — Human Verdict — 2026-04-24 16:12 UTC
- **Structured**: approve (design stable)
- **Next action**: suggest user open @kernel-plan to move to ideation

---

## Turn 3 — Awaiting Review — 2026-04-25 09:10 UTC  (different day, kernel-plan session)

**Agent**: kernel-plan  |  **Target slug**: sysmgr_pwrmgr

### In scope
- .opencode/docs/sysmgr_pwrmgr_design.md
- .opencode/docs/sysmgr_pwrmgr_ideas.md  (5 ranked)

### Agent summary posted to human
5 candidate mechanisms ranked; 2 dropped via dedup (one matches idea_ledger:L002 rejected last round, one matches state/bad_plans.md).  Top 3 remaining: (#1) hoist-invariant, (#3) drop-redundant-refcount, (#4) merge-two-atomic-reads.

### Human verdict
- **Per-idea**:
  - #1 hoist-invariant → approve
  - #3 drop-redundant-refcount → approve
  - #4 merge-two-atomic-reads → defer (reopen once #1 lands)
- **Next action**: write plan for #1 + #3 to .opencode/plans/sysmgr_pwrmgr_plan.md, post Turn 4 for plan approval
-->
