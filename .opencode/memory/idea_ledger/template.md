# Idea Ledger — <target>

Target slug: `<target_slug>`
First created: `<YYYY-MM-DD>`
Last updated: `<YYYY-MM-DD>`

All mechanisms verdicted by the human reviewer on this target live here.  Append-only; never delete rows.

## Schema

Every idea gets a stable `id` (e.g. `L001`, `L002`).  IDs are never reused even after rejection.  All fields below are required unless marked optional.

- **id**: `L<3-digit>`
- **mechanism**: one-line description (the *what*, not the target function)
- **scope**: files / functions / structs that would be touched
- **status**: `approved` | `landed` | `reverted` | `rejected` | `deferred`
- **verdicted_by**: `kernel-research` | `kernel-plan` | `os-opt-manager` | `kernel-function-research`
- **verdicted_at**: free-text label for which turn / gate / stage produced the verdict (e.g. `kernel-plan turn 3`, `pipeline Gate D`)
- **iteration**: pipeline iteration K if applicable, otherwise omit
- **approved_on** / **rejected_on** / **landed_on** / **deferred_on**: UTC timestamp of the verdict
- **rationale**: paraphrase of the human's reasoning — keep to ≤ 3 sentences; quote verbatim only when the wording is uniquely important
- **delta_pct** (landed only): aggregate instruction-count delta from the tester
- **compare_level** (landed only): total | process | thread | lib | function
- **validation_path** (landed only): `.opencode/bench/<artifact_slug>_validation.md`
- **reopen_trigger** (deferred only): the condition under which the idea becomes viable again (e.g. "when hot_path X drops out of top 5")
- **related_ids** (optional): other ledger rows this one is orthogonal to or supersedes

## Approved (pending implementation)

<!--
Example row:

### L001 hoist invariant lock check out of sched_tick_load_update
- **scope**: kernel/sched/sched_indicator.c :: sched_tick_load_update
- **status**: approved
- **verdicted_by**: kernel-plan
- **verdicted_at**: kernel-plan turn 2
- **approved_on**: 2026-04-24 15:30 UTC
- **rationale**: human confirmed the check is loop-invariant on the hot path; expected 3-5% reduction
-->

## Landed (pipeline run later confirmed a win)

<!--
Example:

### L001 hoist invariant lock check out of sched_tick_load_update
- **scope**: kernel/sched/sched_indicator.c :: sched_tick_load_update
- **status**: landed
- **verdicted_by**: os-opt-manager
- **verdicted_at**: pipeline decision stage
- **iteration**: 1
- **approved_on**: 2026-04-24 15:30 UTC
- **landed_on**: 2026-04-24 19:02 UTC
- **delta_pct**: -3.1
- **compare_level**: function
- **validation_path**: .opencode/bench/sysmgr_pwrmgr_validation.md
- **rationale**: clean A/B, no regression in adjacent paths
-->

## Rejected

<!--
Example:

### L002 inline hot kworker thread entry
- **scope**: kernel/workqueue.c :: process_one_work
- **status**: rejected
- **verdicted_by**: kernel-plan
- **verdicted_at**: kernel-plan turn 1
- **rejected_on**: 2026-04-24 15:32 UTC
- **rationale**: human reported prior i-cache blow-up on phone X; blanket "inline" is a bad pattern here
- **related_ids**: —
-->

## Deferred

<!--
Example:

### L003 batch eid writes in hyperhold iotab
- **scope**: drivers/hyperhold/iotab.c :: hyperhold_write_eid
- **status**: deferred
- **verdicted_by**: kernel-plan
- **verdicted_at**: kernel-plan turn 4
- **deferred_on**: 2026-04-26 09:10 UTC
- **rationale**: hot path is currently dominated by compression, not eid writes; revisit after compression work lands
- **reopen_trigger**: aggregate eid_write instruction share > 5% at function level
-->

## Superseded / Historical Notes

Free-prose section for cross-cutting notes the human wants preserved (e.g. "after a rebase onto new kernel — all ledger entries must re-check offsets").
