---
name: optimization-funnel
description: Ideation protocol for generating exactly five diverse-scope optimization ideas with mandatory dedup against bad-plans, memory stores, idea ledgers, and prior-iteration landed plans.
---

# Optimization Funnel

## Ideation Protocol

1. generate exactly five ideas
1a. **Scope diversity requirement.** Each of the 5 ideas MUST carry a `scope:` tag — one of:
    - `function` — change inside a single function body
    - `call-site` — caller-callee restructuring (hoist/inline/share pre-work across callers, push work into a different layer)
    - `data-flow` — change how data moves between functions (batch, coalesce, cache, eliminate round-trip)
    - `subsystem` — rework a contract or layer within the subsystem boundary (collapse indirection, remove dead policy)
    - `architectural` — data structure layout, lock granularity, lifetime/allocator, state-machine compression

    Among the 5 ideas, **at least 3 distinct scope tags** must appear. A funnel that emits 5 `function`-scope ideas is invalid and must be re-run after the researcher has done a structural pass over the target — see `research-discipline/SKILL.md` step 5 (Structural Audit).

    If after honest structural analysis only `function`-scope wins genuinely exist, the handoff packet MUST contain a `scope_justification:` block listing the call-site / data-flow / subsystem / architectural alternatives considered and why each is non-applicable (e.g. "lock split blocked by ownership semantics in `foo.c:123`", "dead-policy excision blocked by ABI stability"). Without this block the funnel result is rejected by the plan reviewer with reason `scope_justification_missing`.
2. drop repeated bad plans (see "Mandatory Dedup Sources" below — this step is not vibes, it is a file check)
3. rank first by likely instruction-count reduction on the hot path, then by risk and implementation cost
4. show only the top idea  *(default — pipeline sub-agents, auto flow)*  —  **when the caller is the `kernel-plan` primary agent** (or any agent that loads `human-interaction-memory/SKILL.md`), show ALL five in `<target>_ideas.md` with full per-idea analysis, since the human triages each idea independently
5. wait for explicit approval  *(auto flow: model approval on the top idea; primary-agent human flow: per-idea verdicts captured in the decision log + idea ledger)*
6. write the detailed plan only after approval  *(primary-agent human flow: only the approved ideas become plan candidates)*

## Mandatory Dedup Sources

Before emitting the five ideas, Read these files (use `ls` first if unsure which subsystem file exists):

1. **Global rejects** — `.opencode/state/bad_plans.md`.  Patterns here apply to every subsystem.
2. **Subsystem rejects** — `.opencode/state/<subsystem>-bad_plans.md` if it exists (e.g. `wq-threadpool-opt-bad_plans.md`, `hyperhold-io-opt_bad_plans.md`, `memmgr-reclaim-bad_plans.md`).  Check `ls .opencode/state/` for the current list.
3. **Target memory** — `.opencode/memory/targets/<target>.md` if the task names a concrete target (file, function, or subsystem slug).  Past runs record both what worked and what didn't.
4. **Subsystem memory** — `.opencode/memory/subsystems/<subsystem>.md` when present.  Broader context from past runs on the same area.
5. **Global lessons** — `.opencode/memory/global_lessons.md`.  Cross-subsystem heuristics and anti-patterns.
6. **Prior-iteration landed plans** — when `.opencode/state/current_task.json` → `auto_iterate.iteration_history` is non-empty, Read every `.opencode/plans/<prior_slug>_plan.md` listed there.  Those mechanisms are already LANDED in the tree; re-proposing them adds nothing.  See `.opencode/skills/iterative-optimization/SKILL.md`.
7. **Idea ledger** — if `.opencode/memory/idea_ledger/<target_slug>.md` exists for the target, Read it.  Every row with status `rejected` or `landed` is a dedup source on par with `bad_plans.md`.  `deferred` rows MAY be re-proposed only if the `reopen_trigger` has plausibly fired — the funnel handoff MUST state which trigger fired and why.  The ledger is populated by primary-agent human workflows (`kernel-plan`, optionally `kernel-research`) and updated by the pipeline's decision stage when a patch lands.  See `.opencode/skills/human-interaction-memory/SKILL.md` → "Dedup Feedback Loop".  Use `ls .opencode/memory/idea_ledger/` if unsure whether the file exists.

An idea is a "repeated bad plan" and MUST be dropped when any of the following is true:

- the mechanism (not just the wording) matches an entry in one of the files above marked `rejected` / `bad` / `failed`
- a prior attempt on the same target tried the same mechanism and resulted in a fail/inconclusive verdict
- the idea duplicates a prior-iteration LANDED mechanism (from `iteration_history`) — under iterative mode
- the idea matches a `rejected` or `landed` row in the idea ledger
- the idea violates a constraint recorded in target or subsystem memory (e.g. "this lock must be held across X", "this path is ABI-stable")

When you drop an idea for this reason, call it out in the handoff — "dropped: <idea>; matches <file>:<entry>; reason: <why it failed before or is already landed>" — so the reviewer can audit the dedup.

## Ranking Questions

### Per-idea instruction-count plausibility

- does the idea plausibly remove instructions from the measured or suspected hot path
- does it remove repeated work, branches, loads/stores, or redundant synchronization
- is the expected instruction-count gain likely to survive real build and runtime validation
- does it keep correctness, locking, lifetime, and logic boundaries defensible
- has this mechanism been tried before on this target? (if yes → is the new variant materially different, or is it the same idea in disguise?)

### Per-idea scope and leverage

- what scope tag does this idea carry (`function` / `call-site` / `data-flow` / `subsystem` / `architectural`) — see Ideation Protocol step 1a
- how many distinct call sites or paths benefit if this lands? (one hot path / a few callers / the whole subsystem)
- is the benefit one-time (this hot path only) or amortized across the subsystem (every future caller pays less)
- does this idea unblock future structural wins, or is it a terminal local-minimum patch
- if all 5 ideas converge to a single scope tag, the funnel handoff MUST carry a `scope_justification` block — otherwise the plan reviewer rejects with `scope_justification_missing`
