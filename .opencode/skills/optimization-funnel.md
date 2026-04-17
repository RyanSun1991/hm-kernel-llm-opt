# Optimization Funnel

## Ideation Protocol

1. generate exactly five ideas
2. drop repeated bad plans (see "Mandatory Dedup Sources" below — this step is not vibes, it is a file check)
3. rank first by likely instruction-count reduction on the hot path, then by risk and implementation cost
4. show only the top idea
5. wait for explicit approval
6. write the detailed plan only after approval

## Mandatory Dedup Sources

Before emitting the five ideas, Read these files (use `ls` first if unsure which subsystem file exists — NEVER glob):

1. **Global rejects** — `.opencode/state/bad_plans.md`.  Patterns here apply to every subsystem.
2. **Subsystem rejects** — `.opencode/state/<subsystem>-bad_plans.md` if it exists (e.g. `wq-threadpool-opt-bad_plans.md`, `hyperhold-io-opt_bad_plans.md`, `memmgr-reclaim-bad_plans.md`).  Check `ls .opencode/state/` for the current list.
3. **Target memory** — `.opencode/memory/targets/<target>.md` if the task names a concrete target (file, function, or subsystem slug).  Past runs record both what worked and what didn't.
4. **Subsystem memory** — `.opencode/memory/subsystems/<subsystem>.md` when present.  Broader context from past runs on the same area.
5. **Global lessons** — `.opencode/memory/global_lessons.md`.  Cross-subsystem heuristics and anti-patterns.
6. **Prior-iteration landed plans** — when `.opencode/state/current_task.json` → `auto_iterate.iteration_history` is non-empty, Read every `.opencode/plans/<prior_slug>_plan.md` listed there.  Those mechanisms are already LANDED in the tree; re-proposing them adds nothing.  See `.opencode/skills/iterative-optimization.md`.

An idea is a "repeated bad plan" and MUST be dropped when any of the following is true:

- the mechanism (not just the wording) matches an entry in one of the files above marked `rejected` / `bad` / `failed`
- a prior attempt on the same target tried the same mechanism and resulted in a fail/inconclusive verdict
- the idea duplicates a prior-iteration LANDED mechanism (from `iteration_history`) — under iterative mode
- the idea violates a constraint recorded in target or subsystem memory (e.g. "this lock must be held across X", "this path is ABI-stable")

When you drop an idea for this reason, call it out in the handoff — "dropped: <idea>; matches <file>:<entry>; reason: <why it failed before or is already landed>" — so the reviewer can audit the dedup.

## Minimum Ranking Questions

- does the idea plausibly remove instructions from the measured or suspected hot path
- does it remove repeated work, branches, loads/stores, or redundant synchronization
- is the expected instruction-count gain likely to survive real build and runtime validation
- does it keep correctness, locking, lifetime, and logic boundaries defensible
- has this mechanism been tried before on this target? (if yes → is the new variant materially different, or is it the same idea in disguise?)
