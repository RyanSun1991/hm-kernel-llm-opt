# Memory Accumulation

## Goal

Every non-trivial pipeline run should improve future runs.

## Long-Term Memory Layers

1. target memory
2. subsystem memory
3. global lessons
4. rejected or bad-plan memory

## Update Rules

- if the run discovers stable subsystem structure, update target or subsystem memory
- if the run rejects an idea pattern, append it to bad-plan memory
- if the run finds a reusable optimization heuristic, append it to global lessons
- if the run exposes a validation pitfall, append it to global lessons or validation notes

## Minimum Memory Outputs

- a target memory note under `.opencode/memory/targets/`
- optional subsystem memory note under `.opencode/memory/subsystems/`
- update `.opencode/memory/global_lessons.md` if the result generalizes

## Interaction with Iterative Close-Loop Mode

When `.opencode/skills/iterative-optimization.md` is loaded and `auto_iterate.enabled` is true, memory updates happen at the **end of each pass** (before the manager auto-starts the next pass).  The target memory file accumulates the mechanism list across iterations — it is the single most important input for the next pass's researcher dedup.

Prior-iteration plan files in `.opencode/plans/` are ALSO retained (never deleted).  Memory and plans serve complementary roles: memory records "what we learned"; per-iteration plans record "what we landed".  Both are inputs to the next iteration's researcher and plan reviewer.
