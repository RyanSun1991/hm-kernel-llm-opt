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

## Interaction with Primary-Agent Human Workflows

When a primary agent that loads `.opencode/skills/human-interaction-memory.md` (e.g. `kernel-research`, `kernel-plan`) runs, memory writes happen at **two** points:

1. **Live, per human turn** — the decision log at `.opencode/memory/human_decisions/<target_slug>.md` and the idea ledger at `.opencode/memory/idea_ledger/<target_slug>.md` get appended to / updated on every human verdict, before the agent ends its turn.  This is what makes sessions resumable after compaction.  See `.opencode/skills/human-interaction-memory.md` for the full protocol.
2. **End of session, as usual** — when the human-agent dialogue concludes, the primary agent still runs the normal promotion step above: distill stable findings into target memory, promote reusable heuristics into `global_lessons.md`, append newly-rejected mechanism patterns to `bad_plans.md` if they generalize.

The idea ledger carries the raw per-idea record (dedup dataset); the target / subsystem / global memory carries the boiled-down insight (reuse dataset).  The two are complementary — ledger prevents re-proposing a specific mechanism; target memory tells a future researcher "this whole direction is usually dead on this subsystem."

If the target later enters the full `os-opt-manager` pipeline and a patch lands, the pipeline's decision stage can update the idea ledger row's status from `approved` to `landed` with `delta_pct` and `validation_path`, closing the loop between primary-agent ideation and pipeline verification.
