# Human Interaction Memory

This skill defines how **primary agents that iterate with a human expert** persist that dialogue to durable memory.  It is the complement to `memory-accumulation.md`: the latter records stable structural/heuristic findings at the end of a non-trivial run, while this skill records **per-turn human decisions in real time** so sessions survive compaction and new sessions can resume from disk.

Primary agents that load this skill:

- `kernel-research` — iteratively refines `<target>_design.md` with human feedback
- `kernel-plan` — funnels optimization ideas past a human expert, records per-idea verdicts, produces `<target>_plan.md`
- `kernel-function-research` — existing per-function deep-dive; may optionally adopt the same stores

Any sub-agent under the full `os-opt-manager` pipeline does NOT use this skill — the pipeline has its own stage-gate handoffs.  This skill is specifically for primary agents that own a human-facing iterative loop.

## Three Persistent Stores

All paths are relative to the project root; always resolve with `git rev-parse --show-toplevel` before writing.

### 1. Decision Log — chronological, per target

`.opencode/memory/human_decisions/<target_slug>.md`

- Append-only narrative log of every significant human-agent exchange on this target.
- Captures paraphrased rationale + verbatim quotes worth preserving + the artifact(s) that were in play.
- Survives compaction; replayable into a new session.
- Template: `.opencode/memory/human_decisions/template.md`.

### 2. Idea Ledger — per-target set of mechanism verdicts

`.opencode/memory/idea_ledger/<target_slug>.md`

- Structured registry of every optimization mechanism the human has verdicted on this target.
- Statuses: `approved` | `landed` | `reverted` | `rejected` | `deferred`.
- Feeds the dedup step in `optimization-funnel.md` — ideas matching a rejected entry MUST be dropped on the next round with a citation.
- Survives across sessions and (if a pipeline run later lands a patch) across pipeline passes.
- Template: `.opencode/memory/idea_ledger/template.md`.

### 3. Design Doc Amendments — inline, in the same design file

`.opencode/docs/<target>_design.md`

- When the human says "dig deeper into X" / "also consider Y" during research, the agent appends a new section `## Research Iteration <N> — Questions` (the human's ask) and `## Research Iteration <N> — Findings` (the agent's update after re-investigating).
- Do NOT overwrite prior sections.  Do NOT create `_design_v2.md`.  The whole chronology must be readable in one Read so a fresh session can reconstruct context.

## Target Slug Convention

The `<target_slug>` used for `human_decisions/` and `idea_ledger/` files is the **base** target slug — the same one used by the pipeline's `current_task.json` → `base_slug`, shared across pipeline iterations if the target later enters a full pipeline run:

- target `sysmgr/pwrmgr` → slug `sysmgr_pwrmgr` → `.opencode/memory/human_decisions/sysmgr_pwrmgr.md` and `.opencode/memory/idea_ledger/sysmgr_pwrmgr.md`
- target `__pm_idle_enter` (a function) → slug `__pm_idle_enter`
- normalization: replace `/` with `_`, strip leading/trailing separators

## When To Write

### Before asking the human for input

When the agent has produced an artifact (updated design doc, new ideas list, plan draft) and is about to ask the human to review it, it MUST:

1. Save the artifact to disk.
2. Append a `## Turn <N> — Awaiting Review — <UTC timestamp>` block to the decision log with:
   - artifact paths in scope
   - the one-paragraph summary the agent is about to show the human
   - the agent's own short recommendation (so the human sees both sides)
   - status marker `pending-human-review`
3. Post the review request to the human.  End the turn.

This ordering is critical: the append happens **before** the review request is posted so that if the session is killed between the post and the reply, the pending state is already on disk.

### After parsing a human reply

Before doing anything else — before updating artifacts, before proposing a new idea — the agent MUST:

1. Append a `## Turn <N> — Human Verdict — <UTC timestamp>` block to the decision log containing:
   - structured verdict (e.g. `approve` / `revise` / `reject` / `needs-more-research` / `defer` / `accept`)
   - key rationale points (paraphrased + any verbatim quotes worth preserving, marked with `>`)
   - any scope additions the human asked for (bulleted)
   - the agent's parse confidence (`high` / `medium` / `low`) — if `low`, a clarification question was asked and no advance happened
2. Update the idea ledger (only for verdicts that touch a specific mechanism):
   - per-idea verdicts in `kernel-plan` → add new rows under Approved / Rejected / Deferred
   - `reject` on an already-approved idea → move row from Approved to Rejected with reason
   - `accept` after a pipeline later lands the patch → set idea status to `landed`, fill in delta_pct and validation path (this write can come from the pipeline's decision stage too)
3. Then act on the verdict (update design doc, regenerate ideas, write plan, etc.).

### When iterating on "needs more research" (for `kernel-research`)

1. Read the latest `## Turn <N> — Human Verdict` block in the decision log to pick up the human's questions verbatim.
2. Re-investigate only within the scope the human named.
3. Append `## Research Iteration <N+1> — Questions` (copy the human's ask from the decision log) and `## Research Iteration <N+1> — Findings` (new analysis) to the same `<target>_design.md` file.
4. Write the artifact, then run the "Before asking the human" sequence above to post Turn N+1.

## What To Write

### Decision Log Entry — template

```markdown
## Turn <N> — <Awaiting Review | Human Verdict> — <YYYY-MM-DD HH:MM UTC>

**Agent**: <kernel-research | kernel-plan | kernel-function-research>  |  **Target slug**: <slug>

### In scope
- <artifact path 1>
- <artifact path 2>

### Agent summary posted to human
<one paragraph, same text as the review request>

### Agent recommendation
<one sentence — the agent's own read, e.g. "Design looks complete; no blockers flagged.">

### Human verdict  (only on the "Human Verdict" variant)
- **Structured**: <approve | revise | reject | needs-more-research | defer | accept>
- **Parse confidence**: <high | medium | low>
- **Per-idea (kernel-plan turns only)**: idea #1 <approve|reject|defer>, idea #2 <...>, ...
- **Rationale points**:
  - <paraphrase>
  - > <verbatim quote if noteworthy>
- **Scope additions / follow-up questions**:
  - <bullet>
- **Next action**: <what the agent will do on its next turn>
```

### Idea Ledger Update — write rules

See `.opencode/memory/idea_ledger/template.md` for the ledger structure.  Update rules:

- **Adding an approved idea**: fill `status: approved`, record turn number, rationale, approved-on timestamp.
- **Moving approved → rejected** (human later changes mind): set `status: rejected`, add `rejected_on`, `reason`.
- **Deferred**: set `status: deferred`, add `reason`, `reopen_trigger` (the condition under which it becomes viable again, if the human named one).
- **Moving approved → landed** (after a later pipeline run lands the patch): set `status: landed`, add `delta_pct`, `validation_path`, `landed_on`.  The `kernel-plan` agent does NOT write this — it is written by the pipeline's decision stage if/when the patch later runs through the full flow.
- **Never delete entries.**  Historical verdicts are part of the dedup dataset.

## Dedup Feedback Loop

`optimization-funnel.md` lists the dedup sources a researcher/planner must check before emitting ideas.  When the idea ledger exists for the target (i.e. any primary-agent human-mode work has happened on it), it is **added** to that list:

- any idea whose mechanism matches a `rejected` entry in `.opencode/memory/idea_ledger/<target_slug>.md` MUST be dropped with a citation like `dropped: <idea>; matches idea_ledger:<ledger-id>; reason: <verbatim reason>`.
- any idea whose mechanism matches a `landed` entry behaves like a prior-iteration landed plan: do not re-propose.
- `deferred` entries are NOT auto-dropped; the agent MAY re-propose if new evidence suggests the `reopen_trigger` has been met — but must call that out explicitly.

## Interaction With Existing Long-Term Memory

`.opencode/memory/targets/<target>.md`, `.opencode/memory/subsystems/<subsystem>.md`, and `.opencode/memory/global_lessons.md` still hold stable structural facts, good directions, bad plans, validation notes — unchanged by this skill.

The new stores are **additive**, not replacements:

| Store | Stable | Per-target | Granularity | Lifetime |
|---|---|---|---|---|
| `memory/targets/<target>.md` | ✓ | ✓ | prose summary | forever |
| `memory/subsystems/<subsystem>.md` | ✓ | per-subsystem | prose summary | forever |
| `memory/global_lessons.md` | ✓ | global | prose summary | forever |
| `memory/human_decisions/<target_slug>.md` | ✓ | ✓ | chronological log | forever |
| `memory/idea_ledger/<target_slug>.md` | ✓ | ✓ | per-idea row | forever |
| `state/bad_plans.md` | ✓ | global | per-mechanism | forever |

When a `kernel-research` or `kernel-plan` session ends cleanly (human says "done, this is good"), the agent still runs the normal `memory-accumulation.md` promotion step — distill stable findings into the target / subsystem / global memory.  This skill adds the live per-turn writes on top; it does not replace end-of-session promotion.

## Safety Rules

- Never store API keys, hardware serials, or any secret the human pastes by accident.  If a reply contains what looks like a credential, redact it in the decision log (replace with `[REDACTED]`) and flag it back to the human in the next turn.
- Never write the human's unrelated chat (jokes, scheduling, off-topic) to the decision log.  Only the verdict and its rationale belong there.
- If the human explicitly says "don't record this" about a specific sentence, honor it — do not write that sentence to any memory file.  The verdict itself still gets recorded, without the sentence.

## Resuming After Session Compaction Or A New Session

Because every verdict is persisted to disk before the agent ends its turn, a fresh session on the same target can recover by reading:

1. `.opencode/memory/human_decisions/<target_slug>.md` — full chronology of what the human already decided, including the most recent `Awaiting Review` block if a turn was pending
2. `.opencode/memory/idea_ledger/<target_slug>.md` — approved / rejected / deferred ideas (persistent dedup)
3. `.opencode/docs/<target>_design.md` — full research chronology via the append-only `Research Iteration <N>` sections
4. `.opencode/plans/<target>_plan.md` — current plan draft if `kernel-plan` had started one

The agent on resume MUST:

1. Re-read the four sources above.
2. Acknowledge the resumption to the human, summarize the last turn's state in one paragraph.
3. Ask the human whether to continue from where the last turn left off or redirect.
4. Continue normally.

Do NOT try to "catch up" by re-running prior investigation — the artifacts are on disk.
