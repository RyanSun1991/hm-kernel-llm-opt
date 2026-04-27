# Idea Ledger

Per-target registry of every optimization mechanism verdicted by a human reviewer.  Primary-agent workflows (`kernel-research`, `kernel-plan`, `kernel-function-research`) populate this ledger during human-in-the-loop dialogue; the full `os-opt-manager` pipeline reads it for dedup and MAY update `landed` status on decision if/when a patch lands through the pipeline.

## When Populated

Ideas are written here whenever a primary agent that loads `.opencode/skills/human-interaction-memory.md` receives a per-idea verdict from the human expert.  Typical triggers:

- `kernel-plan` presents 3–5 ranked candidate ideas → human approves / rejects / defers each → ledger gets N new rows
- `kernel-research` discovers a design pitfall and the human explicitly says "this whole direction is wrong" → ledger gets a rejected row
- A later `os-opt-manager` pipeline run lands an approved idea → the decision stage updates the row's status to `landed` with `delta_pct` and `validation_path`

## File Layout

- one file per target, named `<target_slug>.md`
- `<target_slug>` is the **base** target slug (same convention as the pipeline's `base_slug`) — shared across sessions, primary-agent runs, and pipeline iterations
- use `template.md` as the starting point when creating a new file

## Read Rules

Anyone generating optimization ideas for a target MUST read the target's ledger (if it exists) before the funnel step.  Ideas that match a `rejected` mechanism MUST be dropped with a citation; ideas matching a `landed` mechanism MUST be dropped as already-landed; `deferred` entries MAY be re-proposed only if the `reopen_trigger` has plausibly fired, and that reasoning must be stated in the funnel handoff.

See `.opencode/skills/human-interaction-memory.md` — "Dedup Feedback Loop".

## Write Rules

Only agents that own a human-facing dialogue loop (the primary agents listed above) or the pipeline's `os-opt-manager` decision stage write to the ledger.  Sub-agents inside the pipeline MUST NOT edit the ledger directly — they return handoff packets and let the manager persist.

Never delete entries.  Rejected ideas with full rationale are as valuable as approved ones because they prevent re-proposal.
