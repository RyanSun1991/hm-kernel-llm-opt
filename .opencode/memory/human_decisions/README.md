# Human Decision Log

Per-target chronological log of every significant human-agent exchange under primary-agent workflows that load `.opencode/skills/infra/human-interaction-memory/SKILL.md` (e.g. `kernel-research`, `kernel-plan`, `kernel-function-research`).

## File Layout

- one file per target, named `<target_slug>.md`
- `<target_slug>` matches the idea-ledger convention (and the pipeline's `base_slug` if the target later enters a pipeline run)
- see `template.md` for the block structure

## Purpose

This log exists for **three** reasons:

1. **Audit** — every advance past a human-approval turn must be traceable to a human verdict.
2. **Resumption** — after a session compaction or a brand-new session, the agent reconstructs conversation state by reading the latest turn block here.
3. **Dedup context** — when `kernel-research` is iterating on "needs-more-research", it reads the latest turn block to pick up the human's questions; the log is the single source of truth for what the human asked.

## Read Rules

- **Agent on resume** reads the whole file, scans for the most recent block whose status is `pending-human-review` or whose verdict is missing, and rebuilds from there.
- **Researcher at an iteration** reads only the latest `Turn <N> — Human Verdict` block to extract the human's questions/scope additions.
- **Planner re-proposing ideas** reads recent turn blocks to avoid repeating conversational ground.
- Pipeline sub-agents (inside an `hm-opt-manager` run) do NOT read this file; the idea ledger carries the distilled verdicts they care about.

## Write Rules

Only the primary agent that owns the current human-facing loop writes to this file.  Append-only; never edit past blocks.  When redacting a credential the human pasted, leave the surrounding structure intact and replace the secret with `[REDACTED]` in place.

Never capture off-topic chat — only the review content belongs here.
