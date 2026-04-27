# Memory System

This repository uses `.opencode/memory/` as the long-term memory layer for OpenCode-driven optimization work.

## Layout

- `targets/`: target-specific notes for files or directories
- `subsystems/`: broader subsystem knowledge reused across many tasks
- `global_lessons.md`: reusable heuristics, anti-patterns, and validation lessons
- `human_decisions/`: chronological human-agent decision logs, populated by primary-agent human workflows (`kernel-research`, `kernel-plan`) — see `.opencode/skills/human-interaction-memory.md`
- `idea_ledger/`: per-target approved / landed / rejected / deferred mechanism ledger; populated by primary-agent human workflows and consulted by the optimization-funnel dedup step of both primary agents and pipeline sub-agents

## Usage Rules

- load relevant memory before fresh research
- update memory only with stable, reusable findings
- do not store volatile or one-off chat noise
- when in doubt, put transient findings into `.opencode/docs/` and only promote them to memory after review
