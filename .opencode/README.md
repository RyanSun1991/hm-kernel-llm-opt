# OpenCode Multi-Agent Workspace

This directory is the canonical OpenCode-facing workspace for kernel analysis and optimization in this repository.

## Layout

- `agents/`: primary OpenCode agent prompt files
- `pipelines/`: one-click pipeline preset cards
- `skills/`: reusable capability packs for agent loading
- `docs/`: living subsystem design notes and bootstrap docs
- `memory/`: long-term memory across runs
- `state/`: persistent task and ideation state
- `plans/`: approved implementation plans
- `reviews/`: independent review outputs
- `bench/`: validation plans and before/after evidence
- `patches/`: exported patches when needed

## Working Rules

1. For one-shot startup, begin with `agents/kernel-pipeline-starter.md`.
2. For manual routing, begin with `agents/os-opt-manager.md`.
3. Research before optimization.
4. Use Sequential Thinking MCP first.
5. Use Kernel Index MCP early.
6. Save durable findings under `docs/`.
7. Promote stable reusable findings into `memory/`.
8. Save approved plans under `plans/`.
9. Save reviewer output under `reviews/`.

## Current Canonical Bootstrap

For memmgr and reclaim work, read `docs/memmgr-reclaim_bootstrap.md` before new exploration.
