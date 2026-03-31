# OpenCode Multi-Agent Workspace

This directory is the canonical OpenCode-facing workspace for kernel analysis and optimization in this repository.

## Layout

- `commands/`: slash-command files for one-click pipeline triggering in OpenCode
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

## Configuration

- `config.yaml`: workspace-level settings (session language, etc.)
  - `language: zh-CN` for Chinese, `language: en` for English (default)
  - Applied automatically via `skills/language-config.md`

## Working Rules

0. Read `config.yaml` and load `skills/language-config.md` at the start of every session to apply the configured language.
1. For one-shot startup, begin with `agents/kernel-pipeline-starter.md`.
2. For manual routing, begin with `agents/os-opt-manager.md`.
3. Treat instruction-count reduction as the default primary optimization goal unless the staged task explicitly overrides it.
4. Research before optimization.
5. Use Sequential Thinking MCP first.
6. Use Kernel Index MCP early.
7. Route every optimization plan through `agents/kernel-plan-reviewer.md` before implementation.
8. Route every implemented patch through `agents/kernel-code-reviewer.md` immediately after implementation.
9. Route reviewed patches to `agents/kernel-tester-agent.md` only when code review requires Build MCP and Auto-Test MCP validation and preconditions are available.
10. Save durable findings under `docs/`.
11. Promote stable reusable findings into `memory/`.
12. Save approved plans under `plans/`.
13. Save reviewer output under `reviews/`.
14. Save validation output under `bench/`.

## Current Canonical Bootstrap

For memmgr and reclaim work, read `docs/memmgr-reclaim_bootstrap.md` before new exploration.
