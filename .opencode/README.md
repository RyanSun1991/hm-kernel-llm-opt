# OpenCode Multi-Agent Workspace

This directory is the canonical OpenCode-facing workspace for kernel analysis and optimization in this repository.

## Layout

- `commands/`: slash-command files for one-click pipeline triggering in OpenCode
- `agents/`: primary OpenCode agent prompt files
- `pipelines/`: one-click pipeline preset cards
- `skills/`: reusable capability packs for agent loading
- `docs/`: living subsystem design notes and bootstrap docs
- `memory/`: long-term memory across runs
  - `targets/`, `subsystems/`, `global_lessons.md`: stable structural / heuristic memory
  - `human_decisions/`: per-target chronological log populated by primary-agent human workflows (`kernel-research`, `kernel-plan`, `kernel-function-research`) — see `skills/human-interaction-memory.md`
  - `idea_ledger/`: per-target approved / landed / rejected / deferred mechanism ledger; populated by the primary-agent human workflows and consulted by both primary agents and pipeline sub-agents during the optimization-funnel dedup step
- `state/`: persistent task and ideation state
- `plans/`: approved implementation plans
- `reviews/`: independent review outputs from the pipeline's review sub-agents
- `bench/`: validation plans and before/after evidence
- `patches/`: exported patches when needed

## Configuration

- `config.yaml`: workspace-level settings (session language, etc.)
  - `language: zh-CN` for Chinese, `language: en` for English (default)
  - Applied automatically via `skills/language-config.md`

## Entry Points — Three Routes For Three Kinds Of Work

The workspace supports three complementary entry points depending on what the user is trying to do:

### 1. Full automated pipeline — for end-to-end optimization runs

`@os-opt-manager` (or `/optimize_generic`, `/optimize_hyperhold`, etc.) runs the complete `research → plan review → implement → code review → tester → decision` pipeline with sub-agents delegated by the manager. Use this when the target is well-understood and you want an automated land-it run.

### 2. Primary-agent human-in-the-loop — for expert-driven iterative work

Two standalone primary agents own their own multi-turn dialogue with a human expert and write artifacts + memory live every turn:

- `@kernel-research` (`/research`) — iterative research. Produces `.opencode/docs/<target_slug>_design.md` as a living document grown across many turns. Explain-only; no optimization ideation.
- `@kernel-plan` (`/plan`) — iterative ideation + planning. Reads the design doc + memory + idea ledger, runs the 5-idea funnel, triages per-idea with the human, writes `.opencode/plans/<target_slug>_plan.md`. Precondition: design doc exists.

Output of `kernel-plan` (`<target_slug>_plan.md` + idea-ledger rows) is directly consumable by route 1 — run `/optimize_generic <target>` after `@kernel-plan` and the pipeline will pick up the plan and land the approved ideas.

All human verdicts in routes 2 and 3 are persisted live to `memory/human_decisions/` and `memory/idea_ledger/` before the turn ends, so sessions survive compaction and new sessions resume from disk.

### 3. Per-function deep dive — for a single-shot explainer

`@kernel-function-research` (`/function_detail`) produces a complete design + callee-graph report on ONE kernel function in a single pass. Explain-only. Useful when routes 1 or 2 need a deeper understanding of a specific function before continuing.

## Working Rules

0. Read `config.yaml` and load `skills/language-config.md` at the start of every session to apply the configured language.
1. For automated end-to-end runs: `agents/os-opt-manager.md` (central hub) or `agents/kernel-pipeline-starter.md` (legacy alias).
2. For expert-driven iterative research: `agents/kernel-research.md` (`/research`).
3. For expert-driven iterative ideation + planning: `agents/kernel-plan.md` (`/plan`) — reads the design doc produced in step 2.
4. For a single-shot function deep dive: `agents/kernel-function-research.md` (`/function_detail`).
5. Treat instruction-count reduction as the default primary optimization goal unless the staged task explicitly overrides it.
6. Research before optimization.
7. Use Sequential Thinking MCP first.
8. Use Kernel Index MCP early.
9. In the pipeline route (1): route every optimization plan through `agents/kernel-plan-reviewer.md` before implementation; route every implemented patch through `agents/kernel-code-reviewer.md`; route reviewed patches to `agents/kernel-tester-agent.md` only when code review requires Build MCP + Auto-Test MCP validation.
10. Save durable findings under `docs/`.
11. Promote stable reusable findings into `memory/`.
12. Save approved plans under `plans/`.
13. Save reviewer output under `reviews/` (pipeline only).
14. Save validation output under `bench/`.

## Current Canonical Bootstrap

For memmgr and reclaim work, read `docs/memmgr-reclaim_bootstrap.md` before new exploration.
