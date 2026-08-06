# OpenCode Agent Workbench Workspace

This directory is the canonical OpenCode-facing workspace for kernel analysis and
optimization in this repository. Since the Agent Workbench migration (design:
`docs/Agent_Workbench_Design_EN.md`) it hosts **two lanes**: the interactive
workbench (default) and the automated pipeline (explicit `/optimize_*` recipes).

## Layout

- `CLAUDE.md`: the thin constitution (mirrors repo-root `AGENTS.md`)
- `agents/`: the 7 generic workbench roles (assistant · researcher · architect ·
  implementer · reviewer · validator · coordinator) — see `agents/README.md`
  - `agents/profiles/`: named role+skill compositions (thin agent files, from M3)
  - `agents/legacy/`: the pre-workbench pipeline cast (hm-opt-manager + specialists);
    the fallback chain; deleted only after the live old-vs-new comparison is archived
- `commands/`: slash-command files — `/optimize_*` (pipeline recipes), `/research`
  (researcher role), `/plan` (architect role), `/function_detail` (legacy explainer)
- `pipelines/`: pipeline preset cards (consumed by the pipeline lane)
- `skills/`: 3-tier skill library (`role/` · `scenario/` · `infra/`) indexed by
  `skills/_registry.yaml` — see `skills/README.md`
- `templates/workspace/`: tracked template for task workspaces
  (instantiate via `bash scripts/new_workspace.sh <task-slug>`)
- `local/`: git-ignored runtime state — task workspaces
  (`local/workspaces/<task-slug>/`) and sediment staging
- `docs/`: living subsystem design notes and bootstrap docs
  (`docs/harness_engineer_system.md` = pipeline-lane spec)
- `memory/`: long-term memory across runs
  - `targets/`, `subsystems/`, `global_lessons.md`: stable structural / heuristic memory
  - `human_decisions/`: per-target chronological log from human-in-the-loop sessions
    (see `skills/infra/human-interaction-memory/SKILL.md`)
  - `idea_ledger/`: per-target approved / landed / rejected / deferred mechanism ledger
- `state/`: pipeline-lane persistent state (`current_task.json`, bad-plan ledgers).
  Workbench tasks do NOT use it — their truth lives in `local/workspaces/`.
- `plans/` · `reviews/` · `bench/` · `patches/`: pipeline-lane artifact directories
  (workbench-lane artifacts live inside each task workspace's `artifacts/`)

## Configuration

- `config.yaml`: workspace-level settings (session language)
  - `language: zh-CN` for Chinese, `language: en` for English (default)
  - Applied automatically via `skills/infra/language-config/SKILL.md`

## Entry Points

### 1. Just talk (default — workbench lane)

Open OpenCode and ask. The default agent is `assistant`: simple questions get direct
answers; for bigger tasks it proposes "open a workspace + bring in <role> with
<skills>" and **waits for your confirmation**. Ordinary prompts never start a
pipeline.

### 2. Pick a role or profile

Tab-switch or `@researcher` / `@architect` / `@implementer` / `@reviewer` /
`@validator` — or a preloaded composition from `agents/profiles/`
(e.g. `@reclaim-investigator`). Roles suggest skills from the registry; you confirm.
Task state persists in `local/workspaces/<task-slug>/` (capsule = resume carrier;
say "continue <task-slug>" in a new session).

### 3. Human-in-the-loop commands

- `/research` — researcher role, iterative living design doc with per-turn human
  verdict persistence
- `/plan` — architect role, 5-idea optimization funnel with per-idea human triage

### 4. Automated pipeline (explicit recipes)

`/optimize_generic` · `/optimize_hyperhold` · `/optimize_memmgr_reclaim` ·
`/optimize_workqueue` run the full staged pipeline (research → plan review GATE →
implement → code review GATE → tester A/B → decision) under
`docs/harness_engineer_system.md` rules. Entry agent: `@coordinator` driving the
workbench roles (see `skills/infra/pipeline/recipe-execution/SKILL.md`); the legacy
`@hm-opt-manager` chain (`agents/legacy/`) remains the fallback until the live
old-vs-new comparison is archived.

## Working Rules

0. Read `config.yaml` and apply `skills/infra/language-config/SKILL.md` at the start
   of every session.
1. Workbench lane: follow `skills/infra/agent-core/SKILL.md` (output contract, six
   verbs, capsule upkeep, status gating, permission discipline). The user owns
   routing.
2. Pipeline lane: follow `docs/harness_engineer_system.md` + `skills/infra/pipeline/`
   (stage gates, handoff packets, hub-and-spoke delegation).
3. Research before optimization; use Sequential Thinking MCP first and Kernel Index
   MCP early.
4. Durable findings → `docs/`; stable reusable findings → `memory/`; approved plans →
   `plans/`; reviews → `reviews/`; validation evidence → `bench/`.
5. Artifact headers carry `status:` + `produced_by:` receipts; status promotions have
   role-owned conditions (agent-core §6).

## Current Canonical Bootstrap

For memmgr and reclaim work, read `docs/memmgr-reclaim_bootstrap.md` before new
exploration.
