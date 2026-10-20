@architect @.opencode/agents/architect.md

Target: <REPLACE_WITH_FILE_DIRECTORY_SUBSYSTEM_OR_FUNCTION>
Steering (optional): <REPLACE_OR_DELETE — e.g. "lock-free only", "must not touch ABI">
Continue from (optional): <REPLACE_OR_DELETE — e.g. "turn 4" if resuming after a pause>

Objective: Run the 5-idea optimization funnel against the target's existing design doc + memory + idea ledger, triage each idea with a human expert, and write a concrete plan covering only human-approved ideas to `.opencode/plans/<target_slug>_plan.md`. Every per-idea verdict is persisted live to `.opencode/memory/idea_ledger/<target_slug>.md` and `.opencode/memory/human_decisions/<target_slug>.md` before the turn ends. This is kernel-optimization planning: use the preloaded scenario funnel (optimization-funnel), not the generic plan-funnel.

Precondition: `.opencode/docs/<target_slug>_design.md` must already exist (from `/research` or an earlier pipeline run). If missing, stop and suggest `/research` first.

Do NOT implement code, do NOT run tests, do NOT delegate. After the plan is approved, point the user to `/optimize_generic` for implementation through the pipeline lane.

Legacy fallback: `@kernel-plan` (agents/legacy/) remains available — until the live old-vs-new comparison is archived — if the new role underperforms on a target.

Skill packs:
- @.opencode/skills/infra/language-config/SKILL.md
- @.opencode/skills/infra/agent-core/SKILL.md
- @.opencode/skills/scenario/kernel-opt/optimization-funnel/SKILL.md
- @.opencode/skills/scenario/kernel-opt/perf-bottleneck-playbooks/SKILL.md
- @.opencode/skills/scenario/kernel-opt/instruction-count-first/SKILL.md
- @.opencode/skills/infra/human-interaction-memory/SKILL.md
- @.opencode/skills/infra/hub-bridge/SKILL.md
- @.opencode/skills/infra/memory-accumulation/SKILL.md

Memory packs:
- @.opencode/memory/global_lessons.md

Bootstrap docs:
- @.opencode/docs/memory_system.md

Config:
- @.opencode/config.yaml
