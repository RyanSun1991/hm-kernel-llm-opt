@kernel-plan @.opencode/agents/kernel-plan.md

Target: <REPLACE_WITH_FILE_DIRECTORY_SUBSYSTEM_OR_FUNCTION>
Steering (optional): <REPLACE_OR_DELETE — e.g. "lock-free only", "must not touch ABI">
Continue from (optional): <REPLACE_OR_DELETE — e.g. "turn 4" if resuming after a pause>

Objective: Run the 5-idea optimization funnel against the target's existing design doc + memory + idea ledger, triage each idea with a human expert, and write a concrete plan covering only human-approved ideas to `.opencode/plans/<target_slug>_plan.md`. Every per-idea verdict is persisted live to `.opencode/memory/idea_ledger/<target_slug>.md` and `.opencode/memory/human_decisions/<target_slug>.md` before the turn ends.

Precondition: `.opencode/docs/<target_slug>_design.md` must already exist (from `@kernel-research` or an earlier pipeline run). If missing, the agent will stop and ask you to run `@kernel-research` first.

Do NOT implement code, do NOT run tests, do NOT delegate. After the plan is approved, the agent will point you to `/optimize_generic` or `@kernel-code-agent` for implementation.

Skill packs:
- @.opencode/skills/language-config/SKILL.md
- @.opencode/skills/optimization-funnel/SKILL.md
- @.opencode/skills/instruction-count-first/SKILL.md
- @.opencode/skills/handoff-contract/SKILL.md
- @.opencode/skills/memory-accumulation/SKILL.md
- @.opencode/skills/human-interaction-memory/SKILL.md

Memory packs:
- @.opencode/memory/global_lessons.md

Bootstrap docs:
- @.opencode/docs/memory_system.md

Config:
- @.opencode/config.yaml
