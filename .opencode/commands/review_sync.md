@coordinator @.opencode/agents/coordinator.md

Profile: sync_review @.opencode/pipelines/sync_review.md
Target: <REPLACE_WITH_TARGET_FILE_OR_SUBSYSTEM>
Objective: Focused instruction-count-aware review for lock scope, waiter ordering, refcount lifetime, and race-sensitive code. No implementation stage — review and analysis only.

Skill packs:
- @.opencode/skills/infra/agent-core/SKILL.md
- @.opencode/skills/scenario/kernel-opt/instruction-count-first/SKILL.md
- @.opencode/skills/role/research-discipline/SKILL.md
- @.opencode/skills/scenario/kernel-opt/domain-sync/SKILL.md
- @.opencode/skills/infra/hub-bridge/SKILL.md
- @.opencode/skills/infra/pipeline/handoff-contract/SKILL.md
- @.opencode/skills/role/validation-flight-check/SKILL.md
- @.opencode/skills/infra/language-config/SKILL.md
- @.opencode/skills/infra/pipeline/stage-gate-enforcement/SKILL.md
- @.opencode/skills/infra/pipeline/recipe-execution/SKILL.md
- @.opencode/skills/infra/pipeline/delegate/SKILL.md
- @.opencode/skills/role/review-checklists/SKILL.md

Memory packs:
- @.opencode/memory/global_lessons.md

Bootstrap docs:
- @.opencode/docs/harness_engineer_system.md

Config:
- @.opencode/config.yaml
