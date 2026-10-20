# Synchronization Review Pipeline

## Intent

Focused instruction-count-aware review pipeline for lock scope, waiter ordering, refcount lifetime, and race-sensitive changes.

## Specialist Bias

- research: `researcher` + `domain-sync` pack (legacy: `basic-mechanism-sync-opt`)
- plan review: `reviewer` (legacy: `kernel-plan-reviewer`)
- code review: `reviewer` (legacy: `kernel-code-reviewer`)

## Load First

- `.opencode/config.yaml`
- `.opencode/skills/infra/language-config/SKILL.md`
- `.opencode/skills/infra/agent-core/SKILL.md`
- `.opencode/docs/harness_engineer_system.md`
- `.opencode/skills/infra/pipeline/stage-gate-enforcement/SKILL.md`
- `.opencode/skills/infra/pipeline/recipe-execution/SKILL.md`
- `.opencode/skills/infra/pipeline/delegate/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/instruction-count-first/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/domain-sync/SKILL.md`
- `.opencode/skills/role/research-discipline/SKILL.md`
- `.opencode/skills/infra/pipeline/handoff-contract/SKILL.md`
- `.opencode/skills/role/review-checklists/SKILL.md`
- `.opencode/skills/role/validation-flight-check/SKILL.md`

## Execution Shape

1. identify protected data and ownership assumptions
2. map lock and state-machine boundaries
3. estimate whether instruction-count reduction is compatible with synchronization safety
4. produce plan or code review verdict
