# Workqueue Full Pipeline

This is a domain-specific preset, not the generic default. Use `generic_full` when the target is any arbitrary directory or file and you want automatic routing.

## Intent

Full workqueue and thread-pool optimization pipeline with instruction count as the primary objective, idea ranking, and bad-plan memory.

## Specialist Bias

- research and ideation: `researcher` + `domain-workqueue` pack (legacy: `wq-threadpool-opt`)
- plan review: `reviewer` (legacy: `kernel-plan-reviewer`)
- implementation: `implementer` (legacy: `kernel-code-agent`)
- code review: `reviewer` (legacy: `kernel-code-reviewer`)
- tester: `validator` (legacy: `kernel-tester-agent`)

## Load First

- `.opencode/config.yaml`
- `.opencode/skills/infra/language-config/SKILL.md`
- `.opencode/skills/infra/pipeline/stage-gate-enforcement/SKILL.md`
- `.opencode/skills/infra/agent-core/SKILL.md`
- `.opencode/skills/infra/pipeline/recipe-execution/SKILL.md`
- `.opencode/skills/infra/pipeline/delegate/SKILL.md`
- `.opencode/skills/role/review-checklists/SKILL.md`
- `.opencode/docs/harness_engineer_system.md`
- `.opencode/skills/scenario/kernel-opt/instruction-count-first/SKILL.md`
- `.opencode/skills/role/research-discipline/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/domain-workqueue/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/optimization-funnel/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/perf-bottleneck-playbooks/SKILL.md`
- `.opencode/skills/infra/pipeline/handoff-contract/SKILL.md`
- `.opencode/skills/role/implementation-guardrails/SKILL.md`
- `.opencode/skills/role/validation-flight-check/SKILL.md`

## Execution Shape

1. API and worker-loop understanding
2. ranked instruction-count-focused ideation
3. approved plan
4. plan review gate
5. minimal patch and implementation handoff
6. code review gate
7. conditional tester validation
