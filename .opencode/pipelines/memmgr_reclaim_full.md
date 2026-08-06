# Memmgr Reclaim Full Pipeline

## Intent

Deep reclaim and allocator-coupling analysis with instruction-count-first optimization and explicit review gates.

## Specialist Bias

- research: `researcher` + `domain-reclaim` pack (legacy: `memmgr-reclaim-research`)
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
- `.opencode/docs/memmgr-reclaim_bootstrap.md`
- `.opencode/skills/scenario/kernel-opt/instruction-count-first/SKILL.md`
- `.opencode/skills/role/research-discipline/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/domain-reclaim/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/optimization-funnel/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/perf-bottleneck-playbooks/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/memory-tlb-optimization/SKILL.md`
- `.opencode/skills/infra/pipeline/handoff-contract/SKILL.md`
- `.opencode/skills/role/validation-flight-check/SKILL.md`

## Execution Shape

1. reclaim model and trigger mapping
2. pressure and watermark reasoning with instruction-count focus
3. ranked optimization ideas
4. approved implementation plan
5. plan review gate
6. implementation handoff
7. code review gate
8. conditional tester validation and trace comparison
