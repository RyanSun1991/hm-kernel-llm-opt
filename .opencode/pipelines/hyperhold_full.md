# Hyperhold Full Pipeline

## Intent

Full-spectrum instruction-count-first analysis and optimization pipeline for Hyperhold, swap I/O, hpio, iotab, eid, and hot serialization paths.

## Specialist Bias

- research: `researcher` + `domain-hyperhold-io` pack (legacy: `hyperhold-io-opt`)
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
- `.opencode/skills/scenario/kernel-opt/domain-hyperhold-io/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/optimization-funnel/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/perf-bottleneck-playbooks/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/memory-tlb-optimization/SKILL.md`
- `.opencode/skills/infra/pipeline/handoff-contract/SKILL.md`
- `.opencode/skills/role/implementation-guardrails/SKILL.md`
- `.opencode/skills/role/validation-flight-check/SKILL.md`

## Execution Shape

1. research and design model
2. instruction-count-aware ideation
3. approved plan
4. plan review gate
5. minimal implementation and handoff
6. code review gate
7. conditional tester validation via Build MCP and Auto-Test MCP as needed
