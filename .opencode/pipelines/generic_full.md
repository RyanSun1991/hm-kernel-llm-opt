# Generic Full Pipeline

## Intent

A domain-agnostic pipeline for analyzing and optimizing any kernel directory, subsystem, or file target with instruction count as the primary optimization objective.

## Behavior

- start with the coordinator hub (legacy: hm-opt-manager), not a fixed specialist
- route automatically by target path, symbols, and discovered code semantics
- require research before optimization
- require ranked ideation before implementation
- require plan review before implementation
- require code review before tester validation
- run tester validation after code review only when executable verification is needed and feasible
- accumulate long-term memory while working

## Load First

- `.opencode/config.yaml`
- `.opencode/skills/infra/pipeline/stage-gate-enforcement/SKILL.md`
- `.opencode/skills/infra/agent-core/SKILL.md`
- `.opencode/skills/infra/pipeline/recipe-execution/SKILL.md`
- `.opencode/skills/infra/pipeline/delegate/SKILL.md`
- `.opencode/skills/role/review-checklists/SKILL.md`
- `.opencode/skills/infra/language-config/SKILL.md`
- `.opencode/docs/harness_engineer_system.md`
- `.opencode/skills/scenario/kernel-opt/instruction-count-first/SKILL.md`
- `.opencode/skills/role/research-discipline/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/domain-reclaim/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/domain-hyperhold-io/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/domain-workqueue/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/domain-sync/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/optimization-funnel/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/perf-bottleneck-playbooks/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/memory-tlb-optimization/SKILL.md`
- `.opencode/skills/infra/pipeline/handoff-contract/SKILL.md`
- `.opencode/skills/role/implementation-guardrails/SKILL.md`
- `.opencode/skills/role/validation-flight-check/SKILL.md`
- `.opencode/skills/infra/memory-accumulation/SKILL.md`
- `.opencode/docs/memory_system.md`
- `.opencode/skills/infra/pipeline/stage-gate-enforcement/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/build-and-sign/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/flash-device-operations/SKILL.md`
- `.opencode/skills/scenario/kernel-opt/ab-test-comparison/SKILL.md`

## Execution Shape

1. target classification and routing
2. design understanding and instruction-count hotspot model
3. ranked ideas with bad-plan filtering
4. approved plan
5. plan review gate
6. minimal implementation and implementation handoff
7. code review gate
8. conditional tester validation
9. validation and memory update
