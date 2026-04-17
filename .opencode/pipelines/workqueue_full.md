# Workqueue Full Pipeline

This is a domain-specific preset, not the generic default. Use `generic_full` when the target is any arbitrary directory or file and you want automatic routing.

## Intent

Full workqueue and thread-pool optimization pipeline with instruction count as the primary objective, idea ranking, and bad-plan memory.

## Specialist Bias

- research and ideation: `wq-threadpool-opt`
- plan review: `kernel-plan-reviewer`
- implementation: `kernel-code-agent`
- code review: `kernel-code-reviewer`
- tester: `kernel-tester-agent`

## Load First

- `.opencode/config.yaml`
- `.opencode/skills/language-config.md`
- `.opencode/docs/harness_engineer_system.md`
- `.opencode/skills/instruction-count-first.md`
- `.opencode/skills/research-discipline.md`
- `.opencode/skills/optimization-funnel.md`
- `.opencode/skills/handoff-contract.md`
- `.opencode/skills/implementation-guardrails.md`
- `.opencode/skills/validation-flight-check.md`

## Execution Shape

1. API and worker-loop understanding
2. ranked instruction-count-focused ideation
3. approved plan
4. plan review gate
5. minimal patch and implementation handoff
6. code review gate
7. conditional tester validation
