# Synchronization Review Pipeline

## Intent

Focused instruction-count-aware review pipeline for lock scope, waiter ordering, refcount lifetime, and race-sensitive changes.

## Specialist Bias

- research: `basic-mechanism-sync-opt`
- plan review: `kernel-plan-reviewer`
- code review: `kernel-code-reviewer`

## Load First

- `.opencode/docs/harness_engineer_system.md`
- `.opencode/skills/instruction-count-first.md`
- `.opencode/skills/research-discipline.md`
- `.opencode/skills/handoff-contract.md`
- `.opencode/skills/validation-flight-check.md`

## Execution Shape

1. identify protected data and ownership assumptions
2. map lock and state-machine boundaries
3. estimate whether instruction-count reduction is compatible with synchronization safety
4. produce plan or code review verdict
