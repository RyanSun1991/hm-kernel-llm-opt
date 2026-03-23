# Synchronization Review Pipeline

## Intent

Focused review pipeline for lock scope, waiter ordering, refcount lifetime, and race-sensitive changes.

## Specialist Bias

- research: `basic-mechanism-sync-opt`
- review: `kernel-reviewer`

## Load First

- `.opencode/skills/research-discipline.md`
- `.opencode/skills/validation-flight-check.md`

## Execution Shape

1. identify protected data and ownership assumptions
2. map lock and state-machine boundaries
3. produce risk note
4. produce final review verdict
