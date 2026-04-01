# Stage-Gate Enforcement

This skill defines hard gates that no agent may bypass. Load this skill at every session start alongside `handoff-contract.md`.

## Pipeline Stages and Ownership

| Stage | Owner Agent | Entry Condition | Exit Condition |
|-------|-------------|-----------------|----------------|
| 1. Intake | `kernel-pipeline-starter` | User request | Delegated to manager with expanded task |
| 2. Routing | `os-opt-manager` | Starter delegation | Delegated to correct specialist |
| 3. Research | Specialist researcher | Manager delegation | Design doc + plan written, delegated to plan reviewer |
| 4. Plan Review | `kernel-plan-reviewer` | Research handoff with plan | Review written, delegated to coder (if approved) or back to researcher (if rejected) |
| 5. Implementation | `kernel-code-agent` | Approved plan review | Code changes + handoff, delegated to code reviewer |
| 6. Code Review | `kernel-code-reviewer` | Coder handoff | Review written, delegated to tester (if required) or manager/user (if skipped) |
| 7. Tester Validation | `kernel-tester-agent` | Code review requires validation | Validation report written, delegated to manager/user |
| 8. Decision | `os-opt-manager` or User | Review/validation complete | Memory updated, next cycle or done |

## Hard Gates — Violations Are Forbidden

### Gate 1: Research Before Plan

- A plan MUST NOT be written without a design doc that identifies the hot path, subsystem boundary, and instruction-count hypothesis.
- The researcher MUST hand off to `kernel-plan-reviewer` with a complete handoff packet. It is FORBIDDEN to hand off directly to `kernel-code-agent`.

### Gate 2: Plan Review Before Implementation

- `kernel-code-agent` MUST NOT begin implementation unless `.opencode/reviews/*_plan_review.md` exists and contains `decision: approve`.
- If the plan review says `needs revision` or `reject`, the work MUST return to the researcher or manager. It is FORBIDDEN to proceed to implementation.

### Gate 3: Code Review After Implementation

- Every implementation MUST be followed by delegation to `kernel-code-reviewer`.
- It is FORBIDDEN to skip code review, even for "small" or "obvious" changes.
- The coder MUST NOT self-review.

### Gate 4: Tester Only When Code Review Requests It

- `kernel-tester-agent` MUST only be invoked when `kernel-code-reviewer` explicitly sets tester decision to `required` or `recommended`.
- If tester is `skipped`, the flow goes directly to the manager/user.

## Mandatory Delegation Checklist

Every agent MUST complete this checklist before finishing:

1. [ ] I have written my required output artifact to the correct `.opencode/` subdirectory
2. [ ] I have prepared the handoff packet per `handoff-contract.md`
3. [ ] I know which agent is next according to the stage table above
4. [ ] I am delegating to that agent NOW with the full handoff packet
5. [ ] I am telling the user which agent to open next

## Anti-Drift Rules

These rules prevent agents from silently absorbing work that belongs to other stages:

- **Researchers**: You MUST NOT write code. You MUST NOT review code. You MUST NOT run tests. Your job ends when the plan is handed to the plan reviewer.
- **Plan Reviewer**: You MUST NOT implement code. You MUST NOT run tests. Your job ends when you approve/reject and delegate.
- **Coder**: You MUST NOT review your own code. You MUST NOT run validation tests. Your job ends when you delegate to the code reviewer.
- **Code Reviewer**: You MUST NOT fix code. You MUST NOT run tester validation. Your job ends when you delegate to tester or manager.
- **Tester**: You MUST NOT modify code. You MUST NOT re-review. Your job ends when you report results and delegate to manager.

## Context Refresh Protocol

If your conversation has exceeded 10 exchanges or you feel uncertain about the current stage:

1. Re-read this file (`stage-gate-enforcement.md`)
2. Re-read `harness_engineer_system.md` Section "Stage Order"
3. Check `.opencode/state/current_task.json` for the current stage
4. Verify what artifacts exist under `.opencode/reviews/`, `.opencode/plans/`, `.opencode/bench/`
5. Resume from the correct stage

## Delegation Message Template

When delegating, use this structure:

```
## Delegation to [next-agent-name]

**Current Stage**: [your stage name]
**Next Stage**: [next stage name]
**Target**: [subsystem/file]
**Primary Metric**: instruction count (or override)

### Handoff Packet
- **Hot path**: ...
- **Evidence baseline**: ...
- **Files in scope**: ...
- **Risks**: ...
- **Open questions**: ...

### Required Next Action
[Specific instruction for the receiving agent]

### Required Reading
- [list of artifact paths the next agent must read]
```