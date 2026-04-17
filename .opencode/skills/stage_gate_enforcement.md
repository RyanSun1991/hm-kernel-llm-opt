# Stage-Gate Enforcement

This skill defines hard gates that no agent may bypass. Load this skill at every session start alongside `handoff-contract.md`.

## Pipeline Stages and Ownership

| Stage | Owner Agent | Entry Condition | Exit Condition |
|-------|-------------|-----------------|----------------|
| 1. Intake + Routing | `os-opt-manager` | User request | Config loaded, delegated to correct specialist |
| 2. Research | Specialist researcher | Manager delegation | Design doc + plan written, **returns to manager** |
| 3. Plan Review | `kernel-plan-reviewer` | Manager delegation with plan | Review written, **returns to manager** |
| 4. Implementation | `kernel-code-agent` | Manager delegation (plan approved) | Code changes + handoff, **returns to manager** |
| 5. Code Review | `kernel-code-reviewer` | Manager delegation with coder handoff | Review written, **returns to manager** |
| 6. Tester Validation | `kernel-tester-agent` | Manager delegation (code review requires it) | Validation report written, **returns to manager** |
| 7. Decision | `os-opt-manager` | All sub-agent results received | Memory updated, next cycle or done |

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
- When tester IS invoked, the delegation MUST include: stock image path, feature image path, device target, and test case name.

### Gate 5: A/B Comparison Before Decision

- The tester MUST flash and test BOTH the stock image AND the feature image.
- It is FORBIDDEN to report a test result based on only the feature image without a stock baseline.
- The tester MUST produce a comparison artifact showing metric deltas (stock vs feature).
- If either flash fails, the tester MUST report the failure explicitly and MUST NOT fabricate comparison data.
- If the stock flash fails, report as infrastructure failure — not a patch failure.
- Both test runs MUST use identical parameters (same test case, same duration, same device).

## Mandatory Completion Checklist

Every agent MUST complete this checklist before finishing:

1. [ ] I have written my required output artifact to the correct `.opencode/` subdirectory
2. [ ] I have prepared the handoff packet per `handoff-contract.md`
3. [ ] I know which agent is next according to the stage table above

**If I am `os-opt-manager`**: I use the delegate tool NOW to hand off to the next sub-agent with the full handoff packet. I do NOT stop to ask the user to manually continue.

**If I am a sub-agent** (specialist, reviewer, coder, tester): I return my results with the handoff packet to the manager. I do NOT attempt to delegate to other agents myself.

## Anti-Drift Rules

These rules prevent agents from silently absorbing work that belongs to other stages:

- **Researchers**: You MUST NOT write code. You MUST NOT review code. You MUST NOT run tests. Your job ends when the plan is handed to the plan reviewer.
- **Plan Reviewer**: You MUST NOT implement code. You MUST NOT run tests. Your job ends when you approve/reject and delegate.
- **Coder**: You MUST NOT review your own code. You MUST NOT run validation tests. Your job ends when you delegate to the code reviewer.
- **Code Reviewer**: You MUST NOT fix code. You MUST NOT run tester validation. Your job ends when you delegate to tester or manager.
- **Tester**: You MUST NOT modify code. You MUST NOT re-review. Your job ends when you report results and delegate to manager.

## Context Refresh Protocol

If your conversation has exceeded 10 exchanges or you feel uncertain about the current stage:

1. Re-read this file (`.opencode/skills/stage_gate_enforcement.md`)
2. Re-read `.opencode/docs/harness_engineer_system.md` Section "Stage Order"
3. Read `.opencode/state/current_task.json` for the current stage
4. Enumerate the existing artifacts with Bash: `ls .opencode/reviews/ .opencode/plans/ .opencode/bench/`. **Do NOT glob `.opencode/**`** — it hangs.
5. Resume from the correct stage

## Delegation Message Template

When the manager delegates, or when a sub-agent returns results, use this structure:

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