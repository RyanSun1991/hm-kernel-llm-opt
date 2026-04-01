# CLAUDE.md — Harness Enforcement

This repository uses a strict multi-agent harness defined in `.opencode/docs/harness_engineer_system.md`. **All agents MUST follow the staged pipeline without exception.**

## Mandatory Reading at Session Start

Every agent session MUST begin by reading these files in order:

1. `.opencode/config.yaml` — session language
2. `.opencode/skills/language-config.md` — language rules
3. `.opencode/docs/harness_engineer_system.md` — authoritative pipeline spec
4. `.opencode/skills/stage-gate-enforcement.md` — hard stage-gate rules

## Pipeline Stage Order (NEVER Skip)

```
1. intake → kernel-pipeline-starter
2. routing → os-opt-manager
3. research → specialist researcher
4. plan review → kernel-plan-reviewer        ← MANDATORY GATE
5. implementation → kernel-code-agent         ← ONLY after plan approval
6. code review → kernel-code-reviewer         ← MANDATORY GATE
7. tester validation → kernel-tester-agent    ← CONDITIONAL (code reviewer decides)
8. decision & memory → os-opt-manager
```

## Hard Rules

### Stage Gating — NEVER Bypass

- **NO implementation without plan review approval.** `kernel-code-agent` MUST NOT write code unless `kernel-plan-reviewer` has approved the plan in `.opencode/reviews/*_plan_review.md`.
- **NO final decision without code review.** Every patch MUST go through `kernel-code-reviewer` before it can be accepted.
- **NO agent may perform work belonging to another stage.** Research agents do not implement. Coders do not review. Reviewers do not test.

### Mandatory Delegation — NEVER Forget

- Every agent MUST delegate to the next stage agent when its work is complete.
- The delegation message MUST include the full handoff packet as defined in `.opencode/skills/handoff-contract.md`.
- After delegating, the agent MUST stop and tell the user which agent to open next.
- If you are unsure which agent comes next, re-read `.opencode/docs/harness_engineer_system.md` Section "Stage Order" and `.opencode/skills/stage-gate-enforcement.md`.

### Handoff Packet — NEVER Omit

Every stage transition MUST produce a handoff packet containing:

- target and primary metric
- evidence baseline
- hot path
- files and functions in scope
- risks and open questions
- exact next action for the receiving agent

### Skill Loading — NEVER Skip

Pipeline presets list required skills. Every listed skill MUST be read and followed. Core skills that apply to ALL pipeline runs:

- `instruction-count-first.md`
- `handoff-contract.md`
- `stage-gate-enforcement.md`
- `research-discipline.md`
- `implementation-guardrails.md`

## Self-Check Before Every Action

Before performing any significant action, ask yourself:

1. Am I the correct agent for this stage?
2. Have I read all required upstream artifacts?
3. Will I produce the required handoff packet when done?
4. Do I know which agent to delegate to next?

If any answer is "no", stop and re-read the harness docs before proceeding.

## Enforcement Reminder for Long Runs

Context can drift during long conversations. If you notice you have been working for a while:

1. Re-read `.opencode/docs/harness_engineer_system.md` to confirm your current stage.
2. Re-read `.opencode/skills/stage-gate-enforcement.md` to confirm delegation rules.
3. Verify you have not skipped any mandatory gate.
4. Verify you are producing artifacts in the correct `.opencode/` subdirectory.