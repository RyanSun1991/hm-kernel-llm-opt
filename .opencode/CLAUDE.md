# CLAUDE.md — Harness Enforcement

This repository uses a strict multi-agent harness defined in `.opencode/docs/harness_engineer_system.md`. **All agents MUST follow the staged pipeline without exception.**

## Mandatory Reading at Session Start

Every agent session MUST begin by reading these files in order:

1. `.opencode/config.yaml` — session language
2. `.opencode/skills/language-config.md` — language rules
3. `.opencode/docs/harness_engineer_system.md` — authoritative pipeline spec
4. `.opencode/skills/stage_gate_enforcement.md` — hard stage-gate rules

## Pipeline Stage Order (NEVER Skip)

```
1. intake + routing → os-opt-manager (entry agent, central hub)
2. research → specialist researcher            ← delegated by manager, returns to manager
3. plan review → kernel-plan-reviewer          ← MANDATORY GATE, returns to manager
4. implementation → kernel-code-agent          ← ONLY after plan approval, returns to manager
5. code review → kernel-code-reviewer          ← MANDATORY GATE, returns to manager
6. tester A/B validation → kernel-tester-agent  ← CONDITIONAL: flash stock, test, flash feature, test, compare, returns to manager
7. decision & memory → os-opt-manager
```

## Hard Rules

### File Discovery in `.opencode/` — NEVER Use glob

OpenCode's glob tool gets stuck on `.opencode/**` patterns (dot-prefixed directories are not enumerated) and repeatedly reports "no matches" even when the files exist. This blocks the whole pipeline.

**Do NOT call glob on anything under `.opencode/` — not `.opencode/**`, not `.opencode/**/*.md`, not any variation.**

Use these alternatives instead:

1. **Read a file by exact path.** The agent specs and handoff packets name the specific files you need (e.g. `.opencode/docs/harness_engineer_system.md`, `.opencode/skills/stage_gate_enforcement.md`). Open them directly with Read. If the task names a plan, review, or bench artifact, it already carries the concrete path.
2. **Enumerate a directory with Bash `ls`.** When you genuinely need to discover what's in a subdirectory (e.g. "which memory files exist under `.opencode/memory/targets/`?"), run `ls .opencode/memory/targets/` or `ls -la .opencode/memory/targets/*.md 2>/dev/null`. Bash sees dotfiles correctly.
3. **Search content with Grep.** For "find files that mention X in `.opencode/`", use Grep — it traverses dot-prefixed directories fine.

When a doc below refers to something like `.opencode/reviews/*_plan_review.md`, treat the wildcard as *describing what to write to*, not as an instruction to list. If you need to know whether a review already exists, `ls` that directory first.

### Stage Gating — NEVER Bypass

- **NO implementation without plan review approval.** `kernel-code-agent` MUST NOT write code unless `kernel-plan-reviewer` has approved the plan in `.opencode/reviews/*_plan_review.md`.
- **NO final decision without code review.** Every patch MUST go through `kernel-code-reviewer` before it can be accepted.
- **NO test verdict without A/B comparison.** The tester MUST flash and test BOTH stock and feature images. Single-image test results are FORBIDDEN as the basis for a verdict.
- **NO agent may perform work belonging to another stage.** Research agents do not implement. Coders do not review. Reviewers do not test.

### Hub-and-Spoke Delegation — NEVER Bypass

- **Only `kernel-pipeline-starter` and `os-opt-manager` use the delegate tool.** They are the only agents with `delegate: true`.
- **All sub-agents (specialists, reviewers, coder, tester) return results to the manager.** They complete their work, write artifacts, output the handoff packet, and finish. The manager then reads the results and delegates to the next stage.
- The manager MUST NOT stop and ask the user to manually continue between stages. When a sub-agent returns, the manager immediately proceeds.
- Sub-agents MUST NOT attempt to delegate to other agents. They return to whoever called them (the manager).
- If you are unsure which agent comes next, re-read `.opencode/docs/harness_engineer_system.md` Section "Stage Order" and `.opencode/skills/stage_gate_enforcement.md`.

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
- `stage_gate_enforcement.md`
- `research-discipline.md`
- `implementation-guardrails.md`
- `flash-device-operations.md` (when tester validation is active)
- `ab-test-comparison.md` (when tester validation is active)

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
2. Re-read `.opencode/skills/stage_gate_enforcement.md` to confirm delegation rules.
3. Verify you have not skipped any mandatory gate.
4. Verify you are producing artifacts in the correct `.opencode/` subdirectory.