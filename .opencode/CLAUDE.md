# CLAUDE.md — Harness Enforcement

This repository uses a strict multi-agent harness defined in `.opencode/docs/harness_engineer_system.md`. **All agents MUST follow the staged pipeline without exception.**

## Mandatory Reading at Session Start

Every agent session MUST begin by reading these files in order:

1. `.opencode/config.yaml` — session language
2. `.opencode/skills/language-config/SKILL.md` — language rules
3. `.opencode/docs/harness_engineer_system.md` — authoritative pipeline spec
4. `.opencode/skills/stage-gate-enforcement/SKILL.md` — hard stage-gate rules

## Pipeline Stage Order (NEVER Skip)

```
1. intake + routing → hm-opt-manager (entry agent, central hub)
2. research → specialist researcher            ← delegated by manager, returns to manager
3. plan review → kernel-plan-reviewer          ← MANDATORY GATE, returns to manager
4. implementation → kernel-code-agent          ← ONLY after plan approval, returns to manager
5. code review → kernel-code-reviewer          ← MANDATORY GATE, returns to manager
6. tester A/B validation → kernel-tester-agent  ← CONDITIONAL: flash stock, test, flash feature, test, compare, returns to manager
7. decision & memory → hm-opt-manager
```

## Hard Rules

### Sub-Agents — Skills Are Already Inlined, Do NOT Read Them Again

When a command launches the manager (e.g. `/optimize_generic`), OpenCode `@`-inlines every skill document listed in the command's `Skill packs:` section into the session's prompt context.  **That context propagates to every sub-agent the manager delegates to.**  The skill content is already visible to sub-agents — they do not need to Read any file under `.opencode/skills/` to see it.

**Sub-agents MUST NOT Read files under `.opencode/skills/` at runtime.**  OpenCode's sub-agent sessions do not always run with this repo as the current working directory.  A relative path like `.opencode/skills/X.md` can resolve to `$HOME/.opencode/skills/X.md` — a different file, a missing one, or a stale copy from another project.

If a sub-agent's prompt says "follow the `optimization-funnel` protocol" or similar, that protocol is already in the sub-agent's context — apply its rules from memory, do not try to re-Read the file.

For dynamic project-local state the sub-agent truly needs to read at runtime (e.g. `.opencode/state/bad_plans.md`, `.opencode/memory/targets/<target>.md`), always resolve the project root first with Bash `git rev-parse --show-toplevel` (falling back to `pwd`) and use the absolute path — never rely on CWD for `.opencode/...` resolution.

### File Discovery in `.opencode/`

When a doc below refers to something like `.opencode/reviews/*_plan_review.md`, treat the wildcard as *describing what to write to*, not as an instruction to list. If you need to know whether a review already exists, `ls` that directory first.

### Stage Gating — NEVER Bypass

- **NO implementation without plan review approval.** `kernel-code-agent` MUST NOT write code unless `kernel-plan-reviewer` has approved the plan in `.opencode/reviews/*_plan_review.md`.
- **NO final decision without code review.** Every patch MUST go through `kernel-code-reviewer` before it can be accepted.
- **NO test verdict without A/B comparison.** The tester MUST flash and test BOTH stock and feature images. Single-image test results are FORBIDDEN as the basis for a verdict.
- **NO agent may perform work belonging to another stage.** Research agents do not implement. Coders do not review. Reviewers do not test.

### Hub-and-Spoke Delegation — NEVER Bypass

Only agents with `permission: skill: "delegate": "allow"` in their front-matter (currently only `hm-opt-manager`) may delegate. Load `.opencode/skills/delegate/SKILL.md` for the full mechanism — the `task(subagent_type=...)` tool is the delegation primitive when a native `delegate()` runtime function is not exposed.

- **All sub-agents (specialists, reviewers, coder, tester) return results to the manager.** They complete their work, write artifacts, output the handoff packet, and finish. The manager then reads the results and delegates to the next stage.
- The manager MUST NOT stop and ask the user to manually continue between stages. When a sub-agent returns, the manager immediately proceeds.

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

- `instruction-count-first/SKILL.md`
- `handoff-contract/SKILL.md`
- `stage-gate-enforcement/SKILL.md`
- `research-discipline/SKILL.md`
- `implementation-guardrails/SKILL.md`
- `flash-device-operations/SKILL.md` (when tester validation is active)
- `ab-test-comparison/SKILL.md` (when tester validation is active)

## Verifying Delegation Actually Hit a Real Sub-Agent

If you suspect the manager is hallucinating a sub-agent instead of actually delegating (see the "How to Delegate" section in `hm-opt-manager.md`), here are the four signals, strongest first:

1. **Agent status line switches.**  When `task` succeeds, OpenCode's UI shows the sub-agent's name as the active agent while it runs.  If the status line stays on the manager, delegate was not called.
2. **Identity banner in the trace.**  Every sub-agent is required to print a unique banner as its first line:
   ```
   === <agent-name> v1 — acknowledging target: <X> ===
   ```
   The banner text is defined inside the sub-agent's own prompt, which the manager cannot see.  If you see the banner, the real sub-agent ran.  If you see a manager-formatted "Delegation to X" markdown block but no banner, the manager hallucinated.
3. **Artifact check.**  Each sub-agent writes to a fixed path:
   - research → `.opencode/docs/<target>_design.md` + `.opencode/plans/<target>_plan.md`
   - plan-reviewer → `.opencode/reviews/<artifact>_plan_review.md`
   - coder → `.opencode/patches/<artifact>.patch` + actual source edits
   - code-reviewer → `.opencode/reviews/<artifact>_code_review.md`
   - tester → `.opencode/bench/<artifact>_validation.md`
   
   `ls` the expected path before and after a delegate step.  No new file = nothing real ran.
4. **Tool-call trace.**  OpenCode's debug view shows `tool_call: task({subagent_type: "...", ...})` entries.  Zero task calls = zero real sub-agents.

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
2. Re-read `.opencode/skills/stage-gate-enforcement/SKILL.md` to confirm delegation rules.
3. Verify you have not skipped any mandatory gate.
4. Verify you are producing artifacts in the correct `.opencode/` subdirectory.