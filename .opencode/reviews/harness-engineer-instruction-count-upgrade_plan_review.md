# Harness Engineer Instruction-Count Upgrade Plan Review

## Target Artifact

`.opencode/plans/harness-engineer-instruction-count-upgrade_plan.md`

## Review Type

Plan review

## Decision

approve

## Findings

- The proposed topology cleanly separates research, plan review, implementation, code review, and tester validation.
- The plan correctly makes instruction count the default primary optimization objective without pretending correctness or synchronization constraints are secondary.
- The implementation scope covers the real control-plane assets that need to change: agent prompts, skills, pipeline cards, staged prompt generation, and profile state.

## Risk Summary

- The harness remains prompt-driven, so stage discipline is enforced by workflow assets rather than hard runtime policy.
- Existing legacy docs and aliases may remain in the repository for compatibility, which can create naming drift if not documented clearly.

## Missing Validation

- Dry-run proof that the upgraded staged prompt is readable and complete in a live OpenCode session.
- A sample task that explicitly exercises manager -> plan reviewer -> coder -> code reviewer -> tester handoff.

## Instruction-Count Assessment

The plan is sound. It does not claim direct instruction-count wins in kernel code itself; instead it improves the quality and discipline of the harness so that future optimization plans are more likely to target and validate instruction-count reductions correctly.

## Required Follow-Up

- Update the workflow assets and staged prompt generation as proposed.
- Preserve a compatibility path for the legacy `kernel-reviewer` name while standardizing on `kernel-code-reviewer`.
- Record residual gaps after implementation.
