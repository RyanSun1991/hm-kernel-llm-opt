# Harness Engineer Instruction-Count Upgrade Code Review

## Target Artifact

Updated `.opencode/` harness assets, `configs/pipeline_profiles.yaml`, and `src/hmopt/opencode/pipeline.py`

## Review Type

Code review

## Decision

approve

## Findings

- The new role split is coherent: `kernel-plan-reviewer`, `kernel-code-reviewer`, and `kernel-tester-agent` now have distinct ownership boundaries.
- The staged profile and prompt path now explicitly carry the primary goal and the new reviewer/tester role names.
- Research, implementation, review, and validation prompts now consistently prioritize instruction-count reduction while still guarding correctness, locking, lifecycle safety, and logic completeness.

## Risk Summary

- The new stage order is still enforced socially through `.opencode` assets and staged prompts; there is no hard execution engine preventing a user from skipping a stage manually.
- The repository still contains legacy references to `kernel-reviewer` in older docs, so compatibility messaging matters.

## Missing Validation

- `pytest` is not available in the current environment, so the existing `tests/test_opencode_pipeline.py` changes were validated through direct static review rather than the project test runner.
- A live OpenCode dry run with Build MCP and Auto-Test MCP was not executed in this task.

## Instruction-Count Assessment

The change is appropriate for the harness layer. It does not directly reduce runtime instruction count in kernel code, but it meaningfully increases the chance that future plans, reviews, and tester reports stay focused on instruction-count outcomes instead of drifting into generic performance discussion.

## Required Follow-Up

- Run an end-to-end dry run through the upgraded generic pipeline.
- Clean up remaining old-doc references over time so `kernel-code-reviewer` becomes the obvious default name everywhere.
- If desired, add stronger machine-readable stage status fields to `.opencode/state/current_task.json` in a later pass.
