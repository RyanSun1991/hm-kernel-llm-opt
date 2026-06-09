# Harness Engineer Instruction-Count Upgrade Plan

## Objective

Upgrade the current `.opencode` multi-agent harness into a fuller harness engineer system with:

- instruction count as the default primary optimization objective
- a dedicated plan review gate between research and implementation
- a dedicated code review gate focused on patch quality and hidden regressions
- a dedicated tester role responsible for Build MCP and Auto-Test MCP validation
- explicit handoff contracts between all stages

## Current Gaps

1. research agents are not uniformly instruction-count-first
2. there is no dedicated plan reviewer gate before coding
3. the current reviewer role mixes plan review, code review, and validation expectations
4. build and auto-test responsibilities are spread across implementation and review prompts
5. handoff expectations exist implicitly but are not standardized

## Target Topology

1. `kernel-pipeline-starter`
2. `hm-opt-manager`
3. research specialist
4. `kernel-plan-reviewer`
5. `kernel-code-agent`
6. `kernel-code-reviewer`
7. `kernel-tester-agent`

## Target Stage Order

1. intake and staging
2. research and instruction-count hypothesis
3. plan authoring
4. plan review
5. implementation
6. code review
7. tester validation
8. final decision and memory update

## Implementation Scope

### A. Agent and skill assets

Update `.opencode/agents/` and `.opencode/skills/` so that:

- research and manager assets explicitly prioritize instruction count
- new roles exist for plan review, code review, and tester validation
- implementation no longer owns build and auto-test execution as a primary responsibility
- handoff packets are required and structured

### B. Pipeline and profile assets

Update:

- `.opencode/pipelines/*.md`
- `configs/pipeline_profiles.yaml`
- `src/hmopt/opencode/pipeline.py`

so that staged prompts and task state capture:

- primary metric
- dedicated reviewers
- tester role
- handoff contract
- stage-specific review and test status

### C. Harness documentation

Add or update `.opencode/docs/` documentation so that the workflow is easy to understand without reading every agent file.

## Expected Outputs

- updated `.opencode/agents/*`
- updated `.opencode/skills/*`
- updated `.opencode/pipelines/*`
- updated `configs/pipeline_profiles.yaml`
- updated `src/hmopt/opencode/pipeline.py`
- updated or added `.opencode/docs/harness_engineer_system.md`
- review note capturing remaining gaps and risks

## Acceptance Criteria

The upgrade is complete when:

1. every default pipeline clearly states instruction count as the primary optimization objective
2. implementation is blocked on plan review approval
3. tester validation is separated from implementation
4. code review is separated from plan review
5. task state captures reviewer and tester roles plus stage status
6. handoffs are explicitly described in durable artifacts
7. the default generic workflow is understandable from one doc plus the agent files
