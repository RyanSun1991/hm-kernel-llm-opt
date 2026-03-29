# Harness Engineer System

This document defines the upgraded `.opencode` harness engineer system for this repository.

## Primary Objective

The default optimization objective is to reduce instruction count on the hot path while preserving correctness, locking guarantees, memory safety, lifecycle safety, and logical completeness.

## Agent Topology

1. `kernel-pipeline-starter`
2. `os-opt-manager`
3. research specialist
4. `kernel-plan-reviewer`
5. `kernel-code-agent`
6. `kernel-tester-agent`
7. `kernel-code-reviewer`

## Stage Order

1. intake and staging
2. routing and scope confirmation
3. research and instruction-count hypothesis
4. plan review
5. implementation
6. build and auto-test validation
7. code review
8. decision, memory update, and next-step routing

## Role Summary

### Research Specialist

- understand subsystem structure
- locate hot path
- explain where instruction count is spent
- produce optimization plan and design context

### Plan Reviewer

- review the proposed optimization plan before coding
- challenge whether the plan can plausibly reduce instruction count
- reject plans that are vague, unmeasurable, or correctness-risky

### Coder

- implement only approved plans
- keep changes minimal and measurable
- prepare review and tester handoff notes

### Code Reviewer

- review code quality and risk only
- examine instruction-count tradeoffs, deadlocks, memory leaks, logical completeness, and other regressions
- do not replace the tester role

### Tester

- trigger Build MCP and Auto-Test MCP
- collect validation artifacts
- compare instruction-count outcome directly or through approved proxies
- decide whether the patch is validated, inconclusive, or failed
- hand the result to the code reviewer

## Communication Contract

Every stage must produce a handoff packet that includes:

- target and primary metric
- evidence baseline
- hot path
- files and functions in scope
- risks and open questions
- exact next action for the receiving agent

## Canonical Handoffs

### Research -> Plan Review

The researcher must hand over:

- subsystem boundary
- hot path
- instruction-count thesis
- exact files and symbols
- risk notes
- proposed validation path

### Plan Review -> Implementation

The plan reviewer must state:

- whether the plan is approved
- whether the instruction-count thesis is credible
- which risks must be preserved against
- what implementation and tester must later confirm

### Implementation -> Tester

The code agent must state:

- exact changed files
- exact changed symbols
- expected hot-path win
- expected build path
- expected auto-test path
- open correctness or concurrency risks

### Tester -> Code Review

The tester must state:

- build result
- auto-test result
- runtime evidence, if any
- whether the instruction-count thesis still looks plausible
- missing validation

### Code Review -> Manager / Human

The code reviewer must state:

- final code review decision
- instruction-count review result
- residual correctness and performance risks
- readiness for final decision

## Standard Outputs

- design docs: `.opencode/docs/*.md`
- plans: `.opencode/plans/*.md`
- plan review: `.opencode/reviews/*_plan_review.md`
- code review: `.opencode/reviews/*_code_review.md`
- validation: `.opencode/bench/*_validation.md`
- memory updates: `.opencode/memory/*.md`
