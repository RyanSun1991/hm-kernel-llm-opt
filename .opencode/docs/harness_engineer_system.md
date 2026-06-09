# Harness Engineer System

This document defines the upgraded `.opencode` harness engineer system for this repository.

## Language Configuration

The workspace supports configurable session language. Read `.opencode/config.yaml` for the `language` field and load `.opencode/skills/language-config/SKILL.md` at session start. All agent dialogue, analysis prose, review verdicts, and documentation prose must follow the configured language. Code, commit messages, and technical identifiers remain in English. See the language-config skill for full rules.

## Primary Objective

The default optimization objective is to reduce instruction count on the hot path while preserving correctness, locking guarantees, memory safety, lifecycle safety, and logical completeness.

## Agent Topology

1. `hm-opt-manager` — **entry agent and central hub** (handles intake, routing, and stage chaining)
2. research specialist (domain-specific sub-agent)
3. `kernel-plan-reviewer` (sub-agent)
4. `kernel-code-agent` (sub-agent)
5. `kernel-code-reviewer` (sub-agent)
6. `kernel-tester-agent` (conditional sub-agent)
7. `kernel-pipeline-starter` — legacy alias, redirects to `hm-opt-manager`

## Stage Order

1. intake, config loading, and routing (`hm-opt-manager`)
2. research and instruction-count hypothesis (specialist, returns to manager)
3. plan review (returns to manager)
4. implementation (returns to manager)
5. code review (returns to manager)
6. conditional flash + A/B test validation: flash stock, test stock, flash feature, test feature, compare (returns to manager)
7. decision, memory update, and next-step routing (`hm-opt-manager`)

### Iterative Close-Loop Mode (Optional)

When a command carries `Auto-Iterate: N` (with N ≥ 2) and loads `.opencode/skills/iterative-optimization/SKILL.md`, stage 7 for a **pass** verdict does not end the session — the manager automatically starts pass K+1 on the same target, treating all prior-pass plans/patches as LANDED context. The researcher must propose orthogonal new mechanisms each pass. Iteration stops when N passes complete, the researcher returns `no_more_ideas`, two consecutive passes land within noise, or a failure hits the back-edge stall cap. See `iterative-optimization/SKILL.md` for the full protocol.

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
- prepare review handoff notes and optional tester context

### Code Reviewer

- review code quality and risk only
- examine instruction-count tradeoffs, deadlocks, memory leaks, logical completeness, and other regressions
- decide whether tester validation is required based on risk, environment, and validation preconditions

### Tester

- verify relay connectivity and device visibility via Flash MCP
- trigger Build MCP to verify stock and feature images exist
- execute A/B comparison: flash stock image, run test, flash feature image, run test
- use Flash MCP for device flashing and Auto-Test MCP for test execution
- compare instruction-count outcome (stock vs feature delta) directly or through approved proxies
- if flamegraph/hitrace/hiperf artifacts are available, perform differential analysis
- decide whether the patch is validated, inconclusive, or failed
- provide post-review A/B validation evidence when code review requests tester execution

## Communication Contract

Every stage must produce a handoff packet that includes:

- target and primary metric
- evidence baseline
- hot path
- files and functions in scope
- risks and open questions
- exact next action for the receiving agent

## Hub-and-Spoke Delegation Model

Only agents with `permission: skill: "delegate": "allow"` in their front-matter (currently only `hm-opt-manager`) may delegate. Load `.opencode/skills/delegate/SKILL.md` for the full mechanism — the `task(subagent_type=...)` tool is the delegation primitive.

**Sub-agents (specialists, reviewers, coder, tester) do NOT delegate.** They complete their work, write their artifacts, and return their handoff packet to the manager. The manager then reads the artifacts, checks stage-gate conditions, and delegates to the next stage.

Flow:
```
starter → manager → specialist → [returns to manager] → plan-reviewer → [returns to manager] → coder → [returns to manager] → code-reviewer → [returns to manager] → tester → [returns to manager] → decision
```

The manager MUST NOT stop and ask the user to continue between stages. When a sub-agent returns, the manager immediately proceeds to the next stage.

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
- what implementation must satisfy before code review

### Implementation -> Code Review

The code agent must state:

- exact changed files
- exact changed symbols
- expected hot-path win
- open correctness or concurrency risks
- suggested validation path if tester execution becomes necessary

### Code Review -> Tester (Conditional)

The code reviewer must state:

- whether tester is required, recommended, or skipped
- trigger conditions and scope for tester execution
- required build/auto-test/benchmark evidence
- risk hypotheses the tester should validate
- stock image path (baseline kernel without patches)
- feature image path (kernel with the optimization patch)
- device target for flash
- test case name and parameters

### Tester -> Manager / Human

The tester must state:

- stock flash result and stock test result
- feature flash result and feature test result
- instruction-count delta (stock vs feature)
- hot path changes and new hotspots (if any)
- flamegraph diff path (if available)
- whether the instruction-count thesis still looks plausible
- missing validation
- verdict: pass, fail, or inconclusive
- recommended next route: accept, iterate, or reject

### Code Review -> Manager / Human (When Tester Is Skipped)

The code reviewer must state:

- final code review decision
- instruction-count review result
- residual correctness and performance risks
- readiness for final decision and whether tester was skipped with reason

## Standard Outputs

- design docs: `.opencode/docs/*.md`
- plans: `.opencode/plans/*.md`
- plan review: `.opencode/reviews/*_plan_review.md`
- code review: `.opencode/reviews/*_code_review.md`
- validation: `.opencode/bench/*_validation.md`
- memory updates: `.opencode/memory/*.md`
