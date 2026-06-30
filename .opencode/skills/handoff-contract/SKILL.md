---
name: handoff-contract
description: Non-negotiable handoff packet requirements for every stage transition — defines mandatory artifacts, delegation message structure, naming conventions, and receiving-agent verification rules.
---

# Handoff Contract

## Non-Negotiable Rule

Every agent MUST produce a written handoff packet before the next stage begins. No stage transition is valid without the mandatory artifacts listed below. If an agent cannot produce a required artifact, it MUST explicitly state the blocker and halt the pipeline — it MUST NOT silently skip the handoff.

## Enforcement

- The manager Reads the artifact only when its decision needs the details — not to "review" it inline.
- The receiving agent MUST verify that all mandatory artifacts from the previous stage exist and are non-empty before starting its own work.
- If any mandatory artifact is missing, the receiving agent MUST refuse to proceed and report the gap back to the manager.
- The manager MUST NOT route to the next stage until the current stage's handoff is complete.

---

## Stage 1: Manager

**Agent**: `hm-opt-manager`

**Mandatory deliverables**:

| Artifact | Location | Description |
|----------|----------|-------------|
| Task state | `.opencode/state/current_task.json` | Updated with profile, target, objective, pipeline_card, active_agent |
| Delegation message | (inline) | Fully expanded task statement with all context |

**Delegation message MUST include**:

- Profile and pipeline preset name
- Target path or subsystem
- Objective (primary metric)
- Full list of loaded skill packs (by path)
- Full list of loaded bootstrap docs (by path)
- Full list of loaded memory files (by path)
- Configured language from `config.yaml`
- Explicit instruction: "Run the COMPLETE pipeline through ALL stages — research, plan review, implementation, code review, and conditional tester validation. Do NOT stop after a single stage."

---

## Stage 2: Manager -> Research Specialist

**Agent**: `hm-opt-manager` -> research specialist (e.g., `kernel-source-research`, `memmgr-reclaim-research`, etc.)

**Mandatory delegation includes**:

- Target and subsystem scope
- Primary metric (instruction count unless overridden)
- Pipeline stage context: "You are in Stage 3 of 8. After your research, the manager will route your output to kernel-plan-reviewer."
- Required skill packs and docs to load
- Language setting
- Explicit instruction: "After completing your research, produce the handoff packet and signal the manager to proceed to plan review."

---

## Stage 3: Research Specialist -> Plan Reviewer

**Agent**: research specialist -> `kernel-plan-reviewer` (routed via manager)

**Mandatory deliverables**:

| Artifact | Location | Description |
|----------|----------|-------------|
| Design document | `.opencode/docs/[component]_design.md` | Subsystem structure, hot path, concurrency model, IC waste sources |
| Optimization plan | `.opencode/plans/[component]_optimization_plan.md` | Concrete proposal with IC hypothesis |
| Handoff packet | (inline or appended to plan) | See fields below |

**Handoff packet MUST include ALL of the following**:

1. **Target**: exact subsystem, directory, or file
2. **Hot path**: specific function call chain with file:line references
3. **Primary-metric hypothesis**: the bottleneck class + primary metric (per `perf-bottleneck-playbooks` Stage 0) and the win mechanism keyed to it — for `compute-bound`: which instructions are removed and why; for `memory-tlb-bound`: which TLB flushes/shootdowns/page-walks are cut; for `ipc-bound`: which round-trip is eliminated/batched. State `bottleneck_class:` explicitly so the reviewer and tester judge against the right metric.
4. **Baseline evidence**: how current IC waste was identified (MCP query, trace, code inspection)
5. **Files in scope**: exact file paths
6. **Functions and structs in scope**: exact symbol names
7. **Top risks**: correctness, locking, lifetime, memory, API, logic
8. **Rejected alternatives**: at least 2 alternatives considered and why they were dropped
9. **Validation path**: how the expected win will be confirmed or falsified
10. **Design doc path**: path to the design document produced
11. **Plan path**: path to the optimization plan produced
12. **Next action**: "Route to kernel-plan-reviewer for plan review"

**Failure mode**: If any of fields 1-12 is missing, plan reviewer MUST reject the handoff and request completion.

---

## Stage 4: Plan Reviewer -> Coder

**Agent**: `kernel-plan-reviewer` -> `kernel-code-agent` (routed via manager)

**Mandatory deliverables**:

| Artifact | Location | Description |
|----------|----------|-------------|
| Plan review | `.opencode/reviews/[component]_plan_review.md` | Formal review verdict |

**Review document MUST include ALL of the following**:

1. **Decision**: one of `APPROVED`, `NEEDS_REVISION`, or `REJECTED`
2. **IC assessment**: is the instruction-count hypothesis credible? (yes/no with reasoning)
3. **Must-keep semantics**: which behaviors, invariants, or guarantees the implementation MUST preserve
4. **Must-not-cross boundaries**: scope limits the coder MUST NOT exceed
5. **Expected IC mechanism**: the specific code transformation expected to reduce instructions
6. **Required validation steps**: what the coder must prepare for code review and potential tester
7. **Key risks**: risks the coder must watch for during implementation
8. **Revision notes**: if NEEDS_REVISION, exact changes required before re-review
9. **Next action**: if APPROVED → "Route to kernel-code-agent for implementation"; if NEEDS_REVISION → "Route back to research specialist for revision"; if REJECTED → "Pipeline halted — plan rejected"

**Gate rule**: Only `APPROVED` plans proceed to implementation. `NEEDS_REVISION` routes back to the researcher. `REJECTED` halts the pipeline with a summary to the user.

---

## Stage 5: Coder -> Code Reviewer

**Agent**: `kernel-code-agent` -> `kernel-code-reviewer` (routed via manager)

**Mandatory deliverables**:

| Artifact | Location | Description |
|----------|----------|-------------|
| Code changes | Kernel source files (in-repo edits or patch) | The actual implementation |
| Patch export (optional) | `.opencode/patches/[component].patch` | When explicit export is needed |
| Implementation summary | `.opencode/bench/[component]_after_patch.md` | Post-implementation handoff |

**Implementation summary MUST include ALL of the following**:

1. **Changed files**: exact file paths with line ranges
2. **Changed functions/symbols**: the authoritative list of function names whose body the patch edited, to be forwarded to the tester for per-function instruction-count comparison. Mirrors the `## Modified functions` section in `.opencode/bench/after_patch.md`. Use `none` if only macros / headers / Kconfig were touched.
3. **Hot path modified**: which part of the hot path was changed and how
4. **Expected IC reduction**: specific reasoning about instruction-count improvement
5. **Correctness argument**: why the change preserves semantics, locking, and lifetime guarantees
6. **Known tradeoffs**: any tradeoffs accepted (e.g., slight memory increase for IC win)
7. **Open risks**: risks the code reviewer should focus on
8. **Validation suggestions**: recommended build/test/profiling commands for tester
9. **Approved plan path**: reference to the plan this implements
10. **Plan review path**: reference to the plan review that approved it
11. **Next action**: "Route to kernel-code-reviewer for code review"

---

## Stage 6: Code Reviewer -> Tester (or Manager/User)

**Agent**: `kernel-code-reviewer` -> `kernel-tester-agent` OR manager/user (routed via manager)

**Mandatory deliverables**:

| Artifact | Location | Description |
|----------|----------|-------------|
| Code review | `.opencode/reviews/[component]_code_review.md` | Formal code review verdict |

**Code review document MUST include ALL of the following**:

1. **Decision**: one of `APPROVED`, `NEEDS_REVISION`, or `REJECTED`
2. **IC assessment**: does the patch likely reduce instruction count as claimed?
3. **Correctness findings**: any bugs, races, deadlocks, leaks, or logic gaps found
4. **Risk summary**: residual risks after review
5. **Scope compliance**: did the patch stay within the approved plan boundaries?
6. **Tester decision**: one of `REQUIRED`, `RECOMMENDED`, or `SKIPPED`
7. **Tester rationale**: why tester is required/recommended/skipped
8. **Tester scope** (if REQUIRED/RECOMMENDED): exact build commands, test targets, profiling instructions, and risk hypotheses to validate
9. **Test method** (if REQUIRED/RECOMMENDED): `lmbench-suite` (default) or `instruction-count` — which validation the tester runs. For `instruction-count`, also pass the `compare_level` + target names (`compare_process` / `compare_thread` / `compare_lib` / `compare_function`). For `lmbench-suite`, `compare_level` is ignored; the verdict is on the benchmark delta with a ~2% noise floor. See `ab-test-comparison-lmbench/SKILL.md`.
10. **Regression watch list** (if REQUIRED/RECOMMENDED): specific regressions the tester should look for
11. **Next action**: if tester REQUIRED → "Route to kernel-tester-agent for validation"; if tester SKIPPED → "Pipeline complete — report to user with final summary"

**Gate rule**: `NEEDS_REVISION` routes back to the coder. `REJECTED` halts with summary. `APPROVED` with tester REQUIRED routes to tester. `APPROVED` with tester SKIPPED completes the pipeline.

---

## Stage 7: Tester -> Manager/User

**Agent**: `kernel-tester-agent` -> manager/user

**Mandatory deliverables**:

| Artifact | Location | Description |
|----------|----------|-------------|
| Validation report | `.opencode/bench/[component]_validation.md` | Full validation results |

**Validation report MUST include ALL of the following**:

1. **Validation scope**: what was tested and why (state the `test_method` used — `lmbench-suite` or `instruction-count`)
2. **Build result**: PASS / FAIL / SKIPPED (with exact error if FAIL)
3. **Auto-test result**: PASS / FAIL / SKIPPED / NOT_APPLICABLE (with details)
4. **Trace or benchmark result**: collected evidence or NOT_AVAILABLE
   - compare result: level, target names, aggregate baseline / candidate / delta / delta_pct, pairs_compared, any missing pairs
   - per-pair breakdown for the cases that moved most
   - **per-modified-function compare rows** (one per function touched by the patch) with baseline / candidate / delta / delta_pct when the
   - tester was given the patch diff — see `ab-test-comparison.md` "Per-Modified-Function Comparison"
   - for `test_method: lmbench-suite`: the lmbench **digest** instead of the IC rows above — a per-benchmark-group table (stock vs feature, direction-aware `delta% / improvement%`), the HM-vs-Linux weighted-gap line, and discounted high-dispersion anomalies (see `ab-test-comparison-lmbench/SKILL.md`)
5. **IC outcome**: instruction-count thesis CONFIRMED / PLAUSIBLE / INCONCLUSIVE / DISPROVED (for `lmbench-suite`: report the benchmark outcome instead — IMPROVED / FLAT / REGRESSED)
6. **Correctness outcome**: no regressions / regressions found (with details)
7. **Missing validation**: what could not be validated and why
8. **Final decision**: one of `PASS`, `FAIL`, `INCONCLUSIVE`, `SKIPPED`
9. **recommended next route**: `accept` | `kernel-code-agent` | `kernel-source-research` | `iterate` | `reject` — see `kernel-tester-agent.md` → "Recommended Next Route" for which failure maps to which agent, and `os-opt-manager.md` → "Feedback Routing Table" for the manager-side rules

---

## Stage 8: Pipeline Completion

**Agent**: `hm-opt-manager` -> user

After the final stage (tester report or code-review-with-tester-skipped), the manager MUST produce a **pipeline summary** that includes:

1. **Target and objective**
2. **Pipeline stages executed**: list each stage with its verdict
3. **Artifacts produced**: paths to all documents, plans, reviews, patches, and validation reports
4. **IC outcome**: final instruction-count assessment
5. **Memory updates**: what was persisted to `.opencode/memory/`
6. **Open items**: any remaining risks or follow-up work
7. **Final status**: COMPLETED / COMPLETED_WITH_CAVEATS / HALTED

---

## Artifact Naming Convention

All artifacts MUST follow this naming pattern:

| Stage | Pattern | Example |
|-------|---------|---------|
| Design doc | `[component]_design.md` | `hp_iotab_design.md` |
| Optimization plan | `[component]_optimization_plan.md` | `hp_iotab_optimization_plan.md` |
| Plan review | `[component]_plan_review.md` | `hp_iotab_plan_review.md` |
| After-patch summary | `[component]_after_patch.md` | `hp_iotab_after_patch.md` |
| Code review | `[component]_code_review.md` | `hp_iotab_code_review.md` |
| Validation report | `[component]_validation.md` | `hp_iotab_validation.md` |
| Patch export | `[component].patch` | `hp_iotab.patch` |

The `[component]` token MUST be consistent across all artifacts for the same task. It is derived from the target path or the task ID.
