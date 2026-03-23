# OpenCode Multi-Agent Collaboration Workflow for JKernel Analysis and Optimization

## 1. Objective

This document defines a practical multi-agent collaboration scheme for analyzing and optimizing JKernel-style kernel code in this repository, using OpenCode as the interactive front end and HMOPT as the execution and evidence backbone.

The design is grounded in what already exists in the repo today:

- OpenCode-oriented specialist prompts in `agent/`
- HMOPT runtime agents in `src/hmopt/agents/`
- LangGraph orchestration in `src/hmopt/orchestration/graph.py`
- MCP services for kernel retrieval, sequential thinking, Git, build, and auto-test
- Existing memmgr reclaim exploration notes in `agent/memmgr_memory_reclaim_navigator.md`

The goal is not to create uncontrolled parallelism. The goal is to create a staged, auditable, artifact-driven multi-agent workflow that supports:

- deep code understanding before change proposals
- hotspot-driven optimization rather than intuition-only tuning
- explicit human approval gates
- reusable design artifacts under `.opencode/`
- clean handoff from analysis to patching to validation

## 2. Current Repository Baseline

### 2.1 Existing OpenCode-facing agent assets

The repo already contains OpenCode-style agent prompt drafts:

- `agent/os-opt-manager-full.md`
- `agent/kernel-source-research.md`
- `agent/wq-threadpool-opt.md`
- `agent/memmgr_memory_reclaim_navigator.md`

These files already establish the right direction:

- a manager agent for delegation
- a research-first kernel source analysis agent
- a stateful optimization agent with bad-plan memory
- a memmgr/reclaim bootstrap knowledge document

### 2.2 Existing HMOPT automated agents

The repo also already has Python agent implementations:

- `ConductorAgent`: decides whether to continue optimizing
- `CoderAgent`: generates a patch
- `VerifierAgent`: runs build and tests
- `ProfilerAgent`: collects profiling artifacts
- `TraceAnalystAgent`: interprets runtime hotspots and traces

This means the repo already has an execution-plane multi-agent loop. What is missing is a stronger OpenCode-facing control plane that can:

- route kernel-domain tasks to the correct specialists
- force research discipline before optimization
- define artifact contracts
- connect specialist findings to the HMOPT execution loop

### 2.3 Existing MCP surfaces

The repo already exposes the exact MCP services needed for a serious kernel optimization workflow:

- Kernel Index MCP: `scripts/run_mcp_server.sh`
- Sequential Thinking MCP: `scripts/run_seq_mcp_server.sh`
- Git MCP: `scripts/run_git_mcp_server.sh`
- Build MCP: `scripts/run_build_mcp_server.sh`
- Auto-Test MCP: `scripts/run_auto_test_mcp_server.sh`
- Combined startup: `scripts/run_all_mcp_servers.sh`

This is a major advantage. The workflow should be designed around these real tools instead of inventing new infrastructure.

## 3. Design Principles

The proposed scheme follows six hard rules.

### 3.1 Research before optimization

No optimization proposal is considered valid until the agent has established:

- subsystem boundary
- hot path or suspected hot path
- protected/shared data
- lifecycle constraints
- dependency radius

### 3.2 Staged collaboration, not free-form parallelism

Multiple agents can exist, but their outputs must be staged:

1. intake and routing
2. research and graph building
3. optimization ideation
4. plan authoring
5. implementation
6. verification
7. review

### 3.3 Artifact-first operation

Every phase must persist artifacts so future sessions do not rediscover the same structure.

### 3.4 Human approval gates

The human should approve at these points:

- start of heavy MCP indexing if desired by workflow policy
- selected top optimization idea
- implementation of a plan
- final merge/landing decision

### 3.5 Cross-check through a reviewer

A reviewer agent must validate:

- concurrency semantics
- lifecycle safety
- regression risk
- validation completeness

### 3.6 Reuse the existing HMOPT loop

OpenCode specialists should not replace HMOPT runtime agents. They should feed them:

- research docs
- ranked optimization ideas
- approved implementation plans
- validation intent

## 4. Two-Layer Multi-Agent Model

The recommended architecture uses two layers.

## 4.1 Layer A: OpenCode control plane

This is the human-facing, domain-aware layer. It runs in OpenCode and manages reasoning, routing, and planning.

Recommended agents:

- `os-opt-manager`
- `kernel-source-research`
- `memmgr-reclaim-research`
- `hyperhold-io-opt`
- `basic-mechanism-sync-opt`
- `wq-threadpool-opt`
- `kernel-code-agent`
- `kernel-reviewer`

## 4.2 Layer B: HMOPT execution plane

This is the automated loop already present in Python:

- `TraceAnalystAgent`
- `ConductorAgent`
- `CoderAgent`
- `VerifierAgent`
- `ProfilerAgent`

Recommended responsibility split:

- OpenCode specialists produce analysis, plans, and approval checkpoints.
- HMOPT automated agents execute patching, profiling, verification, and iterative refinement.

This keeps the system explainable to the user while still benefiting from an automated closed loop.

## 5. Recommended Agent Topology

## 5.1 `os-opt-manager`

### Role

Primary orchestration and routing agent.

### Responsibilities

- classify the user request
- route the request to the correct specialist
- decide whether the task starts with research or optimization
- enforce specialist startup protocol
- request reviewer pass after a plan or patch is ready
- maintain workflow discipline across sessions

### Required improvements over current draft

The current manager draft only routes basic workqueue and sync categories. It should be extended to cover the repo’s actual hotspot domains:

- `memmgr`
- `reclaim`
- `reclaim_async`
- `reclaim_sync`
- `page alloc`
- `vmpressure`
- `psi`
- `memview`
- `hyperhold`
- `zswap`
- `hpio`
- `iotab`
- `eid`
- `zsmalloc`
- `mutex`
- `rwlock`
- `refcount`
- `waiter`
- `race`

## 5.2 `kernel-source-research`

### Role

General kernel implementation research specialist.

### Responsibilities

- create or update `.opencode/docs/[component]_design.md`
- build subsystem boundary understanding
- map API, structs, hot path, and concurrency model
- generate architectural diagrams
- establish the shared technical baseline for later optimizers

### Recommended usage

Use this agent first when:

- the task is broad
- the subsystem is unfamiliar
- the user requests deep design understanding
- the manager cannot confidently route to a narrower specialist

## 5.3 `memmgr-reclaim-research`

### Role

Repo-specific specialist for `sysmgr/memmgr` reclaim, pressure, and allocation slow paths.

### Responsibilities

- bootstrap from `agent/memmgr_memory_reclaim_navigator.md`
- research reclaim entry points and reclaim-control flow
- analyze sync versus async reclaim
- map reclaim instances, watermark logic, PSI, vmpressure, and page allocator coupling
- maintain:
  - `.opencode/docs/memmgr-reclaim_bootstrap.md`
  - `.opencode/docs/memmgr-reclaim_design.md`
  - `.opencode/docs/memmgr-reclaim_trace.md`

### Why it should exist

This repo is clearly memmgr-heavy. The reclaim navigator already captures valuable structure. That knowledge should be promoted from an ad hoc note into a formal specialist bootstrap.

## 5.4 `hyperhold-io-opt`

### Role

Specialist optimizer for Hyperhold, swap, and I/O-heavy reclaim paths.

### Responsibilities

- focus on `sysmgr/memmgr/mem/swap/hyperhold/**`
- analyze hot data paths around:
  - `hpio`
  - `iotab`
  - `eid`
  - inflight tracking
  - serialization
  - compression and non-compression branches
  - radix or index structures
  - wait and lock behavior
- write:
  - `.opencode/docs/hyperhold_io_design.md`
  - `.opencode/plans/hyperhold-io-[component]_optimization_plan.md`

### Recommended trigger keywords

- `hyperhold`
- `zswap`
- `swap io`
- `hpio`
- `iotab`
- `eid`
- `compression`
- `zsmalloc`

## 5.5 `basic-mechanism-sync-opt`

### Role

Cross-cutting synchronization and state-machine optimization specialist.

### Responsibilities

- inspect lock scope and lock ownership
- analyze waiter queues and wakeup semantics
- check refcount and lifetime transitions
- identify contention amplification
- assess whether lock splitting, sharding, or state compression is safe

### Recommended trigger keywords

- `mutex`
- `rwlock`
- `futex`
- `semaphore`
- `condvar`
- `refcount`
- `race`
- `wait`
- `contention`

## 5.6 `wq-threadpool-opt`

### Role

Existing specialist for workqueue and thread-pool optimization.

### Responsibilities

- preserve the current five-ideas workflow
- keep `.opencode/state/.wq_opt_temp_ideas.json`
- keep `.opencode/state/wq-threadpool-opt-bad_plans.md`
- show only the top-ranked idea to the human
- write a detailed plan only after approval

### Recommendation

This agent already has the best stateful ideation pattern in the repo. That pattern should be generalized and reused by other optimization specialists.

## 5.7 `kernel-reviewer`

### Role

Independent reviewer for plans and patches.

### Responsibilities

- review `.opencode/plans/*.md`
- review patch scope and safety
- validate concurrency and lifetime assumptions
- check validation coverage
- write `.opencode/reviews/[artifact]_review.md`

### Review criteria

- correctness risk
- race windows
- lock-ordering issues
- lifetime or ownership leaks
- incomplete benchmark plan
- hidden cross-file coupling

## 5.8 `kernel-code-agent`

### Role

Implementation specialist for turning approved plans into minimal patches and validation-ready code changes.

### Responsibilities

- read approved plans and related design docs
- implement the minimum safe patch scope
- prepare build or runtime validation handoff
- export patch artifacts when needed

## 6. MCP Mapping by Agent

Each specialist should use the MCP services in a disciplined order.

## 6.1 Mandatory order for research and optimization specialists

1. `sequential_thinking`
2. Kernel Index MCP
3. local file reading
4. Git MCP if historical context is needed
5. Build MCP and Auto-Test MCP only after plan approval or patch generation

This order matters. It prevents agents from proposing changes before they have a coherent model of the code.

## 6.2 Kernel Index MCP usage

Primary tools already exposed by the repo:

- `kernel_index_code`
- `kernel_symbol_graph`
- `kernel_hotspot_context`

Recommended usage pattern:

- `kernel_index_code` for implementation understanding and patch planning
- `kernel_symbol_graph` for caller/callee and dependency radius
- `kernel_hotspot_context` when runtime hotspots are known from trace or profiling artifacts

## 6.3 Sequential Thinking MCP usage

Use `sequential_thinking` to structure:

- decomposition of the subsystem
- hypotheses about hotspot causes
- tradeoff analysis between candidate optimizations
- reviewer reasoning with explicit assumptions

This should become a hard requirement for research, optimizer, and reviewer agents.

## 6.4 Git MCP usage

Use Git MCP for:

- retrieving local diffs
- reviewing historical changes to a subsystem
- checking branch state during implementation and review

## 6.5 Build MCP usage

Use Build MCP when:

- an implementation plan is approved
- a patch needs reproducible build validation
- device/profile-specific build paths matter

## 6.6 Auto-Test MCP usage

Use Auto-Test MCP when:

- a patch affects runtime behavior on device
- the workload is phone-driven or UI-triggered
- the optimization claim requires target-device verification

The built-in `basic_swipe` case already provides a usable foundation for lightweight regression/perf checks.

## 7. Recommended Artifact Layout

The repo should standardize all OpenCode collaboration artifacts under `.opencode/`.

```text
.opencode/
  agents/
    os-opt-manager.md
    kernel-source-research.md
    memmgr-reclaim-research.md
    hyperhold-io-opt.md
    basic-mechanism-sync-opt.md
    wq-threadpool-opt.md
    kernel-code-agent.md
    kernel-reviewer.md

  docs/
    memmgr-reclaim_bootstrap.md
    memmgr-reclaim_design.md
    memmgr-reclaim_trace.md
    hyperhold_io_design.md
    zsmalloc_design.md

  state/
    current_task.json
    .temp_ideas.json
    bad_plans.md
    specialist_sessions.json

  plans/
    memmgr-reclaim-[topic]_optimization_plan.md
    hyperhold-io-[topic]_optimization_plan.md
    sync-[topic]_optimization_plan.md
    wq-threadpool-opt-[component]_optimization_plan.md

  reviews/
    [artifact]_review.md

  bench/
    baseline.md
    validation_matrix.md
    after_patch.md

  patches/
    0001-*.patch
```

This separates:

- research state
- temporary ideation state
- approved plans
- review output
- validation evidence
- generated patches

## 8. Standard Specialist Protocol

Every specialist agent should follow the same startup contract.

### 8.1 Required startup behavior

When delegated a task, the specialist should:

1. acknowledge the task
2. state the inferred subsystem and files of interest
3. wait for the human to authorize heavy MCP indexing if the workflow requires manual approval
4. begin with Sequential Thinking MCP
5. use Kernel Index MCP early
6. read existing `.opencode/docs/*` before proposing changes
7. write findings back to `.opencode/`

### 8.2 Recommended standard prompt block

Use this shared block in all specialist prompt files:

```text
Acknowledge receipt of the task.
State the inferred subsystem, probable hot path, and initial file scope.
If the workflow requires human approval for heavy indexing, wait for the HUMAN USER to authorize MCP indexing.
When authorized:
1. Use Sequential Thinking MCP first.
2. Use Kernel Index MCP early for implementation lookup, caller/callee analysis, dependency expansion, and hotspot context.
3. Read existing .opencode/docs/* documents before proposing changes.
4. Persist findings to .opencode/docs/, plans to .opencode/plans/, and review-ready summaries to .opencode/bench/ when applicable.
5. Do not propose optimization before identifying hot paths, protected data, ownership boundaries, and lifecycle constraints.
```

## 9. Recommended Staged Workflow

## 9.1 Phase 0: Intake and routing

`os-opt-manager` receives the user request and classifies it.

Routing policy:

- `memmgr`, `reclaim`, `page alloc`, `vmpressure`, `psi`, `memview`
  - route to `memmgr-reclaim-research`
- `hyperhold`, `zswap`, `hpio`, `iotab`, `eid`, `zsmalloc`
  - route to `hyperhold-io-opt`
- `mutex`, `rwlock`, `futex`, `refcount`, `race`, `wait`
  - route to `basic-mechanism-sync-opt`
- `workqueue`, `worker`, `thread pool`
  - route to `wq-threadpool-opt`
- broad or ambiguous tasks
  - route to `kernel-source-research` first

## 9.2 Phase 1: Research and baseline document

The research specialist:

- reads existing design docs
- reads repo-specific bootstrap docs
- performs sequential thinking
- uses Kernel Index MCP to build symbol and dependency context
- writes or updates a design document

No optimization ideas should be proposed in this phase unless the subsystem understanding is already mature and documented.

## 9.3 Phase 2: Hotspot evidence enrichment

If runtime evidence exists, the workflow should hand off to HMOPT execution-plane tooling:

- `ProfilerAgent` gathers or normalizes performance artifacts
- `TraceAnalystAgent` explains hotspot classes and suspicious code regions
- Kernel Index MCP enriches those hotspots with structural context

This phase converts “interesting code” into “measured optimization target”.

## 9.4 Phase 3: Optimization ideation

The optimizer specialist:

- reads design docs
- reads hotspot findings
- checks bad-plan memory
- generates exactly five candidate ideas
- drops ideas matching rejected patterns
- ranks valid ideas by impact versus risk
- presents only the top idea

This pattern is already proven in `wq-threadpool-opt` and should become the standard pattern for all optimizer agents.

## 9.5 Phase 4: Approval and detailed plan

After the human approves the top-ranked idea:

- the specialist writes a detailed plan under `.opencode/plans/`
- the plan must include:
  - exact files to touch
  - functions and structs involved
  - state or lock changes
  - expected instruction-path reduction
  - correctness risks
  - benchmark and validation plan

## 9.6 Phase 5: Implementation

Implementation should preferably be executed through `kernel-code-agent` and then, when needed, through the HMOPT execution plane:

- `kernel-code-agent` applies the approved plan at repository level
- `ConductorAgent` refines the next action
- `CoderAgent` generates a minimal patch
- Git MCP records and inspects diff state as needed

This keeps implementation disciplined and tied to an approved plan rather than free-form patching.

## 9.7 Phase 6: Verification

`VerifierAgent` validates build and tests.

Depending on the component, the workflow may additionally invoke:

- Build MCP for containerized target builds
- Auto-Test MCP for phone-driven runtime checks
- `ProfilerAgent` for before/after profiling

Validation outputs should be written into `.opencode/bench/`.

## 9.8 Phase 7: Review and loopback

`kernel-reviewer` reviews the plan or patch.

If review fails:

- record the concern
- feed the reason back into bad-plan memory or risk notes
- resume ideation with the next-best idea

If review passes:

- mark the plan as implementation-ready or landing-ready

## 10. State Management Model

The optimizer specialists should all use a common state pattern.

### 10.1 Required state files

- `.opencode/state/current_task.json`
- `.opencode/state/[specialist]_temp_ideas.json`
- `.opencode/state/[specialist]_bad_plans.md`

### 10.2 Required behavior

- pending ideas remain in temp state
- rejected ideas are appended to bad-plan memory
- approved ideas are removed from temp state and turned into formal plans
- bad-plan memory is reused across sessions

This avoids repetitive low-value proposals and gives the workflow real memory.

## 11. Integration with Existing HMOPT Orchestration

The current repo already has a LangGraph optimization loop. The proposed OpenCode multi-agent design should be treated as the front-end control plane for that loop, not as a competing orchestration system.

Recommended mapping:

- OpenCode `os-opt-manager`
  - decides specialist routing and workflow phase
- OpenCode research/optimizer specialists
  - produce design docs and approved plans
- HMOPT `TraceAnalystAgent`
  - converts trace artifacts into hotspot narratives
- HMOPT `ConductorAgent`
  - decides whether to continue iterative optimization
- HMOPT `CoderAgent`
  - produces patch diffs
- HMOPT `VerifierAgent`
  - runs build and tests
- HMOPT `ProfilerAgent`
  - gathers validation traces

This gives the repo a hybrid model:

- human-steered, domain-specific planning in OpenCode
- automated optimization execution in HMOPT

That is stronger than an OpenCode-only prompt setup and also stronger than a fully automated black-box loop.

## 12. Recommended Improvements to the Current Repo

## 12.1 First batch to implement

Create or formalize these OpenCode agents first:

- `os-opt-manager`
- `kernel-source-research`
- `memmgr-reclaim-research`
- `hyperhold-io-opt`
- `kernel-reviewer`

These five cover most of the repo’s likely kernel analysis and optimization work.

## 12.2 Second batch

Add:

- `basic-mechanism-sync-opt`
- upgraded `wq-threadpool-opt`

These complete the cross-cutting optimization coverage.

## 12.3 Promote reclaim navigator into bootstrap documentation

The existing `agent/memmgr_memory_reclaim_navigator.md` should become the basis for:

- `.opencode/docs/memmgr-reclaim_bootstrap.md`

All memmgr-related specialists should read it before new exploration.

## 12.4 Replace placeholder prompt files in `src/hmopt/agents/prompts/`

The current prompt markdown files for:

- `conductor.md`
- `coder.md`
- `trace_analyst.md`

are placeholders. They should be upgraded so HMOPT runtime agents consume the same artifact model as OpenCode specialists:

- design doc summary
- hotspot summary
- approved plan
- validation matrix

## 12.5 Standardize artifact contracts

Every approved task should produce:

- design doc
- findings or trace doc
- optimization plan
- validation plan
- review output

This standardization is more important than adding more agents.

## 13. Example User Flows

## 13.1 Broad memmgr analysis request

```text
@os-opt-manager
Analyze the reclaim and vmpressure interaction in sysmgr/memmgr.
Use the multi-agent workflow.
I want design understanding first, then hotspot-focused optimization ideas, then a reviewed plan.
Save all artifacts under .opencode/.
```

Expected flow:

1. manager routes to `memmgr-reclaim-research`
2. research doc is written
3. hotspot evidence is enriched if traces exist
4. optimizer proposes top idea
5. reviewer checks final plan

## 13.2 Hyperhold / hp_iotab optimization request

```text
@os-opt-manager
Analyze and optimize sysmgr/memmgr/mem/swap/hyperhold/hp_iotab.c and related Hyperhold code.
Use deep design understanding first, then MCP-based dependency analysis, then optimization ideation, then reviewed implementation planning.
Save all artifacts under .opencode/.
```

Expected flow:

1. manager routes to `hyperhold-io-opt`
2. specialist builds or updates `hyperhold_io_design.md`
3. specialist proposes five ideas internally and shows top one
4. approved plan is written
5. reviewer validates
6. HMOPT execution-plane agents implement and verify if requested

## 13.3 Synchronization risk second opinion

```text
@os-opt-manager
Review whether the planned reclaim optimization changes lock scope, waiter behavior, or refcount lifetime.
Use a sync-focused second opinion and produce a reviewer-ready risk note.
```

Expected flow:

1. manager routes to `basic-mechanism-sync-opt`
2. sync specialist produces a focused risk note
3. `kernel-reviewer` consumes that note in final review

## 14. Final Recommendation

The best path for this repo is not “many agents in parallel by default”. The best path is:

- one manager
- several domain specialists
- one reviewer
- a strict artifact model
- staged execution
- reuse of the existing HMOPT automated loop

In practical terms, the target architecture should be:

1. OpenCode manager routes the task.
2. Research specialist establishes subsystem understanding using Sequential Thinking MCP and Kernel Index MCP.
3. Optimizer specialist generates ranked ideas with bad-plan memory.
4. Human approves only the top idea.
5. Specialist writes a detailed implementation and validation plan.
6. Reviewer validates architecture and risk.
7. HMOPT agents execute patching, build/test, profiling, and iteration.

This design is tightly aligned with the repository as it exists now, improves the current `agent/` prompt drafts, and gives OpenCode a realistic multi-agent operating model for kernel analysis and optimization work.
