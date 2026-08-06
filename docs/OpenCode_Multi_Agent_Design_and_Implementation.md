# OpenCode-First Multi-Agent Design and Implementation

## 1. Scope

This document describes the current multi-agent system in this repository with **OpenCode as the primary control plane**.

It covers:

- the architecture
- implemented agents, presets, skills, and memory
- the relationship between OpenCode and HMOPT
- the current automation boundary

This document reflects the **implemented state of the repo**, not just a proposal.

## 2. Design Goal

The goal is to let a user point OpenCode at:

- a file
- a directory
- a subsystem
- or a hotspot-related target

and then start a **full staged analysis and optimization workflow** with:

- automatic routing
- research before optimization
- instruction-count-first optimization targeting by default
- ranked ideation
- approval-gated planning
- dedicated plan review before coding
- implementation handoff
- code review after implementation
- tester validation using Build MCP and Auto-Test MCP
- validation planning
- long-term memory accumulation

The default philosophy is:

1. OpenCode decides and coordinates.
2. MCP provides deep retrieval and external actions.
3. HMOPT provides execution, verification, and automation hooks.
4. Durable artifacts are saved under `.opencode/`.

## 3. System Architecture

The system is split into two planes.

### 3.1 OpenCode control plane

This is the primary user-facing layer.

Implemented under:

- [.opencode/agents](/mnt/d/work/hm-kernel-llm-opt/.opencode/agents)
- [.opencode/pipelines](/mnt/d/work/hm-kernel-llm-opt/.opencode/pipelines)
- [.opencode/skills](/mnt/d/work/hm-kernel-llm-opt/.opencode/skills)
- [.opencode/docs](/mnt/d/work/hm-kernel-llm-opt/.opencode/docs)
- [.opencode/memory](/mnt/d/work/hm-kernel-llm-opt/.opencode/memory)

Responsibilities:

- task intake
- target classification
- specialist routing
- research discipline
- idea ranking
- plan writing
- implementation handoff
- review coordination
- memory promotion

### 3.2 HMOPT execution plane

This is the repo’s Python automation layer.

Implemented under:

- [src/hmopt/agents](/mnt/d/work/hm-kernel-llm-opt/src/hmopt/agents)
- [src/hmopt/orchestration/graph.py](/mnt/d/work/hm-kernel-llm-opt/src/hmopt/orchestration/graph.py)
- [src/hmopt/cli.py](/mnt/d/work/hm-kernel-llm-opt/src/hmopt/cli.py)

Responsibilities:

- evidence generation
- patch proposal
- build and test verification
- profiling and evaluation
- report generation

OpenCode is the control plane. HMOPT is the execution plane.

## 4. Core Design Principles

### 4.1 Research before optimization

No optimization idea is considered valid until the system establishes:

- subsystem boundary
- likely hot path
- likely instruction-count-heavy path
- protected data
- ownership and lifecycle constraints
- dependency radius

### 4.2 Artifacts over chat-only reasoning

All meaningful outputs should persist to `.opencode/`.

### 4.3 Routing first, specialization second

The system starts from a generic entry and narrows to the correct specialist.

### 4.4 Approval-gated optimization

The optimizer ranks ideas, but only the top idea is shown and approved before planning proceeds.

### 4.5 Memory accumulation

The system is designed to improve over time by promoting reusable results into long-term memory.

## 5. Implemented OpenCode Agent Stack

### 5.1 Entry agent

- [kernel-pipeline-starter.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/agents/legacy/kernel-pipeline-starter.md)

Purpose:

- one-shot startup
- load preset card
- load skill packs
- load memory and bootstrap docs
- stage the task
- delegate to the manager

### 5.2 Manager

- [os-opt-manager.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/agents/os-opt-manager.md)

Purpose:

- route by target or semantics
- choose research-first flow
- enforce specialist startup protocol
- require memory usage and memory promotion

### 5.3 Research and optimizer specialists

- [kernel-source-research.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/agents/legacy/kernel-source-research.md)
- [memmgr-reclaim-research.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/agents/legacy/memmgr-reclaim-research.md)
- [hyperhold-io-opt.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/agents/legacy/hyperhold-io-opt.md)
- [basic-mechanism-sync-opt.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/agents/legacy/basic-mechanism-sync-opt.md)
- [wq-threadpool-opt.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/agents/legacy/wq-threadpool-opt.md)

Purpose:

- build subsystem understanding
- generate ranked optimization ideas
- write design docs
- write approved plans

### 5.4 Implementation and review

- [kernel-code-agent.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/agents/legacy/kernel-code-agent.md)
- [kernel-plan-reviewer.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/agents/legacy/kernel-plan-reviewer.md)
- [kernel-code-reviewer.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/agents/legacy/kernel-code-reviewer.md)
- [kernel-tester-agent.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/agents/legacy/kernel-tester-agent.md)

Purpose:

- turn approved plans into minimal patches
- review plan quality before coding
- review patch safety after coding
- execute build and auto-test validation
- enforce instruction-count-first validation depth
- promote stable lessons into memory

## 6. Implemented Pipeline Presets

Pipeline presets are OpenCode-facing workflow cards under [.opencode/pipelines](/mnt/d/work/hm-kernel-llm-opt/.opencode/pipelines).

### 6.1 Generic default

- [generic_full.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/pipelines/generic_full.md)

This is the recommended default for:

- arbitrary file targets
- arbitrary directories
- unknown subsystems
- “just optimize this path” style requests

It deliberately starts with:

- manager routing
- automatic specialist selection
- long-term memory loading

### 6.2 Domain-specific presets

- [hyperhold_full.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/pipelines/hyperhold_full.md)
- [memmgr_reclaim_full.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/pipelines/memmgr_reclaim_full.md)
- [workqueue_full.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/pipelines/workqueue_full.md)
- [sync_review.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/pipelines/sync_review.md)

These should be used when the target domain is already known and a narrower bias improves quality or speed.

## 7. Implemented Skill Packs

Skill packs are repo-local reusable instruction sets under [.opencode/skills](/mnt/d/work/hm-kernel-llm-opt/.opencode/skills).

Implemented packs:

- [instruction-count-first.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/skills/scenario/kernel-opt/instruction-count-first/SKILL.md)
- [research-discipline.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/skills/role/research-discipline/SKILL.md)
- [optimization-funnel.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/skills/scenario/kernel-opt/optimization-funnel/SKILL.md)
- [handoff-contract.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/skills/infra/pipeline/handoff-contract/SKILL.md)
- [implementation-guardrails.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/skills/role/implementation-guardrails/SKILL.md)
- [validation-flight-check.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/skills/role/validation-flight-check/SKILL.md)
- [memory-accumulation.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/skills/infra/memory-accumulation/SKILL.md)

These are loaded by presets and starter prompts to avoid repeating complex instructions in every user request.

## 8. Long-Term Memory Design

Long-term memory is implemented under [.opencode/memory](/mnt/d/work/hm-kernel-llm-opt/.opencode/memory/README.md).

Structure:

- `targets/`: per-target memory
- `subsystems/`: broader subsystem memory
- `global_lessons.md`: reusable optimization and validation lessons

Reference docs:

- [memory_system.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/docs/memory_system.md)
- [global_lessons.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/memory/global_lessons.md)
- [target_memory_template.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/memory/target_memory_template.md)

Current implemented behavior:

- the pipeline session generator infers memory file paths from the target
- those memory paths are injected into the generated OpenCode prompt
- manager and specialists are instructed to read and update memory

## 9. Session Staging and Prompt Generation

The implemented staging layer is in:

- [pipeline.py](/mnt/d/work/hm-kernel-llm-opt/src/hmopt/opencode/pipeline.py)
- [pipeline_profiles.yaml](/mnt/d/work/hm-kernel-llm-opt/configs/pipeline_profiles.yaml)

It does the following:

1. load a pipeline profile
2. create `.opencode/` workspace directories if needed
3. generate a task id
4. generate a fully expanded OpenCode prompt
5. infer memory files for the target
6. write:
   - [.opencode/state/current_task.json](/mnt/d/work/hm-kernel-llm-opt/.opencode/state/current_task.json)
   - [.opencode/state/current_prompt.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/state/current_prompt.md)

This is the core of the “one-click staging” behavior.

## 10. CLI and Launcher Implementation

### 10.1 CLI

Implemented in [cli.py](/mnt/d/work/hm-kernel-llm-opt/src/hmopt/cli.py).

Relevant commands:

- `list-pipeline-profiles`
- `start-pipeline`
- `resume-pipeline`
- `index-kernel`
- `analyze-artifacts`
- `optimize`

### 10.2 Shell launcher

Implemented in [run_opencode_pipeline.sh](/mnt/d/work/hm-kernel-llm-opt/scripts/run_opencode_pipeline.sh).

It can:

- start MCP services
- optionally build kernel index
- stage the pipeline prompt
- optionally launch `opencode` if available

## 11. MCP Integration Model

The OpenCode-first system relies on:

- Kernel Index MCP
- Sequential Thinking MCP
- Git MCP
- Build MCP
- Auto-Test MCP

Starter, manager, and specialists are instructed to use:

1. Sequential Thinking MCP first
2. Kernel Index MCP early
3. Git/Build/Auto-Test MCP as needed later

## 12. Relationship to HMOPT Runtime Agents

The OpenCode-first system does not replace HMOPT runtime agents.

Implemented runtime pieces include:

- prompt-driven `ConductorAgent`
- prompt-driven `CoderAgent`
- prompt-driven `TraceAnalystAgent`
- `ReviewerAgent`
- verification and profile loop in [graph.py](/mnt/d/work/hm-kernel-llm-opt/src/hmopt/orchestration/graph.py)

This gives a hybrid structure:

- OpenCode handles intent, routing, staged collaboration, and memory
- HMOPT handles automated execution and evaluation

## 13. Current Strengths

The current implementation already supports:

- OpenCode-first startup
- generic default pipeline
- domain-specific preset pipelines
- reusable skills
- long-term memory path injection
- staged task state
- review stage in runtime orchestration
- MCP-oriented analysis flow

## 14. Current Automation Boundary

The repo supports **one-click staging**, not yet full unattended OpenCode conversation execution.

Current behavior:

- CLI/script can generate the full prompt and state automatically
- CLI/script can launch `opencode` if the binary exists
- the user still typically pastes the staged prompt into OpenCode

This boundary is intentional because:

- optimization choice usually needs approval
- patch landing usually needs approval
- review findings may need human arbitration

## 15. Recommended Default Usage

For most tasks, the recommended entry is:

- profile: `generic_full`
- target: the file or directory you want optimized

Then allow the manager to route to the right specialist automatically.

## 16. Summary

The current OpenCode-first multi-agent system is a **generic staged orchestration framework** with:

- one-shot startup
- automatic routing
- reusable skill packs
- durable artifacts
- long-term memory
- HMOPT-backed execution hooks

Its default form is no longer a workqueue-specific template. The current default architecture is the generic pipeline plus specialist routing.
