# OpenCode-First Multi-Agent Execution Guide

## 1. Purpose

This guide explains how to run the current OpenCode-first multi-agent pipeline in practice.

The intended default is:

- use `generic_full`
- point it at a file or directory
- let the manager route automatically
- use OpenCode as the primary driver

## 2. Before You Start

You should have:

1. the repo dependencies installed
2. a valid `configs/app.yaml`
3. MCP services available if you want the full experience

Important config areas:

- `project.repo_path`
- `indexing.clangd.compile_commands_dir`
- LLM connectivity

## 3. Main Entry Modes

There are three practical ways to start.

### 3.1 Recommended default: stage with CLI, then run in OpenCode

```bash
python3 -m hmopt.cli start-pipeline \
  --profile generic_full \
  --target sysmgr/memmgr/mem/swap/hyperhold/hp_iotab.c
```

This writes:

- [.opencode/state/current_task.json](/mnt/d/work/hm-kernel-llm-opt/.opencode/state/current_task.json)
- [.opencode/state/current_prompt.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/state/current_prompt.md)

Then open OpenCode and paste `current_prompt.md`.

### 3.2 Wrapper flow

```bash
bash scripts/run_opencode_pipeline.sh \
  --profile generic_full \
  --target sysmgr/memmgr/mem/swap/hyperhold/hp_iotab.c \
  --start-mcp
```

This can also:

- start MCP services
- optionally build index
- optionally launch `opencode`

### 3.3 Manual OpenCode entry

If you do not want staging first, enter OpenCode and start with:

```text
@os-opt-manager
Profile: generic_full
Target: sysmgr/memmgr/mem/swap/hyperhold/hp_iotab.c
Objective: Analyze and optimize this target using the full generic pipeline with automatic routing, implementation, review, validation, and memory updates.
```

This works, but the staged CLI flow is cleaner because it also records task state and memory references.

## 4. Which Profile Should You Use?

### 4.1 Use `generic_full` by default

Use it when:

- you only know the target path
- you are not sure which subsystem it belongs to
- you want the manager to classify and route automatically

### 4.2 Use a domain-specific profile when you already know the domain

- `hyperhold_full`
- `memmgr_reclaim_full`
- `workqueue_full`
- `sync_review`

## 5. Listing Available Profiles

```bash
python3 -m hmopt.cli list-pipeline-profiles
```

This shows:

- title
- description
- specialist hint
- preset card
- validation mode

## 6. Typical End-to-End Flow

### Step 1: Start MCP services

Either manually:

```bash
bash scripts/run_mcp_server.sh
bash scripts/run_seq_mcp_server.sh
bash scripts/run_git_mcp_server.sh
bash scripts/run_build_mcp_server.sh
bash scripts/run_auto_test_mcp_server.sh
```

or via wrapper:

```bash
bash scripts/run_opencode_pipeline.sh \
  --profile generic_full \
  --target path/to/target \
  --start-mcp
```

### Step 2: Build kernel index if needed

```bash
python3 -m hmopt.cli index-kernel \
  --config configs/app.yaml \
  --repo-path /path/to/kernel \
  --compile-commands-dir /path/to/kernel
```

### Step 3: Stage the pipeline

```bash
python3 -m hmopt.cli start-pipeline \
  --profile generic_full \
  --target path/to/file/or/dir
```

### Step 4: Open OpenCode

If you have an `opencode` binary and want the launcher to try opening it:

```bash
bash scripts/run_opencode_pipeline.sh \
  --profile generic_full \
  --target path/to/file/or/dir \
  --launch-opencode
```

Otherwise, open OpenCode yourself.

### Step 5: Paste the staged prompt

Paste the content of:

- [.opencode/state/current_prompt.md](/mnt/d/work/hm-kernel-llm-opt/.opencode/state/current_prompt.md)

This is the canonical prompt for the staged run.

### Step 6: Let the pipeline run in OpenCode

Expected flow:

1. `kernel-pipeline-starter`
2. `os-opt-manager`
3. selected specialist
4. `kernel-code-agent` if implementation is approved
5. `kernel-reviewer`

### Step 7: Inspect outputs

Primary output locations:

- [.opencode/docs](/mnt/d/work/hm-kernel-llm-opt/.opencode/docs)
- [.opencode/plans](/mnt/d/work/hm-kernel-llm-opt/.opencode/plans)
- [.opencode/reviews](/mnt/d/work/hm-kernel-llm-opt/.opencode/reviews)
- [.opencode/bench](/mnt/d/work/hm-kernel-llm-opt/.opencode/bench)
- [.opencode/memory](/mnt/d/work/hm-kernel-llm-opt/.opencode/memory)

## 7. Long-Term Memory During Execution

The generic pipeline includes memory by default.

When you stage a run, the session generator injects memory references such as:

- `.opencode/memory/targets/<target>.md`
- `.opencode/memory/global_lessons.md`

This means specialists should:

1. read existing memory first
2. do the run
3. promote stable reusable findings back into memory

Memory should contain:

- stable structural facts
- recurring hotspot patterns
- bad-plan history
- validation lessons

## 8. How to Resume

If a run is interrupted:

```bash
python3 -m hmopt.cli resume-pipeline
```

This shows:

- current task state
- current staged prompt
- current profile
- current target

## 9. How to Bring in Runtime Evidence

You can attach runtime artifacts when staging mentally in OpenCode, or use HMOPT CLI to analyze them directly.

Example:

```bash
python3 -m hmopt.cli analyze-artifacts \
  --config configs/app.yaml \
  --artifact flamegraph:outputs/flamegraph.json \
  --artifact hitrace:outputs/hitrace.json \
  --artifact hiperf:outputs/hiperf.json
```

For runtime-assisted patch loop:

```bash
python3 -m hmopt.cli analyze-artifacts \
  --config configs/app.yaml \
  --artifact flamegraph:outputs/flamegraph.json \
  --with-patch \
  --with-verify \
  --with-profile \
  --legacy-pipeline
```

OpenCode remains the control plane. HMOPT remains the execution plane.

## 10. Minimal Recommended Commands

If you only want the shortest practical path:

```bash
bash scripts/run_opencode_pipeline.sh \
  --profile generic_full \
  --target path/to/target \
  --start-mcp
```

Then paste:

- `.opencode/state/current_prompt.md`

into OpenCode.

## 11. Current Best Practice

For new work, use:

- profile: `generic_full`
- target: exact path you want optimized

Only switch to a domain-specific preset when you already know the problem belongs to a narrow domain and you want a stronger specialist bias.

## 12. Copy-Paste Startup Prompts and Use Cases

The following prompts can be copied directly into OpenCode.

### 12.1 Generic single-file optimization

Use when:

- you want the full pipeline for one specific file
- you do not want to decide the specialist manually

```text
@os-opt-manager
Profile: generic_full
Target: path/to/file.c
Objective: Full analysis and optimization pipeline for this file with automatic routing, implementation, review, validation, and memory updates.
```

### 12.2 Generic directory optimization

Use when:

- you want the manager to classify a whole directory or subsystem

```text
@os-opt-manager
Profile: generic_full
Target: path/to/directory
Objective: Full analysis and optimization pipeline for this directory with automatic routing, implementation, review, validation, and memory updates.
```

### 12.3 Research and plan only

Use when:

- you want design understanding and a reviewed plan
- you do not want implementation yet

```text
@os-opt-manager
Profile: generic_full
Target: path/to/target
Objective: Research this target deeply, build design understanding, generate ranked optimization ideas, and stop at the reviewed plan stage before implementation.
```

### 12.4 Hyperhold or swap-path optimization

Use when:

- the target clearly belongs to Hyperhold, swap I/O, hpio, iotab, or eid paths

```text
@os-opt-manager
Profile: hyperhold_full
Target: sysmgr/memmgr/mem/swap/hyperhold/hp_iotab.c
Objective: Full analysis and optimization pipeline for the Hyperhold path, including design understanding, ranked ideas, implementation, review, validation, and memory updates.
```

### 12.5 Memmgr reclaim optimization

Use when:

- the target is reclaim, allocator slow path, vmpressure, or PSI related

```text
@os-opt-manager
Profile: memmgr_reclaim_full
Target: sysmgr/memmgr/mem/reclaim
Objective: Full reclaim analysis and optimization pipeline, including trigger mapping, pressure analysis, ranked ideas, implementation, review, validation, and memory updates.
```

### 12.6 Workqueue or thread-pool optimization

Use when:

- the target clearly belongs to workqueue, dispatch, or worker-loop paths

```text
@os-opt-manager
Profile: workqueue_full
Target: kernel/workqueue.c
Objective: Full workqueue and thread-pool optimization pipeline with hotspot-guided research, ranked ideas, implementation, review, validation, and bad-plan filtering.
```

### 12.7 Synchronization or lifecycle review

Use when:

- you want correctness-first review around locking, waiters, refcount, or race windows

```text
@os-opt-manager
Profile: sync_review
Target: path/to/target
Objective: Review synchronization semantics, lock scope, waiter behavior, refcount lifetime, race windows, and regression risks for this target.
```

### 12.8 Generic optimization with runtime artifacts

Use when:

- you already have flamegraph, hitrace, or hiperf data

```text
@os-opt-manager
Profile: generic_full
Target: path/to/target
Objective: Full analysis and optimization pipeline for this target using runtime evidence, automatic routing, implementation, review, validation, and memory updates.
Artifacts:
- flamegraph: outputs/flamegraph.json
- hitrace: outputs/hitrace.json
- hiperf: outputs/hiperf.json
```

### 12.9 Review-only or no-code flow

Use when:

- you want analysis, review, and optimization direction
- you do not want code edits in this run

```text
@os-opt-manager
Profile: generic_full
Target: path/to/target
Objective: Analyze this target, build the design model, review existing code or plan quality, identify optimization directions, and stop before implementation.
```

### 12.10 Memory-heavy optimization run

Use when:

- you want the run to explicitly emphasize memory reuse and memory promotion

```text
@os-opt-manager
Profile: generic_full
Target: path/to/target
Objective: Full analysis and optimization pipeline for this target with automatic routing, implementation, review, validation, and explicit long-term memory promotion into target memory, subsystem memory, and global lessons.
```

## 13. Summary

The OpenCode-first execution model is:

1. stage with `start-pipeline` or `run_opencode_pipeline.sh`
2. use `generic_full` by default
3. paste the generated prompt into OpenCode
4. let the starter and manager route automatically
5. inspect `.opencode/` artifacts and memory
6. use HMOPT runtime commands when build, profiling, or artifact-driven execution is needed
