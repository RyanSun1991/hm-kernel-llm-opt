---
name: delegate
description: >-
  Defines the delegate mechanism for hub-and-spoke orchestration — maps conceptual
  delegation to the available task(subagent_type=...) tool, standardizes the handoff
  packet format inside tool calls, and documents the gating via
  permission: skill: "delegate": "allow"/"deny".
  **Hard rule: ONLY the pipeline hub may delegate — coordinator (new chain) or
  hm-opt-manager (legacy chain); every other agent MUST NOT.**
depends_on:
  - handoff-contract
  - stage-gate-enforcement
---

# Delegate Skill

## Purpose

The pipeline spec describes a `delegate(agent, task, context)` primitive. This skill bridges the gap when the OpenCode runtime does **not** expose a native `delegate` tool. It defines a **drop-in equivalent** using the available `task(subagent_type=...)` tool while preserving the same semantic contract: spawn a sub-agent, hand off a complete packet, and receive results back.

The gating mechanism for which agents may delegate is the `permission: skill:` block in each agent's front-matter under `.opencode/agents/`. **Only the pipeline hub has `"delegate": "allow"` — `coordinator` on the new workbench chain (the default since M4) and `hm-opt-manager` (`agents/legacy/`) on the legacy fallback chain; all other agents have `"delegate": "deny"`. You MUST NEVER delegate unless you are one of those two hub agents.** (Workbench roles' one-shot `consult` via `task: ask` is a different, user-confirmed verb — see agent-core §4 — not pipeline delegation.)

| Front-matter permission | Meaning |
|---|---|
| `permission: skill: "delegate": "allow"` | Agent MAY delegate using `task(subagent_type=...)` |
| `permission: skill: "delegate": "deny"` | Agent MUST NOT delegate; always returns results to the caller |

## Available Tools

The `delegate` tool from the pipeline spec is **not exposed as an MCP tool** in `opencode.jsonc`. The `task()` tool is the authorized substitute for spawning sub-agents. Only the hub agent of the active chain may use it for stage delegation.

## Mapping: Stage → subagent_type

Every agent under `.opencode/agents/` (including subdirectories such as `legacy/`) is registered with OpenCode and can be spawned via `task(subagent_type=NAME)` where `NAME` is the agent's `name` field from its front-matter. Use the cast that matches the chain you are running:

| Stage | New chain (default) `subagent_type` | Legacy chain `subagent_type` |
|---|---|---|
| research — generic | `researcher` | `kernel-source-research` |
| research — reclaim | `researcher` (brief names `domain-reclaim`) | `memmgr-reclaim-research` |
| research — hyperhold/IO | `researcher` (brief names `domain-hyperhold-io`) | `hyperhold-io-opt` |
| research — sync/mech | `researcher` (brief names `domain-sync`) | `basic-mechanism-sync-opt` |
| research — workqueue | `researcher` (brief names `domain-workqueue`) | `wq-threadpool-opt` |
| plan review (GATE) | `reviewer` (plan-review brief) | `kernel-plan-reviewer` |
| implementation | `implementer` | `kernel-code-agent` |
| code review (GATE) | `reviewer` (code-review brief) | `kernel-code-reviewer` |
| tester A/B | `validator` | `kernel-tester-agent` |

Do not mix casts within one run. (`kernel-reviewer` is a deprecated legacy alias file — never target it; use `kernel-code-reviewer` on the legacy chain.)

## How to Delegate (Canonical Form — hub agent only)

Use the `task` tool with these conventions (new-chain example):

```
task(
  subagent_type="researcher",               # MUST match an agent name from the table above
  description="3-5 word description",        # e.g. "research sched_indicator paths"
  prompt="""FULL TASK STATEMENT HERE

## Handoff Packet
- **Current Stage**: research
- **Domain pack to apply**: domain-reclaim   (new chain: name the pack the command inlined)
- **Target**: <target path or subsystem>
- **Primary Metric**: <Stage-0 class metric; instruction count by default>
- **Hot path**: <specific call chain with file:line>
- **Evidence baseline**: <how the current waste was identified>
- **Files in scope**: <exact file paths>
- **Functions in scope**: <exact symbol names>
- **Risks**: <correctness, locking, lifetime risks>
- **Open questions**: <any unresolved questions>

## Required Reading
- `.opencode/docs/<component>_design.md` (if exists)
- `.opencode/memory/targets/<target>.md` (if exists)
- `.opencode/plans/<prior>_plan.md` (if iterative mode)

## Required Outputs
- `.opencode/docs/<artifact_slug>_design.md`
- `.opencode/plans/<artifact_slug>_plan.md`

## Required Next Action
After completing research, return the handoff packet to the hub.
""",
  command="<the original command that triggered this>",
)
```

## The Handoff Packet Must Be Inside the Prompt

Critical rule from the pipeline docs: **"The handoff packet goes inside the tool call's arguments, not as a user-facing message."** The `task` tool's `prompt` parameter is the equivalent container. Every delegation call's `prompt` MUST include:

1. **Pipeline context**: which stage the sub-agent is in, who sent them, who comes next
2. **Target and scope**: exact subsystem, file, or function (new chain: plus the domain pack to apply)
3. **Primary metric**: the Stage-0 class metric (instruction count by default)
4. **Evidence and baseline**: what is already known, what artifacts to read
5. **Required outputs**: what files the sub-agent must write
6. **Termination rule**: `"After completing your work, return the full handoff packet to the hub."`

## Receiving Results (hub agent only)

The `task` tool returns a single result message. When the hub receives results from a sub-agent:

1. **Read the artifacts** it produced from `.opencode/` (design doc, plan, review, patch, etc.)
2. **Check stage-gate conditions** — does the required artifact exist? Does the review say `approve`?
3. **Update `.opencode/state/current_task.json`** — advance `current_stage`, record `gates_passed`, set `pending_action.next_agent`
4. **Delegate to the next stage** via another `task(subagent_type=...)` call with the accumulated handoff packet

## Sub-Agent Constraint

Only the active chain's hub may use this skill — enforced by the harness via `permission: skill: "delegate": "allow"/"deny"` in each agent's front-matter. Sub-agents complete their stage, write their artifacts, return the packet, and stop.

## Example: Full Pipeline Delegation Sequence (new chain)

```
# Stage 1: Intake → Research
task(subagent_type="researcher", prompt="...handoff packet (names domain pack)...")

  # sub-agent returns
  → Read .opencode/docs/<slug>_design.md
  → Read .opencode/plans/<slug>_plan.md
  → Update current_task.json: current_stage="plan_review"

# Stage 2: Research → Plan Review
task(subagent_type="reviewer", prompt="...plan-review brief...")

  # sub-agent returns
  → Read .opencode/reviews/<slug>_plan_review.md
  → Check decision; if approve → advance
  → Update current_task.json: gates_passed += "plan_review:iter1"

# Stage 3: Plan Review → Implementation
task(subagent_type="implementer", prompt="...handoff packet...")

  # etc. — legacy chain: same sequence with the legacy cast names
```

## When task() Is Unavailable

If the `task` tool itself is unavailable (e.g., the runtime only exposes basic file/MCP tools), use the `general` subagent type from `task` if available, or fall back to reading/writing files directly and producing a summary message. The pipeline cannot be auto-driven without some sub-agent spawning mechanism — report the gap to the user rather than faking delegation.
