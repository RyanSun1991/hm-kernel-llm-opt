---
name: delegate
description: Defines the delegate mechanism for hub-and-spoke orchestration — maps conceptual delegation to the available task(subagent_type=...) tool, standardizes the handoff packet format inside tool calls, and documents the gating via permission: skill: "delegate": "allow"/"deny". **Hard rule: ONLY hm-opt-manager may delegate; all other agents MUST NOT.**
depends_on:
  - handoff-contract
  - stage-gate-enforcement
---

# Delegate Skill

## Purpose

The pipeline spec describes a `delegate(agent, task, context)` primitive. This skill bridges the gap when the OpenCode runtime does **not** expose a native `delegate` tool. It defines a **drop-in equivalent** using the available `task(subagent_type=...)` tool while preserving the same semantic contract: spawn a sub-agent, hand off a complete packet, and receive results back.

The gating mechanism for which agents may delegate is the `permission: skill:` block in each agent's front-matter in `.opencode/agents/*.md`. **Only `hm-opt-manager` has `"delegate": "allow"`; all other agents have `"delegate": "deny"`. You MUST NEVER delegate unless you are `hm-opt-manager`.**

| Front-matter permission | Meaning |
|---|---|
| `permission: skill: "delegate": "allow"` | Agent MAY delegate using `task(subagent_type=...)` |
| `permission: skill: "delegate": "deny"` | Agent MUST NOT delegate; always returns results to the caller |

## Available Tools

The `delegate` tool from the pipeline spec is **not exposed as an MCP tool** in `opencode.jsonc`. The `task()` tool is the authorized substitute for spawning sub-agents. Only `hm-opt-manager` can load this skill (enforced via `permission: skill: "delegate": "allow"` in its front-matter).

## Mapping: Pipeline Agent Name → subagent_type

Every agent in `.opencode/agents/*.md` is registered with OpenCode and can be spawned via `task(subagent_type=NAME)` where `NAME` is the agent's `name` field from its front-matter.

| Pipeline role | agent file | `subagent_type` value |
|---|---|---|
| **Generic researcher** | `kernel-source-research.md` | `kernel-source-research` |
| Reclaim researcher | `memmgr-reclaim-research.md` | `memmgr-reclaim-research` |
| Hyperhold/IO researcher | `hyperhold-io-opt.md` | `hyperhold-io-opt` |
| Sync/mech researcher | `basic-mechanism-sync-opt.md` | `basic-mechanism-sync-opt` |
| Workqueue researcher | `wq-threadpool-opt.md` | `wq-threadpool-opt` |
| Plan reviewer | `kernel-plan-reviewer.md` | `kernel-plan-reviewer` |
| Code agent | `kernel-code-agent.md` | `kernel-code-agent` |
| Code reviewer | `kernel-code-reviewer.md` | `kernel-code-reviewer` |
| Tester | `kernel-tester-agent.md` | `kernel-tester-agent` |
| Legacy code reviewer | `kernel-reviewer.md` | `kernel-reviewer` |

## How to Delegate (Canonical Form)

Only `hm-opt-manager` may use this form. Use the `task` tool with these conventions:

```
task(
  subagent_type="kernel-source-research",   # MUST match agent name in .opencode/agents/
  description="3-5 word description",        # e.g. "research sched_indicator paths"
  prompt="""FULL TASK STATEMENT HERE

## Handoff Packet
- **Current Stage**: research
- **Target**: <target path or subsystem>
- **Primary Metric**: instruction count
- **Hot path**: <specific call chain with file:line>
- **Evidence baseline**: <how current IC waste was identified>
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
After completing research, return the handoff packet to the manager.
""",
  command="<the original command that triggered this>",
)
```

## The Handoff Packet Must Be Inside the Prompt

Critical rule from the pipeline docs: **"The handoff packet goes inside the tool call's arguments, not as a user-facing message."** The `task` tool's `prompt` parameter is the equivalent container. Every delegation call's `prompt` MUST include:

1. **Pipeline context**: which stage the sub-agent is in, who sent them, who comes next
2. **Target and scope**: exact subsystem, file, or function
3. **Primary metric**: instruction count (default) or override
4. **Evidence and baseline**: what is already known, what artifacts to read
5. **Required outputs**: what files the sub-agent must write
6. **Termination rule**: `"After completing your work, return the full handoff packet to the manager."`

## Receiving Results (hm-opt-manager only)

The `task` tool returns a single result message. When `hm-opt-manager` receives results from a sub-agent:

1. **Read the artifacts** it produced from `.opencode/` (design doc, plan, review, patch, etc.)
2. **Check stage-gate conditions** — does the required artifact exist? Does the review say `approve`?
3. **Update `.opencode/state/current_task.json`** — advance `current_stage`, record `gates_passed`, set `pending_action.next_agent`
4. **Delegate to the next stage** via another `task(subagent_type=...)` call with the accumulated handoff packet

## Sub-Agent Constraint

Only `hm-opt-manager` may use this skill — enforced by the harness via `permission: skill: "delegate": "allow"/"deny"` in each agent's front-matter.

## Example: Full Pipeline Delegation Sequence (hm-opt-manager only)

```
# Stage 1: Intake → Research
task(subagent_type="kernel-source-research", prompt="...handoff packet...")

  # sub-agent returns
  → Read .opencode/docs/<slug>_design.md
  → Read .opencode/plans/<slug>_plan.md
  → Update current_task.json: current_stage="plan_review"

# Stage 2: Research → Plan Review
task(subagent_type="kernel-plan-reviewer", prompt="...handoff packet...")

  # sub-agent returns
  → Read .opencode/reviews/<slug>_plan_review.md
  → Check decision; if approve → advance
  → Update current_task.json: gates_passed += "plan_review:iter1"

# Stage 3: Plan Review → Implementation
task(subagent_type="kernel-code-agent", prompt="...handoff packet...")

  # etc.
```

## When task() Is Unavailable

If the `task` tool itself is unavailable (e.g., the runtime only exposes basic file/MCP tools), use the `general` subagent type from `task` if available, or fall back to reading/writing files directly and producing a summary message. The pipeline cannot be auto-driven without some sub-agent spawning mechanism — report the gap to the user rather than faking delegation.
