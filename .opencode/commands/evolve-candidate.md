---
description: Run or inspect an Evolution candidate by ID or task file, with full or bounded stage scope
agent: coordinator
subtask: false
---

@coordinator @.opencode/agents/coordinator.md

Profile: evolution_stage
Evolution request: $ARGUMENTS
Objective: Run exactly the requested scope of one Evolution candidate through service-approved stages using the task-local workspace and independent workbench roles.

Syntax: `/evolve-candidate <candidate-id|absolute task.json> [full|stage|status|research|plan|implement|review|validate]`.
Quote a task path containing spaces. The scope defaults to `full`; a path-only
invocation remains supported. `stage` executes only the currently permitted service
stage. A named scope executes only that step (the `plan` scope includes its research
prerequisite and independent plan review). `status` is read-only. Apply the scope
table and stop conditions in evolution-execution before dispatching any role.

Tool identifiers below are registered MCP base names. Use the actual tool name
exposed by this OpenCode session, including its server prefix. Bind all calls to
one configured connection and its `evolution_discovery_profiles().store_root`;
if multiple connections expose these tools and the selected store is ambiguous,
stop for the connection to be identified. Never mix state between connections.

Resolve the explicit candidate ID through `evolution_show`; for a task file, read
the candidate ID from its Evolution binding and verify it against that same store.
An ID must exactly identify one stored candidate. If the input is missing, unknown,
ambiguous, or has an unsupported scope, report the input requirement. Do not select
a previous dispatch, the first candidate, or the optimization singleton state.
If owner confirmation is missing, show the evidence and the operator CLI action,
then stop. Invoking this command does not perform owner or curator decisions.

Apply evolution-execution for this recipe's state and gate protocol. A coordinator
dispatches the existing researcher/architect/reviewer/implementer/validator roles through
task(). It does not perform their work or expand their permissions. The explicit
recipe authorizes bounded continuation across accepted service gates; individual
runtime permissions and owner/curator decisions remain in effect.
Fresh child sessions need explicit target-repository and Workbench roots, selected
skill references, task/handoff paths and the task's submission schema in their
brief. Do not assume they inherit this command's expanded context.

Skill packs:
- @.opencode/skills/infra/agent-core/SKILL.md
- @.opencode/skills/infra/language-config/SKILL.md
- @.opencode/skills/infra/pipeline/evolution-execution/SKILL.md
- @.opencode/skills/role/research-discipline/SKILL.md
- @.opencode/skills/role/plan-funnel/SKILL.md
- @.opencode/skills/role/review-checklists/SKILL.md
- @.opencode/skills/role/implementation-guardrails/SKILL.md
- @.opencode/skills/role/validation-flight-check/SKILL.md

Config:
- @.opencode/config.yaml
