---
description: Run or resume an explicit bounded batch of approved Evolution candidate IDs
agent: coordinator
subtask: false
---

@coordinator @.opencode/agents/coordinator.md

Evolution batch request: $ARGUMENTS

Syntax: `/evolve-batch <profile> <full|stage|plan|implement|review|validate> <candidate-id> [more-ids]`
or `/evolve-batch resume <batch-id>`.
Only one to ten explicitly selected IDs; no implicit queue-wide run. Bind all tools
to one configured connection through evolution_discovery_profiles. Use actual
MCP-prefixed tool names. Load the batch and execution Skills below, then read the
archived method snapshot for a resumed batch. Inspect existing workers before
resuming; do not duplicate an active child. Status uses evolution_read with kind
execution_batch. Creation and next follow evolution-batch. Each item follows the
existing evolution-execution role handoffs, permissions, scope and evidence gates.
Forward the handoff's frozen execution Skill version to all child roles. On retry
an existing isolated candidate may retain an older execution method than this
batch; record both references and use the handoff version for that candidate.
No automatic owner/curator action, merge, push or external messages.

Skill packs:
- @.opencode/skills/infra/agent-core/SKILL.md
- @.opencode/skills/infra/language-config/SKILL.md
- @.opencode/skills/infra/pipeline/evolution-batch/SKILL.md
- @.opencode/skills/infra/pipeline/evolution-execution/SKILL.md

Config:
- @.opencode/config.yaml
