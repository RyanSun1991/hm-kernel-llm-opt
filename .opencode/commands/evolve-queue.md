---
description: Show candidate approvals, readiness and complete dossiers without executing work
agent: assistant
subtask: false
---

@assistant @.opencode/agents/assistant.md

Queue request: $ARGUMENTS

Syntax: `/evolve-queue <profile> [approved|pending|ready|completed|rejected|all]`.
Default approved. Bind to one configured connection via evolution_discovery_profiles.
Call evolution_candidates, show candidate ID, owner, approval IDs, stage and readiness
or reason. Respect next_offset/has_more, including pages with no ready items.
For a requested candidate detail use evolution_dossier and linked evidence hashes.
Use actual MCP tool names including the exposed prefix. Never interpret an old
receipt as permission to skip current gates. Do not select IDs or start work for the user.
Offer the concrete `/evolve-research` or `/evolve-candidate` / `/evolve-batch` next
step appropriate to the returned state. No role delegation in this read-only entry.

Skill packs:
- @.opencode/skills/infra/agent-core/SKILL.md
- @.opencode/skills/infra/language-config/SKILL.md
