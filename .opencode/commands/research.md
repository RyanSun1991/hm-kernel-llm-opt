@researcher @.opencode/agents/researcher.md

Target: <REPLACE_WITH_FILE_DIRECTORY_SUBSYSTEM_OR_FUNCTION>
Scope hint (optional): <REPLACE_OR_DELETE — e.g. "focus on the reclaim fast path", "lifecycle of struct swp_slot">

Objective: Iteratively build a living design document at `.opencode/docs/<target_slug>_design.md` with a human expert in the loop. Every turn rebuilds state from disk (design doc + decision log + memory) and appends a new `## Research Iteration <N>` section; every human verdict is persisted to `.opencode/memory/human_decisions/<target_slug>.md` before the turn ends. Separate facts / inferences / hypotheses and cite `file:line` evidence for every fact. Do NOT propose optimizations (that is the architect's work via `/plan`), do NOT write plans or patches, do NOT delegate to other agents. Suggest matching scenario skills from the registry (≤3, with the trigger that matched) and wait for confirmation before loading them.

Legacy fallback: `@kernel-research` (agents/legacy/) remains available — until the live old-vs-new comparison is archived — if the new role underperforms on a target.

Skill packs:
- @.opencode/skills/infra/language-config/SKILL.md
- @.opencode/skills/infra/agent-core/SKILL.md
- @.opencode/skills/role/research-discipline/SKILL.md
- @.opencode/skills/infra/human-interaction-memory/SKILL.md
- @.opencode/skills/infra/hub-bridge/SKILL.md
- @.opencode/skills/infra/memory-accumulation/SKILL.md

Memory packs:
- @.opencode/memory/global_lessons.md

Bootstrap docs:
- @.opencode/docs/memory_system.md

Config:
- @.opencode/config.yaml
