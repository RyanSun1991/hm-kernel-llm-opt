@kernel-research @.opencode/agents/kernel-research.md

Target: <REPLACE_WITH_FILE_DIRECTORY_SUBSYSTEM_OR_FUNCTION>
Scope hint (optional): <REPLACE_OR_DELETE — e.g. "focus on the reclaim fast path", "lifecycle of struct swp_slot">

Objective: Iteratively build a living design document at `.opencode/docs/<target_slug>_design.md` with a human expert in the loop. Every turn rebuilds state from disk (design doc + decision log + memory) and appends a new `## Research Iteration <N>` section; every human verdict is persisted to `.opencode/memory/human_decisions/<target_slug>.md` before the turn ends. Do NOT propose optimizations (that is `@kernel-plan`), do NOT write plans or patches, do NOT delegate to other agents.

Skill packs:
- @.opencode/skills/language-config/SKILL.md
- @.opencode/skills/research-discipline/SKILL.md
- @.opencode/skills/handoff-contract/SKILL.md
- @.opencode/skills/memory-accumulation/SKILL.md
- @.opencode/skills/human-interaction-memory/SKILL.md

Memory packs:
- @.opencode/memory/global_lessons.md

Bootstrap docs:
- @.opencode/docs/memory_system.md

Config:
- @.opencode/config.yaml
