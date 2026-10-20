@kernel-function-research @.opencode/agents/legacy/kernel-function-research.md

Target function: <REPLACE_WITH_FUNCTION_NAME>
Kernel file (optional, disambiguates same-name statics across TUs): <REPLACE_OR_DELETE>
Callee-graph depth (2–6, default 3): 3
Caller-graph depth (0–2, default 1): 1

Objective: Produce a complete design + implementation + multi-level callee-graph report for the named function. Explain-only — do NOT propose optimizations and do NOT hand off to the optimization pipeline. Write the artifact to `.opencode/docs/function_<sym>_detail.md` (with a `__<basename>` suffix if the symbol is a static duplicated across TUs).

Skill packs:
- @.opencode/skills/infra/language-config/SKILL.md
- @.opencode/skills/role/research-discipline/SKILL.md
- @.opencode/skills/infra/pipeline/handoff-contract/SKILL.md

Memory packs:
- @.opencode/memory/global_lessons.md

Config:
- @.opencode/config.yaml
