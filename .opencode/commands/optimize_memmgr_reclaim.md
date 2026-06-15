@hm-opt-manager @.opencode/agents/hm-opt-manager.md

Profile: memmgr_reclaim_full @.opencode/pipelines/memmgr_reclaim_full.md
Target: sysmgr/memmgr/mem/reclaim
Objective: Deep reclaim and allocator-coupling analysis with instruction-count-first optimization. Focus on pressure signals, watermark reasoning, and page-allocator hot paths.
Auto-Iterate: 1     # Set to N to auto-run N close-loop passes on clean verdicts; 1 = single pass (default). See iterative-optimization skill.

Skill packs:
- @.opencode/skills/instruction-count-first/SKILL.md
- @.opencode/skills/research-discipline/SKILL.md
- @.opencode/skills/optimization-funnel/SKILL.md
- @.opencode/skills/handoff-contract/SKILL.md
- @.opencode/skills/implementation-guardrails/SKILL.md
- @.opencode/skills/validation-flight-check/SKILL.md
- @.opencode/skills/memory-accumulation/SKILL.md
- @.opencode/skills/hub-bridge/SKILL.md
- @.opencode/skills/iterative-optimization/SKILL.md
- @.opencode/skills/language-config/SKILL.md
- @.opencode/skills/stage-gate-enforcement/SKILL.md
- @.opencode/skills/build-and-sign/SKILL.md
- @.opencode/skills/flash-device-operations/SKILL.md
- @.opencode/skills/ab-test-comparison/SKILL.md

Memory packs:
- @.opencode/memory/global_lessons.md

Bootstrap docs:
- @.opencode/docs/harness_engineer_system.md
- @.opencode/docs/memmgr-reclaim_bootstrap.md

Config:
- @.opencode/config.yaml
