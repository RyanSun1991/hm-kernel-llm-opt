@os-opt-manager @.opencode/agents/os-opt-manager.md

Profile: generic_full @.opencode/pipelines/generic_full.md
Target: sysmgr/pwrmgr
Objective: Analyze and optimize this target using the full generic pipeline with automatic routing, research, implementation, review, validation, and memory updates.
Auto-Iterate: 1     # Set to N to auto-run N close-loop passes on clean verdicts; 1 = single pass (default). See iterative-optimization skill.

Skill packs:
- @.opencode/skills/instruction-count-first.md
- @.opencode/skills/research-discipline.md
- @.opencode/skills/optimization-funnel.md
- @.opencode/skills/handoff-contract.md
- @.opencode/skills/implementation-guardrails.md
- @.opencode/skills/validation-flight-check.md
- @.opencode/skills/memory-accumulation.md
- @.opencode/skills/iterative-optimization.md
- @.opencode/skills/language-config.md
- @.opencode/skills/stage_gate_enforcement.md
- @.opencode/skills/build-and-sign.md
- @.opencode/skills/flash-device-operations.md
- @.opencode/skills/ab-test-comparison.md

Memory packs:
- @.opencode/memory/global_lessons.md

Bootstrap docs:
- @.opencode/docs/harness_engineer_system.md

Config:
- @.opencode/config.yaml
