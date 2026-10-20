@coordinator @.opencode/agents/coordinator.md

Profile: memmgr_reclaim_full @.opencode/pipelines/memmgr_reclaim_full.md
Target: sysmgr/memmgr/mem/reclaim
Objective: Deep reclaim and allocator-coupling analysis with instruction-count-first optimization. Focus on pressure signals, watermark reasoning, and page-allocator hot paths.
Auto-Iterate: 1     # Set to N to auto-run N close-loop passes on clean verdicts; 1 = single pass (default). See iterative-optimization skill.

Skill packs:
- @.opencode/skills/infra/agent-core/SKILL.md
- @.opencode/skills/scenario/kernel-opt/instruction-count-first/SKILL.md
- @.opencode/skills/role/research-discipline/SKILL.md
- @.opencode/skills/scenario/kernel-opt/domain-reclaim/SKILL.md
- @.opencode/skills/scenario/kernel-opt/optimization-funnel/SKILL.md
- @.opencode/skills/scenario/kernel-opt/perf-bottleneck-playbooks/SKILL.md
- @.opencode/skills/scenario/kernel-opt/memory-tlb-optimization/SKILL.md
- @.opencode/skills/infra/pipeline/handoff-contract/SKILL.md
- @.opencode/skills/role/implementation-guardrails/SKILL.md
- @.opencode/skills/role/validation-flight-check/SKILL.md
- @.opencode/skills/infra/memory-accumulation/SKILL.md
- @.opencode/skills/infra/hub-bridge/SKILL.md
- @.opencode/skills/scenario/kernel-opt/iterative-optimization/SKILL.md
- @.opencode/skills/infra/language-config/SKILL.md
- @.opencode/skills/infra/pipeline/stage-gate-enforcement/SKILL.md
- @.opencode/skills/scenario/kernel-opt/build-and-sign/SKILL.md
- @.opencode/skills/scenario/kernel-opt/flash-device-operations/SKILL.md
- @.opencode/skills/scenario/kernel-opt/ab-test-comparison/SKILL.md
- @.opencode/skills/scenario/kernel-opt/ab-test-comparison-lmbench/SKILL.md
- @.opencode/skills/infra/pipeline/recipe-execution/SKILL.md
- @.opencode/skills/infra/pipeline/delegate/SKILL.md
- @.opencode/skills/role/review-checklists/SKILL.md

Memory packs:
- @.opencode/memory/global_lessons.md

Bootstrap docs:
- @.opencode/docs/harness_engineer_system.md
- @.opencode/docs/memmgr-reclaim_bootstrap.md

Config:
- @.opencode/config.yaml
