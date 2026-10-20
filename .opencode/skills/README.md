# Skill Library — 3 tiers + one registry

Skills are reusable capability packs. Since the M1 reorg (Agent Workbench design §5)
they live in three tiers, and `_registry.yaml` is the single index — there are no
per-skill sidecar files.

```
skills/
  _registry.yaml     # the single index: name / tier / roles / applies_when / not_for / risk / cost
  role/              # domain-free discipline, one per responsibility
                     #   research-discipline · plan-funnel · review-checklists ·
                     #   implementation-guardrails · validation-flight-check
  scenario/          # problem-space packs
    kernel-opt/      #   kernel performance optimization: perf-bottleneck-playbooks ·
                     #   instruction-count-first · memory-tlb-optimization ·
                     #   optimization-funnel · iterative-optimization · build-and-sign ·
                     #   flash-device-operations · ab-test-comparison[-lmbench] ·
                     #   domain-{reclaim,hyperhold-io,workqueue,sync} packs
    kernel-understand/  # explanation-only work (forbids optimization framing)
    bug-fix/            # correctness-only work
  infra/             # cross-cutting contracts and bridges
                     #   agent-core (the base contract every role loads) ·
                     #   language-config · team-memory · hub-bridge ·
                     #   human-interaction-memory · memory-accumulation
    pipeline/        # pipeline-recipe process machinery — loaded by the pipeline lane
                     #   only: stage-gate-enforcement · handoff-contract ·
                     #   recipe-execution (the coordinator's manual) · delegate
```

## How loading works (two lanes)

- **Pipeline lane** (`/optimize_*`, `/research`, `/plan`, …): the command file's
  `Skill packs:` section lists exact `@.opencode/skills/<tier>/<name>/SKILL.md`
  paths and OpenCode inlines them at launch. Sub-agents must NOT re-Read skill
  files at runtime (see `.opencode/CLAUDE.md`).
- **Workbench lane** (roles, from M2): a role reads `_registry.yaml` once
  (metadata only, ~80 tokens/skill), matches the brief against `applies_when` /
  `not_for`, suggests ≤3 skills with reasons, and reads a SKILL.md full text only
  after the user confirms. Profiles preload a fixed list and skip the suggestion
  step.

## Adding a skill

1. Create `<tier>/<name>/SKILL.md` with `name:` + `description:` frontmatter
   (`name` must equal the directory name).
2. Add one entry to `_registry.yaml` — same `name`, correct `tier`, honest
   `applies_when` / `not_for` triggers, `roles` affinity, and `risk` tag
   (R0 read-only · R1 doc-write · R2 source/build · R3 device/publish).
3. Keep SKILL.md ≤500 lines; split details into `references/` inside the skill
   directory when it grows past that.
4. Run `python scripts/lint_skill_registry.py` — it must pass before commit.

No role file is edited to add a skill: roles discover skills through the
registry (dynamic suggestions) or through profiles (static preloads).
