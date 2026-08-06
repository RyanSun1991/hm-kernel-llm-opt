# .opencode Constitution

This file governs every OpenCode session in this repository. It is deliberately thin:
behavior contracts live in skills, responsibility in role files, task state in
workspaces. Canonical carrier: repo-root `AGENTS.md` (OpenCode convention) — this file
is its `.opencode/`-local mirror and the Claude Code compatibility fallback.

Two lanes share `.opencode/`:

## Lane 1 — Workbench (the default)

Everyday unit of work = **one role + a small selected skill set + a lightweight task
workspace**.

- **Default entry is `assistant`.** An ordinary prompt NEVER implicitly starts a
  pipeline; there is no stage a session is obliged to advance to.
- **The user owns routing.** Roles suggest next steps (handoff / consult / fork, with
  forwardable briefs); only the user triggers them. Agents never seize the
  conversation.
- **7 generic roles** live in `.opencode/agents/`: `assistant` · `researcher` ·
  `architect` · `implementer` · `reviewer` · `validator` · `coordinator`. Role prompts
  are domain-free; domain knowledge comes from `skills/scenario/` packs. Profiles in
  `agents/profiles/` are preloaded role+skill compositions.
- **Every role loads `.opencode/skills/infra/agent-core/SKILL.md`** — the base
  contract: per-turn output format, the six interaction verbs, workspace/capsule
  upkeep, artifact status gating, composition receipts, permission discipline.
- **Skills are discovered through `.opencode/skills/_registry.yaml`** (the only
  index): match `applies_when`/`not_for`, suggest ≤3 with reasons, load a SKILL.md
  full text only after the user confirms. ≤4 active non-core skills.
- **Task truth lives in `.opencode/local/workspaces/<task-slug>/`** (git-ignored
  runtime state; created from `.opencode/templates/workspace/`). The `capsule.md` is
  the ONLY carrier for handoff, consult, resume, and post-compaction recovery — never
  pass chat history between roles.
- **Permission ceilings are runtime-enforced** by each role's frontmatter, as
  pattern-scoped `edit` maps: a role may write only its own artifact directories
  (workspaces + e.g. docs/memory for researcher, plans for architect, reviews for
  reviewer, bench for validator, state for coordinator) — **source is denied for
  every role except implementer, whose every edit asks** (destructive commands
  denied). Read-only bash runs freely; mutating bash asks. Device/R3 operations run
  through MCP tools that frontmatter cannot gate — their per-action approval is a
  validator contract obligation, enforced by discipline and the runbook, not by the
  runtime. Only coordinator holds `task: allow` + delegate. Skills can never widen
  permissions (they degrade to advisory, per-action ask, or a handoff suggestion).
- **Execution rights ≠ claim rights.** Artifact headers carry
  `status: draft | reviewed | approved | validated | superseded`; promotions have
  role-owned conditions (plan→approved needs a review verdict; patch→ready-to-land
  needs approved review + passing build; perf claim→validated needs comparable A/B
  evidence). Producing an artifact never includes the right to promote it.

## Lane 2 — Pipeline recipes (explicit only)

`/optimize_*` commands run the staged optimization pipeline on the workbench role
chain — `coordinator` as hub, delegating to researcher / reviewer / implementer /
validator (the legacy `@hm-opt-manager` chain in `agents/legacy/` remains the
fallback until the live old-vs-new comparison is archived):

```
intake → research → plan review (GATE) → implementation → code review (GATE) → tester A/B → decision
```

- Stage gates, handoff packets, and delegation rules come from
  `.opencode/skills/infra/pipeline/` and apply **only inside recipe runs** — they are
  not a general law of the workbench.
- Authoritative spec: `.opencode/docs/harness_engineer_system.md`. Hard rules: no
  implementation without plan-review approval; no acceptance without code review; no
  test verdict without stock-vs-feature A/B; sub-agents return to the hub, never
  chain onward; state persists in `.opencode/state/current_task.json` before every
  delegation.
- **Recipe sub-agents must NOT re-Read `.opencode/skills/` files at runtime** — the
  launching command already `@`-inlined every listed skill pack into their context,
  and a sub-agent's CWD may not be the repo root. This includes the workbench roles
  when they are delegated to inside a recipe: their session-start Reads
  (agent-core, registry, role skill) apply to interactive sessions only — inside a
  recipe the inlined packs are authoritative and the registry/suggestion round is
  skipped. (In interactive sessions, workbench roles DO read the registry and
  confirmed skills — always via absolute paths, see below.)
- To verify a delegation really ran: the OpenCode status line switches to the
  sub-agent, its identity banner appears, and its expected artifact exists on disk
  afterwards. A narrated "Delegation to X" with no banner and no artifact means
  nothing ran.

## Rules that apply to both lanes

1. **Session start**: read `.opencode/config.yaml` and apply
   `.opencode/skills/infra/language-config/SKILL.md` (session language).
2. **Path resolution**: resolve the repo root once (`git rev-parse --show-toplevel`,
   fall back to `pwd`) and use absolute paths for every `.opencode/...` read — a
   relative path can resolve into `$HOME/.opencode/...`.
3. **Wildcards in docs describe write targets, not list commands** — to check whether
   `.opencode/reviews/*_plan_review.md` exists, `ls` the directory; never glob
   `.opencode/**`.
4. **Memory**: recall before proposing (`memory_recall` / target + subsystem files
   under `.opencode/memory/`); log reusable lessons, not task noise; the four planes
   (workspace · capsule · personal journal · team hub) never auto-merge.

## The golden rule (design §2)

Create a **role** only when responsibility or authority changes. Create a **skill**
when domain or method changes. Create a **profile** when a useful composition repeats.
Create a **workflow** only when several activities need repeatable coordination. Task
truth lives in the workspace; reusable truth lives in Team Memory / the Skill Hub.

Full design: `docs/Agent_Workbench_Design_EN.md` (CN counterpart alongside).
