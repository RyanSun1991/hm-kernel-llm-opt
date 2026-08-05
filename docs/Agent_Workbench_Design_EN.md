# HMOPT Interactive Agent Workbench — Complete Design & Usage Guide

Status: Final (integrates the external "Workbench v3" design after evaluation)
Version: 2.1 · 2026-08-05 (adds implementation approach and usage guide on top of v2.0)
Audience: HMOPT maintainers, OpenCode integration engineers, all team users
Chinese counterpart: `Agent_Workbench_Design_CN.md`

---

## Part I — Design

### 1. Problem & Core Decision

**Two findings from real adoption feedback:**

1. The fully automated pipeline is not what most users want — they want to interact with
   each agent directly, get structured output, and **decide themselves which agent to
   engage next**.
2. Agents are over-specialized — instruction-count objectives and hot-path analysis are
   hard-wired into prompts; non-optimization scenarios (understanding, debugging,
   refactoring) are unusable. Seven research agents are really "one role + different
   domain knowledge".

**Core decision:** rebuild the harness as a **composable, user-controlled interactive
workbench**:

```
Everyday unit of work = one stable role + a small selected skill set + a lightweight task workspace
```

- Default entry is `assistant`; ordinary prompts **never** implicitly start a pipeline;
- The user owns routing: role switches, consultations, and handoffs are all explicit —
  agents suggest, never seize control;
- The existing automated pipeline is preserved as a compatibility recipe (coordinator
  driving the same roles);
- The product-level shift: **from "the manager owns a pipeline" to "the user owns a
  task workspace."**

Implementation constraint: everything lands as markdown / frontmatter / file conventions —
no new services, no plugins, no event sourcing (an explicit trade-off against the
platform-grade external Workbench v3 proposal; see §11).

### 2. Design Rules (the Golden Rule)

> **Create a role only when responsibility or authority changes. Create a skill when
> domain or method changes. Create a profile when a useful role+skill composition
> repeats. Create a workflow only when several activities need repeatable coordination.
> Create a policy when a safety/quality requirement must be enforced independently.
> Task truth lives in the workspace; reusable truth lives in Team Memory / Skill Hub.**

Anti-patterns (forbidden): creating roles like "memory-researcher / hotpath-reviewer"
(use a profile); putting "who to hand off to next" inside skills; skills granting
permissions; passing whole conversation history between roles; promoting one successful
conversation directly into a shared skill.

### 3. Layered Architecture (five layers, one concern each)

```
┌ Task workspace / capsule ──── State layer: task truth lives in files, not in any agent's head
├ Profiles ─────────────────── Reuse layer: named role+skill compositions (optional layer)
├ Skill library (3 tiers) ──── Capability layer: domain knowledge, methods, how-to
├ Roles (7) ────────────────── Responsibility layer: mission / process skeleton / output contract / permission ceiling
└ agent-core base contract ─── Behavior layer: shared I/O contract, interaction verbs, capsule upkeep
```

Decoupling map — five things currently welded into agent prompts, and where each goes:

| Currently coupled in prompts | Moves to | Enforcement |
|---|---|---|
| Domain knowledge (reclaim/IC/hotpath) | scenario skill packs | Role prompts contain zero domain vocabulary (acceptance item) |
| Permission boundaries (prose "don't edit code") | role frontmatter `permission` | **Runtime-enforced by OpenCode** |
| Process rules (stage gates, next-agent) | infra/pipeline pack, loaded only by coordinator | Ordinary roles never see stage gates |
| Task state | workspace / capsule files | State survives role switches and session restarts |
| Quality requirements | artifact status gating (§8.3) | Enforced at status promotion, not by forced stage order |

### 3.5 Target `.opencode/` Directory Layout (vs. current state)

Two lanes coexist: the **workbench lane** adds `agents/` (rebuilt), `skills/`
(reorganized), and `local/workspaces/`; the **pipeline lane**'s artifact and memory
directories all stay in place. `local/` is git-ignored runtime state (the convention
PR #39 established for sediment_staging).

```
.opencode/
├── CLAUDE.md                    # M2: becomes the "thin constitution"; pipeline enforcement moves into the pipeline skill pack
├── config.yaml                  # unchanged
├── skill-memory.lock            # unchanged (hub broadcast output)
│
├── agents/                      # ── responsibility layer (rebuilt in M2) ──
│   ├── assistant.md             # 7 generic roles: assistant / researcher / architect /
│   ├── researcher.md            # implementer / reviewer / validator (mode: all)
│   ├── architect.md             # + coordinator (mode: primary)
│   ├── implementer.md
│   ├── reviewer.md
│   ├── validator.md
│   ├── coordinator.md
│   ├── profiles/                # ── reuse layer (M3) ── thin agent files; OpenCode discovers subdirectories
│   │   ├── reclaim-investigator.md      # converted from the 4 domain research agents
│   │   ├── hyperhold-io.md · workqueue.md · sync-mechanism.md
│   │   ├── kernel-understand.md         # non-optimization scenarios (prove universality)
│   │   └── bug-fix.md
│   └── legacy/                  # M2–M4 transition aliases (hm-opt-manager + 14 others); deleted in M4
│
├── skills/                      # ── capability layer (reorganized in M1, §5) ──
│   ├── _registry.yaml
│   ├── role/                    # research-discipline / plan-funnel / review-checklists /
│   │                            # implementation-guardrails / validation-flight-check
│   ├── scenario/
│   │   └── kernel-opt/          # all current optimization skills: perf-bottleneck-playbooks /
│   │                            # IC / memory-tlb / ab-test* / iterative / build-and-sign / flash-device
│   └── infra/
│       ├── agent-core/          # §7 base contract (new)
│       ├── team-memory/ hub-bridge/ language-config/
│       └── pipeline/            # stage-gate + handoff-contract + delegate (coordinator only)
│
├── local/                       # ── state layer (git-ignored runtime) ──
│   ├── workspaces/<task-slug>/  # task.md / capsule.md / artifacts/ / decisions.md (M2, §8)
│   └── sediment_staging/        # existing team-memory dir, untouched
│
├── memory/                      # unchanged: pipeline-lane memory + team-memory sediment source
│   ├── global_lessons.md · targets/ · subsystems/ · human_decisions/ · idea_ledger/
├── state/                       # bad_plans.md unchanged; current_task.json is a compatibility pointer in M2–M3, converges in M4
│
├── commands/                    # /optimize_* unchanged (pipeline recipe entry); plan/research point to new roles
├── pipelines/                   # recipe cards unchanged; consumed only by coordinator + pipeline pack
│
├── docs/                        # harness_engineer_system.md scoped to the pipeline lane from M2; bootstraps unchanged
└── bench/ plans/ reviews/ patches/   # pipeline-lane artifact dirs, kept in place;
                                      # workbench-lane artifacts go to local/workspaces/<slug>/artifacts/
```

Fate of the current 14 top-level items at a glance: **reorganized** skills/;
**rebuilt** agents/ (old files into legacy/); **new** agents/profiles/,
skills/infra/agent-core, local/workspaces/; **unchanged** memory/, commands/,
pipelines/, bench/, plans/, reviews/, patches/, config.yaml, skill-memory.lock;
**revised** CLAUDE.md (thin constitution), docs/harness_engineer_system.md (scoped to
the pipeline lane), state/current_task.json (converges in M4).

### 4. Role Catalog (7 roles)

Canonical names + aliases, `mode: all` (directly conversable; also delegatable by the
coordinator):

| Role | Alias | Mission (domain-free) | Permission ceiling (frontmatter-enforced) |
|---|---|---|---|
| `assistant` | — | **Default entry**: answer simple questions, make small changes, recognize when a workspace is worthwhile, recommend roles/skills | read: allow · edit: ask · bash: ask |
| `researcher` | merges 7 research variants | Build a trustworthy system model: separate facts/inferences/hypotheses, cite evidence, produce research notes | read: allow · **edit: deny** · read-only bash allow |
| `architect` | planner | Evidence to options: alternatives, trade-offs, decision records, plan (5-idea funnel is an optional skill) | source **edit: deny** · plan artifacts write: allow |
| `implementer` | coder | Implement an accepted plan: minimal diff, record assumptions/deviations; **never self-approves** | edit: ask (profile may pre-approve) · destructive: deny |
| `reviewer` | plan+code merged | Independently challenge research/plan/patch in a **clean context**: verdict + required changes | **edit: deny** · writes review artifacts only |
| `validator` | tester | Validate/falsify claims via build/test/benchmark/device: distinguish implementation vs hypothesis vs infrastructure failure | operational tools: ask · device flash: **explicit per-action approval** |
| `coordinator` | orchestrator (optional) | Only for pipeline recipes or genuinely parallel work: decompose/delegate/join; owns no domain truth, writes no source | mode: primary · delegate allow · edit: deny |

Role prompts contain only: mission, generic process skeleton, output contract, skill
loading rules, permission notes. **The permission ceiling is the design's only runtime
enforcement** — implemented natively by OpenCode, zero new code.

### 5. Skill Library (3 tiers + registry + 4 loading channels)

#### 5.1 Layout

```
.opencode/skills/
  _registry.yaml     # the single registry (no per-skill sidecar files)
  role/              # role skills: research-discipline / plan-funnel / review-checklists
                     # implementation-guardrails / validation-flight-check
  scenario/          # scenario packs: kernel-opt/ (all current optimization skills)
                     # kernel-understand/ · bug-fix/
  infra/             # agent-core / team-memory / hub-bridge / language-config
                     # pipeline/ (stage-gate + handoff + delegate; coordinator only)
```

#### 5.2 Registry entry (absorbs the high-value fields of v3's manifest)

```yaml
- name: memory-tlb-optimization
  tier: scenario/kernel-opt
  class: optimization-method     # tag: domain|method|scenario|review|validation|tool|output
  roles: [researcher, architect, reviewer, validator]
  applies_when: ["memory-management syscall optimization", "TLB/page-table paths"]
  not_for: ["explanation-only tasks", "latency claims without memory evidence"]
  conflicts: []
  context_cost: ~400 lines       # >500 lines must split into references/
  risk: R0                       # R0 read-only · R1 doc-write · R2 source/build · R3 device/publish
```

#### 5.3 Loading: four channels + three disciplines

```
Priority: ① explicit user request > ② profile preload > ③ trigger suggestion (suggest, user confirms) > ④ role defaults
```

Disciplines: **progressive disclosure** (registry keeps only name+description+applies_when
resident, ~80 tokens/skill; full text loads on activation) · **≤4 active non-core skills**,
**≤3 suggestions with reasons** · **composition receipts** (§8.4) close the feedback loop
to refine trigger wording.

### 6. Profiles (the reuse layer)

A profile is a **named composition** of role + default skills + optional skills +
permission preferences. It answers 90% of "I want a custom agent" requests without
creating roles. Implemented as thin agent files (selectable via Tab/@ in OpenCode):

```markdown
--- # .opencode/profiles/reclaim-investigator.md
description: Reclaim subsystem investigator (researcher + reclaim domain pack)
mode: primary
base_role: researcher
skills: [research-discipline, kernel-opt/perf-bottleneck-playbooks, kernel-opt/domain-reclaim]
optional_skills: [kernel-opt/memory-tlb-optimization]
---
Work per the researcher role contract with reclaim domain context preloaded.
```

Scope precedence: team-curated (hub) < project repo < personal config < explicit session
choice; permissions are outside this override chain. The 4 existing domain research
agents become 4 profiles.

### 7. Interaction Model

#### 7.1 Six interaction verbs (the user arbitrates ownership)

| Verb | Conversation ownership | Meaning |
|---|---|---|
| continue | current role | same responsibility continues |
| add/remove skill | current role | method changes, responsibility doesn't |
| **consult** | **not transferred** | bounded advice: one-shot @target-role, compact conclusion returns |
| **handoff** | transferred | responsibility/permission boundary changes: forward an (editable) brief; target becomes conversation owner |
| fork | new branch | copy the capsule into a new workspace branch to compare alternatives |
| recipe | coordinator | explicitly start a pipeline recipe (`/optimize_*`) |

#### 7.2 Per-turn output contract (role report)

① identity banner → ② structured result → ③ artifacts written → ④ **Next options
(1–3 items: verb + target role + reason + a directly forwardable brief draft)** →
⑤ open questions & confidence → ⑥ capsule update.
In interactive/guided modes the agent must wait for the user's choice — never
auto-transfer.

#### 7.3 Clean-context review

The reviewer receives **requirements + artifacts + evidence + decision records**, not the
implementer's persuasive self-narrative. Consult mode (fresh subagent context) satisfies
this naturally. Prevents author bias from contaminating review.

#### 7.4 Multi-agent eligibility gate

Before any parallel/multi-agent work, all must hold: ≥2 genuinely independent branches ·
minimal shared mutable state · clear I/O per branch · a join rule · a budget · a
measurable reason one role + skills is insufficient. Otherwise: one role + skills.

### 8. Task Workspace (lightweight, file-based)

#### 8.1 Layout

```
.opencode/local/workspaces/<task-slug>/
  task.md          # objective / scope / constraints / state (ready|running|waiting-user|done)
  capsule.md       # Task Capsule: the current projection (the key artifact, below)
  artifacts/       # research-note.md / plan.md / review.md / validation.md …
  decisions.md     # decisions & rejected alternatives (append-only)
```

Long-term replacement for the singleton `current_task.json`; fork = copy the directory.
Not built: events.jsonl, snapshots, optimistic concurrency, SQLite.

#### 8.2 Task Capsule

```markdown
# Capsule: reclaim-race investigation
objective: diagnose the shrink_node race
scope: mm/vmscan.c · commit abc1234 · symbol shrink_node
constraints: [preserve ABI, device ops need approval]
active: researcher + [research-discipline, domain-reclaim] · mode: guided
confirmed_facts:
  - Lock X covers callback Y (evidence: vmscan.c:137-155)
open_questions: [may callback Y sleep?]
decisions: [profile contention before considering lock splitting]
artifacts: [artifacts/research-note.md]
```

The active role updates it at the end of every turn (mandatory per agent-core).
**Handoff/consult passes the capsule + artifact references — never chat history**; after
compaction only the capsule is re-injected. One file solves handoff, resume, and
compaction at once.

#### 8.3 Artifact status gating (govern status, not process)

Artifact headers carry `status: draft | reviewed | approved | validated | superseded`.
Drafts are free; **status promotion has conditions**: patch → `ready-to-land` requires an
approved review (+ passing build); performance claims → `validated` require comparable
A/B evidence (baseline + candidate + matching metric + noise floor). Corrections create
new versions; old ones become superseded. Enforced by reviewer/validator at the role
level — no Policy Engine service.

#### 8.4 Composition receipt

One header line per artifact:
`produced_by: researcher + [domain-reclaim, method-data-flow] · 2026-08-05`.
One line of cost buys the hub an evidence stream of "which skill compositions produce
accepted artifacts" for skill curation.

### 9. Pipeline Compatibility

- `/optimize_*` commands remain = coordinator + pipeline skill pack driving the same
  roles; stage gates exist only inside that pack;
- Old agents remain as aliases until M4; the regression suite = golden cases for the
  4 optimize commands;
- The Python/LangGraph orchestration is untouched by this design (deferred wholesale).

### 10. Team Memory / Skill Hub Integration

- agent-core makes memory_recall/log/feedback a universal behavior for all roles;
- Receipts + artifact acceptance outcomes feed the hub's skill-curation evidence stream;
- Four planes stay separate: task state (workspace) ≠ checkpoint (capsule) ≠ personal
  experience (journal) ≠ team curation (hub); one successful conversation never
  auto-rewrites a shared skill.

### 11. Absorb/Reject Record vs. External Workbench v3

**Absorbed (12 items, all downgraded to file/convention level):** user-owns-workspace
framing · assistant default role · the golden rule · Task Capsule · six interaction
verbs · clean-context review · multi-agent eligibility gate · permission ceilings +
R0–R3 risk tags · artifact status gating · profiles · skill-selection numeric discipline
with applies_when/not_for · composition receipts and non-optimization scenarios in the MVP.

**Rejected (scale mismatch):** event-sourced workspace · Capability Broker & logical
capability vocabulary · Policy Engine service · workflow recipe compiler & condition
language · Python/LangGraph contract unification · workbench TS plugin · 17 versioned
JSON schemas with runtime validation · the full /task /consult command suite (native
OpenCode suffices) · full Composition Lock pinning.

---

## Part II — Implementation Approach

### 12. Mechanism → carrier map (how the whole design lands)

| Mechanism | Carrier | New executable code |
|---|---|---|
| Roles & permission ceilings | 7 agent markdown files + frontmatter `permission` | 0 (pure config) |
| agent-core contract | 1 infra skill file (output contract / verbs / capsule rules / status gating / eligibility gate) | 0 |
| Skill tiers & triggers | directory reorg + `_registry.yaml` | 1 registry lint script |
| Profiles | thin agent files | 0 |
| Workspace / capsule | directory convention + templates | 0 (optional init script) |
| Status gating / receipts | artifact header convention + reviewer/validator contract clauses | 0 |
| Pipeline compatibility | coordinator (slimmed manager) + pipeline skill pack | 1 agent rewrite |

Total new executable code ≈ one registry lint script; everything else is markdown/config.

### 13. Migration Plan (M1–M4, each independently shippable)

| Phase | Content | DoD |
|---|---|---|
| **M1** skill library reorg | 3-tier directory reorg, `_registry.yaml` (applies_when/not_for/risk/cost), skill slimming audit (≤500 lines), reference-path updates | 4 optimize commands pass regression; registry lint green; **zero behavior change** |
| **M2** roles + base + workspace | agent-core contract skill; 7 role files (permission frontmatter); workspace/capsule templates; old agents aliased | Human-routed end-to-end run of a real task (assistant→researcher→consult reviewer→architect→implementer→reviewer→validator) with capsule-only handoffs; researcher edit attempt rejected at runtime |
| **M3** profiles + scenarios | 4 domain agents → 4 profiles; kernel-understand and bug-fix non-optimization profiles; receipts live; suggest policy on | Same researcher covers ≥3 domains by swapping skills only; non-optimization tasks show zero optimization vocabulary |
| **M4** coordinator + wrap-up | manager rebuilt as coordinator; `/optimize_*` on the new chain; old agents deleted; usage guide published | 4 optimize commands pass golden regression |

### 14. Verification Methods

- **Regression**: freeze golden outputs (artifact lists + key fields) for the 4 optimize
  commands before M1; run every phase;
- **Comparison**: in M2 compare old vs new researcher output on the same task (domain
  depth must not drop);
- **Permissions**: one boundary-violation case per role (researcher attempts edit;
  reviewer attempts source change);
- **Tokens**: measure base context tokens before/after M1 (quantify progressive
  disclosure);
- **Triggers**: test applies_when/not_for with 10 positive + 10 negative task
  descriptions.

---

## Part III — Usage Guide

### 15. Getting Started: three entry paths

1. **Just ask** (default assistant): open OpenCode and talk — simple questions get
   answered directly; for complex tasks the assistant proposes "open a workspace + bring
   in researcher" and **waits for your confirmation**;
2. **Pick a role**: Tab-switch or `@researcher`;
3. **Use a profile**: `@reclaim-investigator` jumps straight into the preloaded
   researcher + domain-pack composition.

### 16. Typical Scenarios

**Scenario A — ordinary question (no workspace)**

```
You: what does sc->priority do in shrink_node?
assistant: (direct explanation with file:line) … Simple question — no task needed.
```

**Scenario B — deep investigation (one role + skill suggestions)**

```
You: @researcher investigate the suspected race in shrink_node
researcher: === researcher — acknowledging: shrink_node race ===
  Workspace opened: workspaces/shrink-node-race/.
  Suggested skills (confirm?): domain-reclaim (path match mm/) · method-lifetime (keyword "race")
You: ok
researcher: (investigates…)
  ## Result: lock X covers callback Y, but path Z has a window (evidence: vmscan.c:137-155)
  Artifact: artifacts/research-note.md (status: draft)
  ## Next options
  1. consult reviewer — independently challenge the race analysis (brief draft attached)
  2. handoff architect — develop fix options from the findings
  3. continue — resolve the sleep semantics of path Z
You: 1
```

**Scenario C — full change flow (human-routed, brief forwarding)**

```
researcher produces the note → you forward the suggested brief to architect
architect: two options (A split lock / B defer work), trade-off table, decision record → plan.md (draft)
You: pick A, handoff implementer
implementer: (requests edit approval) → patch + correctness argument → suggests review
You: consult reviewer (clean context: plan + patch + evidence only)
reviewer: verdict approved; patch may promote to ready-to-land (after build passes)
You: handoff validator → lmbench A/B → validation.md, claim validated
```

At every step you may: edit the brief before forwarding, ask for a redo, add/remove
skills, or simply stop (any artifact is usable as-is).

**Scenario D — fork to compare alternatives**

```
You: fork — try option B on the other branch
→ workspaces/shrink-node-race-b/ (capsule copied); both branches proceed; compare validations
```

**Scenario E — legacy automated pipeline (unchanged)**

```
/optimize_workqueue        # coordinator drives the full flow, stage gates enforced, as before
```

### 17. User Quick Reference

| Want to… | Do this |
|---|---|
| Switch roles | Tab or `@role`, or accept a handoff from Next options |
| Add/remove a skill | "load memory-tlb" / "drop the IC skill" |
| Ask why a skill was suggested | "why this skill?" (registry trigger reasons) |
| Independent review without losing the floor | consult reviewer (returns to you) |
| Save and resume later | workspaces persist; reopen and say "continue shrink-node-race" |
| Compare two approaches | fork |
| Run the old automated optimization | `/optimize_*` |

### 18. User Discipline (Do / Don't)

- **Do**: run reviews via consult (clean context); don't cite artifacts as conclusions
  before they reach validated; have agents record key decisions into decisions.md; ask
  the team to promote good compositions into profiles.
- **Don't**: paste whole chat history to the next role (pass capsule + artifacts);
  bypass the implementer's edit approval; ask the researcher to edit code directly (the
  permission denial is by design, not a bug).

---

## Part IV — Acceptance & Risks

### 19. Acceptance Criteria

- **Modularity**: core role prompts contain no subsystem paths and no IC assumptions;
  domain skills reusable by ≥3 roles;
- **User control**: ordinary prompts never implicitly start a pipeline; transfers
  require the user; skill-suggestion reasons are inspectable;
- **Permissions**: skills cannot widen permissions; researcher/reviewer edit:deny holds
  in all tests; R3 always needs per-action approval;
- **State**: tasks resume (capsule load = continue); forks never overwrite;
- **Compatibility**: all 4 optimize commands green;
- **Efficiency**: base context tokens drop after progressive disclosure (quantified);
  most ordinary tasks complete with a single role.

### 20. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Pipeline regression (biggest, M4) | dedicated phase; aliases in parallel; golden cases built before the switch |
| assistant grows into a new god-manager | edit:ask ceiling + contract: "recognize and hand off, don't carry it yourself" |
| Wrong/missed skill loading | applies_when/not_for dual triggers + suggest with confirmation + receipt feedback to refine descriptions |
| Capsule upkeep forgotten | mandatory item in the agent-core output contract; reviewer checks |
| Domain depth loss after role abstraction | depth moves (scenario packs + bootstraps), not lost; M2 old-vs-new comparison |
| Workspace vs current_task.json dual track | compatibility pointer during M2–M3; converge in M4 |
