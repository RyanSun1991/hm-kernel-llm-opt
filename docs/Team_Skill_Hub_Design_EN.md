# Team-Level Skill / Memory Repository Closed-Loop Design (Team Skill Hub)

| Item | Value |
|---|---|
| Document status | Draft v2.3 (pending team review) |
| Date | 2026-06-09 |
| Scope | Team-oriented evolution of the `.opencode/` harness; adds a standalone central repo `hm-skill-hub` |
| Language strategy | Prose en-US; paths / field names / code / CLI / commit messages all in English |
| Related docs | `.opencode/docs/harness_engineer_system.md`, `.opencode/docs/memory_system.md` |
| Diagrams | `docs/Team_Skill_Hub_Design_Diagrams_CN.md` (7 Mermaid diagrams: closed loop / dual engines / sedimentation funnel / skills layout / runtime composition / roadmap / read path) |
| Revisions | v2.3 (review feedback, 2026-06-09): ① §6.1 converges on "one file per record + frontmatter + path-encoded scope + CI consistency check", eliminating the contradiction with the Phase 0 multi-record example; ② §7 adds `subsumes[]/subsumed_by[]` fields + a constraint requiring frontmatter to carry all fields; ③ §10.1 introduces a unified "merge relation classification table" (dup / contradiction / temporal / conditional / subsumption / selector / evidence, seven paths), 10.1.a forbids delete on temporal/conditional/selector conflicts, 10.1.b adds subsumption (LLM entailment judgment); ④ §11.5 wires up subsumption → promotion + a ≥2-instance guard against spurious generalization; ⑤ §17 issue 7 elevated to a P1 blocking prerequisite + adds the path-encoded scope decision. v2.2 (mem0/EverOS research): §3 research rows + §8 LLM salience pass + §10.1 two-tier merge + §11.5 promotion detector + §12 retrieval and runtime composition rewrite + §14/§15 sync. v2.1: §6.2 skill/knowledge decision. v2.0: §6.2 skills layout + streamlining |

---

## 0. TL;DR

The existing `.opencode/` mixes together **two assets of fundamentally different nature**:

- **Skills (procedural instructions)** — "programs" that can be measured by eval and optimized.
- **Knowledge (facts/memory)** — "learned state" that is continuously appended and needs dedup and conflict resolution.

**The spine of the design**: split the two apart, govern them with two separate merge engines, then funnel each member's local experience through a single "sedimentation funnel" with verification gates into a standalone, semver-versioned central repo `hm-skill-hub`, and feed it back into the pipeline as pinned versions — forming a closed loop.

- **Skills** → governed with **SkillOpt**: the skill document = "trainable external parameters" of a frozen model; a change is accepted only if it is **strictly better** on a held-out eval suite; use **GEPA Pareto** to resolve the collapse of multi-member edits.
- **Knowledge** → governed with **memU/Mem0/Zep**: layered, typed, stable IDs, **append + dedup + conflict resolution** (not git line-level merge), bi-temporal retention of superseded records.

Four guarantees: **iterable** (scheduled optimization jobs), **sedimentable** (three tiers + L0–L3), **closed-loop** (consume → distill → promote → release → re-consume), and **reusable + stably available** (SKILL.md standard + semver + lockfile + eval-gate + local fallback + redaction gate).

---

## 1. Background and gaps

`.opencode/` already has the embryo of "process orchestration + memory sedimentation + skill reuse": a staged pipeline (`research → plan review → implement → code review → test → decision`, with hard gates), path-loaded instruction packs (skills), and layered memory:

| Storage | Role |
|---|---|
| `memory/targets/` `memory/subsystems/` | Structural facts |
| `memory/global_lessons.md` | Heuristics / anti-patterns |
| `memory/human_decisions/` | Human-machine decision timeline (written to disk in real time) |
| `memory/idea_ledger/` | Per-mechanism rulings, stable IDs (`L001`), state machine `approved/landed/rejected/...` |
| `state/bad_plans.md` | Reverse dedup |

**Three things are still missing for a team-level closed loop**: ① a cross-member aggregation mechanism (sedimentation only lives locally); ② a unified, machine-checkable quality gate; ③ automated iteration (skills are edited by hand, with no verification gate).

**Key observation**: the existing local promotion chain "idea_ledger row → distill → global_lessons" is exactly the chain we want to scale up across members and across repos. This design **externalizes it + adds gates + adds mergers**, rather than starting over.

---

## 2. Goals and non-goals

| Goal | Main support |
|---|---|
| Iterable | Scheduled optimization jobs (§11) |
| Sedimentable | Three tiers + L0–L3 (§4) |
| Closed-loop | Closed-loop architecture (§5) + eval-gate (§9) |
| Reusable | SKILL.md standard + hub/local overlay (§12) |
| Stably available | eval-gate + semver + lockfile + local fallback (§13) |

**Non-goals**: do not move `.opencode/` wholesale (the execution surface stays close to the code); no heavy infrastructure in the first phase (prefer git + files + lightweight indexing); no model fine-tuning (reuse SkillOpt's zero inference overhead); no chasing real-time sync (batch at the minute-to-day scale).

---

## 3. Industry research mapping

| Source | Core mechanism | Role |
|---|---|---|
| **SkillOpt** (arXiv 2605.23904) | Skill = trainable external state; bounded add/delete/replace edits; accepted only if strictly better on a held-out validation set; textual learning rate + rejected-edit buffer + slow updates; produces `best_skill.md` | Skills engine B + quality gate |
| **memU** | Three layers Resource→Item→Category; Memory-as-File-System; RAG + LLM dual retrieval; `where` scope | Knowledge layering and retrieval; hub/local overlay |
| **Mem0 / Mem0g** (v0.1.x paper version; v3 OSS already degraded, see below) | Two-stage extract → conflict detection + resolution: Phase 1 extracts facts with `FACT_RETRIEVAL_PROMPT`, Phase 2 resolves candidates against the nearest k records via ADD / UPDATE / DELETE / NOOP tool calls | **Local online** resolution (engine A first tier, §10.1.a) + Curator merger inspiration |
| **EverOS** (EverMind-AI, v1.0 / 2026-06) | markdown as source of truth; six memory categories (including procedural Skill as a first-class citizen); synchronous lightweight extraction + asynchronous OME offline reorganization; skill/profile clustering triggers; cascade incremental re-embedding; LanceDB runs BM25+vector+scalar filtering in a single query; explicitly **does not** include conflict/decay/quality gates | **markdown-as-source-of-truth + rebuildable index** pattern; **hybrid retrieval** template (§12); **promotion candidate clustering** (§11.5); counter-reference: the missing governance layer is exactly the hub's differentiation |
| **Zep / Graphiti** | Bi-temporal, facts carry an expiry time | Superseded records are not deleted (`superseded`) |
| **ExpeL** | Success/failure pool → extract insights ADD/UPVOTE/DOWNVOTE/EDIT | Promotion scoring and decay |
| **GEPA** | Reflective evolution + Pareto frontier (keeps complementary candidates, not a single optimum) | Avoids multi-member edit collapse |
| **Voyager** | Growing skill library, self-verifies before admission, composable | Skill-library paradigm + compositional reuse |
| **Anthropic Agent Skills** | SKILL.md open standard; plugin distribution; project-scope version-controlled sharing; marketplace security scanning | Distribution and interoperability standard + governance |

**⚠️ mem0 v3 OSS note** (since 2026-04): the open-source `main.py` collapses the two stages into a single-pass **ADD-only** flow of "`ADDITIVE_EXTRACTION_PROMPT` + content-hash dedup"; the LLM-driven UPDATE/DELETE and graph memory are **kept only in the paid Platform**. At the planning level this means: when §10.1.a references mem0, the OSS version only gives "extraction + hash dedup + indexing/retrieval", and the **intelligent UPDATE/DELETE/NOOP resolution logic must be reimplemented yourself following the v0.1.x paper prompts** (or evaluate the paid Platform).

**Conclusion**: SkillOpt gives the "skill-optimization engineering paradigm", memU gives the "memory-asset organization paradigm", and mem0/EverOS give the "online extraction + hybrid retrieval" engineering pattern, **yet all of them lack a cross-member curation closed loop (semver/eval-gate/L0–L3/dual review)** — and that is exactly the differentiating moat of this design. The best-fit approach: keep `.opencode/` as the execution surface and add the hub as the asset surface; the **local tier** borrows from mem0/EverOS to capture the latency and cost dividends of "cheap online resolution + hybrid retrieval", while the **central tier** keeps the hub's homegrown heavy governance (CI + eval-gate + dual review) to turn "experience sedimentation" into "verifiable iteration".

---

## 4. Core design principles (the spine)

### 4.1 Govern the two asset classes separately (the most important decision)

| | **Skills (procedural)** | **Knowledge (facts/memory)** |
|---|---|---|
| Growth | In-place edit (add/delete/replace) | Append + merge + dedup |
| Quality gate | eval-suite regression gate | Evidence + provenance + conflict resolution |
| Consumption | `@`-inline into context | On-demand retrieval (RAG + ledger) |
| Merge engine | **B**: gated competitive editing + Pareto | **A**: set merge + dedup + conflict resolution |

> **Anti-pattern**: governing both with the same git line-level merge — knowledge ends up duplicated/self-contradictory, and skills get overwritten by someone's bad week of experience. **Two engines, two gates, kept separate.**

> **v2.2 addendum**: engine A (Knowledge) is split into two tiers in engineering — **local online** (mem0/EverOS-style: run ADD/UPDATE/DELETE/NOOP at each close-out so `local/memory/` does not rot) + **central batch** (Curator + CI + eval-gate). See §10.1. Engine B (Skills) is always central batch and is **not** run locally (individual eval is noisy and tends to pollute instead).

### 4.2 Three tiers + L0–L3 maturity

Three tiers (the engineering form of memU raw→item→category):

- **Tier 0 run traces** (local, per member): live session artifacts (`current_task.json`, `*_design.md`, `plans/reviews/bench`). Low signal-to-noise, may contain secrets, **not shared directly**; they are the distillation input.
- **Tier 1 candidate sediment** (local→staging): schema-structured units distilled from Tier 0, with provenance + evidence + confidence score, **proposed for promotion**.
- **Tier 2 core shared** (central repo): the `best_skill.md` / curated knowledge after verification, dedup, and merge, **fed back**.

Maturity (orthogonal to the three tiers, characterizing "how trustworthy, how widely reusable"):

| Level | Criterion | Scope |
|---|---|---|
| **L0** draft | Local draft | Author's own project |
| **L1** candidate | Structurally complete + initial evidence | Under PR review |
| **L2** stable | Team review + eval passing | Whole team |
| **L3** core | Successfully reused across sub-teams | Organization gold standard (`skills/core/`) |

Promotion is step by step (L0→L1→L2→L3), each step gated (§9).

### 4.3 The three-archetype axis: the unique home for each artifact

When "the two asset classes" land in directories they expand into **three archetypes** — the single axis that decides where each file goes:

> **Is it a "program" (process, team-shared, optimizable by eval), or a "product of one run" (evidence/trace, personal local)?**

| Archetype | Meaning | Home |
|---|---|---|
| **procedural-shared** | agents / skills / commands / pipelines / harness specs / `*_template.md` | **hub** |
| **knowledge-curated** | facts / rules / patterns / anti_patterns / idea_ledger (a stable move can graduate into a technique skill, §6.2) | **hub** `knowledge/` (holds only distilled essence) |
| **run-evidence-local** | plans / reviews / bench / patches / per-run design / current_task | business repo `local/`; pure runtime gitignored |

Mutually exclusive and exhaustive: every file in `.opencode/` falls into exactly one (see §6.4). **Key corollary**: `agents/` and `skills/` are both procedural-shared and go into the hub together; `bench/`/`reviews/` are run-evidence-local, stay in the business repo, and only the distilled essence is promoted to the hub.

### 4.4 Traceability first

Every Tier 2 record must link back to the run/member/evidence that produced it. **No provenance = not admissible.**

---

## 5. Overall architecture and closed loop

```
   ┌──────────────── hm-skill-hub (Tier 2, team-shared, semver) ────────────────┐
   │   skills/ (engine B: SkillOpt + Pareto)     knowledge/ (engine A: memU merge) │
   └──────▲────────────────────────────────────────────────┬───────────────┘
          │ (4) release: eval-gate passes → bump version+tag+scorecard   │ (1) consume: submodule pin
          │                                                   │   + skill-memory.lock + retrieval
   (3) promote/merge (Curator + CI)                           ▼
      dedup/conflict/eval/Pareto/redaction          ┌──────────────────────────┐
          ▲                                       │  pipeline run (per member)  │
      staging/ (Tier 1 candidates)                 └────────────┬─────────────┘
          ▲                                                     │ (2) close-out: distill Tier0→1
          └──────────  validated delta / anti-pattern / ledger row ◄───┘
```

1. **Consume**: the pipeline starts by pulling the pinned version from the hub (submodule + lock).
2. **Distill**: during a run it produces Tier 0→1 candidates at close-out points (`hmopt sediment`).
3. **Promote/merge**: candidates are packaged into a PR; Curator + CI run dedup/conflict/eval/redaction and merge per engine.
4. **Release**: if eval-gate passes, bump the version, tag, and emit a scorecard; the pipeline re-pins.

Each loop has a gate, so dirty data does not snowball.

---

## 6. Repository layout

### 6.1 Central repo `hm-skill-hub` (semver tagged)

```text
hm-skill-hub/
  registry.yaml                # manifest: skill list + version + eval status + owner
  schemas/                     # JSON-Schema per record type (lint gate)
  skills/                      # Tier 2 procedural skills (engine B) —— internal layout see §6.2
  knowledge/                   # Tier 2 curated memory (engine A, memU/Mem0 layering)
    global/lessons/  global/anti_patterns/
    subsystems/<s>.md
    targets/<slug>/{facts/, decisions/, idea_ledger.md}
    index/                     # vector index manifest or rebuild recipe
  evidence/{benchmarks/, regressions/}    # verifiable evidence
  eval/{task_suites/, scorecards/}        # SkillOpt verification-gate assets
  policies/{promotion,merge,deprecation}.md
  staging/<member>/<date>/*.json          # Tier 1 inbound candidates
  tools/{sediment,run_evals,lint,dedup,redact}.py  merge_curator.md
  .github/workflows/ci.yml     # lint + secret-scan + eval-gate + index-build
```

**Storage form convergence (v2.3, corresponds to §17 issue 7 / Phase 0.5 blocker)**: `knowledge/` uniformly uses "**one file per record**"; a file = YAML frontmatter (all schema fields) + markdown body. A different ID = a different file, git line-level conflicts almost vanish, and dedup/conflict/subsumption are handed to the Curator for semantic handling (§10).

> ⚠️ **Convergence action**: the Phase 0 example (`A001-*.md` is `### A001` multiple records piled into one category file, with custom fields `lesson/applies_when/...`) is **inconsistent** with this principle and does not align with `memory_item.schema.json` fields. Phase 0.5 must converge **first** (§17 issue 7, listed as a P1 blocking prerequisite): ① one record per file; ② frontmatter uses standard schema fields (no per-category self-extended markdown fields allowed); ③ `parse_memory.py` outputs a standard schema object.

**Path is scope (v2.3)**: the file path **encodes** scope, **redundant with and must be consistent with** the frontmatter `scope` field, enforced by CI (inconsistency = reject). This lets scalar filtering (§12.1) first coarse-filter by directory, then fine-filter by reading the frontmatter:

```text
knowledge/
  global/{heuristics,anti_patterns,bad_plans,validation_pitfalls}/<ID>.md   # scope.level=global
  subsystems/<subsystem>/<ID>.md                                            # scope.level=subsystem
  targets/<slug>/{facts,decisions}/<ID>.md                                  # scope.level=function|call-site|...
  targets/<slug>/idea_ledger/<Lxxx>.md                                      # one file per idea
  index/                       # derived cache: vector / BM25 / scalar manifest + rebuild recipe (manifest.yaml)
```

> **Source-of-truth invariant (corroborated by EverOS)**: the `*.md` files and their YAML frontmatter are the hub's **single source of truth**; the vector/BM25/scalar under `index/` are all **derived cache** — deleting `index/` loses no knowledge, it can be rebuilt by re-indexing the whole markdown tree. This invariant simultaneously yields three guarantees: ① a natural disaster-recovery path; ② a replaceable index backend (faiss / pgvector / LanceDB can be swapped freely without affecting data); ③ any member can manually edit via git, and PR review is plain-text readable. **Forbidden**: any write path that "writes the index first, markdown lags".

### 6.2 Internal layout of skills/

Kernel work spans many dimensions — directory/submodule/file/function/move, etc. Building a tree by topology causes combinatorial explosion and rots on every rebase. **The single principle**:

> The skill tree is layered only by "skill kind/stability"; **dimensions finer than subsystem (dir/file/function) are not skills but knowledge**, mounted at runtime by the selector.

The home of each dimension:

| Kernel dimension | Home |
|---|---|
| Process / cross-cutting | `skills/core/<name>/` |
| Optimization move (mechanism) | `skills/technique/<name>/` |
| Subsystem | `skills/domain/<subsystem>/` ← **the only layer in skills that touches topology** |
| Directory (dir glob) | the domain skill's `applies_to.path_globs` (no separate directory) |
| File | `knowledge/targets/<slug>/facts/` + selector |
| Function (function/symbol) | `knowledge/` idea_ledger + `symbol_selectors` |

Only 3 dimensions are real skill folders; file/function all sink into `knowledge/`, eliminating the explosion at the root:

```text
skills/
  core/                          # procedural · topology-independent · across all subsystems (L3, SkillOpt focus)
    optimization-funnel/  instruction-count-first/  stage-gate-enforcement/
    handoff-contract/  research-discipline/  ab-test-comparison/  ...
  technique/                     # reusable "optimization moves" · topology-independent · named by mechanism (not by target)
    hoist-loop-invariant/  batch-coalescing/  lock-granularity-reduction/
    branch-elimination/  redundant-load-elimination/  inline-tradeoff/  ...
  domain/                        # the only layer touching topology · only down to subsystem granularity
    mm-reclaim/   { SKILL.md, references/ }     # ← the current memmgr-reclaim series
    hyperhold-io/  workqueue-threadpool/  sync-primitives/  sched/  fs/  ...
  _registry/
    skills.yaml                  # full index: name/kind/version/maturity/eval_id/applies_to
    subsystem_selectors.yaml     # subsystem → {path_globs, symbol_selectors} centralized binding (single-point maintenance of volatile)
```

Each skill is a folder (Anthropic SKILL.md standard): `SKILL.md` (a short "when to use + how to use", loaded once selected) + `best_skill.md` (SkillOpt artifact) + `evals/` + `candidates/` (Pareto frontier) + `scorecards/` + `references/` (heavy material, loaded on demand).

**Topology is mounted by selector, not hardcoded paths** (domain skill frontmatter):

```yaml
name: mm-reclaim
kind: domain
applies_to:
  subsystems: [mm/reclaim]
  path_globs: ["mm/vmscan.c", "mm/*reclaim*"]
  symbol_selectors: ["shrink_*", "*_reclaim", "kswapd*"]
requires: [core/optimization-funnel, technique/hoist-loop-invariant, technique/batch-coalescing]
eval_id: eval/task_suites/mm_reclaim_suite
```

`resolver.py` (§12) takes a target (e.g. `mm/vmscan.c::shrink_node`): ① the selector matches `domain/mm-reclaim`; ② follow `requires` to pull in core+technique; ③ retrieve and mount that function's `knowledge/`. Selectors resolve against the **current clangd/scip index**, so after a rebase they re-resolve automatically and the skill body stays untouched.

**Compose rather than enumerate** (Voyager-style): do not pre-build a big skill for every `(subsystem × move)`; instead let the pipeline preset compose small skills at load time. The `(subsystem × technique)` matrix is absorbed by **load-time composition**, not enumerated in the tree.

**Anti-explosion: the skill vs knowledge decision**

**Primary criterion (ask this first)**: a **practice/process** (the AI *executes* it, tuned by *rewording*, with good/bad measured by eval) → **skill**; a **fact/conclusion/lesson** (the AI *consults* it, maintained by *adding/removing/correcting entries*, not by rewording) → **knowledge**. The clean cut: "check the checklist" is a skill, "the checklist contents" are knowledge.

The three below are only **exclusion criteria** (failing any one → it must be knowledge; passing all three still does not directly equal a skill — the primary criterion still rules):

1. **Reusable**: holds only for one file/function → knowledge.
2. **Stable**: moves with the file/symbol (changes on rebase) → knowledge.
3. **Optimizable**: the wording is not something you would repeatedly tune, eval cannot tell the difference → knowledge.

Example: `bad_plans` / `global_lessons` pass all three yet are still **knowledge** (lessons that get consulted; the *process* "dedup against bad_plans before proposing a plan" is written into the `optimization-funnel` skill). If a lesson stabilizes into a "fixed steps + reusable + eval-measurable" move, it can **graduate** into `technique/` (knowledge is the raw ore, technique is the refined product).

→ **Never build a per-file / per-function skill.**

**Per-layer eval**: core uses a cross-subsystem big suite (broadest signal, hardest, the §15 long pole); technique uses the task subset that the move applies to; domain uses representative tasks of that subsystem. The three `evals/` are independent and do not pollute each other.

### 6.3 Consumer side `.opencode/` (shared + local overlay)

```text
.opencode/                # (inside the business repo hm-kernel-llm-opt)
  hub/                    # git submodule, pinned to a version of hm-skill-hub (read-only)
  local/                  # this member's execution-surface artifacts (run-evidence + Tier 1 in-flight memory)
    runs/<run_id>/{plans,reviews,bench,patches, <target>_design.md}  # evidence, recommended to commit and retain
    memory/               # in-flight working memory (Tier 1): targets/ human_decisions/ idea_ledger/
    sediment_staging/     # candidate packages produced by hmopt sediment (→ PR to hub)
  state/current_task.json # pure runtime, gitignore
  skill-memory.lock       # locks the hub version (semver + SHA)
  resolver.py             # at load time: hub (shared) first, then overlay local (personal in-flight)
```

This is exactly memU's `where` scope (team vs personal) and Anthropic's project-scope vs personal-scope.

### 6.4 Directory migration home table + two-repo view

The original flat `.opencode/` is split in two; **no directory stays put in its old place**, each is assigned per the §4.3 axis:

| Original directory | Archetype | New location |
|---|---|---|
| `skills/` `agents/` `commands/` `pipelines/` `docs/` (harness specs) | procedural | **hub** corresponding directory |
| `*_template.md` | procedural (a template is a program) | **hub** (instances land in local) |
| `memory/idea_ledger/` | knowledge | authoritative version in **hub** `knowledge/targets/<slug>/`; in-flight copy in `local/memory/` |
| `memory/targets|subsystems|global_lessons` | knowledge | distill → **hub** `knowledge/`; in-flight copy in `local/memory/` |
| `memory/human_decisions/` | run-evidence→knowledge | raw timeline stays in `local/` (needs redaction); stable summary → hub `decisions/` |
| `state/bad_plans.md` | knowledge | distill → **hub** `knowledge/global/anti_patterns/` |
| `state/current_task.json` | run-evidence (pure state) | **local only** gitignore |
| `plans/` `reviews/` `bench/` | run-evidence | business repo `local/runs/`; validated delta/anti-pattern → hub |
| `patches/` | run-evidence | **business repo** (with the code), not into the hub |

```
┌─ hm-skill-hub  (asset surface · shared · semver) ─────────────────────────────────┐
│   PROCEDURAL (engine B)                       KNOWLEDGE (engine A)            │
│     skills/{core,technique,domain}  agents/   knowledge/   evidence/      │
│     commands/  pipelines/  docs/              ▲ holds only "distilled essence"  │
└──────────▲─────────────────────────────────────┼────────────────────────┘
           │ (1) pin: submodule + lock (read-only)      │ (3) promote: distill + eval gate
┌─ business repo /.opencode/ (execution surface) ────────────────────┴───────────────────────┐
│   hub/  ← submodule read-only pinned                                           │
│   local/  runs/<id>/{plans,reviews,bench,patches}  memory/  sediment_staging/ │
│   state/current_task.json (gitignore)   skill-memory.lock   resolver.py   │
└────────────────────────────────────────────────────────────────────────────┘
```

**Path compatibility (must be handled in Phase 0)**: the current harness hardcodes relative paths like `.opencode/skills/X.md`, and moving into `hub/` would break them. Two options: ① symlink `.opencode/skills→hub/skills` to keep the old paths (minimal change); ② bulk-rewrite + `resolver.py` for unified resolution. Recommend ① as a fallback first, then migrate to ②.

---

## 7. Data model and Schema

**Typed memory** — each Knowledge record declares a `type`, one of five:

| type | Meaning |
|---|---|
| `fact` | Stable structural fact |
| `rule` | Operational rule |
| `pattern` | Reusable positive pattern |
| `anti_pattern` | Anti-pattern |
| `playbook_step` | Process step fragment |

> `pattern` / `playbook_step` are *consulted* record entries ("there exists such a move/step"); when one stabilizes into an *executable*, eval-measurable process, it graduates into a `technique` / `core` skill (the §6.2 primary criterion).

**`memory_item` key fields** (the full JSON-Schema is in `schemas/`):

```
id(stable, prefix F/G/A/R) · type · title · body
scope{level: function|call-site|data-flow|subsystem|architectural|global, subsystem, target_slug}
applies_when(conditional applicability range, used for "conditional divergence" coexistence judgment, §10.1)   ← v2.3
source[]{kind: commit|review|bench|doc|run_id, ref}   ← required, no provenance = reject
evidence{delta_pct, compare_level, confirmations}
maturity(L0-L3) · status(active|superseded|deprecated) · score
invalidation(invalidation condition, e.g. "must re-check offset after rebase")
supersedes[] · superseded_by[]                         ← temporal/contradiction relations (bi-temporal)
subsumes[] · subsumed_by[]                             ← v2.3 generalization-containment relations (§10.1 / §11.5)
valid_from · valid_until · contributor · created_at
```

> **Frontmatter constraint (v2.3, CI-enforced)**: the YAML frontmatter of every knowledge file on disk **must** carry all the required fields above, and per-category self-extended markdown fields are **not allowed**; **the scope encoded by the file path must be consistent with the frontmatter `scope`** (§6.1). `subsumes[]/subsumed_by[]` and `supersedes[]/superseded_by[]` are both **relation edges** — these are the first batch of landable edges should a graph layer (mem0g / Graphiti) be introduced in the future; for now they are fields only, no graph storage is built.

**skill frontmatter** (compatible with SKILL.md, see §6.2): `name/kind/version/maturity/applies_to/requires/eval_id/owners/status`.

**Skill update manifest** (must accompany every skill PR, otherwise CI rejects): binds `edit_ops` + `task_suite` + `metrics{pass_rate, instr_count_delta, regression_rate}` + `baseline_version`.

**idea_ledger**: reuse the existing structure (stable ID, state machine, never delete), only ① JSON-ify fields for machine merge; ② externalize to the hub; ③ merge across members via the Curator.

---

## 8. Sedimentation timing and promotion

**Tier 0→1 (distillation) trigger**: reuse existing close-out points — the pipeline decision stage, a human-machine session "done", the end of each auto-iterate pass. Products land in `local/sediment_staging/*.json` (tagged `maturity: L1`).

**Two-stage extraction (v2.2)**:

1. **Rule-mapping extractor** (deterministic, always runs): `extractors.py` maps bench delta → fact, review rejection → anti_pattern, ledger state-machine change → idea record. Guarantees that structured fields such as `delta_pct / compare_level / source[]` are not lost.
2. **LLM salience pass** (heuristic, can be turned off): take the free text left over from rule extraction (design summaries, reviewer notes, human-machine decision dialogues), run a `FACT_RETRIEVAL_PROMPT`-style extraction once to capture reusable insights that **do not fit the rule templates** (both mem0 and EverOS use this; extracting with LLM alone tends to miss structured metrics, extracting with rules alone tends to miss the atypical "this is actually reusable" insight, and stacking the two stages is the most robust). The LLM-pass product defaults to `confidence: tentative` and needs subsequent confirmations to raise maturity.

**Close-out cadence (corroborated by EverOS)**: synchronous close-out points only do **cheap extraction** (~≤100ms scale, rules + optional lightweight LLM); the heavy dedup / salience aggregation / cross-run correlation are handed to **asynchronous offline jobs** (§11). The synchronous flow **does not block** the main pipeline.

**Tier 1→2 (promotion) trigger** (satisfy one of + pass the §9 three gates): ① ≥2 independent tasks reproduce the gain; ② a single-task gain is significant and has bench evidence; ③ a highly reusable failure lesson (→ anti_pattern).

**Contribution cadence**: auto-staging (continuous) + batch PR (weekly/milestone); `hmopt sediment` packages candidates into a single "sedimentation PR" for unified dedup and to avoid spamming.

---

## 9. Quality gates (three of them)

```
candidate(L1) → [gate1 Schema/Lint/redaction] → [gate2 evidence] → [gate3 curation+eval] → stable(L2/L3)
                CI automatic                    automatic        Curator + human + eval-gate
```

1. **Schema/Lint/redaction**: pass the §7 schema; `redact.py` + CI secret-scan hard-scan for device serial numbers/keys, hit = reject (a leak in the team repo is an amplified incident).
2. **Evidence**: knowledge needs a citation (`validation_path`/`delta_pct`/`confirmations≥N`); a skill edit needs eval results. No evidence → stays at L1.
3. **Curation + eval**: the Curator pre-processes dedup/conflict/generalization (§10), then **two reviewers** sign off (1 domain + 1 process). Skills additionally pass the eval-gate (strictly better on the held-out suite). **No exemptions**: an exception can only be downgraded to an L1 candidate + owner sign-off + re-review.

**Scoring** (promotion/retrieval ranking + decay): `score = w1·evidence strength + w2·confirmation count + w3·recency + w4·generalization scope − w5·counterexamples − w6·staleness (decay after invalidation triggers)`.

---

## 10. Merge mechanisms (two engines)

### 10.1 Engine A — Knowledge: set merge + dedup + conflict resolution (never line-level merge)

Since v2.2 it is split into **local online** + **central batch** two tiers. The same classifier core, run at two different times and at two different permission levels.

**Why two tiers**: relying only on the central Curator would let `local/memory/` accumulate a week of debt before reaching the batch process — duplicates, self-contradiction, retrieval-quality degradation — and cleaning up only at the Curator stage is already too late. mem0 / EverOS corroborate: online resolution can run cheaply (small candidate set, approximate nearest neighbor) and is a prerequisite for retrieval quality.

#### 10.1.0 Merge relation classification table (v2.3 core)

A merge decision is **not** a binary "duplicate? / contradiction?", but classifies the **relation** between the incoming and the nearest k existing records into one of the seven categories below. **Iron rule: except for "clear contradiction with stronger new evidence", no branch physically deletes** — this is the fundamental safeguard for the review feedback "don't mistakenly delete historical facts".

| Relation | Decision | Handling | Who decides | **Never** |
|---|---|---|---|---|
| **duplicate** | Semantically near-duplicate, same scope same conclusion | Merge source[], `confirmations += 1` | hash + embedding (cheap) | — |
| **contradiction** | Same (target, mechanism, **same condition**) asserts the opposite | Stronger new evidence → old record `superseded` + `valid_until`, `superseded_by` cross-link; otherwise escalate | embedding + LLM | delete the old record |
| **temporal staleness** | The old record **was right, is now outdated** (e.g. a new kernel version changed the behavior) | Old record `superseded` + `valid_until=now`, **kept auditable** | LLM (looks at valid_from / version) | **delete it as an error** |
| **conditional divergence** | Both records are **right, with different applicable conditions** | **Coexist**, each writes its own `applies_when` / `scope` | LLM | dedup it as a contradiction |
| **subsumption (generalization-containment)** | One record is the **generalization** of another (B subsumes A) | **Keep both**: A as target-level evidence; B promoted to a pattern/technique candidate; A enters B's `source[]` + cross-link `subsumes/subsumed_by` | **LLM entailment judgment** | dedup A and swallow it into B |
| **selector drift** | The same symbol **changed path/offset after rebase** | **Re-resolve the selector**, update `invalidation`, the knowledge body stays untouched | clangd/scip index | delete the knowledge |
| **evidence divergence** | Same mechanism same delta, **different `compare_level`** | **Merge**, disambiguate by `compare_level` (total/process/function are not directly comparable) | rules | treat it as a contradiction |
| **novel** | No above relation with the nearest k | ADD | — | — |

**Tier assignment**: `duplicate / temporal / conditional / contradiction / evidence` are cheap or a single LLM call → **run at both tiers** (local 10.1.a + central 10.1.b); `selector drift` depends on the code index → local resolver at load time + central CI; **`subsumption` needs LLM entailment judgment and is more expensive → central 10.1.b only** (the local latency budget cannot bear it, and generalization is a cross-member signal).

#### 10.1.a Local online (run once at each close-out, per member, independent)

```
# Trigger: sediment close-out point. Latency budget: per record ≤ 1 LLM call + 1 ANN query (~1-3s)
for item in just_sedimented(local):
    if hash_seen(item): merge_provenance(...); continue   # cheap dedup, before the LLM
    nearest = vector_search(local.index, item, k=5, filter=scope)
    rel = classify_relation(item, nearest)      # §10.1.0 table (local runs only the cheap 5 categories)
    apply(rel, local.memory)                     # see the "no delete" discipline below
```

- **Scope**: resolve only within `local/memory/<member>/`, **not** across members; **does not run subsumption** (left to central).
- **Key discipline (review feedback ③)**: local `apply` does **not delete** any of the four categories `temporal / conditional / selector / evidence` — temporal → `superseded`+`valid_until`; conditional → coexist (write `applies_when`); selector → re-resolve + update `invalidation`; evidence → merge by `compare_level`. The only thing that ever writes a tombstone is "contradiction with stronger new evidence", and even that tombstone is `superseded`, not a physical delete. **Local false-delete must be ≈ 0** (a P1-8 PoC hard metric).
- **Dependency**: `pip install mem0ai` can provide the "index + ANN + dedup" infrastructure; **the relation-classification prompt is homegrown** (extend mem0 v0.1.x paper `DEFAULT_UPDATE_MEMORY_PROMPT` to the seven paths, because mem0 v3 OSS is already ADD-only).
- **markdown remains the source of truth**: the products of online resolution are **written directly back to markdown frontmatter + body**, and the index is incrementally rebuilt by cascade (§12).

#### 10.1.b Central batch (PR / nightly, cross-member)

```
# Trigger: sedimentation PR or nightly Curator. Latency budget: minutes
for item in incoming_from_all_members:
    rel = classify_relation(item, hub)     # §10.1.0 all seven paths, including LLM entailment judgment
    match rel.kind:
        case duplicate:    merge_provenance(rel.target, item); confirmations += 1
        case temporal:     rel.target.status="superseded"; rel.target.valid_until=now(); link(item, rel.target)
        case conditional:  add(item)        # coexist, verify the two applies_when do not overlap
        case evidence:     merge_by_compare_level(rel.target, item)
        case contradiction:
            if stronger_evidence(item): rel.target.status="superseded"; item.supersedes=[rel.target.id]; add(item)
            elif high_risk: escalate_to_human()
            else: drop_with_citation(item)
        case subsumption:  # ← v2.3 newly added third category (review feedback ④)
            general, specific = orient(item, rel.target)   # which generalizes which
            specific.subsumed_by += [general.id]; general.subsumes += [specific.id]
            general.source += specific.as_source()         # specific becomes general's evidence, not swallowed
            emit_promotion_signal(general)                 # → §11.5 (promotes only with ≥2 instances)
        case novel:        add(item)
```

- **Scope**: across all members, across the hub's full knowledge. Thresholds are stricter than local: tighten similarity, conflicts go through dual review (§9 gate 3).
- **subsumption ≠ duplicate ≠ contradiction** (review feedback ④): example "in `shrink_node`, hoisting `sc->priority` reduces repeated reads" (A, target-level) vs "in the mm reclaim hot loop, loop-invariant state should be hoisted out of the loop" (B, pattern-level) — B is the **generalization** of A. Handling: keep A as target evidence, promote B to a pattern/technique candidate, and **A becomes B's `source`/evidence rather than being deduped and swallowed**. This is the engine of the `knowledge → technique skill` graduation channel.
- **Guard against spurious generalization**: subsumption only immediately **builds the link** (cheap, safe); the generalization record B actually **promotes** to a technique still requires the §11.5 gate of **≥2 different subsumed instances** + the §9 three gates. A single A is not enough to spawn a technique.
- **New responsibility**: **merge same-semantic clusters across members** (different members' different wordings of the same fact) → merge provenance + accumulate confirmations.

**CRDT discipline (throughout both tiers)**: append + tombstone (`active/superseded/deprecated`) instead of delete; bi-temporal `valid_from/until` catches "rebase invalidates the offset". **Local** tombstones are not pushed immediately; **central** tombstones are release artifacts.

### 10.2 Engine B — Skills: SkillOpt verification gate + GEPA Pareto (never set merge)

```
def merge_skill_edit(skill, edit):
    if edit in bad_edits: return REJECT          # rejected-edit buffer
    edit = clip_to_budget(edit, textual_lr)      # textual learning rate: bounded edit
    cand = apply(skill, edit); s = run_evals(cand, suite)
    if s.strictly_better_than(skill.score):      # accepted only if strictly better
        skill = cand; write_scorecard(s)
    else: bad_edits.append(edit)
    pareto = update_pareto(pareto, cand, per_instance_scores)   # keep complementary candidates
    return skill, pareto
```

**Why Pareto**: when N members each submit an edit, a single global eval score makes "complementary but mutually exclusive" edits collapse to a local optimum. The Pareto frontier keeps candidates that "are each optimal on certain instances" (`candidates/`), and periodically merges complementary lessons — **this is the correct answer to "everyone sediments, unified intake without overwriting each other".** The textual learning rate = the bounded edit budget per release; slow updates = batch-merge per release cycle.

> **In one sentence**: knowledge relies on "set merge + dedup + conflict resolution"; skills rely on "gated competitive editing + Pareto". Two asset classes, two engines.

---

## 11. Closed-loop optimization job (scheduled)

A nightly/weekly "Skill/Memory optimization job":

```
(1) Collect    aggregate candidates from each project → staging/
(2) Normalize  standardize per schema + denoise + redact
(3) Cluster    embedding clustering (engine A dedup prerequisite)
(4) Optimize   run SkillOpt bounded edits over skills (engine B); early semi-automatic: auto-open PR + manual merge
(5) Validate   held-out suite A/B, emit scorecard
(6) Promote    bump only the version that gains, semver + tag, update registry.yaml
(7) Broadcast  generate release notes, for the business repo to pin
```

**Early safety constraint**: step (4) must connect to `bad_edits` + Pareto + redaction, and is **semi-automatic by default** (auto-open PR, manual merge); open it up only after building trust.

### 11.5 Automatic promotion-candidate detector (v2.2 newly added)

The main flow's L1→L2 and knowledge→technique skill promotions are started by manual PR, and **the signal is easily buried**. EverOS's `trigger_skill_clustering.py` / `trigger_profile_clustering.py` validated that "cluster **repeated patterns** → auto-open a PR for human review" is an engineering-feasible middle tier: **only automate candidate detection, the decision is still reviewed by a human at the §9 gates** — governance does not yield, humans are not drowned.

**Two input paths (v2.3)**:
- **Clustering signal**: embedding-cluster the hub knowledge (along the mechanism + scope dimensions).
- **Subsumption signal** (fed in by §10.1.b): a generalization record B's `subsumes[]` accumulates to **≥2 different subsumed instances** (different targets / different contributors) — a stronger graduation signal than pure clustering (the "specific→general" relation has already been explicitly established).

```
(1) Gather       collect (a) embedding clusters + (b) generalization records with subsumes[] ≥ 2
(2) Threshold    intra-cluster confirmations sum ≥ N (default 3) and across ≥ 2 contributors;
                 the subsumption path requires ≥ 2 different subsumed instances (guard against spurious generalization)
(3) Distill      call the LLM to distill the cluster/generalization record into "move + applicable conditions + evidence list (including subsumed instances)"
(4) PR-Open      auto-open a promotion PR (label promote-candidate), CODEOWNERS take over
(5) Guard        the promotion PR still goes through the §9 three gates (schema/evidence/curation+eval)
```

**Two applicable scenarios**:
- **L1 → L2**: the staging area hits the same fact N times across members → propose promotion to hub `knowledge/global/` or `knowledge/subsystems/`.
- **knowledge → technique skill**: under the same mechanism the anti_pattern/heuristic cluster ≥ N, **or** one pattern already `subsumes` ≥2 target-level instances → propose **graduation** to `skills/technique/<mechanism>/` (when the §6.2 primary criterion "practice/process" is met). The subsumed concrete instances are **kept** as that technique's `evidence`, not deleted.

**Discipline**: the detector can only **propose suggestions**, **not** merge by itself; any promote-candidate PR must be explicitly approved by a human, no exemptions.

---

## 12. Retrieval and runtime composition (v2.2 rewrite)

**Before v2.1** this section only wrote the one sentence "hub first then local + RAG"; v2.2 completes the entire **read path** — this is the full value zone of mem0 / EverOS, and the original draft was severely under-designed.

### 12.1 Three retrieval query types, one hybrid retrieval stack

`resolver.py` issues retrievals against hub + local at each pipeline stage. **The inputs come in three types**, **the underlying stack is the same one**:

| query type | Trigger point | Input | Main consumer |
|---|---|---|---|
| **target-anchored** | research / plan / code stage | the current target slug + symbol (e.g. `mm/vmscan.c::shrink_node`) | knowledge mounted after the domain skill selector hits |
| **mechanism-anchored** | plan-review / code-review | candidate mechanism (`hoist-loop-invariant` etc.) | technique skill context + related anti_pattern |
| **free-form** | any moment, agent asks explicitly | free text | fallback general retrieval |

**Hybrid retrieval stack (EverOS LanceDB pattern)**:

```
def retrieve(query, scope_filter, k=5):
    # 1) Scalar pre-filter (schema fields hit directly, cheap)
    cands = scalar_filter(
        index, status="active",
        maturity_in={"L2","L3"},  # exclude L0/L1 by default, adjustable for graduated rollout
        scope=scope_filter,        # subsystem / target_slug / level
    )
    # 2) Hybrid score: BM25 + vector cosine + entity match + temporal recency
    v_scores  = vector_topk(cands, embed(query), k=4*k)
    bm_scores = bm25_topk(cands, query, k=4*k)
    ent_bonus = entity_match_bonus(cands, extract_entities(query))
    fused     = rrf_fuse(v_scores, bm_scores) + ent_bonus
    # 3) score-field weighting (§9: promotion scoring feeds back into ranking, recent / high-confirmations first)
    fused    *= sigmoid(item.score)
    return topk(fused, k)
```

Four things mem0 / EverOS taught us: ① **scalar filter before vector** (the cost differs by orders of magnitude; the schema's existing `scope.level / maturity / status / scope.subsystem` are used directly); ② **BM25 + vector fusion** (pure vector flops on term-hit scenarios — symbol names like `shrink_node` are poorly approximated by vectors, and BM25 saves the day); ③ **the `score` field (§9) must feed back into ranking** — it is currently used only for promotion ranking and the read path never connected it, which is an obvious bug; ④ **each stage has a token budget** — more retrieval is not always better; the mem0 paper gives a 7K vs 25K tokens/query comparison, and excessive context degrades the decision in reverse.

### 12.2 Runtime composition (replaces v2.1's "overlay" phrasing)

The resolver's resolution order is as follows; **hub and local are not a simple overlay, but each contribute a different facet**:

```
resolve(target, stage)
├─ hub.skills/   per §6.2 selector hits domain → pull requires → core + technique
├─ hub.knowledge call retrieve() for target-anchored + mechanism-anchored, take top-k
└─ local.memory  call retrieve() for the same query, take top-k (in-flight, including personal un-promoted ideas)
   → merge & dedup (the same stable ID defers to hub; local only supplements the un-promoted)
   → trim to context budget (per-stage token cap)
   → inject into agent context
```

**Context budget (per pipeline stage)**:

| stage | skills | knowledge top-k | knowledge token cap |
|---|---|---|---|
| research | core full + domain selector hits | 8 | 3K |
| plan / plan-review | + technique requires | 5 | 2K |
| implement | + technique requires | 3 | 1.5K |
| code-review | core + anti-patterns first | 5 | 2K |
| test / decision | only anti_pattern + heuristic | 3 | 1K |

The numbers are a starting point, adjusted by scorecard feedback. **At every stage, once over budget, drop the ones with low `maturity`, low `score`, weak `evidence` first.**

### 12.3 Index: derived cache, markdown as source of truth

- **Storage**: `hub/knowledge/index/` and `local/memory/index/` (faiss files or a LanceDB directory, one of the two, aligning with the §17 decision). Phase 1 starts with faiss + sqlite-fts5 at the lowest cost, Phase 3+ evaluates LanceDB (runs hybrid retrieval + scalar filtering in a single query).
- **Incremental re-embedding (cascade-style, corroborated by EverOS)**: watchdog watches the markdown tree, diffs `content_sha256`, re-embeds only changed records; crash recovery relies on a sqlite state queue. **Full rebuild is forbidden** as a routine path.
- **Rebuild recipe**: each release ships `index/manifest.yaml` (embedding model + chunking parameters + rebuild command), and any member rebuilds with a one-line command.

### 12.4 Cross-tool and versioning/degradation

- **Version locking**: `skill-memory.lock` (semver + SHA) pins the hub version, equivalent to a package lockfile, reproducible and drift-proof; each run records the consumed version.
- **Failure degradation (availability)**: the hub is submodule-pinned locally (a vendored copy), so **a central-repo outage does not block**; the resolver detects unreachability and falls back to the last successful snapshot with a warning.
- **Cross-tool**: SKILL.md is an open standard and can be consumed by OpenCode / Claude Code / Codex simultaneously, so the hub is a "team-private skill marketplace".
- **Observability**: each retrieve records `{query, scope, returned_ids, latency, token_used}` to `local/runs/<id>/retrieval.jsonl`, feeding subsequent score decay and identification of never-retrieved records (→ deprecation candidates).

---

## 13. Governance · stability · availability (summary)

The three first-class documents under `policies/` solidify the rules: `promotion` (§8 triggers + §9 three gates + promotion path), `merge` (§10 two engines + dual review + no exemptions + CODEOWNERS), `deprecation` (invalidation governance). Release cadence: weekly minor versions, monthly stable versions, with `skills/core/` going through stricter review.

| Guarantee | Mechanism |
|---|---|
| Rollbackable | Each update has a tag + scorecard; `git revert` rolls back |
| Regression-resistant | **CI eval-gate**: failing forbids release (the fundamental mechanism of safe feedback) |
| Resistant to destructive rewrites | Textual learning rate + rejected-edit buffer + slow updates (the SkillOpt trifecta) |
| Auditable | Each record links back to source + reviewer + scorecard |
| Pollution-resistant | The candidate layer (L1) and the stable layer (L2/L3) are physically isolated |
| Multi-version coexistence | Different projects pin different versions, upgrading gradually |
| Highly available | lockfile + local fallback + resolver degradation |
| Invalidation governance | `superseded/deprecated` status + bi-temporal + periodic cleanup (kept auditable, not physically deleted) |
| Leak-proof | Redaction gate + CI secret-scan |
| Observable read path | Each retrieve writes `retrieval.jsonl`; long-unhit records auto-enter the deprecation candidates; latency / token_used feed §14 tuning directly |

---

## 14. Phased roadmap

| Phase | Period | Goal | Deliverable | Risk |
|---|---|---|---|---|
| **0 Extract** | 1–2w | Run the two repos with zero behavior change | Move `skills/agents/pipelines/commands/docs` to the hub + submodule pin; **path compatibility** (symlink/rewrite, §6.4); repo skeleton + schemas + registry | Low |
| **1 Distill + read path + local online resolution** | 3–5w | Tier0→1 structured **+ resolver read path online + local mem0 integration evaluation** | `hmopt sediment` (with LLM salience pass, §8); `memory export` converts to standard objects; **`resolver.py` + hybrid retrieval (§12) + context budget**; **local mem0 online resolution integration PoC** (§10.1.a): evaluate whether to reuse the mem0ai package + homegrown UPDATE prompt (bypass v3 OSS degradation) | Medium (mem0 v3 uncertainty) |
| **2 Curate + merge + promotion candidate detection** | 3–6w | Central batch merge online (engine A second tier) | Curator + lint/secret-scan/dedup CI; the three `policies/` documents; **automatic promotion-candidate detector (§11.5)**: opens promote-candidate PRs | Medium |
| **3 eval gate** ★ | 6–10w | Safe feedback (engine B) | **Build the core task suite** (the long pole) + CI eval-gate + scorecard; semi-automatic optimizer | **High** |
| **4 Auto-optimize** | 10w+ | Closed-loop auto-iteration | Scheduled job (§11); release cadence; `skill-memory.lock` drift-proofing | Medium |

---

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **The eval suite is the long pole** (real-machine A/B is slow/expensive/noisy) | Phase 3 focus; start with static proxy metrics + small-sample real-machine, densify gradually; honestly label it the critical path |
| Over/under sedimentation | Scoring + decay + N-confirmation threshold; L0–L3 filtering |
| Secret leakage | Redaction gate + CI secret-scan (mandatory) |
| Hot-file merge contention | One file per record + stable ID, sidestepping line-level merge |
| Mixing two asset classes in one engine | §4.1 mandatory separation |
| Multi-member edit collapse | GEPA Pareto (§10.2) |
| Skills dimension explosion | core/technique/domain three layers + file/function into knowledge + selector (§6.2) |
| eval exemption loophole | No exemptions; an exception can only be downgraded to a candidate + sign-off + re-review |
| Path hardcoding migration | Phase 0 symlink or rewrite + resolver (§6.4) |
| **mem0 v3 OSS capability shrinkage** | OSS has degraded to ADD-only + hash dedup (§3 note); when §10.1.a borrows mem0, this design **brings its own UPDATE/DELETE prompt** (per the v0.1.x paper version) to avoid coupling with the OSS degradation; Phase 1 must explicitly PoC-verify |
| **Retrieval quality degradation (new risk surface)** | Add retrieval observability (§12.4); build a small retrieval eval set ("given a query, does it hit the expected ID"), run a baseline once at the end of Phase 1, regress before each release |
| **markdown ↔ index drift** | cascade incremental re-embedding + content_sha256 check; each release generates `index/manifest.yaml` with the rebuild command; CI runs a "rebuild the index once → compare" check on every PR |
| **Local resolution mistakenly deletes historical facts** (review feedback ③) | The seven-path classifier never deletes temporal/conditional/selector/evidence (§10.1.a); the P1-8 PoC sets a **false-delete rate ≈ 0** hard metric, with the temporal/conditional sub-categories counted separately |
| **subsumption over-generalization** (review feedback ④) | Central LLM judgment only; linking is cheap but promotion requires **≥2 different subsumed instances** + the §9 three gates (§11.5); `skills/core/` candidates may require a higher instance gate (§17 issue 8) |
| **Opening Phase 1 before schema converges** | §17 issue 7 elevated to a P1 blocking prerequisite; Phase 0.5 DoD not met = do not start (implementation plan §1) |

---

## 16. Appendix

**Key CLI (proposed additions to `hmopt`)**

```bash
hmopt sediment [--bundle --open-pr]    # Tier0→1 distill / package candidates into a PR
hmopt skill-lock --update <semver>     # update skill-memory.lock
hmopt skill-eval <skill> --suite <s>   # run the skill eval locally, emit a scorecard
```

**Glossary**

| Term | Meaning |
|---|---|
| Tier 0/1/2 | run traces / candidate sediment / core shared |
| L0–L3 | draft / candidate / stable / core |
| Engine A / B | knowledge merge / skill merge |
| Three archetypes | procedural-shared / knowledge-curated / run-evidence-local |
| Textual learning rate | the bounded edit budget per release (SkillOpt) |
| Pareto frontier | a set of candidates each optimal on certain eval instances (GEPA) |
| eval-gate | the held-out verification regression gate before release — the fundamental mechanism of safe feedback |
| selector | the topology binding in skill frontmatter that resolves against the code index (subsystem/glob/symbol) |

---

## 17. Open decisions for the team

1. The hub repo's name and ownership, and which GitHub org to put it in.
2. eval ground truth: pure real-machine A/B vs static proxy vs hybrid (determines the Phase 3 schedule).
3. Whether to adopt the `technique/` layer (moves are currently implicit in the funnel scope labels).
4. Retrieval backend: **faiss + sqlite-fts5 (Phase 1 start) vs pgvector (aligns with the existing `storage/`) vs LanceDB (runs hybrid retrieval + scalar filtering in one query, the EverOS route)**. Recommend faiss from Phase 1, evaluate LanceDB in Phase 3.
5. The `skills/core/` owner team and the promotion reviewers.
6. **mem0 dependency strategy for local online resolution** (v2.2 newly added): ① fully homegrown reimplementation of the v0.1.x paper prompt; ② use the `mem0ai` OSS package for infrastructure + bring your own resolution prompt (avoiding the v3 degradation); ③ evaluate mem0 Platform. The decision affects the Phase 1 schedule.
   > **Decided (Phase 1 PoC, 2026-06-09): choose ① (homegrown), do not introduce a `mem0ai` runtime dependency.** Rationale: (a) mem0 v3 OSS has degraded to ADD-only + hash dedup (§3 note), so UPDATE/DELETE has to be reimplemented anyway; (b) this design's **seven-path relation classification** (§10.1.0) is finer than mem0's four-path ADD/UPDATE/DELETE/NOOP, so the classification logic must be homegrown; (c) homegrown keeps **offline determinism** (CI can run it, no LLM-gateway dependency) and builds in the "temporal/conditional/selector/evidence never delete" discipline. Implementation: `src/hmopt/memory/local_curator.py` (a deterministic heuristic classifier + an injectable LLM to override ambiguous items) + `curator_benchmark.py` (48 cases, accuracy 1.0 / false-delete 0). **Re-evaluation point retained**: when `local/memory/` grows in volume and ANN recall becomes the bottleneck, re-evaluate choosing ② to borrow `mem0ai`'s indexing/ANN infrastructure (infrastructure only, the classification prompt still homegrown) — at which point the §12.3 index backend (faiss/LanceDB) is decided together.
7. **Convergence of the markdown and schema on-disk format** (v2.3 elevated to a **P1 blocking prerequisite**, not an ordinary hygiene item): the current example (`A001-*.md` multi-record + custom fields) is inconsistent with `memory_item.schema.json`; downstream lint / dedup / retrieval scalar filter / Curator seven-path classification **all depend on the schema fields being stable**, so convergence must happen **before Phase 1**. Decided direction (§6.1 / §7): ① one record per file + frontmatter with all schema fields; ② **the file path encodes scope** and CI checks the path scope is consistent with the frontmatter scope; ③ `parse_memory.py` outputs a standard schema object, no per-category self-extended fields allowed; ④ the schema also adds `subsumes[]/subsumed_by[]/superseded_by[]/applies_when`. **Remaining open decision**: the path-encoding granularity (whether down to the `targets/<slug>/facts/` level).
8. **The LLM cost and misjudgment of subsumption** (v2.3 newly added): subsumption needs LLM entailment judgment, which is more expensive than dedup; and over-generalization carries risk. A ≥2-instance gate is already added as a fallback (§11.5), but it still needs a decision: the compute-budget cap for the central Curator running subsumption per round + whether to require a higher instance gate (e.g. ≥3) for `skills/core/` candidates.
