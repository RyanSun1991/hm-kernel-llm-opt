# Team Skill Hub — Design & Operations Guide (English)

> A team-level, versioned "experience hub": the experience accumulated while doing AI-driven kernel optimization (which mechanisms work, which are traps, how to optimize a given function, which validation is untrustworthy) is **accumulated across members, auto-validated, and safely reused** — forming a self-evolving closed loop: **consume → distill → curate → gate → release → consume**.
>
> This document covers: the core design philosophy, the directory layout, the interaction flow, MCP integration, and the **concrete commands and execution method** for each step. Chinese version: `Skill_Hub_Design_And_Operations_CN.md`.

---

## 1. One-liner & pain points

**One-liner**: externalize the optimization experience (effective mechanisms, traps, per-function fixes, untrustworthy validations) into a central team repo `hm-skill-hub` that is **quality-gated, versioned, and degradation-proof**, consumed like a "private npm package" and getting more accurate the more it is used.

**Pain point**: today experience stays on each engineer's machine (`.opencode/memory`) and cannot be reused across members/projects/time → repeated exploration, repeated dead-ends. Yet "just share it" risks (1) knowledge contradicting itself / drifting and (2) letting AI self-optimize its own experience falling into the **feedback self-reinforcement trap** (aggregate improves while specific cases regress).

---

## 2. Core design philosophy (key decisions)

### Decision 1: Two asset classes, two engines (most important)
| | **Knowledge (facts / lessons)** | **Skills (procedures / playbooks)** |
|---|---|---|
| e.g. | "shrink_node re-reads sc->priority; hoisting it saves 0.8%" | "before proposing, dedup against bad_plans" workflow |
| evolves by | **append-only + seven-way relation classification**; never physically deleted; contradictions kept bi-temporally | **in-place competitive editing**, protected by an eval gate |
| engine | **Engine A** (set-merge / dedup / conflict / subsumption) | **Engine B** (SkillOpt: bounded edit + Pareto + eval gate) |

> They are split because governing both with one git line-merge → knowledge self-contradicts and skills get overwritten by one person's bad week. **Two engines, two gates, kept separate.**

### Decision 2: Knowledge merge is seven-way relation classification, not duplicate/contradiction binary
Each incoming record vs existing is classified into 7 relations. **Iron rule: nothing is physically deleted except contradiction-with-stronger-evidence**:
`duplicate` (merge provenance) · `contradiction` (loser tombstoned `superseded`, not deleted) · `temporal` (stale kept auditable) · `conditional` (different conditions coexist) · `subsumption` (a general record generalizes a specific one; the specific stays as evidence) · `selector` (path change → re-resolve) · `evidence` (same delta, different measurement → merge).

### Decision 3: Markdown is the single source of truth
Each knowledge record = one `.md` (YAML frontmatter + markdown body). Git-reviewable, hand-editable, index-backend-swappable. **jsonl is ONLY the staging/transport format for a contribution; on landing it becomes `.md`.**

### Decision 4: Skills = trainable external parameters (anti-feedback safety)
Treat skill text as an external parameter of a frozen model. Any edit must be **strictly better with zero regression** on a held-out eval set to be accepted, else it goes to a `bad_edits` buffer. This is the root mechanism that defeats feedback self-reinforcement.

### Decision 5: Gating ≠ Curation
- **Gating = the gatekeeper**: automatic, objective, binary (pass/fail). Only asks "valid / safe / non-regressing"; it does not judge content quality or decide placement.
- **Curation = the editor/librarian**: human-led (tool-assisted). Asks "is it correct / worth sharing / what is its relation to existing records / where does it go / what stable ID".

### Decision 6: Reach the hub via MCP, not a local CLI
Agents run inside a kernel repo where `hmopt` is not on PATH, so they reach the hub through **MCP tools**; any hub call that fails **degrades silently and never blocks the main flow**.

---

## 3. Directory layout

### 3.1 Central repo `hm-skill-hub/`
```
hm-skill-hub/
├── registry.yaml              ★ index/TOC (version, schema hash, skill inventory); maintained by release.py
├── knowledge/                 [Knowledge] append-only, placed by scope (formal content = .md)
│   ├── global/{heuristics,anti_patterns,validation_pitfalls,bad_plans}/   H/A/V/B
│   ├── subsystems/<sub>/       subsystem-level
│   └── targets/<func-slug>/{facts,decisions,idea_ledger}/   F### / L###
├── skills/{core,domain,technique}/<name>/   SKILL.md + best_skill.md + scorecards/*.json
├── staging/<member>/<date>.jsonl  ★ contribution inbox (Tier-1 candidates; jsonl transport; not yet landed)
├── schemas/*.schema.json      the "fixed fields" per record (Phase 0.5 validation)
├── _registry/{subsystem_selectors,mechanisms}.yaml   path/symbol→subsystem, controlled mechanism vocab
├── eval/{task_suites,retrieval,scorecards}/   skill exams / retrieval test set / dashboard
├── policies/*.md              who decides / on what basis (merge/promotion/auto_merge)
├── releases/<version>.md      release notes
└── tools/*.py + ci_local.sh   full toolchain + local CI
```
**Key**: `knowledge/`'s directory structure itself encodes a record's scope (global / subsystems / targets); CI hard-enforces path↔frontmatter scope agreement.

### 3.2 Consumer side (kernel repo) `.opencode/`
```
.opencode/
├── skill-memory.lock          pins the consumed hub version (path/pin/hub_version)
├── memory/{idea_ledger,targets,subsystems,global_lessons.md}   local experience (resolve's --local-memory overlay)
├── local/sediment_staging/    distillation scratch (gitignored; <run>.jsonl + _bundle.jsonl)
├── hub/                        ★ mount the hub here (submodule) so the MCP server discovers it
└── ... (agents / commands / skills / state)
```

### 3.3 Record format (one file = one record)
- ID prefix = type: **F**=fact, **H**=heuristic, **A**=anti_pattern, **V**=validation_pitfall, **B**=bad_plan, **L**=idea.
- **The 9xx band is sediment's provisional IDs**; the stable ID (F001…) is assigned at curation and is hub-wide unique.
- frontmatter = structured fields (machine retrieval/validation); markdown body = human-readable explanation.

---

## 4. Interaction flow (self-evolving closed loop)

```
        ┌────────────  hm-skill-hub (central team repo · semver)────────┐
        │   knowledge/ (Engine A)         skills/ (Engine B)            │
        └──▲──────────────────────────────────────────┬───────────────┘
 ④ release  │ release + broadcast → update .opencode/hub + lock         │ ① consume (resolve)
 /broadcast │                                            ▼
 ③ curate   │  staging/ ◄── ② distill (sediment) ◄── one opt run ◄── ① mount knowledge as context
 + gate     │  (Engine A/B + CI gates + dual review)   (.opencode/memory+bench → candidates)
```

**Six hub touchpoints in the `.opencode` pipeline**:
- ★1 intake: read `skill-memory.lock` for the version
- ★2 before research: `skillhub_resolve(stage=research)` → inject `## Hub context`
- ★3 inside research: hub is the funnel's dedup source #8
- ★4 before plan-review: `skillhub_resolve(stage=plan-review)` → dedup gate
- ★5 decision: `skillhub_sediment` distills back
- ★6 release loop-back: broadcast updates `.opencode/hub` + lock → back to ★1

---

## 5. MCP integration

The platform runs a **skill-hub MCP server** (Docker, port 7338, compose service `hmopt-skillhub-mcp`) exposing three tools:

| MCP tool | Purpose |
|---|---|
| `skillhub_resolve(target, stage, [opencode_dir], [mechanism])` | **READ**: returns a `## Hub context` block (skills + knowledge + bad-plans) |
| `skillhub_sediment([opencode_dir], [contributor], [bundle])` | **WRITE**: distill `.opencode/memory` into candidates + `_bundle.jsonl` |
| `skillhub_status([opencode_dir])` | pinned hub version + reachability |

- The **server** (not the agent) does the `.opencode/` file I/O (mounted kernel repo), so the sub-agent filesystem sandbox does not limit hub access — **any mcp-enabled agent may call these tools**.
- `opencode_dir` defaults to the server's `HMOPT_SKILLHUB_OPENCODE_DIR` (compose default `/workspace/kernel/.opencode`).
- Co-located (with `hmopt` installed) the CLI equivalents are: `hmopt resolve … --local-memory .opencode/memory --run-dir .opencode/state` / `hmopt sediment-opencode --opencode-dir .opencode --bundle`.

---

## 6. Operations & concrete execution

### 6.1 Consume (read)
Before research/plan, the agent calls `skillhub_resolve(target, stage)` and uses the returned `## Hub context` block as context; the audit line is appended to `.opencode/state/retrieval.jsonl`.

### 6.2 Distill (write)
At run close-out, after local memory is written, call `skillhub_sediment(contributor=<you>, bundle=true)` → produces `.opencode/local/sediment_staging/_bundle.jsonl`.
> Non-empty requires local memory in the formats sediment reads (`### L00x`+`status: landed`+`delta_pct`; `## Known Bad Plans`; review `## Decision reject`; bench `verdict: pass`+`delta_pct`).

### 6.3 Contribute to the hub (first commit)
```bash
# ① review + triage (drop test/LLM-meta junk; fix bad mechanism / truncated titles first)
cat .opencode/local/sediment_staging/_bundle.jsonl
# ② place into the hub inbox
cd <hub>; mkdir -p staging/<member>
cp <kernel-repo>/.opencode/local/sediment_staging/_bundle.jsonl staging/<member>/$(date +%F).jsonl
# ③ gating pre-check (commit only when green)
bash tools/ci_local.sh
# ④ commit to the hub
git add staging/<member>/$(date +%F).jsonl && git commit -m "sediment: <member> <date>" && git push   # or open a PR
```
At this point it is an **inbox candidate awaiting review, not yet shared knowledge**.

### 6.4 Curate & land (curation, second commit)
```bash
cd <hub>
python tools/central_curate.py staging/<member>/<date>.jsonl --plan     # preview: assign stable IDs + paths
python tools/central_curate.py staging/<member>/<date>.jsonl --apply    # write knowledge/**/*.md (mechanical part automated)
#   does: seven-way classify (add/merge/conflict/subsumption), assign stable IDs, place by scope, write .md
#   merge/conflict left for a human; fix content quality (mechanism/title) here if not done earlier
bash tools/ci_local.sh                                                  # gating re-check (lint the new md)
# + dual review (1 domain + 1 process; bootstrap may self-approve)
git add knowledge/ && git commit -m "curate: land <ids>" && git push
git rm staging/<member>/<date>.jsonl                                    # inbox may be cleared (or kept for provenance)
```
**Now**: in `knowledge/` and pushed → anyone running `resolve` against this hub can retrieve it = (baseline) shared.

### 6.5 Release / broadcast (formal, version-governed sharing)
```bash
cd <hub>
python tools/nightly.py     # 7-step closed-loop report (dry-run)
python tools/release.py     # bump semver (schema change → major)
python tools/broadcast.py --hub-version=<v> --sha=<sha>   # update each consumer's .opencode/hub + skill-memory.lock
```
The next person's research/plan `resolve` then surfaces your record = **fully closed-loop sharing**.

### 6.6 Local CI (gating, no GitHub)
`tools/ci_local.sh` exactly mirrors the hub CI's five gates; run the whole set with one command locally or in Docker:
```bash
bash tools/ci_local.sh
# the 5 steps: pytest tools/tests/ · lint.py · redact.py --check · dedup.py --check over staging · eval_gate.py
docker compose run --rm hmopt bash -lc "bash hm-skill-hub/tools/ci_local.sh"   # inside Docker
```

---

## 7. Gating vs Curation (quick reference)
| | Gating | Curation |
|---|---|---|
| role | gatekeeper (machine / objective / binary) | editor-librarian (human / judgment / filing) |
| handles | schema · path-scope · id uniqueness · redaction · **unresolved conflict** · skill **eval regression** | **seven-way classification** · content fixes · stable-ID assignment · scope placement · **dual review** |
| tool | `ci_local.sh` (lint/redact/dedup/eval_gate) | `central_curate.py` (--report/--plan/--apply) + human |
| fully automatable? | yes | mechanical part yes (ID/path/file write); judgment stays human |

> dedup has a dual role: `dedup --check` is a gate (conflict → exit 1); its classification is also curation input (central_curate suggests add/merge/conflict from it).

---

## 8. Maturity ladder & promotion (who decides / on what basis)
| Level | Criteria | Lands in | Who decides |
|---|---|---|---|
| L0 draft | local, unstructured | `.opencode/local/` | yourself |
| L1 candidate | schema-complete + initial evidence | `staging/<member>/<date>.jsonl` | yourself |
| L2 stable | passes 3 gates + dual review | `knowledge/` or `skills/domain|technique/` | 1 domain + 1 process |
| L3 core | reused across ≥2 sub-teams + owner sign-off | `skills/core/` | 2 owners + CODEOWNERS |

- **Knowledge→skill graduation**: a mechanism proven on ≥2 independent targets → `promotion_detector` emits a signal → a human graduates it into a `technique` skill (the original records stay as evidence).
- **Versioning**: `release.py` infers semver deterministically (schema change → major).

---

## 9. Safety & degradation invariants
- **Anti-feedback safety**: a skill edit is accepted only if strictly better with zero regression.
- **Never physically delete**: retired records get a `superseded` tombstone; the audit trail stays complete.
- **Silent degradation**: MCP/hub unreachable → returns `hub: unavailable`; the main flow is never blocked.
- **Human gates**: landing needs dual review; auto-merge unlocks only after ≥3 consecutive improvements with zero rollbacks.

---

## 10. Honest boundaries (current state)
- Headline numbers (curator 1.0 / retrieval 1.0 / optimizer 0.67→1.00) are **reproducible but fixture/proxy-constructed** — they prove "control flow / pipeline wiring", not "real-machine capability". A real A/B instruction-count delta is the long pole.
- Contribution-PR human review, `--apply` actually mutating the hub, and broadcast actually releasing are controlled / semi-automatic; the sediment write path is wired but end-to-end needs a fully-provisioned environment to verify.
- Per-item reality (REAL/PARTIAL/STUB) is tracked in `docs/Session_Implementation_Review_CN.md` and `docs/Verify_And_Walkthrough_CN.md`.
