# Team Skill Hub Implementation Plan & Detailed Design

| Item | Value |
|---|---|
| Document status | Draft v1.2 (synced with design v2.3 review revisions) |
| Date | 2026-06-09 |
| Related design | `docs/Team_Skill_Hub_Design_CN.md` (v2.3) + `docs/Team_Skill_Hub_Design_Diagrams_CN.md` (includes the read-path diagram) |
| Scope | Task-level breakdown for Phase 0–4; v1.2 lands the review feedback (schema blocking / retrieval hard gate / conflict taxonomy / subsumption) into task cards |
| Language strategy | Prose in en-US; schema/code/CLI/commit in English |
| Revisions | v1.2 (review feedback): ① P0.5-2 promoted to a **Phase 1 blocking prerequisite gate** (+ path-encoding scope + CI consistency + schema additions subsumes/applies_when); ② P1-8 PoC benchmark expanded to a **seven-way conflict taxonomy** + false-delete≈0 hard metric; ③ P1-10 retrieval promoted to a **hard gate** (3 query classes + must/optional-hit + CI + symbol-name ablation); ④ Phase 2 adds P2-9 subsumption detector, feeding P2-8. v1.1: Phase 1 expanded to 3–5w (read path + local two-tier), Phase 2 adds P2-8 promotion detector |

---

## 0. Overall cadence

```
Phase 0 Extraction      1-2w   ┃ this session ★
Phase 1 Distillation    2-3w   ┃ next session
Phase 2 Curation & merge 3-6w  ┃
Phase 3 eval gate       6-10w  ┃ long pole
Phase 4 Auto-optimization 10w+ ┃
```

**Core constraints**:
- Everything stays on branch `claude/tender-cray-ABIsw`; the hub initially lives as a subdirectory `hm-skill-hub/` within this repo, and will later be split out into a standalone repo via `git subtree split --prefix=hm-skill-hub`.
- Every Phase has a "Definition of Done" (DoD); if it is not met, we do not proceed to the next stage.
- The existing `.opencode/{skills,agents,...}` content is **not touched in Phase 0**; we only build out the structure. Content migration is deferred to a dedicated Phase 0.5 session (with regression verification).

---

## 1. Phase 0 — Extraction (this session)

**Goal**: stand up the dual-repo skeleton with zero behavior change; the hub passes lint and can run a CI placeholder.

| ID | Task | Deliverable (path) | AC (acceptance) | Dependencies |
|---|---|---|---|---|
| P0-1 | Repo skeleton | `hm-skill-hub/{README,CONTRIBUTING,GOVERNANCE,CHANGELOG}.md`, `registry.yaml`, `.gitignore` | 6 files exist, content self-consistent | — |
| P0-2 | Empty directory placeholders | `skills/{core,technique,domain}/`, `knowledge/{global/{lessons,anti_patterns},subsystems,targets,index}/`, `evidence/{benchmarks,regressions}/`, `eval/{task_suites,scorecards}/`, `staging/`, `releases/` (each with a `.gitkeep`) | Directory tree exists | — |
| P0-3 | 7 JSON-Schemas | `schemas/{bad_plan,global_lesson,memory_item,idea,skill_frontmatter,skill_patch,scorecard}.schema.json` | Each is a valid JSON-Schema draft-07 | — |
| P0-4 | Controlled vocabulary | `_registry/{mechanisms,subsystem_selectors}.yaml` | mechanisms starts with ≥ 12 entries (hoist/inline/batch, etc.) | — |
| P0-5 | Three policy documents | `policies/{promotion,merge,deprecation}_policy.md` | Hardens design §8/§9/§10/§13 rules; directly executable by a human | P0-3 |
| P0-6 | Parser + lint CLI | `tools/{parse_memory,lint,redact}.py`, `tools/requirements.txt` | `python tools/lint.py` returns exit 0 on an empty hub; validates schema when examples are present | P0-3 |
| P0-7 | CI placeholder | `.github/workflows/ci.yml` (inside the hub) | Can be activated immediately after the repo split | P0-6 |
| P0-8 | 1 example | `knowledge/global/anti_patterns/A001-*.md`, `knowledge/global/lessons/H001-*.md`, `skills/core/example/SKILL.md` | Passes lint; reusable as a template | P0-3, P0-6 |
| P0-9 | Consumer-side placeholder | `.opencode/skill-memory.lock` (pinning placeholder) | Format self-describing; uses in-repo mode until the hub is split | — |

**DoD**:
- `python hm-skill-hub/tools/lint.py` returns exit 0 on the hub with examples included.
- The directory tree matches design §6.1.
- Any team member can follow `CONTRIBUTING.md` + the example files to author a `bad_plan` from scratch and pass lint.

**Phase 0.5 (standalone mini-session) — includes the Phase 1 blocking prerequisite gate**:

> **Status: P0.5-2 gate ✅ done (this session).** Delivered: ① one-record-per-file frontmatter (A001/H001/B001/V001 backfilled + target examples F001/L001); ② `tools/path_scope.py` path-encoding scope + `lint.py` strict path/frontmatter consistency check; ③ `parse_memory.py` changed to frontmatter→schema object; ④ schema additions `applies_when/subsumes[]/subsumed_by[]/superseded_by[]` (+ `idea.target_slug`); ⑤ `tools/tests/test_tools.py` 22 cases (runnable via pytest or standalone) wired into hub CI. **All gate ACs green**: `python tools/lint.py` exit 0, path/scope inconsistency reported precisely, schema contains the new fields. P0.5-1 (content migration symlink) still pending a standalone session.



- **P0.5-1 Content migration** — move the existing `.opencode/skills`, `agents`, `commands`, `pipelines`, `docs` (the harness-spec portions) into `hm-skill-hub/`, keeping the old paths usable via symlinks under `.opencode/`. With regression: run the existing `/optimize_generic` once to verify pipeline behavior is unchanged.
- **P0.5-2 schema / markdown on-disk format convergence ★blocking prerequisite** (v1.2 upgrade, review feedback ①, corresponds to design §6.1 / §7 / §17 issue 7): lint / dedup / retrieval scalar filter / Curator seven-way classification all depend on stable schema fields, so this **must be completed before Phase 1**. Delivered: ① **one record per file** (eliminate the stacked-multi-record `### A001` style), frontmatter uses the full standard schema fields, **no per-category field self-extension allowed**; ② **path-encoded scope** (`knowledge/targets/<slug>/facts/<ID>.md`, etc., §6.1); ③ `parse_memory.py` upgraded into a "frontmatter → standard schema object" converter; ④ `lint.py` changed to schema-driven + **add path scope vs frontmatter scope consistency check**; ⑤ schema additions `subsumes[]/subsumed_by[]/superseded_by[]/applies_when`; ⑥ one-time backfill of existing content. **AC (gate)**: `python tools/lint.py` returns exit 0 on all examples + any new entry; path/frontmatter scope inconsistency is reported precisely; schema contains the new fields; **if this gate does not pass, Phase 1 does not start**.

---

## 2. Phase 1 — Distillation + read path + local online resolution (3-5w, v1.1 expansion)

**Goal**: the pipeline close-out point can produce a schema-conformant Tier 1 candidate bundle; the resolver read path goes live (hybrid retrieval + context budget); the local online resolution PoC verification is complete.

> **Status (this session)**: P1-1/P1-2/P1-6/P1-7/P1-8/P1-9/P1-10 core landed (`src/hmopt/skillhub/` read path + `src/hmopt/sediment/` write path + `src/hmopt/memory/` local curator + `eval/retrieval/` hard gate). **Measured**: retrieval must-recall@5 = 1.0, each of the three query classes ≥8, symbol-name ablation **hybrid(1.0) > vector(0.8)** proves BM25 saves the day; local seven-way classifier benchmark 48 cases accuracy 1.0, temporal+conditional false-delete = 0, local zero subsumption; sediment produces schema-valid candidates. **Issue 6 (mem0 dependency) decided** (see design §17 issue 6). **Not wired in**: the real pipeline call at P1-3's three close-out hooks (needs a live run; this sandbox has no heavy dependencies), and P1-5 `--open-pr` actually opening a PR (only `--bundle` landed). The retrieval backend is a pure-Python BM25 + token-hashing vector (offline deterministic); faiss / a real embedder can be injected later.

| ID | Task | Deliverable | AC |
|---|---|---|---|
| P1-1 | `hmopt sediment` CLI | `src/hmopt/cli/sediment.py` + Typer registration | Called at the end of the pipeline; walks `.opencode/local/runs/<run_id>/`, extracts bench delta + idea ledger changes + the closed-out design summary → outputs `local/sediment_staging/<run_id>.jsonl` |
| P1-2 | Distillation rule mapping | `src/hmopt/sediment/extractors.py` (bench→facts, review→anti_patterns, ledger→idea record) | Each input class maps to one extractor; unit-test covered |
| P1-3 | Close-out hook integration | Modify the `os-opt-manager` decision stage + the end of `iterative-optimization` pass + primary-agent "done" | All three automatically call `hmopt sediment`; does not block the main flow |
| P1-4 | memory export | `tools/memory_export.py` (one-time script) | Converts the existing `memory/`, `plans/`, `reviews/` into standard objects; the output passes lint |
| P1-5 | Sediment PR tool | `hmopt sediment --bundle --open-pr` | Bundles candidates meeting the promotion trigger conditions into a hub PR; via the GitHub API; this repo → hub repo |
| **P1-6** | **resolver read path** | `src/hmopt/resolver/resolver.py` + unit tests | Inputs `(target, stage)`, resolves in the order of design §12.2: hub.skills selector hit → pull requires → call retrieve to query hub.knowledge + local.memory → merge & dedup → trim to the stage budget. Actually called by each pipeline stage |
| **P1-7** | **Hybrid retrieval + scalar filter** | `src/hmopt/resolver/retrieval.py` (faiss + sqlite-fts5 to start) + `tools/build_index.py` | Implements the design §12.1 pseudocode: scalar pre-filter → BM25 + vector RRF fusion + entity bonus + `score` weighting; returns top-k; retrieval.jsonl written to disk (§12.4 observability) |
| **P1-8** | **Local online resolution PoC** (review feedback ③) | `src/hmopt/memory/local_curator.py` (seven-way classifier, extending the mem0 v0.1.x paper-version prompt) + classification benchmark | One-of-three dependency decision (§17 issue 6). **The benchmark must cover all seven ways**: duplicate / contradiction / **temporal** (once-true now-stale) / **conditional** (both true, different `applies_when`) / **selector** (path changed after rebase) / **evidence** (same delta, different `compare_level`) / novel, ≥ 5 entries each, ≥ 40 total. **AC**: ① seven-way classification accuracy ≥ 0.85; ② **temporal + conditional sub-class false-delete rate ≈ 0** (wrongly deleting a historical/conditional fact is a PoC veto item); ③ end-to-end ≤ 3s/entry; ④ locally does **not** run subsumption (left to the central P2-9) |
| **P1-9** | **LLM salience-extraction pass** | Extend `extractors.py` with an LLM pass | The free-form text the rules leave behind (design summaries / reviewer notes) → run a `FACT_RETRIEVAL_PROMPT`-style extraction; output defaults to `confidence: tentative`; can be turned off via `--no-llm-extract` |
| **P1-10** | **retrieval eval hard gate** (review feedback ②) | `eval/retrieval/queries.yaml` + `tools/run_retrieval_eval.py` + CI integration | ① **≥ 8 entries for each of the three query classes**: target-anchored (`mm/vmscan.c::shrink_node`) / mechanism-anchored (`hoist-loop-invariant`) / free-form ("which plans were recently judged bad plan"); ② each expected ID is marked **must-hit / optional-hit**, computing strict recall and lenient recall separately; ③ **retrieval-logic PRs must run it**, using a **regression gate** early (no worse than the last green) and an absolute line once the corpus is large enough (must-hit recall@5 ≥ 0.8); ④ for **symbol-name queries**, separately report the three ablations BM25-only / vector-only / hybrid, proving hybrid ≥ each single path |

**DoD**:
- **P0.5-2 gate has passed** (schema convergence + path scope check green) — otherwise we do not enter Phase 1.
- Run a complete pipeline against the live system once, automatically dropping ≥ 1 schema-conformant Tier 1 candidate bundle; `hmopt sediment --bundle` can produce a local PR diff (does not have to actually be submitted).
- **The resolver is actually called at each pipeline stage**, retrieval.jsonl is genuinely written to disk; the retrieval hard gate is wired into CI, the hybrid ablation report for symbol-name queries is produced and hybrid ≥ each single path; the must-hit recall@5 baseline is recorded (regression gate in effect).
- **The local online resolution PoC** has run: seven-way classification accuracy ≥ 0.85, **temporal+conditional false-delete ≈ 0**; the mem0 dependency decision (issue 6) is written into the design document.

---

## 3. Phase 2 — Curation + merge (3-6w)

**Goal**: knowledge merge goes live (Engine A), CI strict checks, policies landed.

> **Status: Phase 2 core ✅ landed (this session).** `hm-skill-hub/tools/` adds an independent Curator toolchain (stdlib+pyyaml+jsonschema, offline deterministic): `dedup.py` (P2-2 three-state) · `conflict_resolve.py` (P2-3 bi-temporal, no delete) · `subsumption.py` (P2-9 generalization link-building + ≥2-instance emit) · `promotion_detector.py` (P2-8 clustering + subsumption, two paths) · `central_curate.py` + `merge_curator.md` (P2-1 §10.1.b orchestration) · `similarity.py` / `hub_records.py` (primitives). P2-4 CI dedup gate (hub ci + root ci) · P2-5 PR template · P2-6 policies actual command sections · P2-7 CODEOWNERS. **Tests**: `tools/tests/test_central_curator.py` 15 cases including the §10.1.b mock (subsumption≠dup/contradiction, promote only with ≥2 instances, subsumed instances retain evidence) + on the real hub identifies only H001→F001 with no false positives; hub tools total **43 tests** green. **Follow-ups**: real LLM entailment judgment integration, the Curator-agent actually running in OpenCode, a subsumption compute-budget cap (§17 issue 8) left for the Phase 2 wrap-up / Phase 3.

| ID | Task | Deliverable | AC |
|---|---|---|---|
| P2-1 | Curator-agent prompt | `hm-skill-hub/tools/merge_curator.md` | Loadable by OpenCode; input candidates + existing hub knowledge, output dedup / conflict / resolution decisions |
| P2-2 | Deduplicator | `tools/dedup.py` | Embedding similarity (faiss local) + alias hit; tunable threshold; outputs the three states "merge/new/conflict" |
| P2-3 | Conflict resolution | `tools/conflict_resolve.py` | Same (target, mechanism) with opposite assertions → Zep bi-temporal: the old entry marked `superseded`, `valid_until=now`, the new entry `supersedes=[old.id]` |
| P2-4 | CI: secret-scan + lint + dedup | Extend `.github/workflows/ci.yml` | gitleaks/trufflehog + lint + dedup all passing before merge is allowed |
| P2-5 | Sediment PR template | `hm-skill-hub/.github/PULL_REQUEST_TEMPLATE.md` | Mandatorily lists: candidate source, engine categorization, dual-review checklist |
| P2-6 | policies enhancement | Add an "actual commands" section to the promotion/merge/deprecation documents | Reviewers can execute directly |
| P2-7 | Dual-review configuration | `CODEOWNERS` + GitHub branch protection rules (documentation) | `skills/core/` requires owner + process review |
| **P2-8** | **Automatic promotion-candidate detector** | `tools/promotion_detector.py` (per design §11.5) | Two input paths: (a) hub knowledge clustering (mechanism + scope) with intra-cluster `confirmations` ≥ 3 across ≥ 2 contributors; (b) the `subsumes[] ≥ 2` generalization records **fed in by P2-9** → call the LLM to distill "technique + applicability conditions + evidence (including subsumed instances)" → automatically open a `promote-candidate` PR. **Discipline**: only proposes suggestions, never auto-merges |
| **P2-9** | **subsumption detector** (review feedback ④) | Extend `merge_curator` + `tools/subsumption.py` | Adds a third class of judgment in central batch merge: incoming vs the most recent k hub entries via **LLM entailment judgment**, identifying "generalization containment" (B subsumes A) → build the links `A.subsumed_by/B.subsumes`, A enters B's `source[]` (**does not dedup-swallow A**), emit a promotion signal. **AC**: on the mock set (including cases like "shrink_node hoist sc->priority" vs "reclaim hot-loop hoist loop-invariant") correctly judged as subsumption rather than dup/contradiction; emit to P2-8 only with **≥ 2 instances**; a single instance only builds links, no promotion |

**DoD**: run a PR end-to-end flow — member sediments locally → auto-opens a PR → CI all passes → Curator annotates the merge plan (including the subsumption judgment) → dual reviewers sign off → merge → the hub gains ≥ 1 L2 knowledge record; the **promotion detector**, on a mock knowledge set (one via the clustering path, one via the subsumption path), can identify ≥ 1 reasonable candidate and open a PR, and the specific subsumed instances are retained as evidence in the PR, not deleted.

---

## 4. Phase 3 — eval gate (6-10w) ★long pole

**Goal**: safe feedback of skill modifications (Engine B), the SkillOpt semi-automatic closed loop.

> **Status: Phase 3 core PoC ✅ landed (this session).** The skill being optimized, `skills/core/instruction-count-first/` (the seed `best_skill.md` deliberately incomplete, pass_rate 0.67). P3-1 suite `eval/task_suites/core_optimization_suite/` 9 cases (mm/wq/hyperhold, mechanism + guidance terms + avoid_term). `run_evals.py` (P3-2/P3-3, **pluggable ProxyScorer** proxying real-machine instruction counts) · `skill_optimizer.py` (P3-5/P3-7, bounded edits + strict-improvement gate + bad_edits buffer skipping) · `pareto.py` (P3-6, complementary candidates do not collapse) · `eval_gate.py` + CI (P3-4, reject if pass_rate drops). **DoD closed-loop demo**: the optimizer's one `hoist-invariant` bounded edit takes 0.67→1.00, zero regression, passes the gate and is accepted, emits a scorecard; the regressing edit goes into bad_edits. **Tests** `tools/tests/test_skillopt.py` 11 cases, hub tools total **59 tests** green. **Honest labeling**: the proxy is a keyword/mechanism coverage proxy (not a real-machine speedup proof); what is delivered is the SkillOpt control-flow scaffolding; the real-machine instruction-count estimator / real-machine A/B (P3-3 follow-up) is the long pole.

| ID | Task | Deliverable | AC |
|---|---|---|---|
| P3-1 | Eval sample collection | `eval/task_suites/<suite>/cases/*.yaml` | Each case: input target + expected optimization direction + grading rubric; initially ≥ 20 cases covering mm/wq/hyperhold |
| P3-2 | Eval executor | `tools/run_evals.py` | Given a skill version + task suite, runs all cases, emits `scorecards/<skill>__<semver>.json` |
| P3-3 | Proxy metrics | Static instruction-count estimator + small-sample real-machine A/B interface | Use the proxy early in Phase 3; encrypt on real machines later |
| P3-4 | eval-gate CI | Extend `.github/workflows/ci.yml` | Any `skills/**/` change triggers the evaluator; reject if `metrics.pass_rate` does not increase |
| P3-5 | bounded-edit optimizer | `tools/skill_optimizer.py` | Input rollout traces + the current skill, output bounded add/del/replace edit candidates |
| P3-6 | Pareto frontier | `tools/pareto.py` | Maintains per-instance score; keeps complementary candidates in `skills/<name>/candidates/` |
| P3-7 | bad_edits buffer | `skills/<name>/bad_edits.jsonl` | Edits rejected by eval are stored; the optimizer skips them directly next time |

**DoD**: manually trigger the optimization job → the optimizer proposes a bounded edit on a core skill → eval-gate runs automatically → if strictly better, auto-open a PR; if not better, the edit goes into the bad_edits buffer; a scorecard is produced.

**Reason it is a long pole**: the kernel-optimization ground truth = real-machine A/B instruction-count delta, which is slow/expensive/noisy. The early part of Phase 3 must start with a proxy metric.

---

## 5. Phase 4 — Auto-optimization (10w+)

**Goal**: the closed-loop auto-iteration runs day to day; weekly minor versions / monthly stable versions.

> **Status: Phase 4 core PoC ✅ landed (this session).** `nightly.py` + `nightly.yml` (P4-1, Collect→Normalize→Cluster→Optimize→Validate→Promote→Broadcast, default dry-run semi-automatic, --apply parameterized by the passed-in hub_root) · `release.py` (P4-2, semver inference major/minor/patch + registry rewrite + release notes) · `broadcast.py` (P4-3, regenerates `skill-memory.lock` in-repo/submodule + --open-pr stub) · `dashboard.py` (P4-4, `eval/scorecards/_dashboard.md` per-skill trend) · `auto_merge_gate.py` + `policies/auto_merge_policy.md` (P4-5, trust threshold ≥N improvements + 0 rollbacks before auto-merge). **registry already cut to 0.2.0** reflecting the 4 real skills, the consumer-side lock pinned to 0.2.0. **Tests** `tools/tests/test_phase4.py` 14 cases (including nightly dry-run zero side-effects + --apply temp-hub round-trip), hub tools total **78 tests** green. **policies all changed to English** (user request). **Long pole / follow-ups**: real-machine A/B instruction counts (P3-3), real GitHub PR automation (broadcast/promotion --open-pr actually connected), eval corpus expanded to ≥20 (P3-1), subsumption real LLM entailment judgment.

| ID | Task | Deliverable | AC |
|---|---|---|---|
| P4-1 | Scheduled optimization job | `.github/workflows/nightly.yml` (inside the hub) | nightly runs Collect→Normalize→Cluster→Optimize→Validate→Promote→Broadcast |
| P4-2 | Release tool | `tools/release.py` | Automatically computes the semver bump (patch/minor/major) + tags + generates release notes + updates `registry.yaml` |
| P4-3 | broadcast | `tools/broadcast.py` | After release, automatically opens a PR to the business repo updating `skill-memory.lock` |
| P4-4 | Monitoring dashboard | `eval/scorecards/_dashboard.md` (GitHub-rendered) | Per-skill score trend visualization |
| P4-5 | Semi-automatic → fully-automatic gate | `policies/auto_merge_policy.md` | After the trust threshold (N consecutive eval improvements + 0 rollbacks), auto-merge is allowed; before that it must be manual |

**DoD**: within one complete calendar week, the hub automatically produces ≥ 1 patch version, and after the business repo auto-pins it, the pipeline behavior shows a measurable positive change.

---

## 6. Cross-cutting concerns (spanning all Phases)

| Concern | Measures |
|---|---|
| **Security** | Redaction gate (`redact.py`) + CI secret-scan (gitleaks) + CODEOWNERS; no member can push directly to main, only via PR |
| **Performance** | Full lint runs in < 30s (a large repo needs incremental lint); CI eval runs in < 30min (using a proxy metric + caching) |
| **Path compatibility** | Phase 0.5 falls back to symlinks; Phase 1+ resolves uniformly inside `resolver.py` |
| **Documentation** | Each Phase synchronously updates the `Team_Skill_Hub_Design_CN.md` revision line; this plan document is maintained independently, ticked off after each Phase completes |
| **Rollback** | Tag git at the entry of each Phase; each hub release carries a scorecard for easy rollback diagnosis |

---

## 7. Roles & responsibilities (RACI lite)

| Role | Phase 0–2 | Phase 3 | Phase 4 |
|---|---|---|---|
| **Platform / tooling** | hub skeleton + toolchain (R) | eval executor (R) | scheduled jobs + release (R) |
| **Domain experts** | review examples (C) | eval case design (R) | review anomalies (C) |
| **Process reviewer** | policies review (A) | eval-gate design (A) | auto-merge gate (A) |
| **Business-repo users** | use the existing .opencode unchanged (I) | Phase 3.5 switch to hub-backed (C) | consume new versions (I) |

R=Responsible, A=Accountable, C=Consulted, I=Informed.

---

## 8. Critical path

```
P0-3(schemas) → P0-6(parser/lint) → P0-7(CI)        ← Phase 0 main chain
                       ↓
                P0.5-2(schema/md convergence) ★blocking prerequisite gate ← if not passed, Phase 1 does not start
                       ↓
P1-1(sediment) → P1-2(extractors) → P1-9(LLM salience)            ┐
                       ↓                                              ├→ P1-6(resolver) ← read-path main chain
                P1-4(memory export) → P1-7(hybrid retrieval) → P1-10(retr hard gate) ┘
                       ↓
                P1-8(local seven-way resolution PoC) → issue 6 decision
                       ↓
P2-2(dedup) → P2-3(conflict) → P2-9(subsumption) → P2-8(promotion detection) → P3-2(evaluator) → P3-4(eval-gate) ← long-pole endpoint
                                                                                  ↓
                                                                              P4-1(nightly)
```

**Most critical single points**:
- **P0-3** (schemas) — all subsequent lint / validation / merge rely on it. Landed in Phase 0.
- **P1-6 + P1-7** (resolver + hybrid retrieval) — the read path goes live, determining whether the latency and cost dividends of mem0 / EverOS can be captured; any downstream pipeline loading relies on it.
- **P1-8** (local online resolution PoC) — determines the mem0 dependency strategy, affecting the subsequent allocation of Curator workload in the P2 central layer.

---

## 9. This-session deliverables checklist (Phase 0 actual output)

The files that will exist after completion:

```
hm-skill-hub/
  README.md  CONTRIBUTING.md  GOVERNANCE.md  CHANGELOG.md
  registry.yaml  .gitignore
  schemas/{bad_plan,global_lesson,memory_item,idea,
           skill_frontmatter,skill_patch,scorecard}.schema.json   # 7 files
  _registry/{mechanisms,subsystem_selectors}.yaml
  policies/{promotion,merge,deprecation}_policy.md
  tools/{parse_memory,lint,redact}.py  tools/requirements.txt
  .github/workflows/ci.yml
  skills/{core,technique,domain}/.gitkeep
  skills/core/example/SKILL.md
  knowledge/global/{lessons,anti_patterns}/{.gitkeep, H001-*.md, A001-*.md}
  knowledge/{subsystems,targets,index}/.gitkeep
  evidence/{benchmarks,regressions}/.gitkeep
  eval/{task_suites,scorecards}/.gitkeep
  staging/.gitkeep  releases/.gitkeep
.opencode/
  skill-memory.lock                                                # placeholder
docs/
  Skill_Hub_Implementation_Plan_CN.md                              # this document
```

**Verification**: `python hm-skill-hub/tools/lint.py` returns exit 0 on the examples, schemas cross-reference correctly, the directory tree matches design §6.1.

---

## 10. Out of scope for this session (avoiding scope creep)

- Do not write the Curator-agent prompt (Phase 2)
- Do not write the SkillOpt optimizer, Pareto algorithm, or eval executor (Phase 3)
- Do not actually move the `.opencode/{skills,agents,...}` content (Phase 0.5 standalone session)
- Do not create the standalone GitHub repo (environment-constrained; this session only prepares the structure for a "one-line subtree split in the future")
- Do not wire in the sediment CLI / memory export (Phase 1)
- Do not write the nightly optimization job (Phase 4)
