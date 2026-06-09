# Changelog

All notable changes to `hm-skill-hub`. Semver: MAJOR.MINOR.PATCH.

## [Unreleased]

## [0.2.0] — 2026-06-09

Phases 0.5–4 (schema convergence → read/write path → central curation → eval gate
→ auto-optimization). Cut by `tools/release.py` (minor: four new skills).

### Phase 4 — auto-optimization loop (design §11, plan P4-1..P4-5)

- **P4-1 `nightly.py` + `.github/workflows/nightly.yml`** — the nightly closed
  loop: Collect → Normalize → Cluster → Optimize → Validate → Promote → Broadcast,
  composing the engine-A and engine-B tools. **Dry-run / half-automatic by
  default** (§11 early-safety); `--apply` is the trusted path, parameterized on
  the hub it is handed (never mutates a module-level constant).
- **P4-2 `release.py`** — semver bump inference (major: removal/schema change;
  minor: new/raised skill; patch: knowledge-only) + `registry.yaml` rewrite +
  release notes.
- **P4-3 `broadcast.py`** — regenerates the consumer `skill-memory.lock`
  (in-repo + submodule forms); `--open-pr` is a stub (half-automatic).
- **P4-4 `dashboard.py`** — `eval/scorecards/_dashboard.md` per-skill score trend
  (semver-ordered, ▲/▼ deltas).
- **P4-5 `auto_merge_gate.py` + `policies/auto_merge_policy.md`** — trust
  threshold: auto-merge only after ≥ N eval improvements with 0 rollbacks;
  half-automatic (human merge) until then.
- **registry.yaml** cut to **0.2.0** reflecting the four real skills; consumer
  lock pinned to 0.2.0.
- **Policies are now English-only** (promotion / merge / deprecation /
  auto_merge), per request.
- **Tests**: `tools/tests/test_phase4.py` (14 cases) incl. the nightly dry-run
  safety (mutates nothing) and an `--apply`-on-a-temp-hub round-trip. 78 hub-tool
  tests total.

### Phase 3 — eval gate + SkillOpt (engine B, design §9 / §10.2 / §13)

- **Skill under optimization**: `skills/core/instruction-count-first/` with a
  deliberately-incomplete `best_skill.md` seed (pass_rate 0.67) so the
  optimizer→gate→accept loop is reproducible.
- **P3-1 task suite**: `eval/task_suites/core_optimization_suite/` — 9 cases
  across mm-reclaim / workqueue / hyperhold mapping a hot pattern to its
  expected mechanism + guidance terms (+ a bad_plan avoid_term).
- **P3-2/P3-3 `run_evals.py`** — eval executor + **pluggable** `ProxyScorer`
  (static keyword/mechanism-coverage proxy standing in for real-machine A/B
  instruction-count delta — the long pole, §15); writes `scorecards/<skill>__<semver>.json`.
- **P3-5/P3-7 `skill_optimizer.py`** — bounded-edit optimizer: proposes one
  guidance edit for the heaviest failing mechanism group, accepts only a
  **strictly-better, no-regression** candidate (monotone gate), buffers rejects
  in `bad_edits.jsonl` and skips them next round (textual learning rate + slow
  update). Demo: 0.67 → 1.00 via one `hoist-invariant` edit, zero regression.
- **P3-6 `pareto.py`** — GEPA Pareto frontier (keeps complementary candidates,
  drops dominated) so multi-member edits don't collapse to one local optimum.
- **P3-4 `eval_gate.py` + CI** — re-evaluates each skill's `best_skill.md` vs its
  committed scorecard; any `pass_rate` drop / instance regression rejects the
  change. Wired into hub `ci.yml` + root `skill-hub-ci.yml`.
- **Tests**: `tools/tests/test_skillopt.py` (11 cases) — proxy scoring, strict-
  better/regression logic, Pareto dominance, the optimize loop (improves under
  the gate, no regression), bad_edits skip, and the eval-gate (passes on seed,
  flags a regression). 59 hub-tool tests total.
- **Honesty**: the proxy is keyword/coverage-based, NOT a kernel-speedup claim;
  it exercises the SkillOpt *control flow* end-to-end and the scorer is swappable
  for a static instr-count estimator or a real-machine harness (P3-3 follow-up).

### Phase 2 — central curation + merge (engine A second level, design §10.1.b / §11.5)

- **Curator toolchain** (`tools/`, stdlib + pyyaml + jsonschema, offline-deterministic,
  split-out-ready): `similarity.py` (token-hashing + Jaccard), `hub_records.py`
  (normalized loader reusing parse_memory + path_scope).
- **P2-2 `dedup.py`** — three-state (merge / new / conflict); `--check` fails CI on
  any unresolved conflict.
- **P2-3 `conflict_resolve.py`** — Zep double-time: stronger evidence supersedes
  (status=superseded + valid_until + superseded_by / supersedes), **never deletes**.
- **P2-9 `subsumption.py`** — generalization detector (B subsumes A): builds links,
  carries the specific instance as a `source` of the general (not absorbed), emits a
  promotion signal only at ≥ 2 distinct instances (anti over-generalization).
- **P2-8 `promotion_detector.py`** — two paths: clustering (mechanism+scope,
  confirmations ≥ 3 across ≥ 2 contributors) + subsumption (subsumes ≥ 2); suggests
  `promote-candidate` PRs, never auto-merges; subsumed instances kept as evidence.
- **P2-1 `central_curate.py` + `merge_curator.md`** — orchestrator realizing the
  §10.1.b seven-route decision (subsumption-before-dedup) + the Curator agent prompt.
- **P2-4 CI** — hub `ci.yml` + root `skill-hub-ci.yml` gain a dedup gate; toolchain
  tests cover the new tools.
- **P2-5 / P2-6 / P2-7** — sediment PR template, CODEOWNERS (dual-review), and
  "actual commands" sections in promotion / merge / deprecation policies.
- **Tests**: `tools/tests/test_central_curator.py` (15 cases) incl. the §10.1.b mock
  (subsumption ≠ dup/contradiction; ≥2-instance promotion; evidence preserved) and a
  real-hub assertion that only the designed H001→F001 link is detected (no false
  positives). 43 hub-tool tests total (pytest or standalone).

### Phase 1 — hub-side additions (consumer read/write path lives in the business repo `src/hmopt/`)

- Example skill scaffolds so the resolver's `selector → domain → requires` chain
  is exercisable end-to-end: `skills/domain/mm-reclaim/` + `skills/technique/hoist-loop-invariant/`.
- Retrieval-eval assets (plan P1-10): `eval/retrieval/{corpus.yaml,queries.yaml,baseline.json}`
  — a fixture corpus + 26 labelled queries (target/mechanism/free-form, >=8 each)
  with must/optional-hit tags and bare-symbol ablation discriminators. Lives
  under `eval/` so `lint.py` (knowledge/ + skills/ only) ignores it.
- Example target knowledge renamed to the `_slugify` convention
  (`targets/mm-vmscan-c-shrink-node/`) so target_slug aligns across resolver /
  runs / sediment.

### Phase 0.5 — schema / markdown convergence gate (design §6.1 / §7, blocking gate for Phase 1)

- **One record = one file**: every knowledge record is now a `.md` with YAML
  frontmatter (full schema fields) + markdown body. Dropped the legacy
  heading-delimited (`### A001` + `- **key**: value`) multi-record format.
- **Path encodes scope, CI-enforced consistency**: new `tools/path_scope.py`
  derives the expected schema + scope from a record's path and `lint.py` now
  rejects any path↔frontmatter scope mismatch with a precise message.
- **Schema fields added** (design §7): `memory_item` gains `applies_when`,
  `superseded_by[]`, `subsumes[]`, `subsumed_by[]` (+ a status=superseded ⇒
  superseded_by rule); `global_lesson` / `bad_plan` gain `subsumes[]` /
  `subsumed_by[]`; `idea` gains `target_slug` (path-consistency anchor).
- **Tools rewritten**: `parse_memory.py` is now a frontmatter→schema-object
  converter (one file → one record); `lint.py` is schema-driven + path-scope.
- **Backfill**: A001 / H001 / B001 / V001 migrated to the new format; added a
  worked target example (`targets/mm-vmscan-shrink_node/` facts F001 + idea
  L001) exercising the new fields and a subsumption link (H001 subsumes F001).
- **Tests**: `tools/tests/test_tools.py` (22 cases, runnable via pytest **or**
  standalone) covering parse, path-scope derivation, mismatch detection, and
  end-to-end lint; wired into hub CI.

## [0.1.0] — 2026-05-27

Phase 0 skeleton:

- 7 JSON-Schemas (bad_plan / global_lesson / memory_item / idea / skill_frontmatter / skill_patch / scorecard).
- `_registry/mechanisms.yaml` (16 starter mechanisms) + `subsystem_selectors.yaml` (5 starter subsystems).
- Policies: promotion / merge / deprecation.
- Tools: parse_memory.py · lint.py · redact.py.
- CI scaffold: `.github/workflows/ci.yml` (activates after subtree split).
- One example each: skill (`skills/core/example/`), bad_plan (B001), heuristic (H001), anti-pattern (A001), validation pitfall (V001).
