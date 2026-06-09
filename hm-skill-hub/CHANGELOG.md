# Changelog

All notable changes to `hm-skill-hub`. Semver: MAJOR.MINOR.PATCH.

## [Unreleased]

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
