# Changelog

All notable changes to `hm-skill-hub`. Semver: MAJOR.MINOR.PATCH.

## [Unreleased]

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
