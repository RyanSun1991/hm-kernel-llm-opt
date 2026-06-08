# Changelog

All notable changes to `hm-skill-hub`. Semver: MAJOR.MINOR.PATCH.

## [Unreleased]

## [0.1.0] — 2026-05-27

Phase 0 skeleton:

- 7 JSON-Schemas (bad_plan / global_lesson / memory_item / idea / skill_frontmatter / skill_patch / scorecard).
- `_registry/mechanisms.yaml` (16 starter mechanisms) + `subsystem_selectors.yaml` (5 starter subsystems).
- Policies: promotion / merge / deprecation.
- Tools: parse_memory.py · lint.py · redact.py.
- CI scaffold: `.github/workflows/ci.yml` (activates after subtree split).
- One example each: skill (`skills/core/example/`), bad_plan (B001), heuristic (H001), anti-pattern (A001), validation pitfall (V001).
