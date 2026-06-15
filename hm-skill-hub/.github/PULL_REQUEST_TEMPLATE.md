<!-- Sediment / promotion PR template (design §9 three gates + §10 two engines). -->

## Candidate source

- **Origin run(s) / member**: <!-- run_id(s), contributor -->
- **How produced**: <!-- hmopt sediment --bundle | manual | promotion_detector -->
- **Engine** (pick one per record family):
  - [ ] **Knowledge** (engine A): facts / lessons / anti_patterns / bad_plans / idea_ledger
  - [ ] **Skill** (engine B): `skills/{core,technique,domain}/...` (needs `skill_patch` + eval)

## Gate 1 — schema / lint / redact (CI auto)

- [ ] `python tools/lint.py` green (schema + path↔scope consistency + id uniqueness)
- [ ] `python tools/redact.py --check` green (no secrets)
- [ ] one record = one file, frontmatter complete, path encodes scope

## Gate 2 — evidence

- [ ] every knowledge record has ≥ 1 resolvable `source`/`evidence` ref
- [ ] skill edits carry a `skill_patch` manifest (`task_suite` + `metrics` + `baseline_version`)

## Gate 3 — curation + review

- [ ] `python tools/central_curate.py <batch>.jsonl` run; merge plan attached below
- [ ] **dedup**: no unresolved `conflict` (`python tools/dedup.py <batch>.jsonl --check`)
- [ ] **conflicts** resolved via double-time (superseded, **not** deleted)
- [ ] **subsumption**: general/specific links built; subsumed instances kept as evidence
- [ ] skills only: eval-gate strictly-better (pass_rate ↑ / regression_rate not ↑)

## Merge plan (from Curator)

```yaml
# paste the central_curate merge plan here
```

## Reviews (no exemption)

- [ ] **Domain reviewer** (is the conclusion correct?): @
- [ ] **Process reviewer** (format / reusability / compliance): @
- [ ] `skills/core/` changes: CODEOWNERS owner approval

> CRDT discipline: no physical deletes. Superseded/deprecated records are
> retained for audit. Promotion is suggested, never auto-merged.
