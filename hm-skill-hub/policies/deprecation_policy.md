# Deprecation Policy

Encodes design §13.3. Invalidation governance: when a record moves from "in use"
to "superseded / deprecated", and how to **keep it auditable** without polluting
retrieval.

## State machine

```
   active ──stronger new evidence──▶ superseded   (old record valid_until=now, new record supersedes=[old id])
      │
      │ invalidation fires
      │   OR   long-run score decay
      │   OR   N consecutive counter-examples
      ▼
   deprecated   (excluded from retrieval / inline; not physically deleted)
```

**The status field is schema-enforced** (`status: active | superseded |
deprecated`). **Never physically delete** — this is the floor for auditability
and "why did we do it this way before?" investigations.

## Triggers

| Trigger | Action |
|---|---|
| Stronger evidence for the same (target, mechanism) | Curator auto-sets the old record `superseded`, `valid_until=now()`, and the new record `supersedes=[old.id]` |
| `invalidation` field fires (e.g. `"recheck offset after rebase"`) | mark `deprecated`; attach a deprecation_reason |
| ≥ 3 consecutive counter-example evidence items | mark `deprecated`; write an anti_pattern A### explaining why it was believed correct before |
| The referenced mechanism is removed from `_registry/mechanisms.yaml` | set the record `status=deprecated`, leaving a reference to the successor mechanism |
| A skill's eval pass_rate drops for ≥ 2 consecutive cycles | set the skill `status=deprecated`; keep the `best_skill.md` historical snapshot; switch to another candidate on the Pareto frontier |

## Periodic cleanup (the Phase 4 nightly job)

```
nightly:
  scan all records with status in {superseded, deprecated}
  → keep in the repo (audit trail)
  → exclude from the RAG index rebuild
  → exclude from @-inline context assembly
  → tag with a deprecation_reason if missing
```

## Reactivation (revisit)

`deprecated` is not the end. If new evidence shows the earlier judgment was wrong:
1. Do not change the old record's status (preserve history).
2. Write a **new** record (new ID), referencing the old one in `related_ids`.
3. Annotate `"revisits B017: new evidence in bench/<...>"`.

The idea_ledger's `deferred + reopen_trigger` mechanism is another form of the
same idea — `deferred` is not `deprecated`; it is "re-openable once conditions are
met".

## What not to do

- ❌ Physically delete any record (unless illegal / a leak; using `git
  filter-repo` requires owner-team approval + a recorded ADR).
- ❌ Change a stable ID.
- ❌ Directly reset `confirmations / score`; let the score decay naturally by
  adding counter-evidence.

## Actual commands (reviewers can run these directly)

```bash
# Supersede: stronger new evidence -> old record superseded + valid_until + superseded_by (no delete)
python tools/conflict_resolve.py <winner_path.md> <loser_path.md>

# Mark deprecated (invalidation fired / repeated counter-examples / mechanism retired): hand-edit frontmatter
#   status: deprecated
#   (optionally add deprecation_reason, related_ids pointing to the successor)
python tools/lint.py            # confirm the status change still passes schema (superseded => needs superseded_by)

# Cleanup (Phase 4 nightly): superseded/deprecated stay in the repo but are excluded from index rebuild + @-inline
#   see the nightly job; for now, verify the status field by hand
```

New `memory_item` constraint: `status: superseded` must also set
`superseded_by[]` (schema-enforced), so the double-time chain stays complete and
auditably traceable.
