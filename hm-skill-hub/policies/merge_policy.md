# Merge Policy

Encodes design §10. **Two asset classes, two merge engines** — never mix them.

## Engine A — Knowledge (append-type): set-merge + dedup + conflict resolution

**Never use git line-level merge.** The Curator agent runs on the PR:

```
for item in incoming:
    dup = near_duplicate(item, hub_items)       # embedding similarity >= 0.92
    if dup:
        merge_provenance(dup, item)              # merge source[]; confirmations += 1
        continue
    conflict = contradiction(item, hub_items)    # same (target, mechanism), opposite assertion
    if conflict:
        if stronger_evidence(item):              # evidence/recency weighted (Zep double-time)
            conflict.status = "superseded"
            conflict.valid_until = now()
            item.supersedes = [conflict.id]
            add(item)
        elif high_risk:
            escalate_to_human(item, conflict)
        else:
            drop_with_citation(item, conflict)
    else:
        add(item)
```

**CRDT discipline**: append + tombstone (`active / superseded / deprecated`),
**never delete**.

## Engine B — Skills (edit-type): SkillOpt validation gate + GEPA Pareto

**Never use set-merge.** Each skill change = one `skill_patch` manifest (bounded
add/del/replace):

```
def merge_skill_edit(skill, edit):
    if edit in skill.bad_edits:       return REJECT("known-bad edit")
    edit = clip_to_budget(edit, textual_learning_rate)
    cand = apply(skill, edit)
    score = run_evals(cand, skill.eval_suite)
    if score.strictly_better_than(skill.score):
        skill = cand
        write_scorecard(skill, score)
    else:
        skill.bad_edits.append(edit)
    pareto = update_pareto(pareto, cand, per_instance_scores)   # complementary -> candidates/
```

- **Textual learning rate** = the bounded edit budget per release.
- **Pareto frontier** = when several members propose edits, keep the set that is
  "each best on some instance" in `skills/<name>/candidates/`, and periodically
  merge complementary lessons.

## Dual review

Every PR requires:
- **1 domain reviewer** (is the conclusion correct? is the mechanism sound?);
- **1 process reviewer** (schema compliance, stable-id uniqueness, complete
  double-time fields, not a hit against a deduped source).

A `skills/core/` change requires **2 owners** to sign off.

## No exemption

`metrics.pass_rate` not increasing, or `regression_rate` increasing → always
reject. **The only exception path**: downgrade to an L1 candidate + owner
sign-off + re-review next eval cycle. **"Merge despite eval" is not allowed.**

## Redaction (across every merge)

`tools/redact.py` rejects on a hit for any of:

| pattern | example |
|---|---|
| `aws-akid` | `AKIA...` |
| `ssh-priv` | `-----BEGIN ... PRIVATE KEY-----` |
| `generic-hex-key` | hex string of length ≥ 40 |
| `device-serial` | `serial=...` / `imei=...` |
| `dev-serial-path` | `/dev/ttyUSB<N>` / `/dev/serial/by-id/<…>` |
| `github-pat` | `ghp_...` |
| `slack-token` | `xox?-...` |

After manual redaction, replace with a `[REDACTED]` placeholder to resubmit.

## Actual commands (engine A central batch, Phase 2)

The seven-way relation classification, double-time resolution, and subsumption
link-building are all tooled (`tools/`, stdlib + pyyaml + jsonschema only,
offline-deterministic, runnable in CI straight after the subtree split):

```bash
# 1) one-shot dry-run: run subsumption -> dedup -> conflict -> promotion on a batch, emit a merge plan
python tools/central_curate.py staging/<batch>.jsonl --report=report.md

# 2) run each judgment on its own (CI gate / debugging)
python tools/dedup.py staging/<batch>.jsonl --check    # merge/new/conflict; conflict -> exit 1
python tools/subsumption.py                            # list generalization links (general subsumes specific)

# 3) conflict resolution: stronger incoming -> old record superseded + valid_until (double-time, no delete)
python tools/conflict_resolve.py <winner_path.md> <loser_path.md>
```

`central_curate.py` is the deterministic engine behind `merge_curator.md` (the
Curator agent prompt): the agent's merge plan must agree with it, and CI uses it
as a dry-run.

**Iron rule restated**: except "contradiction with stronger new evidence" (which
goes `superseded`), no branch ever physically deletes; temporal / conditional /
selector / evidence / subsumption all keep both records. Subsumption attaches the
specific instance as a `source` of the generalizing record — **never deduped and
absorbed**.
