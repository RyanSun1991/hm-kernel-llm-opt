# Promotion Policy

Encodes design §8 / §9 / §4.2. Decides when a candidate may be promoted from
Tier 1 (candidate) to Tier 2 (shared), and the L0→L3 maturity ladder.

## Triggers (any one + pass the three gates)

1. **Reproduced gain on ≥ 2 independent tasks** — same-direction bench delta on
   different targets.
2. **Single-task gain that is significant and bench-backed** — attach
   `validation_path` + `delta_pct`.
3. **A failure lesson with high reuse value** — promote into `anti_pattern`
   (A-series) or `bad_plan` (B-series) to prevent repeating the mistake.

## Three quality gates (passed in order)

```
candidate L1 ──▶ gate 1 Schema/Lint/Redact ──▶ gate 2 Evidence ──▶ gate 3 Curation + eval ──▶ L2 stable
                  CI automatic                  automatic           Curator + human + eval-gate
```

**Gate 1 · Schema / Lint / Redact** (CI automatic)
- `python tools/lint.py`: every record validates against its JSON-Schema.
- `python tools/redact.py --check`: a hit on device serial / hex key /
  `/dev/serial/by-id` / ssh private key / AKID / GHP / Slack token → reject.
- **Any failure → block the PR.**

**Gate 2 · Evidence** (CI automatic)
- Knowledge claims: `evidence[]` ≥ 1 item; references must resolve (bench path,
  commit hash, review path).
- Skill edits: must attach a `skill_patch` manifest (with `task_suite` +
  `metrics`); missing either → reject.
- **No evidence → stays at L1, cannot enter the hub.**

**Gate 3 · Curation + eval** (Curator + human + automatic)
- **Curator agent** pre-pass: dedup / conflict / Pareto (see `merge_policy.md`).
- **Dual review sign-off**: 1 domain reviewer (is the conclusion correct?) + 1
  process reviewer (compliance, reusability).
- **Skills** additionally pass the **eval-gate**: A/B on `eval/task_suites/<suite>/`;
  merged only if **strictly better** (pass_rate not down, regression_rate not up).
- **No exemption**: the only way around it is "downgrade to an L1 candidate +
  owner sign-off + re-review next cycle" — never a direct merge.

## L0 → L1 → L2 → L3 promotion path

| Level | Criteria | Action |
|---|---|---|
| **L0 draft** | local only, unstructured | in `.opencode/local/` |
| **L1 candidate** | schema-complete + initial evidence | land in `staging/<member>/<date>/*.jsonl` (one candidate per line; this is what `collect`/`dedup`/`nightly` read), open a PR |
| **L2 stable** | passes the three gates + dual review | merge into `knowledge/` or `skills/domain/` or `skills/technique/` |
| **L3 core** | reused successfully across ≥ 2 sub-teams + owner-team sign-off | promote into `skills/core/` (stricter review + a higher eval bar) |

## Scoring (for promotion ranking + decay)

```
score = w1·evidence_strength + w2·confirmations + w3·recency
      + w4·generality          − w5·counter_evidence − w6·staleness
```

Staleness decay: when an `invalidation` condition fires (e.g. a kernel rebase),
the score decays automatically, triggering `deprecation_policy.md`.

## Anti-examples: what may *not* go straight into the hub

- A single-target / single-function fact (file it under
  `knowledge/targets/<slug>/facts/` first; do not hard-code it into a skill).
- "Folk wisdom" whose wording is not yet stable (no evidence → keep as an L1 draft).
- Raw logs containing an un-redacted device serial / key (redact, then submit).
- A mechanism not registered in `_registry/mechanisms.yaml` (open a registration
  PR first).

## Actual commands (reviewers can run these directly, Phase 2)

```bash
# Gate 1 — schema / lint / redact
python tools/lint.py                          # schema + path/scope consistency + id uniqueness
python tools/redact.py --check                # secret scan

# Gate 3 — curation (read the auto-detection first, then adjudicate)
python tools/central_curate.py staging/<batch>.jsonl --report=report.md
python tools/dedup.py staging/<batch>.jsonl --check     # exit 1 on any unresolved conflict

# Promotion-candidate auto-detection (suggests only, never auto-merges)
python tools/promotion_detector.py            # clustering path + subsumption path
python tools/promotion_detector.py --pr-body  # render the promote-candidate PR body
python tools/subsumption.py --emit-only       # list only generalizations with >= 2 instances
```

The promotion-candidate detector **only suggests**: any `promote-candidate` PR
still passes the three gates + dual review in this file, with no exemption. The
specific subsumed instances are preserved as evidence in the PR, never deleted.
