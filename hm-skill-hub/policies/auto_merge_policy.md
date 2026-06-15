# Auto-Merge Policy

Encodes design §11 early-safety + plan P4-5. Governs when the closed loop may
merge a skill change automatically versus when a human must merge it.

## Default: half-automatic

The loop starts **half-automatic**. The nightly job (`tools/nightly.py`) and the
optimizer (`tools/skill_optimizer.py`) may *propose* and open a PR, but a human
merges. This is the no-exemption default — nothing merges itself until it has
earned trust.

## Trust threshold (when auto-merge is allowed)

A skill becomes auto-merge eligible only after a proven, never-regressing track
record on its eval suite:

- **≥ N eval improvements** in its scorecard history (default `N = 3`), and
- **0 rollbacks** — no released version whose `pass_rate` dropped below the prior
  version.

`tools/auto_merge_gate.py` computes this from `skills/<name>/scorecards/*.json`
and returns `auto` or `human` per skill:

```bash
python tools/auto_merge_gate.py            # decision per skill
python tools/auto_merge_gate.py --min=3    # tune the improvement threshold
```

Until a skill clears the threshold, every change to it requires a human merge,
even if the eval-gate is green.

## Stricter scopes

- **`skills/core/`** (L3 golden-standard skills) never auto-merge on the default
  threshold alone — they additionally require CODEOWNERS approval (see
  `merge_policy.md` and `CODEOWNERS`). Consider a higher `N` for core.
- A single rollback **resets** the trust counter: the skill returns to
  half-automatic until it rebuilds `N` clean improvements.

## Safety interlocks (always on, even after trust)

Auto-merge never bypasses the other gates:

1. **eval-gate** (`tools/eval_gate.py`) must be green — a regression rejects the
   change regardless of trust.
2. **lint + redact** (`tools/lint.py`, `tools/redact.py --check`) must pass.
3. **No physical deletes** of knowledge (engine A discipline, `merge_policy.md`).

If any interlock fails, the change is rejected and the skill is **not** demoted
from trust unless the failure is a genuine eval regression (which is a rollback).

## Escalation back to manual

Revoke auto-merge (return a skill to `human`) when:

- a released version regresses (a rollback is recorded), or
- a downstream consumer reports a real-world regression not caught by the proxy
  eval, or
- the eval suite itself changes materially (the baseline is no longer comparable).

Record the revocation reason alongside the skill (e.g. in its `scorecards/`
trail) so the trust history stays auditable.
