"""Tier 0 -> Tier 1 distillation ("sediment", design §8, plan P1-1/P1-2/P1-9).

At pipeline close-out points a run's artifacts (bench A/B reports, plan/code
review verdicts, idea-ledger state changes) are distilled into schema-valid
Tier-1 candidates and written to `local/sediment_staging/<run_id>.jsonl`, ready
to be bundled into a hub PR.

Two-stage extraction (§8):
  1. deterministic rule extractors (`extractors.py`) — never lose structured
     fields (delta_pct / compare_level / source[]).
  2. optional LLM salience pass (`salience.py`) — capture reusable free-form
     insight the rules miss; products default to confidence=tentative.
"""

from __future__ import annotations

__all__ = ["extractors", "readers", "pipeline", "salience"]
