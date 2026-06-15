"""LLM salience pass (design §8 stage 2, plan P1-9).

The rule extractors capture structured metrics; this pass reads the *free-form*
leftovers (design summaries, reviewer notes) and asks an LLM to surface reusable
insight the rules miss. Products default to `confidence: tentative` and need
later confirmations to mature (§8).

Offline-safe: with no usable LLM (or an unparseable reply) it yields nothing, so
the deterministic write path is never blocked. Disable entirely with
`--no-llm-extract`.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from .extractors import Candidate

FACT_PROMPT = (
    "You are distilling reusable kernel-optimization lessons from a run's notes.\n"
    "Extract at most {max_items} GENERAL, reusable heuristics (not target-specific facts).\n"
    "Return ONLY a JSON array; each item:\n"
    '  {{"lesson": "<=1 line takeaway", "applies_when": "trigger condition", '
    '"do_or_dont": "do: ... / don\'t: ...", "tags": ["t1","t2"]}}\n'
    "If nothing is genuinely reusable, return []."
)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _parse_array(reply: str) -> list[dict[str, Any]]:
    m = _JSON_ARRAY_RE.search(reply or "")
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def _build_messages(text: str, max_items: int):
    system = FACT_PROMPT.format(max_items=max_items)
    try:
        from hmopt.core.llm import ChatMessage

        return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=text)]
    except Exception:  # pragma: no cover - fallback to dicts for non-LLMClient
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]


def llm_salience_pass(
    free_texts: list[str],
    llm: Any | None,
    *,
    run_id: str,
    contributor: str = "llm-salience",
    max_items: int = 5,
) -> list[Candidate]:
    """Extract tentative global_lesson heuristics from free-form run text."""
    if llm is None or not free_texts:
        return []
    text = "\n\n---\n\n".join(t.strip() for t in free_texts if t.strip())
    if not text:
        return []
    try:
        reply = llm.chat(_build_messages(text, max_items))
    except Exception:  # pragma: no cover - never break the pipeline on LLM error
        return []

    out: list[Candidate] = []
    for i, item in enumerate(_parse_array(reply)[:max_items], start=1):
        lesson = str(item.get("lesson") or "").strip()
        if not lesson:
            continue
        record = {
            "id": f"H{950 + i}",
            "lesson": lesson[:300],
            "kind": "heuristic",
            "applies_when": str(item.get("applies_when") or "unspecified")[:300],
            "do_or_dont": str(item.get("do_or_dont") or f"consider: {lesson}")[:300],
            "tags": [str(t) for t in (item.get("tags") or ["llm-salience"])][:6] or ["llm-salience"],
            "evidence": [{"kind": "run_id", "ref": run_id}],
            "confidence": "tentative",
            "added_on": _today(),
            "added_by": f"{contributor} (run {run_id})",
            "status": "active",
        }
        out.append({"schema": "global_lesson", "record": record})
    return out
