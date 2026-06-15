"""Deterministic Tier 0 -> Tier 1 extractors (design §8 stage 1).

Each extractor turns one kind of run artifact into a schema-valid candidate
record (wrapped with its schema name), tagged `maturity: L1` with a `source[]`
back-link to the run. The Curator (Phase 2) assigns final stable ids; the
provisional ids here only need to pass the schema pattern.

Mapping (plan P1-2):
    bench delta (landed, positive)  -> memory_item  fact / pattern
    review rejection                -> global_lesson anti_pattern  (+ bad_plan)
    idea-ledger state change        -> idea
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Candidate = dict[str, Any]  # {"schema": <name>, "record": {...}}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# --- structured inputs ------------------------------------------------------

@dataclass
class BenchResult:
    """An A/B validation outcome distilled from a bench report."""

    mechanism: str
    target: str  # e.g. "mm/vmscan.c::shrink_node"
    delta_pct: float
    compare_level: Literal["total", "process", "thread", "lib", "function"]
    validation_path: str
    verdict: Literal["landed", "reverted", "rejected", "deferred", "approved"] = "landed"
    subsystem: str | None = None
    note: str = ""


@dataclass
class ReviewVerdict:
    """A plan/code review decision on a candidate mechanism."""

    decision: Literal["approve", "reject"]
    mechanism: str
    target_pattern: str
    reason: str
    scope: Literal["function", "call-site", "data-flow", "subsystem", "architectural"] = "function"
    subsystems: list[str] = field(default_factory=lambda: ["*"])
    platforms: list[str] = field(default_factory=list)
    review_ref: str = ""


@dataclass
class LedgerChange:
    """A state-machine transition on an idea-ledger entry."""

    mechanism: str
    target_slug: str
    scope: str
    status: Literal["approved", "landed", "reverted", "rejected", "deferred"]
    rationale: str
    verdicted_by: str = "pipeline"
    verdicted_at: str = ""
    delta_pct: float | None = None
    compare_level: str | None = None
    validation_path: str | None = None
    reopen_trigger: str | None = None


# --- helpers ----------------------------------------------------------------

def _slugify(value: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-") or "task"


def _symbol_of(target: str) -> str | None:
    return target.split("::", 1)[1].strip() if "::" in target else None


# --- extractors -------------------------------------------------------------

def bench_to_fact(b: BenchResult, *, run_id: str, contributor: str, seq: int = 1) -> Candidate:
    """A confirmed positive bench delta -> a target-level memory_item fact."""
    slug = _slugify(b.target)
    sym = _symbol_of(b.target) or b.target
    scope: dict[str, Any] = {"level": "function", "target_slug": slug}
    if b.subsystem:
        scope["subsystem"] = b.subsystem
    record = {
        "id": f"F{900 + seq}",
        "type": "fact",
        "title": f"{b.mechanism} on {sym} gives {b.delta_pct:+.2f}% ({b.compare_level})",
        "body": (
            f"Applying `{b.mechanism}` to `{b.target}` produced a measured "
            f"{b.delta_pct:+.2f}% {b.compare_level}-level instruction-count delta "
            f"(validation: {b.validation_path}). {b.note}".strip()
        ),
        "scope": scope,
        "applies_when": f"target is {sym} and mechanism {b.mechanism} applies",
        "source": [
            {"kind": "run_id", "ref": run_id},
            {"kind": "bench", "ref": b.validation_path},
        ],
        "evidence": {
            "delta_pct": b.delta_pct,
            "compare_level": b.compare_level,
            "confirmations": 1,
        },
        "maturity": "L1",
        "status": "active",
        "contributor": contributor,
        "created_at": _now(),
    }
    return {"schema": "memory_item", "record": record}


def review_to_anti_pattern(v: ReviewVerdict, *, run_id: str, contributor: str, seq: int = 1
                           ) -> Candidate:
    """A review rejection -> a global anti_pattern lesson (Tier-1, tentative)."""
    record = {
        "id": f"A{900 + seq}",
        "lesson": f"{v.mechanism} on {v.target_pattern} was rejected: {v.reason}"[:300],
        "kind": "anti_pattern",
        "applies_when": v.target_pattern,
        "do_or_dont": f"don't: apply {v.mechanism} to {v.target_pattern} — {v.reason}"[:300],
        "tags": [v.mechanism, v.scope],
        "evidence": [{"kind": "review", "ref": v.review_ref or f"run:{run_id}"}],
        "confidence": "tentative",
        "added_on": _today(),
        "added_by": f"{contributor} (run {run_id})",
        "status": "active",
    }
    return {"schema": "global_lesson", "record": record}


def review_to_bad_plan(v: ReviewVerdict, *, run_id: str, seq: int = 1) -> Candidate:
    """A review rejection of a mechanism -> a bad_plan dedup join-target."""
    subs = v.subsystems or ["*"]
    applies_to: dict[str, Any] = {"subsystems": subs}
    if v.platforms:
        applies_to["platforms"] = v.platforms
    record = {
        "id": f"B{900 + seq}",
        "title": f"{v.mechanism} on {v.target_pattern}",
        "mechanism": v.mechanism,
        "target_pattern": v.target_pattern,
        "scope": v.scope,
        "applies_to": applies_to,
        "reason": v.reason[:1500],
        "evidence": [{"kind": "review", "ref": v.review_ref or f"run:{run_id}"}],
        "rejected_on": _today(),
        "rejected_by": f"pipeline (run {run_id})",
        "status": "active",
    }
    return {"schema": "bad_plan", "record": record}


def ledger_to_idea(c: LedgerChange, *, run_id: str, seq: int = 1) -> Candidate:
    """An idea-ledger state change -> an idea record (verdict ledger)."""
    record: dict[str, Any] = {
        "id": f"L{900 + seq}",
        "mechanism": c.mechanism,
        "target_slug": c.target_slug,
        "scope": c.scope,
        "status": c.status,
        "verdicted_by": c.verdicted_by,
        "verdicted_at": c.verdicted_at or f"run {run_id}",
        "rationale": c.rationale[:1500],
    }
    if c.status == "landed":
        record["landed_on"] = _now()
        record["delta_pct"] = c.delta_pct if c.delta_pct is not None else 0.0
        record["compare_level"] = c.compare_level or "function"
        record["validation_path"] = c.validation_path or f"run:{run_id}"
    elif c.status == "deferred":
        record["deferred_on"] = _now()
        record["reopen_trigger"] = c.reopen_trigger or "re-evaluate next cycle"
    elif c.status == "approved":
        record["approved_on"] = _now()
    elif c.status == "rejected":
        record["rejected_on"] = _now()
    return {"schema": "idea", "record": record}
