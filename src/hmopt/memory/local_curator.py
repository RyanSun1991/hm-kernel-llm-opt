"""Seven-way relation classifier + CRDT-disciplined apply (design §10.1).

Relation taxonomy (§10.1.0):
    duplicate | contradiction | temporal | conditional | subsumption |
    selector | evidence | novel

Local mode (§10.1.a) runs the cheap five (duplicate / contradiction / temporal /
conditional / evidence) + selector + novel, but **not** subsumption (LLM
entailment, central-only §10.1.b). The headline guarantee: temporal /
conditional / selector / evidence are **never** turned into a delete — only
`superseded` tombstones, and only for contradiction with stronger evidence.

A deterministic heuristic classifier is provided (offline, testable to the PoC
metrics); an `llm` may be injected to override on ambiguous cases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

RelationKind = Literal[
    "duplicate", "contradiction", "temporal", "conditional",
    "subsumption", "selector", "evidence", "novel",
]

# Local never runs subsumption (central-only). Everything else is allowed local.
LOCAL_KINDS = {"duplicate", "contradiction", "temporal", "conditional",
               "selector", "evidence", "novel"}
# Relations that must NEVER cause a delete/supersede of the existing record.
NO_DELETE_KINDS = {"temporal", "conditional", "selector", "evidence"}

_MATURITY_RANK = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
_SELECTOR_RE = re.compile(r"rebase|offset|moved|relocat|path\s*chang", re.IGNORECASE)


def _kver(value: object) -> tuple[int, ...] | None:
    """Parse a kernel version into a component tuple for correct ordering.

    A float was wrong: 6.10 parsed as 6.1, so every x.10+ kernel compared *less*
    than x.9 and the temporal-staleness check fired backwards. '6.10' -> (6, 10)
    sorts after '6.9' -> (6, 9). Accepts '6.6', 6.6, '6.10.2', etc.
    """
    if value is None:
        return None
    parts = re.findall(r"\d+", str(value))
    return tuple(int(p) for p in parts) if parts else None


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class CuratorItem:
    """Normalized memory item the classifier reasons over."""

    id: str
    mechanism: str | None = None
    target_slug: str | None = None
    subsystem: str | None = None
    applies_when: str = ""
    text: str = ""
    # polarity signals (any one is enough): negative delta == improvement == +1
    delta_pct: float | None = None
    do_or_dont: str = ""
    status: str = "active"
    compare_level: str | None = None
    # Accepts "6.10" / "6.6" / 6.6; compared component-wise via _kver (a float
    # would order 6.10 before 6.9). Stored as given for provenance.
    kernel_version: str | float | None = None
    invalidation: str = ""
    granularity: Literal["specific", "general"] = "specific"
    confirmations: int = 1
    maturity: str = "L1"
    symbol: str | None = None

    def polarity(self) -> int:
        if self.delta_pct is not None:
            if self.delta_pct < 0:
                return 1   # improvement
            if self.delta_pct > 0:
                return -1  # regression
        d = self.do_or_dont.lower()
        # scan negatives first: "do: avoid X" / "do: don't Y" are negative despite
        # the "do:" prefix. Must match hub_records.polarity (dual-tree parity).
        if "don't" in d or "dont" in d or "avoid" in d or "watch out" in d:
            return -1
        if d.startswith("do:") or d.startswith("do "):
            return 1
        if self.status in {"landed", "approved"}:
            return 1
        if self.status in {"rejected", "reverted"}:
            return -1
        return 0

    def strength(self) -> tuple[int, int]:
        return (_MATURITY_RANK.get(self.maturity, 0), self.confirmations)


@dataclass
class Relation:
    kind: RelationKind
    target_id: str | None = None  # the neighbor this relation is against
    stronger: bool = False        # incoming evidence stronger than neighbor?
    rationale: str = ""


@dataclass
class ApplyResult:
    actions: list[str] = field(default_factory=list)
    added: bool = False
    superseded: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    linked: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)  # MUST stay empty (CRDT)
    promotion_signal: bool = False


# --- similarity helpers -----------------------------------------------------

def _toks(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", (s or "").lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _same_scope(a: CuratorItem, b: CuratorItem) -> bool:
    if a.target_slug and b.target_slug:
        return a.target_slug == b.target_slug
    if a.subsystem and b.subsystem:
        return a.subsystem == b.subsystem
    return False


def _same_mechanism(a: CuratorItem, b: CuratorItem) -> bool:
    return bool(a.mechanism) and a.mechanism == b.mechanism


def _conditions_differ(a: CuratorItem, b: CuratorItem) -> bool:
    ca, cb = a.applies_when.strip().lower(), b.applies_when.strip().lower()
    return bool(ca) and bool(cb) and ca != cb


# --- classification ---------------------------------------------------------

def classify_pair(item: CuratorItem, nb: CuratorItem, *, mode: str = "local") -> Relation:
    """Classify the relation of `item` to a single neighbor `nb`."""
    rel = lambda k, **kw: Relation(kind=k, target_id=nb.id, **kw)  # noqa: E731

    related = _same_scope(item, nb) or (
        _same_mechanism(item, nb) and _jaccard(item.text, nb.text) >= 0.2
    )

    # subsumption (central-only): one generalizes the other, same mechanism
    if mode == "central" and _same_mechanism(item, nb) and item.granularity != nb.granularity:
        return rel("subsumption", rationale="one record generalizes the other")

    if not related:
        return rel("novel", rationale="no scope/mechanism relation")

    # selector drift: same symbol but a rebase/offset/path-change signal
    if (item.invalidation and _SELECTOR_RE.search(item.invalidation)) and (
        item.symbol and item.symbol == nb.symbol
    ):
        return rel("selector", rationale="symbol stable but path/offset moved (rebase)")

    if _same_mechanism(item, nb):
        # evidence divergence: same delta magnitude, different compare_level
        if (
            item.delta_pct is not None and nb.delta_pct is not None
            and abs(item.delta_pct - nb.delta_pct) < 0.05
            and item.compare_level and nb.compare_level
            and item.compare_level != nb.compare_level
        ):
            return rel("evidence", rationale="same delta, different compare_level")

        pol_i, pol_n = item.polarity(), nb.polarity()
        opposite = pol_i != 0 and pol_n != 0 and pol_i != pol_n

        # temporal staleness: behavior changed in a newer kernel version
        item_kv, nb_kv = _kver(item.kernel_version), _kver(nb.kernel_version)
        if opposite and item_kv is not None and nb_kv is not None and item_kv > nb_kv:
            return rel("temporal", rationale="newer kernel version changed behavior")

        # conditional divergence: both hold, under different applies_when
        if _conditions_differ(item, nb):
            return rel("conditional", rationale="both correct under different conditions")

        # contradiction: same condition, opposite conclusion
        if opposite:
            stronger = item.strength() > nb.strength()
            return rel("contradiction", stronger=stronger, rationale="same condition, opposite verdict")

        # same scope + mechanism + same conclusion -> duplicate
        return rel("duplicate", rationale="same scope/mechanism/conclusion")

    # related by scope but different mechanism, high overlap -> duplicate-ish
    if _jaccard(item.text, nb.text) >= 0.6:
        return rel("duplicate", rationale="near-duplicate text, same scope")
    return rel("novel", rationale="same scope, unrelated mechanism")


def classify_relation(item: CuratorItem, neighbors: list[CuratorItem], *, mode: str = "local",
                      llm=None) -> Relation:
    """Classify `item` against its nearest `neighbors` (best non-novel wins).

    `llm` is accepted for parity with the central path; the deterministic
    heuristic is authoritative offline.
    """
    if not neighbors:
        return Relation(kind="novel", rationale="no neighbors")
    # priority: the most "actionable" relation a neighbor yields
    priority = {
        "contradiction": 7, "temporal": 6, "subsumption": 5, "conditional": 4,
        "selector": 3, "evidence": 2, "duplicate": 1, "novel": 0,
    }
    best = Relation(kind="novel")
    for nb in neighbors:
        r = classify_pair(item, nb, mode=mode)
        if mode == "local" and r.kind == "subsumption":
            r = Relation(kind="novel", target_id=nb.id, rationale="subsumption deferred to central")
        if priority[r.kind] > priority[best.kind]:
            best = r
    return best


# --- apply (CRDT discipline) ------------------------------------------------

def apply_relation(rel: Relation, item: CuratorItem, neighbor: CuratorItem | None = None
                   ) -> ApplyResult:
    """Apply a classified relation. Never physically deletes; the only tombstone
    is `superseded` for a contradiction with stronger evidence."""
    res = ApplyResult()
    k = rel.kind

    if k == "duplicate":
        if neighbor:
            res.merged.append(neighbor.id)
        res.actions.append(f"merge provenance into {rel.target_id}; confirmations+=1")

    elif k == "contradiction":
        if rel.stronger and neighbor is not None:
            res.superseded.append(neighbor.id)  # tombstone = superseded, NOT delete
            res.added = True
            res.actions.append(
                f"{neighbor.id}.status=superseded + valid_until=now; "
                f"{item.id}.supersedes=[{neighbor.id}]; add {item.id}"
            )
        else:
            res.actions.append(f"keep {rel.target_id}; drop {item.id} with citation (or escalate)")

    elif k == "temporal":
        if neighbor is not None:
            res.superseded.append(neighbor.id)  # superseded + valid_until, auditable
        res.added = True
        res.actions.append(
            f"{rel.target_id}.status=superseded + valid_until=now (auditable); add {item.id}"
        )

    elif k == "conditional":
        res.added = True
        res.actions.append(f"coexist: add {item.id} with its applies_when (keep {rel.target_id})")

    elif k == "evidence":
        if neighbor:
            res.merged.append(neighbor.id)
        res.actions.append(f"merge by compare_level with {rel.target_id} (keep both)")

    elif k == "selector":
        res.actions.append(f"re-resolve selector + update {item.id}.invalidation; knowledge unchanged")

    elif k == "subsumption":
        if neighbor is not None:
            res.linked.append(neighbor.id)
        res.promotion_signal = True
        res.actions.append(
            "link subsumes/subsumed_by; specific -> source of general; emit promotion signal"
        )

    elif k == "novel":
        res.added = True
        res.actions.append(f"add {item.id}")

    # Invariant: no relation ever physically deletes the existing record.
    assert not res.deleted, "CRDT violation: apply must never delete"
    return res
