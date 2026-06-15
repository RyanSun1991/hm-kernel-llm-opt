"""Load hub knowledge records into a normalized view for the central Curator.

Reuses `parse_memory.parse` (frontmatter -> object) + `path_scope` (schema
dispatch) so the Curator tools see exactly what `lint.py` validates. Normalizes
every record family (bad_plan / global_lesson / memory_item / idea) into one
`HubRecord` the dedup / conflict / subsumption / promotion tools reason over.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_memory import parse  # type: ignore[import-not-found]
from path_scope import derive_path_scope  # type: ignore[import-not-found]

HUB = Path(__file__).resolve().parent.parent
_SKIP_DIRS = {"index", "__pycache__"}
_SKIP_NAMES = {"README.md"}
# memory_item types / lesson kinds that are "general" (cross-target) vs specific.
_GENERAL_TYPES = {"rule", "pattern", "playbook_step"}
_GENERAL_LEVELS = {"global", "architectural", "subsystem"}


@dataclass
class HubRecord:
    id: str
    schema: str
    path: str
    fields: dict[str, Any] = field(default_factory=dict)
    body: str = ""

    @property
    def mechanism(self) -> str | None:
        return self.fields.get("mechanism")

    @property
    def aliases(self) -> list[str]:
        out = list(self.fields.get("aliases") or [])
        if self.mechanism:
            out.append(self.mechanism)
        return out

    @property
    def scope_level(self) -> str | None:
        sc = self.fields.get("scope")
        if isinstance(sc, dict):
            return sc.get("level")
        if self.schema in {"global_lesson"}:
            return "global"
        if self.schema == "bad_plan":
            subs = (self.fields.get("applies_to") or {}).get("subsystems")
            return "global" if subs == ["*"] else "subsystem"
        return None

    @property
    def subsystem(self) -> str | None:
        sc = self.fields.get("scope")
        if isinstance(sc, dict) and sc.get("subsystem"):
            return sc["subsystem"]
        subs = (self.fields.get("applies_to") or {}).get("subsystems")
        if isinstance(subs, list) and subs and subs != ["*"]:
            return subs[0]
        return None

    @property
    def target_slug(self) -> str | None:
        sc = self.fields.get("scope")
        if isinstance(sc, dict) and sc.get("target_slug"):
            return sc["target_slug"]
        return self.fields.get("target_slug")

    @property
    def applies_when(self) -> str:
        return str(self.fields.get("applies_when") or "")

    @property
    def status(self) -> str:
        return str(self.fields.get("status") or "active")

    @property
    def contributor(self) -> str:
        return str(
            self.fields.get("contributor")
            or self.fields.get("added_by")
            or self.fields.get("rejected_by")
            or self.fields.get("verdicted_by")
            or ""
        )

    @property
    def confirmations(self) -> int:
        ev = self.fields.get("evidence")
        if isinstance(ev, dict):
            return int(ev.get("confirmations") or 1)
        if isinstance(ev, list):
            return max(1, len(ev))
        return 1

    @property
    def delta_pct(self) -> float | None:
        ev = self.fields.get("evidence")
        if isinstance(ev, dict) and ev.get("delta_pct") is not None:
            return float(ev["delta_pct"])
        if self.fields.get("delta_pct") is not None:
            return float(self.fields["delta_pct"])
        return None

    @property
    def compare_level(self) -> str | None:
        ev = self.fields.get("evidence")
        if isinstance(ev, dict):
            return ev.get("compare_level")
        return self.fields.get("compare_level")

    @property
    def subsumes(self) -> list[str]:
        return list(self.fields.get("subsumes") or [])

    @property
    def granularity(self) -> str:
        """'general' (a generalization candidate) vs 'specific' (one instance).

        Target-relative scope dominates: a `pattern`-typed record pinned to a
        single function/call-site is still a *specific* instance, not a
        generalization (so it can be subsumed, not treated as the generalizer)."""
        if self.scope_level in {"function", "call-site", "data-flow"}:
            return "specific"
        if self.schema == "global_lesson":
            return "general"
        if self.scope_level in _GENERAL_LEVELS:
            return "general"
        if self.fields.get("type") in _GENERAL_TYPES:
            return "general"
        return "specific"

    @property
    def is_claim(self) -> bool:
        """idea_ledger entries are verdicts, not knowledge claims — they don't
        participate in subsumption/generalization."""
        return self.schema in {"memory_item", "global_lesson", "bad_plan"}

    def polarity(self) -> int:
        """+1 improvement / -1 regression / 0 unknown (mirrors local curator)."""
        d = self.delta_pct
        if d is not None:
            return 1 if d < 0 else (-1 if d > 0 else 0)
        do = str(self.fields.get("do_or_dont") or "").lower()
        # scan negatives first: "do: avoid X" / "do: don't Y" are negative despite
        # the "do:" prefix.
        if "don't" in do or "dont" in do or "avoid" in do or "watch out" in do:
            return -1
        if do.startswith("do:") or do.startswith("do "):
            return 1
        st = self.status
        if st in {"landed", "approved"}:
            return 1
        if st in {"rejected", "reverted"}:
            return -1
        if self.schema in {"bad_plan"}:
            return -1  # bad_plan is by definition a "don't"
        return 0

    def strength(self) -> tuple[int, int]:
        rank = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
        conf = {"tentative": 1, "observed": 2, "confirmed": 4}
        m = str(self.fields.get("maturity") or "")
        mr = rank.get(m, conf.get(str(self.fields.get("confidence") or ""), 1))
        return (mr, self.confirmations)

    def text(self) -> str:
        f = self.fields
        parts = [
            self.id,
            str(f.get("title") or ""),
            str(f.get("lesson") or ""),
            str(f.get("mechanism") or ""),
            str(f.get("target_pattern") or ""),
            self.applies_when,
            str(f.get("do_or_dont") or ""),
            str(f.get("reason") or ""),
            str(f.get("rationale") or ""),
            " ".join(map(str, f.get("tags") or [])),
            self.body,
        ]
        return "\n".join(p for p in parts if p)


def record_from_candidate(cand: dict[str, Any]) -> HubRecord:
    """Wrap an incoming sediment candidate {'schema', 'record'} as a HubRecord."""
    rec = dict(cand.get("record") or {})
    rid = str(rec.pop("id", "") or "")
    schema = str(cand.get("schema") or "memory_item")
    body = str(rec.pop("body", "") or "")
    return HubRecord(id=rid, schema=schema, path="<incoming>", fields=rec, body=body)


def load_hub_knowledge(hub_root: str | Path = HUB) -> list[HubRecord]:
    out: list[HubRecord] = []
    root = Path(hub_root) / "knowledge"
    if not root.exists():
        return out
    for p in sorted(root.rglob("*.md")):
        if p.name in _SKIP_NAMES or any(part in _SKIP_DIRS for part in p.parts):
            continue
        rel = "/".join(p.parts[p.parts.index("knowledge") + 1 :])
        ps = derive_path_scope(rel)
        if ps is None:
            continue
        try:
            recs = parse(p)
        except Exception:  # noqa: BLE001 - loader must be resilient
            continue
        for r in recs:
            out.append(HubRecord(id=r.id, schema=ps.schema, path=str(p), fields=r.fields, body=r.body))
    return out
