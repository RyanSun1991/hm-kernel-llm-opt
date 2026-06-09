"""Load hub / local knowledge records from markdown frontmatter.

Post Phase-0.5 convergence the hub stores **one record per file** as YAML
frontmatter + markdown body, and markdown is the single source of truth
(design §6.1). The consumer side reads those files directly and normalizes
every record family (bad_plan / global_lesson / memory_item / idea) into one
flat view that retrieval / scalar-filtering can use uniformly.

We deliberately read *frontmatter* for scope rather than re-deriving it from the
path: the hub's own CI (`tools/path_scope.py`) already guarantees path and
frontmatter agree, so the consumer trusts the frontmatter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    raise ImportError("PyYAML required for skillhub.records") from None

_FRONTMATTER_RE = re.compile(r"^---\n(?P<fm>.*?)\n---\n?(?P<body>.*)$", re.DOTALL)
_SKIP_DIRS = {"index", "__pycache__"}
_SKIP_NAMES = {"README.md"}

# confidence (global_lesson) -> pseudo-maturity, so the maturity scalar filter
# can treat lessons uniformly. Anything curated into the hub is at least stable.
_CONFIDENCE_MATURITY = {"tentative": "L1", "observed": "L2", "confirmed": "L3"}
_MATURITY_RANK = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


@dataclass
class Record:
    """A normalized knowledge record (any family), flattened for retrieval."""

    id: str
    kind: str  # bad_plan | global_lesson | memory_item | idea
    path: str
    fields: dict[str, Any] = field(default_factory=dict)
    body: str = ""
    origin: str = "hub"  # hub | local

    # --- normalized accessors (uniform across families) --------------------

    @property
    def title(self) -> str:
        f = self.fields
        return str(f.get("title") or f.get("lesson") or f.get("mechanism") or self.id)

    @property
    def level(self) -> str | None:
        scope = self.fields.get("scope")
        if isinstance(scope, dict):
            return scope.get("level")
        if self.kind in {"global_lesson", "bad_plan"}:
            subs = (self.fields.get("applies_to") or {}).get("subsystems")
            return "global" if (subs == ["*"] or self.kind == "global_lesson") else "subsystem"
        return None

    @property
    def subsystem(self) -> str | None:
        scope = self.fields.get("scope")
        if isinstance(scope, dict) and scope.get("subsystem"):
            return scope["subsystem"]
        subs = (self.fields.get("applies_to") or {}).get("subsystems")
        if isinstance(subs, list) and subs and subs != ["*"]:
            return subs[0]
        return None

    @property
    def target_slug(self) -> str | None:
        scope = self.fields.get("scope")
        if isinstance(scope, dict) and scope.get("target_slug"):
            return scope["target_slug"]
        return self.fields.get("target_slug")

    @property
    def mechanism(self) -> str | None:
        return self.fields.get("mechanism")

    @property
    def maturity(self) -> str:
        m = self.fields.get("maturity")
        if m:
            return str(m)
        conf = self.fields.get("confidence")
        if conf in _CONFIDENCE_MATURITY:
            return _CONFIDENCE_MATURITY[conf]
        # Anything that made it into the hub is curated => treat as stable.
        return "L2" if self.origin == "hub" else "L1"

    @property
    def status(self) -> str:
        return str(self.fields.get("status") or "active")

    @property
    def score(self) -> float:
        try:
            return float(self.fields.get("score", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def search_text(self) -> str:
        """Concatenated text used for lexical + vector indexing."""
        f = self.fields
        parts = [
            self.id,
            str(f.get("title") or ""),
            str(f.get("lesson") or ""),
            str(f.get("mechanism") or ""),
            str(f.get("target_pattern") or ""),
            str(f.get("applies_when") or ""),
            str(f.get("do_or_dont") or ""),
            str(f.get("reason") or ""),
            str(f.get("rationale") or ""),
            str(f.get("scope") if isinstance(f.get("scope"), str) else ""),
            " ".join(map(str, f.get("tags") or [])),
            self.body,
        ]
        return "\n".join(p for p in parts if p)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError("missing YAML frontmatter")
    fm = yaml.safe_load(m.group("fm"))
    if not isinstance(fm, dict):
        raise ValueError("frontmatter is not a mapping")
    return fm, m.group("body").strip()


def infer_kind(fields: dict[str, Any], path: Path) -> str:
    """Infer record family from frontmatter (path is a tie-breaker)."""
    parts = path.parts
    if "idea_ledger" in parts:
        return "idea"
    if "bad_plans" in parts or ("mechanism" in fields and "rejected_by" in fields):
        return "bad_plan"
    if "mechanism" in fields and "verdicted_by" in fields:
        return "idea"
    if "lesson" in fields and "kind" in fields:
        return "global_lesson"
    if fields.get("type") in {"fact", "rule", "pattern", "anti_pattern", "playbook_step"}:
        return "memory_item"
    # default: dispatch by id prefix
    rid = str(fields.get("id", ""))
    if rid[:1] == "B":
        return "bad_plan"
    if rid[:1] in {"H", "A", "V"}:
        return "global_lesson"
    if rid[:1] == "L":
        return "idea"
    return "memory_item"


def load_record(path: Path, origin: str = "hub") -> Record | None:
    try:
        fm, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    rid = fm.get("id")
    if not isinstance(rid, str) or not rid:
        return None
    fields = {k: v for k, v in fm.items() if k != "id"}
    return Record(id=rid, kind=infer_kind(fm, path), path=str(path), fields=fields, body=body, origin=origin)


def load_records(knowledge_root: str | Path, origin: str = "hub") -> list[Record]:
    """Load all records under a `knowledge/` (or `memory/`) tree."""
    root = Path(knowledge_root)
    if not root.exists():
        return []
    out: list[Record] = []
    for p in sorted(root.rglob("*.md")):
        if p.name in _SKIP_NAMES or any(part in _SKIP_DIRS for part in p.parts):
            continue
        rec = load_record(p, origin=origin)
        if rec is not None:
            out.append(rec)
    return out


def maturity_at_least(maturity: str, floor: str) -> bool:
    return _MATURITY_RANK.get(maturity, 0) >= _MATURITY_RANK.get(floor, 0)
