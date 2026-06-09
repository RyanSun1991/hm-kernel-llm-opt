"""Conflict resolution via Zep-style double-time (engine A, design §10.1 / P2-3).

When two records assert opposite conclusions for the same (target, mechanism,
condition), the **stronger-evidence** record wins WITHOUT deleting the loser:

    loser.status      = "superseded"
    loser.valid_until = now
    loser.superseded_by = [winner.id]
    winner.supersedes  += [loser.id]

If the incoming is NOT stronger, it is dropped-with-citation (or escalated for
high-risk), and nothing in the hub changes. **Never a physical delete** — the
superseded record stays auditable (CRDT discipline).

Pure functions return the mutated frontmatter dicts; `apply_to_files` writes the
tombstone back to the loser's markdown frontmatter (markdown = source of truth).

CLI:
    python tools/conflict_resolve.py <winner_path.md> <loser_path.md>
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hub_records import HubRecord, record_from_candidate  # type: ignore
from parse_memory import parse, parse_frontmatter  # type: ignore


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Resolution:
    decision: str  # supersede | drop | escalate
    winner_id: str | None = None
    loser_id: str | None = None
    actions: list[str] = field(default_factory=list)
    winner_fields: dict | None = None
    loser_fields: dict | None = None


def resolve(incoming: HubRecord, existing: HubRecord, *, high_risk: bool = False) -> Resolution:
    """Resolve a contradiction between `incoming` and an existing hub record."""
    stronger = incoming.strength() > existing.strength()
    if stronger:
        loser = dict(existing.fields)
        loser["status"] = "superseded"
        loser["valid_until"] = _now()
        loser["superseded_by"] = _as_list(loser.get("superseded_by")) + [incoming.id] \
            if existing.schema == "memory_item" else incoming.id
        winner = dict(incoming.fields)
        if incoming.schema == "memory_item":
            winner["supersedes"] = _as_list(winner.get("supersedes")) + [existing.id]
        return Resolution(
            decision="supersede", winner_id=incoming.id, loser_id=existing.id,
            winner_fields=winner, loser_fields=loser,
            actions=[
                f"{existing.id}.status=superseded + valid_until=now",
                f"{existing.id}.superseded_by += {incoming.id}",
                f"{incoming.id}.supersedes += {existing.id}",
                f"add {incoming.id} (auditable; no delete)",
            ],
        )
    if high_risk:
        return Resolution(decision="escalate", winner_id=existing.id, loser_id=incoming.id,
                          actions=[f"escalate to human: weaker incoming {incoming.id} "
                                   f"contradicts {existing.id}"])
    return Resolution(decision="drop", winner_id=existing.id, loser_id=incoming.id,
                      actions=[f"drop {incoming.id} with citation to {existing.id} "
                               f"(weaker evidence); hub unchanged"])


def _as_list(v) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, list) else [v]


def _write_frontmatter(path: Path, new_fields: dict) -> None:
    """Rewrite a record file's frontmatter (preserving the markdown body)."""
    import yaml

    _, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    fm = {"id": _record_id(path), **new_fields}
    text = "---\n" + yaml.safe_dump(fm, sort_keys=False, allow_unicode=True) + "---\n\n" + body + "\n"
    path.write_text(text, encoding="utf-8")


def _record_id(path: Path) -> str:
    return parse(path)[0].id


def apply_to_files(winner_path: Path, loser_path: Path, *, high_risk: bool = False) -> Resolution:
    winner = _load(winner_path)
    loser = _load(loser_path)
    res = resolve(winner, loser, high_risk=high_risk)
    if res.decision == "supersede" and res.loser_fields is not None:
        _write_frontmatter(loser_path, res.loser_fields)
    return res


def _load(path: Path) -> HubRecord:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from path_scope import derive_path_scope  # type: ignore

    rel = "/".join(path.parts[path.parts.index("knowledge") + 1 :]) if "knowledge" in path.parts else path.name
    ps = derive_path_scope(rel)
    rec = parse(path)[0]
    schema = ps.schema if ps else "memory_item"
    return record_from_candidate({"schema": schema, "record": {"id": rec.id, **rec.fields, "body": rec.body}})


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        sys.stderr.write("usage: conflict_resolve.py <winner_path.md> <loser_path.md>\n")
        return 2
    res = apply_to_files(Path(argv[0]), Path(argv[1]))
    print(f"decision: {res.decision}")
    for a in res.actions:
        print(f"  - {a}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
