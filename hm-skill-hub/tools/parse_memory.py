"""Parser for hub Markdown record files (bad_plans, global_lessons, idea_ledger, etc.).

Format: heading-delimited records with `- **key**: value` field blocks.
See policies/promotion_policy.md and the Memory System spec for the grammar.

Stdlib + PyYAML only. CLI:
    python tools/parse_memory.py <path.md>          # print parsed records as JSON
    python tools/parse_memory.py --section <path>   # restrict to a markdown section
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write("PyYAML required: pip install pyyaml\n")
    sys.exit(2)

RECORD_HDR = re.compile(r"^###\s+(?P<id>[A-Z][0-9]{3,})(?:\s+[—\-:]\s+(?P<title>.+?))?\s*$")
SECTION_HDR = re.compile(r"^##\s+(?P<name>.+?)\s*$")
FIELD_LINE = re.compile(r"^-\s+\*\*(?P<key>[A-Za-z_][\w.]*)\*\*\s*:\s*(?P<val>.*)$")
INDENT_RE = re.compile(r"^(?P<indent>    +)(?P<rest>.*)$")
LIST_ITEM = re.compile(r"^\s*-\s+(?P<val>.+)$")


@dataclass
class Record:
    id: str
    title: str = ""
    section: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""
    line_start: int = 0


def _stringify_dates(obj: Any) -> Any:
    """Walk parsed structure; convert date/datetime back to ISO strings so
    JSON-Schema's `format: date` (which requires string type) is satisfied."""
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    if isinstance(obj, _dt.date):
        return obj.isoformat()
    if isinstance(obj, list):
        return [_stringify_dates(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _stringify_dates(v) for k, v in obj.items()}
    return obj


def _coerce(raw: str) -> Any:
    s = raw.strip()
    if not s:
        return None
    # inline YAML list / mapping
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        return yaml.safe_load(s)
    # bare scalar — try YAML scalar coercion (numbers, true/false, dates)
    try:
        val = yaml.safe_load(s)
        if isinstance(val, str) or val is None:
            return s  # keep original whitespace for plain strings
        return val
    except yaml.YAMLError:
        return s


def _read_nested_block(lines: list[str], start: int) -> tuple[Any, int]:
    """Read an indented block starting at `start`. Returns (parsed_value, next_index)."""
    block: list[str] = []
    j = start
    while j < len(lines):
        ln = lines[j]
        if ln.strip() == "":
            block.append("")
            j += 1
            continue
        m = INDENT_RE.match(ln)
        if not m:
            break
        block.append(m.group("rest"))
        j += 1
    # strip trailing empties
    while block and block[-1] == "":
        block.pop()
    if not block:
        return None, j
    # bullet list?
    if all(b.lstrip().startswith("- ") or b == "" for b in block if b):
        items: list[Any] = []
        for b in block:
            if not b:
                continue
            m = LIST_ITEM.match(b)
            if m:
                items.append(_coerce(m.group("val")))
        return items, j
    # YAML mapping fallback
    return yaml.safe_load("\n".join(block)), j


def parse(path: Path) -> list[Record]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[Record] = []
    cur: Record | None = None
    section: str | None = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        if m := SECTION_HDR.match(ln):
            section = m.group("name").strip()
            i += 1
            continue
        if m := RECORD_HDR.match(ln):
            if cur is not None:
                records.append(cur)
            cur = Record(
                id=m.group("id"),
                title=(m.group("title") or "").strip(),
                section=section,
                source_path=str(path),
                line_start=i + 1,
            )
            i += 1
            continue
        if cur is None:
            i += 1
            continue
        if m := FIELD_LINE.match(ln):
            key = m.group("key")
            val_inline = m.group("val")
            if val_inline.strip() == "":
                val, i = _read_nested_block(lines, i + 1)
                cur.fields[key] = val
                continue
            cur.fields[key] = _coerce(val_inline)
            i += 1
            continue
        i += 1
    if cur is not None:
        records.append(cur)
    for r in records:
        r.fields = _stringify_dates(r.fields)
    return records


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    path = Path(argv[0])
    if not path.exists():
        sys.stderr.write(f"file not found: {path}\n")
        return 2
    recs = parse(path)
    print(json.dumps([asdict(r) for r in recs], ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
