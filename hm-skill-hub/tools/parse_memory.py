"""Parser for hub knowledge record files.

v2 (Phase 0.5 convergence, design §6.1 / §7): every knowledge record is **one
file = one record**. The file is YAML frontmatter (all standard schema fields)
followed by a markdown body:

    ---
    id: A001
    type: anti_pattern
    title: ...
    scope: {level: global}
    source: [{kind: review, ref: ...}]
    ...
    ---

    # human-readable title

    free-form markdown body (becomes the `body` field for schemas that have one)

This replaces the legacy heading-delimited (`### A001` + `- **key**: value`)
multi-record format. `parse()` returns the frontmatter as a standard schema
object so `lint.py` can validate it directly.

Stdlib + PyYAML only. CLI:
    python tools/parse_memory.py <path.md>     # print parsed record as JSON
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

FRONTMATTER_RE = re.compile(r"^---\n(?P<fm>.*?)\n---\n?(?P<body>.*)$", re.DOTALL)


class ParseError(ValueError):
    """Raised when a record file cannot be parsed into a frontmatter record."""


@dataclass
class Record:
    id: str
    fields: dict[str, Any] = field(default_factory=dict)
    body: str = ""
    source_path: str = ""
    line_start: int = 1


def _stringify_dates(obj: Any) -> Any:
    """Convert date/datetime back to ISO strings so JSON-Schema's `format: date`
    (which requires a string instance) is satisfied after YAML coercion."""
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    if isinstance(obj, _dt.date):
        return obj.isoformat()
    if isinstance(obj, list):
        return [_stringify_dates(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _stringify_dates(v) for k, v in obj.items()}
    return obj


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a record file into (frontmatter_dict, body_text)."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        raise ParseError("missing YAML frontmatter (file must start with '---' ... '---')")
    try:
        fm = yaml.safe_load(m.group("fm"))
    except yaml.YAMLError as e:
        raise ParseError(f"invalid YAML frontmatter: {e}") from e
    if not isinstance(fm, dict):
        raise ParseError("YAML frontmatter must be a mapping")
    return _stringify_dates(fm), m.group("body").strip()


def parse(path: Path) -> list[Record]:
    """Parse one record file. Returns a single-element list (one file = one record).

    Raises ParseError on malformed files so callers can report precisely.
    """
    text = path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    rid = fm.get("id")
    if not isinstance(rid, str) or not rid:
        raise ParseError("frontmatter is missing a string `id`")
    fields = {k: v for k, v in fm.items() if k != "id"}
    return [Record(id=rid, fields=fields, body=body, source_path=str(path))]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    path = Path(argv[0])
    if not path.exists():
        sys.stderr.write(f"file not found: {path}\n")
        return 2
    try:
        recs = parse(path)
    except ParseError as e:
        sys.stderr.write(f"{path}: {e}\n")
        return 1
    print(json.dumps([asdict(r) for r in recs], ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
