"""Apply a simple double-temporal supersession decision to memory-item JSON.

Input is a JSON object or JSONL candidate. The script does not guess semantic
truth; it stamps the old record as superseded and links the new one with
``supersedes`` so a curator can commit both changes deliberately.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()
    old = _load(args.old)
    new = _load(args.new)
    if isinstance(old, list):
        old = old[0]
    if isinstance(new, list):
        new = new[0]
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    old["status"] = "superseded"
    old["valid_until"] = now
    new.setdefault("supersedes", [])
    if old.get("id") not in new["supersedes"]:
        new["supersedes"].append(old.get("id"))
    new.setdefault("valid_from", now)
    result = {"old": old, "new": new, "decision": "supersede", "resolved_at": now}
    data = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data + "\n", encoding="utf-8")
    else:
        print(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
