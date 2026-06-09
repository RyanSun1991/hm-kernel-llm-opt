"""Export legacy .opencode memory/plans/reviews into hub staging JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hmopt.sediment.extractors import extract_idea_records, extract_review_records, write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opencode", type=Path, default=ROOT / ".opencode")
    parser.add_argument("--output", type=Path, default=ROOT / "hm-skill-hub/staging/legacy_export.jsonl")
    args = parser.parse_args()
    records = []
    records.extend(extract_idea_records(args.opencode / "memory"))
    records.extend(extract_review_records(args.opencode))
    n = write_jsonl(records, args.output)
    print(json.dumps({"output": str(args.output), "records": n}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
