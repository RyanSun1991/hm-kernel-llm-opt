"""Maintain a Pareto frontier over scorecards or candidate score JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _metrics(rec: dict[str, Any]) -> tuple[float, float]:
    m = rec.get("metrics", rec)
    return float(m.get("pass_rate", 0.0)), -float(m.get("regression_rate", 1.0))


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    am = _metrics(a)
    bm = _metrics(b)
    return all(x >= y for x, y in zip(am, bm)) and any(x > y for x, y in zip(am, bm))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scorecards", nargs="*", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()
    records = [json.loads(p.read_text(encoding="utf-8")) for p in args.scorecards]
    frontier = [r for r in records if not any(_dominates(o, r) for o in records if o is not r)]
    data = json.dumps(frontier, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data + "\n", encoding="utf-8")
    else:
        print(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
