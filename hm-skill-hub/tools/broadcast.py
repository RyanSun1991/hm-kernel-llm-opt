"""Update a consuming repo's .opencode/skill-memory.lock for a hub release."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(".."))
    parser.add_argument("--hub-url", default="in-repo:hm-skill-hub")
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha", default="local")
    args = parser.parse_args()
    lock = args.repo / ".opencode" / "skill-memory.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "hub": {"url": args.hub_url, "version": args.version, "sha": args.sha, "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat()},
        "fallback": {"mode": "vendored", "path": "hm-skill-hub"},
    }
    lock.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
