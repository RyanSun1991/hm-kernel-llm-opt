"""Create a local Team Skill Hub release note and update registry.yaml."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import yaml

HUB = Path(__file__).resolve().parent.parent


def bump(version: str, kind: str) -> str:
    major, minor, patch = [int(x) for x in version.split(".")]
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bump", choices=["patch", "minor", "major"], default="patch")
    parser.add_argument("--notes", default="Automated hub release.")
    parser.add_argument("--hub", type=Path, default=HUB)
    args = parser.parse_args()
    registry_path = args.hub / "registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    old = str(registry.get("hub", {}).get("version", "0.0.0"))
    new = bump(old, args.bump)
    registry.setdefault("hub", {})["version"] = new
    registry["hub"]["status"] = "released"
    registry.setdefault("releases", []).append({"version": new, "date": date.today().isoformat(), "notes": args.notes})
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True), encoding="utf-8")
    rel = args.hub / "releases" / f"v{new}.md"
    rel.write_text(f"# v{new}\n\n- Date: {date.today().isoformat()}\n- Bump: {args.bump}\n- Notes: {args.notes}\n", encoding="utf-8")
    print(rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
