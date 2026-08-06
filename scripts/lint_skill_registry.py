#!/usr/bin/env python3
"""Lint `.opencode/skills/_registry.yaml` against the skill directories on disk.

The registry is the skill system's only index: roles read it to learn which skills
exist, and read a SKILL.md full text only after selecting it. That indirection is
cheap but it fails silently — a skill with no entry is invisible to every role, and
an entry with no directory sends a role to Read a file that does not exist.

This script is the governance step that keeps the two sides consistent.

Checks
  1. every registry entry has <tier>/<name>/SKILL.md on disk
  2. every SKILL.md on disk has exactly one registry entry
  3. the entry's `tier` matches the directory the skill actually lives in, and is
     one of the three tiers (role / scenario[/<pack>] / infra[/pipeline]) — a skill
     at the top level of skills/ is a layout regression and fails
  4. SKILL.md frontmatter `name:` matches the directory name
  5. no SKILL.md exceeds MAX_SKILL_LINES without a `references/` split
  6. entry names are unique
  7. `roles:` names are real roles; `risk:` is R0-R3; `class:` is known
  8. `conflicts:` reference existing skills, are symmetric, and are not self-referential
  9. required fields are present and non-empty; a non-core skill with an empty
     `applies_when` is an error (it could never be suggested)
 10. frontmatter `depends_on:` entries name skills that exist on disk

Usage
    python scripts/lint_skill_registry.py            # lint, print report, exit 1 on error
    python scripts/lint_skill_registry.py --quiet    # errors only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

MAX_SKILL_LINES = 500

VALID_ROLES = {
    "assistant",
    "researcher",
    "architect",
    "implementer",
    "reviewer",
    "validator",
    "coordinator",
}

VALID_CLASSES = {
    "core",
    "domain",
    "method",
    "scenario",
    "review",
    "validation",
    "tool",
    "output",
}

VALID_RISKS = {"R0", "R1", "R2", "R3"}

VALID_TIER_ROOTS = ("role", "scenario", "infra")

REQUIRED_FIELDS = ("name", "tier", "class", "roles", "risk", "context_cost")


def find_repo_root(start: Path) -> Path:
    """Walk up until a directory containing `.opencode/skills` is found."""
    for candidate in (start, *start.parents):
        if (candidate / ".opencode" / "skills").is_dir():
            return candidate
    raise SystemExit("error: could not locate a repo root containing .opencode/skills")


def parse_frontmatter(skill_md: Path) -> dict:
    """Return the YAML frontmatter block of a SKILL.md as a dict ({} if unparsable)."""
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    try:
        block = yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError:
        return {}
    return block if isinstance(block, dict) else {}


def discover_skills(skills_root: Path) -> tuple[dict[str, Path], list[str]]:
    """Map skill directory name -> its SKILL.md path, plus duplicate-name errors."""
    found: dict[str, Path] = {}
    duplicates: list[str] = []
    for skill_md in sorted(skills_root.rglob("SKILL.md")):
        # A `references/` subdirectory may hold supporting docs but not a second skill.
        if "references" in skill_md.relative_to(skills_root).parts[:-1]:
            continue
        name = skill_md.parent.name
        if name in found:
            duplicates.append(
                f"duplicate skill directory name '{name}' on disk "
                f"({found[name].parent.relative_to(skills_root)} vs "
                f"{skill_md.parent.relative_to(skills_root)})"
            )
        found[name] = skill_md
    return found, duplicates


def tier_of(skill_md: Path, skills_root: Path) -> str:
    """The tier is the directory path between skills_root and the skill directory."""
    rel = skill_md.parent.relative_to(skills_root)
    return "/".join(rel.parts[:-1]) if len(rel.parts) > 1 else ""


def line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def lint(repo_root: Path) -> tuple[list[str], list[str]]:
    skills_root = repo_root / ".opencode" / "skills"
    registry_path = skills_root / "_registry.yaml"

    errors: list[str] = []
    warnings: list[str] = []

    if not registry_path.is_file():
        return ([f"missing registry: {registry_path.relative_to(repo_root)}"], [])

    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    entries = raw.get("skills") or []
    if not isinstance(entries, list):
        return (["registry `skills:` must be a list"], [])

    on_disk, duplicate_errors = discover_skills(skills_root)
    errors.extend(duplicate_errors)

    seen: set[str] = set()
    by_name: dict[str, dict] = {}

    for index, entry in enumerate(entries):
        label = f"entry #{index + 1}"
        if not isinstance(entry, dict):
            errors.append(f"{label}: must be a mapping")
            continue

        name = entry.get("name")
        if not name:
            errors.append(f"{label}: missing `name`")
            continue
        label = f"skill '{name}'"

        if name in seen:
            errors.append(f"{label}: duplicate registry entry")
            continue
        seen.add(name)
        by_name[name] = entry

        for field in REQUIRED_FIELDS:
            value = entry.get(field)
            if value is None or (isinstance(value, (list, str)) and len(value) == 0):
                errors.append(f"{label}: missing or empty required field `{field}`")

        # --- check 1 + 3: directory exists at the declared, valid tier ----------
        tier = str(entry.get("tier") or "")
        if tier.split("/", 1)[0] not in VALID_TIER_ROOTS:
            errors.append(
                f"{label}: tier '{tier or '<top level>'}' is not under "
                f"{'/'.join(VALID_TIER_ROOTS)} — flat skills are a layout regression"
            )
        declared = skills_root / tier / str(name) / "SKILL.md"
        if str(name) not in on_disk:
            errors.append(
                f"{label}: registry entry has no skill directory "
                f"(expected {declared.relative_to(repo_root)})"
            )
        else:
            actual_md = on_disk[str(name)]
            actual_tier = tier_of(actual_md, skills_root)
            if actual_tier != tier:
                errors.append(
                    f"{label}: tier mismatch — registry says '{tier}', "
                    f"on disk it is '{actual_tier or '<top level>'}'"
                )

            fm = parse_frontmatter(actual_md)

            # --- check 4: frontmatter name matches directory --------------------
            fm_name = str(fm.get("name")) if fm.get("name") else None
            if fm_name is None:
                errors.append(f"{label}: SKILL.md has no parsable frontmatter `name:`")
            elif fm_name != str(name):
                errors.append(
                    f"{label}: frontmatter name '{fm_name}' != directory name '{name}'"
                )

            # --- check 10: depends_on names exist --------------------------------
            depends_on = fm.get("depends_on") or []
            if not isinstance(depends_on, list):
                errors.append(f"{label}: frontmatter `depends_on` must be a list")
            else:
                for dep in depends_on:
                    if str(dep) not in on_disk:
                        errors.append(
                            f"{label}: frontmatter depends_on '{dep}' names no skill on disk"
                        )

            # --- check 5: size ceiling ------------------------------------------
            lines = line_count(actual_md)
            if lines > MAX_SKILL_LINES and not (actual_md.parent / "references").is_dir():
                errors.append(
                    f"{label}: SKILL.md is {lines} lines (>{MAX_SKILL_LINES}) and has no "
                    "references/ split — progressive disclosure requires the split"
                )

        # --- check 7: enum fields ---------------------------------------------
        roles = entry.get("roles") or []
        if not isinstance(roles, list):
            errors.append(f"{label}: `roles` must be a list")
        else:
            for role in roles:
                if role not in VALID_ROLES:
                    errors.append(
                        f"{label}: unknown role '{role}' "
                        f"(valid: {', '.join(sorted(VALID_ROLES))})"
                    )

        klass = entry.get("class")
        if klass is not None and klass not in VALID_CLASSES:
            errors.append(
                f"{label}: unknown class '{klass}' (valid: {', '.join(sorted(VALID_CLASSES))})"
            )

        risk = entry.get("risk")
        if risk is not None and risk not in VALID_RISKS:
            errors.append(f"{label}: risk must be one of {', '.join(sorted(VALID_RISKS))}")

        # --- check 9: triggers ------------------------------------------------
        applies_when = entry.get("applies_when") or []
        if not isinstance(applies_when, list):
            errors.append(f"{label}: `applies_when` must be a list")
        elif not applies_when and klass != "core":
            errors.append(
                f"{label}: empty `applies_when` — the skill can never be suggested "
                "(only `class: core` skills may omit triggers)"
            )
        if not isinstance(entry.get("not_for") or [], list):
            errors.append(f"{label}: `not_for` must be a list")

    # --- check 2: orphan directories ------------------------------------------
    for name, skill_md in sorted(on_disk.items()):
        if name not in seen:
            errors.append(
                f"skill directory '{skill_md.parent.relative_to(skills_root)}' has no "
                "registry entry — it is invisible to every role"
            )

    # --- check 8: conflicts ---------------------------------------------------
    for name, entry in by_name.items():
        conflicts = entry.get("conflicts") or []
        if not isinstance(conflicts, list):
            errors.append(f"skill '{name}': `conflicts` must be a list")
            continue
        for other in conflicts:
            if other == name:
                errors.append(f"skill '{name}': conflicts with itself")
            elif other not in by_name:
                errors.append(f"skill '{name}': conflicts with unknown skill '{other}'")
            elif name not in (by_name[other].get("conflicts") or []):
                errors.append(
                    f"skill '{name}': conflict with '{other}' is not symmetric — "
                    f"add '{name}' to '{other}'.conflicts"
                )

    return (errors, warnings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quiet", action="store_true", help="print errors only")
    parser.add_argument(
        "--repo-root", default=None, help="repo root (default: auto-detect from this file)"
    )
    args = parser.parse_args()

    root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else find_repo_root(Path(__file__).resolve())
    )
    errors, warnings = lint(root)

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)

    if errors:
        print(f"\nregistry lint FAILED — {len(errors)} error(s)", file=sys.stderr)
        return 1

    if not args.quiet:
        skills_root = root / ".opencode" / "skills"
        count = len(discover_skills(skills_root)[0])
        suffix = f", {len(warnings)} warning(s)" if warnings else ""
        print(f"registry lint OK — {count} skills consistent with _registry.yaml{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
