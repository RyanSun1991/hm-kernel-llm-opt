"""Lint the hub: parse every knowledge record (one file = one frontmatter
record), validate it against its JSON-Schema, and enforce path<->frontmatter
scope consistency (design §6.1 / §7, Phase 0.5 gate). Validate skill
frontmatter as well.

Exit codes:
    0 — all records valid
    1 — at least one validation / consistency failure
    2 — usage error

CLI:
    python tools/lint.py                 # lint the whole hub
    python tools/lint.py <path> ...      # lint specific files / dirs

Dispatch (design §6.1): schema is chosen by the **file path** first (so the same
id prefix can mean different things in different locations), with an id-prefix
fallback for paths outside the canonical layout.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import jsonschema
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write("required: pip install pyyaml jsonschema\n")
    sys.exit(2)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_memory import ParseError, parse  # type: ignore[import-not-found]
from path_scope import check_scope_consistency, derive_path_scope  # type: ignore[import-not-found]

HUB = Path(__file__).resolve().parent.parent
SCHEMAS = HUB / "schemas"
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
SKIP_NAMES = {"README.md"}
SKIP_DIRS = {"index"}
# Schemas that carry a markdown `body` field (filled from the file body).
_BODY_SCHEMAS = {"memory_item"}


def load_schema(name: str) -> dict[str, Any]:
    with open(SCHEMAS / f"{name}.schema.json", encoding="utf-8") as f:
        return json.load(f)


def _id_prefix_schema(record_id: str, ctx_path: str) -> str | None:
    """Fallback dispatch by id prefix when the path is not in the canonical layout."""
    p = record_id[:1]
    if "idea_ledger" in ctx_path:
        return "idea" if p == "L" else None
    if p == "B":
        return "bad_plan"
    if p in {"H", "V"}:
        return "global_lesson"
    if p in {"F", "G", "R"}:
        return "memory_item"
    return None


def _knowledge_rel(path: Path) -> str | None:
    """Path relative to the nearest `knowledge/` ancestor, or None."""
    parts = path.parts
    if "knowledge" in parts:
        i = len(parts) - 1 - parts[::-1].index("knowledge")
        return "/".join(parts[i + 1 :])
    return None


def lint_record_file(path: Path) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    try:
        recs = parse(path)
    except (ParseError, OSError) as e:
        return [(str(path), str(e))]

    rel = _knowledge_rel(path)
    ps = derive_path_scope(rel) if rel else None

    for r in recs:
        schema_name = ps.schema if ps else _id_prefix_schema(r.id, str(path))
        if schema_name is None:
            errors.append(
                (str(path), f"{r.id}: unrecognized record path/id — cannot dispatch to a schema "
                            f"(see design §6.1 for the canonical layout)")
            )
            continue

        doc: dict[str, Any] = {"id": r.id, **r.fields}
        if schema_name in _BODY_SCHEMAS and "body" not in doc:
            doc["body"] = r.body

        schema = load_schema(schema_name)
        try:
            jsonschema.validate(instance=doc, schema=schema)
        except jsonschema.ValidationError as e:
            loc = ".".join(str(p) for p in e.absolute_path) or "(root)"
            errors.append((str(path), f"{r.id}@{loc}: {e.message}"))
            continue  # don't pile scope errors on a schema-invalid record

        if ps is not None:
            for msg in check_scope_consistency(ps, r.id, r.fields):
                errors.append((str(path), f"{r.id}: scope mismatch — {msg}"))

    return errors


def lint_skill_md(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        return [(str(path), "missing YAML frontmatter (--- ... ---)")]
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        return [(str(path), f"invalid YAML frontmatter: {e}")]
    schema = load_schema("skill_frontmatter")
    try:
        jsonschema.validate(instance=fm, schema=schema)
    except jsonschema.ValidationError as e:
        loc = ".".join(str(p) for p in e.absolute_path) or "(root)"
        return [(str(path), f"frontmatter@{loc}: {e.message}")]
    return []


def _is_record_md(p: Path) -> bool:
    if p.name in SKIP_NAMES:
        return False
    if any(part in SKIP_DIRS for part in p.parts):
        return False
    return True


def discover(root: Path) -> tuple[list[Path], list[Path]]:
    md_records: list[Path] = []
    skills: list[Path] = []
    for sub in (root / "knowledge", root / "state"):
        if sub.exists():
            md_records.extend(p for p in sub.rglob("*.md") if _is_record_md(p))
    if (root / "skills").exists():
        skills.extend((root / "skills").rglob("SKILL.md"))
    return sorted(md_records), sorted(skills)


def main(argv: list[str]) -> int:
    targets = [Path(a).resolve() for a in argv] or [HUB]
    all_errors: list[tuple[str, str]] = []
    n_records, n_skills = 0, 0
    for t in targets:
        if t.is_dir():
            md, sk = discover(t)
        elif t.name == "SKILL.md":
            md, sk = [], [t]
        elif t.suffix == ".md":
            md, sk = [t], []
        else:
            md, sk = [], []
        for p in md:
            n_records += 1
            all_errors.extend(lint_record_file(p))
        for p in sk:
            n_skills += 1
            all_errors.extend(lint_skill_md(p))
    if all_errors:
        for path, msg in all_errors:
            sys.stderr.write(f"{path}: {msg}\n")
        sys.stderr.write(
            f"\n{len(all_errors)} error(s) across {n_records} record file(s), {n_skills} skill(s)\n"
        )
        return 1
    print(f"OK — {n_records} record file(s), {n_skills} skill(s) validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
