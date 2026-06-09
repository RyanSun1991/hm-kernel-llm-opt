"""Lint the hub: parse knowledge / state markdown files and validate every record
against the corresponding JSON-Schema. Validate skill frontmatter as well.

Exit codes:
    0 — all records valid
    1 — at least one validation failure
    2 — usage error

CLI:
    python tools/lint.py                 # lint the whole hub (cwd's containing repo)
    python tools/lint.py <path> ...      # lint specific paths

Hub layout assumed (see Team_Skill_Hub_Design_CN.md §6.1):
    knowledge/global/anti_patterns/*.md  -> bad_plan or global_lesson (id prefix dispatches)
    knowledge/global/lessons/*.md        -> global_lesson
    knowledge/targets/<slug>/idea_ledger.md -> idea
    state/bad_plans.md, state/<sub>-bad_plans.md -> bad_plan
    skills/<kind>/<name>/SKILL.md -> skill_frontmatter (YAML frontmatter only)
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
from parse_memory import parse  # type: ignore[import-not-found]

HUB = Path(__file__).resolve().parent.parent
SCHEMAS = HUB / "schemas"
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def load_schema(name: str) -> dict[str, Any]:
    with open(SCHEMAS / f"{name}.schema.json", encoding="utf-8") as f:
        return json.load(f)


def dispatch_record(record_id: str, ctx_path: str) -> str | None:
    """Return schema name for a record based on id prefix + file context."""
    p = record_id[0]
    if "idea_ledger" in ctx_path:
        return "idea" if p == "L" else None
    if p == "B":
        return "bad_plan"
    if p in {"H", "A", "V"}:
        return "global_lesson"
    if p in {"F", "G", "R"}:
        return "memory_item"
    return None


def lint_md_records(path: Path) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    recs = parse(path)
    for r in recs:
        schema_name = dispatch_record(r.id, str(path))
        if schema_name is None:
            errors.append((str(path), f"{r.id}: cannot dispatch record to a schema"))
            continue
        schema = load_schema(schema_name)
        doc = {"id": r.id, **(r.fields or {})}
        try:
            jsonschema.validate(instance=doc, schema=schema)
        except jsonschema.ValidationError as e:
            loc = ".".join(str(p) for p in e.absolute_path) or "(root)"
            errors.append((str(path), f"{r.id}@{loc}: {e.message}"))
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


def lint_json_doc(path: Path, schema_name: str) -> list[tuple[str, str]]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=doc, schema=load_schema(schema_name))
        return []
    except (json.JSONDecodeError, jsonschema.ValidationError) as e:
        return [(str(path), f"{schema_name}: {e}")]


def discover(root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    md_records: list[Path] = []
    skills: list[Path] = []
    scorecards: list[Path] = []
    for sub in (root / "knowledge", root / "state"):
        if sub.exists():
            md_records.extend(p for p in sub.rglob("*.md") if p.name not in {"README.md"})
    for p in (root / "skills").rglob("SKILL.md") if (root / "skills").exists() else []:
        skills.append(p)
    if (root / "eval" / "scorecards").exists():
        scorecards.extend(p for p in (root / "eval" / "scorecards").glob("*.json"))
    return md_records, skills, scorecards


def main(argv: list[str]) -> int:
    targets = [Path(a).resolve() for a in argv] or [HUB]
    all_errors: list[tuple[str, str]] = []
    n_records, n_skills, n_scorecards = 0, 0, 0
    for t in targets:
        if t.is_dir():
            md, sk, sc = discover(t)
        elif t.suffix == ".md" and t.name == "SKILL.md":
            md, sk, sc = [], [t], []
        elif t.suffix == ".json" and "scorecards" in str(t):
            md, sk, sc = [], [], [t]
        else:
            md, sk, sc = [t], [], []
        for p in md:
            n_records += 1
            all_errors.extend(lint_md_records(p))
        for p in sk:
            n_skills += 1
            all_errors.extend(lint_skill_md(p))
        for p in sc:
            n_scorecards += 1
            all_errors.extend(lint_json_doc(p, "scorecard"))
    if all_errors:
        for path, msg in all_errors:
            sys.stderr.write(f"{path}: {msg}\n")
        sys.stderr.write(f"\n{len(all_errors)} error(s) across {n_records} record file(s), {n_skills} skill(s), {n_scorecards} scorecard(s)\n")
        return 1
    print(f"OK — {n_records} record file(s), {n_skills} skill(s), {n_scorecards} scorecard(s) validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
