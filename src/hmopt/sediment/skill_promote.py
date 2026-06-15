"""Promote a member `.opencode/skills/<name>/SKILL.md` into the hub skill taxonomy.

Member skills carry only `name` + `description` frontmatter; hub skills are
richer (`kind` core/technique/domain, `version`, `maturity`, `owners`, `status`,
and — to graduate past L0 — an `eval_id` + scorecard the eval gate scores). So
this is a *scaffold*, not a silent copy: it writes a schema-valid hub SKILL.md at
maturity L0/experimental and prints the human follow-ups (add an eval suite, a
`best_skill.md`, and a scorecard) needed before the skill can be promoted.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    raise ImportError("PyYAML required for skill_promote") from None

from ..skillhub.resolver import find_hub_root

_FRONTMATTER_RE = re.compile(r"^---\r?\n(?P<fm>.*?)\r?\n---\r?\n?(?P<body>.*)$", re.DOTALL)
_VALID_KINDS = {"core", "technique", "domain"}


def _resolve_hub(hub_root: str | Path | None) -> Path | None:
    """Accept either a repo root that contains the hub, or the hub dir itself."""
    if hub_root is not None:
        p = Path(hub_root)
        if (p / "schemas").is_dir() or (p / "skills").is_dir():
            return p  # already the hub
        return find_hub_root(p)
    return find_hub_root(Path.cwd())


def _parse(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text.strip()
    fm = yaml.safe_load(m.group("fm")) or {}
    return (fm if isinstance(fm, dict) else {}), m.group("body").strip()


def promote_opencode_skill(
    skill_dir: str | Path,
    *,
    kind: str,
    hub_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Scaffold a hub `skills/<kind>/<name>/SKILL.md` from a member skill dir."""
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}")

    skill_dir = Path(skill_dir)
    src = skill_dir / "SKILL.md"
    if not src.exists():
        raise FileNotFoundError(f"no SKILL.md under {skill_dir}")
    fm, body = _parse(src.read_text(encoding="utf-8"))
    name = str(fm.get("name") or skill_dir.name)
    description = str(fm.get("description") or "").strip()

    hub = _resolve_hub(hub_root)
    if hub is None:
        raise FileNotFoundError("no skill hub found (pass --hub)")
    dest_dir = Path(hub) / "skills" / kind / name
    dest = dest_dir / "SKILL.md"
    if dest.exists() and not force:
        raise FileExistsError(f"{dest} already exists (use --force to overwrite)")

    # additionalProperties:false on the skill schema — only emit allowed keys.
    front: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "version": "0.1.0",
        "maturity": "L0",
        "owners": ["@maintainers"],
        "status": "experimental",
        "requires": [],
    }
    if kind == "technique":
        front["optimization_goal"] = "instruction-count"
    if kind == "domain":
        front["applies_to"] = {"subsystems": [], "path_globs": [], "symbol_selectors": []}

    fm_text = yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip()
    lead = (description + "\n\n") if description else ""
    new_body = (
        f"# {name} ({kind} skill)\n\n"
        f"{lead}"
        f"{body}\n\n"
        f"---\n"
        f"_Promoted from `.opencode/skills/{skill_dir.name}` (member-side). "
        f"Scaffolded at maturity L0; add an eval suite + scorecard before promotion._\n"
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest.write_text(f"---\n{fm_text}\n---\n\n{new_body}", encoding="utf-8")

    next_steps = [
        f"add eval cases under hm-skill-hub/eval/task_suites/ and set `eval_id` in {dest.name}",
        f"add {dest_dir.name}/best_skill.md so the nightly optimizer (engine B) can pick it up",
        "run `python hm-skill-hub/tools/lint.py` to confirm the scaffold validates",
    ]
    if kind == "domain":
        next_steps.insert(0, "fill applies_to.subsystems / path_globs / symbol_selectors")
    return {"name": name, "kind": kind, "skill_md": str(dest), "next_steps": next_steps}
