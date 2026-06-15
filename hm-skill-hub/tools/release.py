"""Release tool: semver bump + registry.yaml + release notes (plan P4-2).

Scans the hub into an inventory, infers a semver bump by diffing against the
committed `registry.yaml`, and (with --apply) rewrites `registry.yaml` + writes
release notes. Tags are cut by CI from the computed version; this tool is the
deterministic, offline brain.

Bump rules (design §14 cadence):
  major  -- a skill removed, or the schema contract changed (breaking).
  minor  -- a new skill, a skill version/maturity raised, or a knowledge promotion.
  patch  -- knowledge-only adds/merges, docs.

CLI:
    python tools/release.py [--bump=auto|patch|minor|major] [--status=stable] [--apply]
"""
from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write("PyYAML required\n")
    sys.exit(2)

HUB = Path(__file__).resolve().parent.parent
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_RANK = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


@dataclass
class Inventory:
    skills: dict[str, dict[str, Any]] = field(default_factory=dict)  # ref -> {version, maturity,...}
    knowledge_count: int = 0
    schema_hash: str = ""


def _semver(v: str) -> tuple[int, int, int]:
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", str(v))
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)  # type: ignore[return-value]


def bump_version(version: str, kind: str) -> str:
    major, minor, patch = _semver(version)
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def scan_inventory(hub_root: str | Path = HUB) -> Inventory:
    root = Path(hub_root)
    inv = Inventory()
    for sk in sorted((root / "skills").rglob("SKILL.md")) if (root / "skills").exists() else []:
        m = _FRONTMATTER_RE.match(sk.read_text(encoding="utf-8"))
        if not m:
            continue
        fm = yaml.safe_load(m.group(1)) or {}
        name, kind = fm.get("name"), fm.get("kind")
        if not name or not kind:
            continue
        inv.skills[f"{kind}/{name}"] = {
            "name": name, "kind": kind, "version": str(fm.get("version", "0.0.0")),
            "maturity": fm.get("maturity", "L0"), "eval_id": fm.get("eval_id", ""),
            "owners": fm.get("owners", []), "status": fm.get("status", "experimental"),
        }
    kn = root / "knowledge"
    if kn.exists():
        inv.knowledge_count = sum(
            1 for p in kn.rglob("*.md")
            if p.name != "README.md" and "index" not in p.parts
        )
    sch = root / "schemas"
    if sch.exists():
        h = hashlib.sha256()
        for p in sorted(sch.glob("*.json")):
            h.update(p.read_bytes())
        inv.schema_hash = h.hexdigest()[:16]
    return inv


def load_registry(hub_root: str | Path = HUB) -> dict[str, Any]:
    p = Path(hub_root) / "registry.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def inventory_from_registry(reg: dict[str, Any]) -> Inventory:
    inv = Inventory()
    for s in reg.get("skills", []) or []:
        inv.skills[f"{s.get('kind')}/{s.get('name')}"] = dict(s)
    inv.knowledge_count = int((reg.get("hub", {}) or {}).get("knowledge_count", 0))
    inv.schema_hash = str((reg.get("hub", {}) or {}).get("schema_hash", ""))
    return inv


def infer_bump(old: Inventory, new: Inventory) -> tuple[str, list[str]]:
    reasons: list[str] = []
    removed = set(old.skills) - set(new.skills)
    added = set(new.skills) - set(old.skills)
    if removed:
        reasons.append(f"skill(s) removed: {sorted(removed)}")
        return "major", reasons
    if old.schema_hash and new.schema_hash and old.schema_hash != new.schema_hash:
        reasons.append("schema contract changed")
        return "major", reasons
    if added:
        reasons.append(f"new skill(s): {sorted(added)}")
    common = set(old.skills) & set(new.skills)
    raised = [
        ref for ref in common
        if _semver(new.skills[ref].get("version", "0.0.0")) > _semver(old.skills[ref].get("version", "0.0.0"))
        or _RANK.get(new.skills[ref].get("maturity"), 0) > _RANK.get(old.skills[ref].get("maturity"), 0)
    ]
    changed = [
        ref for ref in common
        if (new.skills[ref].get("version") != old.skills[ref].get("version")
            or new.skills[ref].get("maturity") != old.skills[ref].get("maturity"))
    ]
    if raised:
        reasons.append(f"skill version/maturity raised: {sorted(raised)}")
    elif changed:
        reasons.append(f"skill version/maturity changed: {sorted(changed)}")  # e.g. a revert
    if added or raised or changed:
        return "minor", reasons
    if new.knowledge_count != old.knowledge_count:
        reasons.append(f"knowledge {old.knowledge_count}→{new.knowledge_count}")
        return "patch", reasons
    reasons.append("no asset change")
    return "patch", reasons


def build_registry(inv: Inventory, version: str, status: str, prev_releases: list,
                   bump: str) -> dict[str, Any]:
    return {
        "hub": {"version": version, "status": status,
                "knowledge_count": inv.knowledge_count, "schema_hash": inv.schema_hash},
        "skills": [inv.skills[r] for r in sorted(inv.skills)],
        "knowledge_indexes": [
            {"path": "knowledge/global/heuristics/"},
            {"path": "knowledge/global/anti_patterns/"},
            {"path": "knowledge/global/validation_pitfalls/"},
            {"path": "knowledge/global/bad_plans/"},
            {"path": "knowledge/subsystems/"},
            {"path": "knowledge/targets/"},
        ],
        "evals": sorted({s["eval_id"] for s in inv.skills.values() if s.get("eval_id")}),
        "releases": [*prev_releases, {"version": version, "bump": bump}],
    }


def render_release_notes(old_v: str, new_v: str, bump: str, reasons: list[str],
                         inv: Inventory) -> str:
    lines = [f"# hm-skill-hub {new_v}", "",
             f"_{bump} release ({old_v} → {new_v})_", "", "## Why", ""]
    lines += [f"- {r}" for r in reasons]
    lines += ["", "## Inventory", "",
              f"- skills: {len(inv.skills)} ({', '.join(sorted(inv.skills)) or '—'})",
              f"- knowledge records: {inv.knowledge_count}",
              f"- schema_hash: `{inv.schema_hash}`", ""]
    return "\n".join(lines)


def compute_release(hub_root: str | Path = HUB, bump: str = "auto") -> dict[str, Any]:
    reg = load_registry(hub_root)
    old_inv = inventory_from_registry(reg)
    new_inv = scan_inventory(hub_root)
    old_v = str((reg.get("hub", {}) or {}).get("version", "0.0.0"))
    auto_bump, reasons = infer_bump(old_inv, new_inv)
    if bump == "auto":
        chosen = auto_bump
    else:
        chosen = bump
        if bump != auto_bump:
            reasons = [*reasons, f"bump overridden {auto_bump}→{bump}"]
    new_v = bump_version(old_v, chosen)
    return {"old_version": old_v, "new_version": new_v, "bump": chosen,
            "reasons": reasons, "inventory": new_inv,
            "prev_releases": reg.get("releases", []) or []}


def apply_release(hub_root: str | Path = HUB, bump: str = "auto", status: str = "active"
                  ) -> dict[str, Any]:
    """Write registry.yaml + release notes for `hub_root` (parameterized — the
    nightly job must operate on the hub it was handed, not a module constant)."""
    hub_root = Path(hub_root)
    r = compute_release(hub_root, bump)
    reg = build_registry(r["inventory"], r["new_version"], status, r["prev_releases"], r["bump"])
    text = ("# Top-level inventory of hub assets. Maintained by tools/release.py.\n"
            + yaml.safe_dump(reg, sort_keys=False, allow_unicode=True))
    (hub_root / "registry.yaml").write_text(text, encoding="utf-8")
    notes = render_release_notes(r["old_version"], r["new_version"], r["bump"],
                                 r["reasons"], r["inventory"])
    (hub_root / "releases").mkdir(exist_ok=True)
    (hub_root / "releases" / f"{r['new_version']}.md").write_text(notes, encoding="utf-8")
    return r


def main(argv: list[str]) -> int:
    bump = next((a.split("=", 1)[1] for a in argv if a.startswith("--bump=")), "auto")
    status = next((a.split("=", 1)[1] for a in argv if a.startswith("--status=")), "active")
    apply = "--apply" in argv
    r = compute_release(HUB, bump)
    print(f"release: {r['old_version']} → {r['new_version']} ({r['bump']})")
    for reason in r["reasons"]:
        print(f"  - {reason}")
    if apply:
        apply_release(HUB, r["bump"], status)
        print(f"wrote registry.yaml + releases/{r['new_version']}.md")
    else:
        print("\n(dry-run; pass --apply to write registry.yaml + release notes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
