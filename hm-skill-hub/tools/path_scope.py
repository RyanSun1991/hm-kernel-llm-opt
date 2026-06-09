"""Path <-> frontmatter scope consistency (design §6.1, Phase 0.5 gate).

The hub encodes a record's scope **in its file path**; the frontmatter `scope`
(and friends) must agree. CI rejects any mismatch — this lets cheap scalar
filtering (design §12.1) pre-filter by directory before reading frontmatter.

`derive_path_scope()` maps a path (relative to `knowledge/`) to the expected
schema + scope. `check_scope_consistency()` compares that against a parsed
record's frontmatter and returns precise error strings.

Recognized layout (design §6.1)::

    knowledge/global/heuristics/<H###>.md          level=global  schema=global_lesson
    knowledge/global/anti_patterns/<A###>.md       level=global  schema=global_lesson
    knowledge/global/validation_pitfalls/<V###>.md level=global  schema=global_lesson
    knowledge/global/bad_plans/<B###>.md           level=global  schema=bad_plan
    knowledge/subsystems/<sub>/bad_plans/<B###>.md level=subsys  schema=bad_plan
    knowledge/subsystems/<sub>/<id>.md             level=subsys  schema=memory_item
    knowledge/targets/<slug>/facts/<F###>.md       target-level  schema=memory_item
    knowledge/targets/<slug>/decisions/<id>.md     target-level  schema=memory_item
    knowledge/targets/<slug>/idea_ledger/<L###>.md target-level  schema=idea
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

# Categories under knowledge/global/ -> (schema, expected global_lesson kind or None)
_GLOBAL_LESSON_CATEGORIES = {
    "heuristics": "heuristic",
    "anti_patterns": "anti_pattern",
    "validation_pitfalls": "validation_pitfall",
}
# scope.level values that are "target-relative" (anything finer than subsystem).
_TARGET_LEVELS = {"function", "call-site", "data-flow", "architectural"}


@dataclass
class PathScope:
    schema: str  # bad_plan | global_lesson | memory_item | idea
    level: str | None = None  # global | subsystem | None(target-relative)
    subsystem: str | None = None
    target_slug: str | None = None
    category: str | None = None  # leaf category for global lessons


def _rel_parts(rel: str | PurePosixPath) -> list[str]:
    p = PurePosixPath(str(rel))
    parts = list(p.parts)
    # tolerate a leading "knowledge/" if a caller passes a hub-relative path
    if parts and parts[0] == "knowledge":
        parts = parts[1:]
    return parts


def derive_path_scope(rel_path: str | PurePosixPath) -> PathScope | None:
    """Derive expected schema + scope from a knowledge-relative file path.

    Returns None when the path is not a recognized record location (caller may
    fall back to id-prefix dispatch and/or flag it).
    """
    parts = _rel_parts(rel_path)
    if len(parts) < 2 or not parts[-1].endswith(".md"):
        return None
    head = parts[0]

    if head == "global":
        category = parts[1]
        if category == "bad_plans":
            return PathScope(schema="bad_plan", level="global", category="bad_plans")
        if category in _GLOBAL_LESSON_CATEGORIES:
            return PathScope(schema="global_lesson", level="global", category=category)
        return None

    if head == "subsystems" and len(parts) >= 3:
        sub = parts[1]
        if parts[2] == "bad_plans":
            return PathScope(schema="bad_plan", level="subsystem", subsystem=sub, category="bad_plans")
        # subsystems/<sub>/<id>.md  (depth 3) -> subsystem-level memory_item
        if len(parts) == 3:
            return PathScope(schema="memory_item", level="subsystem", subsystem=sub)
        return None

    if head == "targets" and len(parts) >= 4:
        slug = parts[1]
        kind = parts[2]
        if kind == "idea_ledger":
            return PathScope(schema="idea", target_slug=slug)
        if kind in {"facts", "decisions"}:
            return PathScope(schema="memory_item", target_slug=slug)
        return None

    return None


def expected_global_lesson_kind(ps: PathScope) -> str | None:
    if ps.category in _GLOBAL_LESSON_CATEGORIES:
        return _GLOBAL_LESSON_CATEGORIES[ps.category]
    return None


def check_scope_consistency(ps: PathScope, record_id: str, fields: dict[str, Any]) -> list[str]:
    """Return error strings where the frontmatter contradicts the path scope."""
    errs: list[str] = []

    if ps.schema == "global_lesson":
        want_kind = expected_global_lesson_kind(ps)
        kind = fields.get("kind")
        if want_kind and kind != want_kind:
            errs.append(
                f"path says global/{ps.category}/ (kind={want_kind}) but frontmatter kind={kind!r}"
            )

    elif ps.schema == "bad_plan":
        subs = (fields.get("applies_to") or {}).get("subsystems")
        if ps.level == "global":
            if subs is not None and subs != ["*"]:
                errs.append(
                    f"global bad_plan must have applies_to.subsystems=['*'], got {subs!r}"
                )
        elif ps.level == "subsystem":
            if not subs or ps.subsystem not in subs:
                errs.append(
                    f"path subsystem={ps.subsystem!r} not in applies_to.subsystems={subs!r}"
                )

    elif ps.schema == "memory_item":
        scope = fields.get("scope") or {}
        level = scope.get("level")
        if ps.level == "subsystem":
            if level != "subsystem":
                errs.append(f"path is subsystems/ but scope.level={level!r} (expected 'subsystem')")
            if scope.get("subsystem") not in (None, ps.subsystem):
                errs.append(
                    f"path subsystem={ps.subsystem!r} != scope.subsystem={scope.get('subsystem')!r}"
                )
        elif ps.target_slug is not None:
            if level in {"global", "subsystem"}:
                errs.append(
                    f"path is targets/{ps.target_slug}/ but scope.level={level!r} "
                    f"(expected a target-relative level, e.g. one of {sorted(_TARGET_LEVELS)})"
                )
            if scope.get("target_slug") not in (None, ps.target_slug):
                errs.append(
                    f"path slug={ps.target_slug!r} != scope.target_slug={scope.get('target_slug')!r}"
                )

    elif ps.schema == "idea":
        slug = fields.get("target_slug")
        if slug is None:
            errs.append(
                f"idea record under targets/{ps.target_slug}/ must declare target_slug "
                f"in frontmatter (path<->scope consistency)"
            )
        elif slug != ps.target_slug:
            errs.append(f"path slug={ps.target_slug!r} != frontmatter target_slug={slug!r}")

    return errs
