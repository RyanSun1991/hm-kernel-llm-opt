"""Resolve Team Skill Hub assets with local fallback.

Resolution order follows the design: pinned hub first, then local in-flight
assets under `.opencode/local/`, while keeping the vendored hub as an offline
fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK = REPO_ROOT / ".opencode" / "skill-memory.lock"


def _lock() -> dict:
    if LOCK.exists():
        return yaml.safe_load(LOCK.read_text(encoding="utf-8")) or {}
    return {}


def hub_root() -> Path:
    data = _lock()
    fallback = data.get("fallback", {}).get("path", "hm-skill-hub")
    return (REPO_ROOT / fallback).resolve()


def resolve_skill(name: str) -> Path:
    local = REPO_ROOT / ".opencode" / "local" / "skills" / name / "SKILL.md"
    if local.exists():
        return local
    for kind in ("core", "technique", "domain"):
        cand = hub_root() / "skills" / kind / name / "SKILL.md"
        if cand.exists():
            return cand
    legacy = REPO_ROOT / ".opencode" / "skills" / f"{name}.md"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(name)


def iter_knowledge() -> Iterable[Path]:
    root = hub_root() / "knowledge"
    if root.exists():
        yield from sorted(root.rglob("*.md"))
    local = REPO_ROOT / ".opencode" / "local" / "memory"
    if local.exists():
        yield from sorted(local.rglob("*.md"))
