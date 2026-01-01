"""Repo scanning helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


def list_source_files(repo_path: str | Path, exts: Iterable[str] | None = None) -> List[Path]:
    root = Path(repo_path)
    exts = set(exts or [".c", ".cc", ".cpp", ".h", ".hpp", ".py"])
    return [p for p in root.rglob("*") if p.suffix in exts and p.is_file()]


def write_snapshot_manifest(manifest: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
