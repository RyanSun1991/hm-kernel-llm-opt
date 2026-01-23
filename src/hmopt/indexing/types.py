"""Shared dataclasses for indexing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class IndexPaths:
    base_dir: Path
    code_root: Path
    runtime_root: Path
    code_dir: Path
    runtime_dir: Path
    code_version: Optional[str] = None
    runtime_version: Optional[str] = None


@dataclass
class Neo4jIndexConfig:
    index_name: str
    node_label: str
