"""Lightweight repo indexer using regex/ctags heuristics."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class SymbolInfo:
    name: str
    file_path: Path
    line: int
    kind: str = "function"


FUNC_PATTERN = re.compile(r"^\s*(?:[\w\*\s]+)?([A-Za-z_][\w\d_]*)\s*\([^;]*\)\s*{")


def _ctags_available() -> bool:
    return shutil.which("ctags") is not None


def _index_with_ctags(root: Path) -> list[SymbolInfo]:
    cmd = ["ctags", "-x", "--c-kinds=fp", "-R", str(root)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    symbols: list[SymbolInfo] = []
    if proc.returncode != 0:
        return symbols
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        name, kind, line_no, file_path = parts[0], parts[1], parts[2], parts[3]
        try:
            symbols.append(SymbolInfo(name=name, file_path=root / file_path, line=int(line_no), kind=kind))
        except ValueError:
            continue
    return symbols


def _index_with_regex(root: Path, patterns: Iterable[str]) -> list[SymbolInfo]:
    symbols: list[SymbolInfo] = []
    for pattern in patterns:
        for path in root.rglob(pattern):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except FileNotFoundError:
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                m = FUNC_PATTERN.match(line)
                if m:
                    symbols.append(SymbolInfo(name=m.group(1), file_path=path, line=idx))
    return symbols


def index_repo(repo_path: str | Path) -> List[SymbolInfo]:
    root = Path(repo_path)
    if _ctags_available():
        symbols = _index_with_ctags(root)
        if symbols:
            return symbols
    return _index_with_regex(root, patterns=["*.c", "*.cc", "*.cpp", "*.h", "*.hpp", "*.cxx", "*.py"])
