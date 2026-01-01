"""Program Structure Graph (PSG) builder."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

from .indexer import SymbolInfo


CALL_PATTERN = re.compile(r"([A-Za-z_][\w\d_]*)\s*\(")


@dataclass
class PsgNode:
    name: str
    file_path: str
    line: int


@dataclass
class PsgEdge:
    src: str
    dst: str
    kind: str = "call_static"


@dataclass
class PsgGraph:
    nodes: Dict[str, PsgNode]
    edges: List[PsgEdge]

    def to_dict(self) -> dict:
        return {
            "nodes": {name: asdict(node) for name, node in self.nodes.items()},
            "edges": [asdict(edge) for edge in self.edges],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def build_psg(symbols: List[SymbolInfo]) -> PsgGraph:
    nodes = {sym.name: PsgNode(sym.name, str(sym.file_path), sym.line) for sym in symbols}
    edges: list[PsgEdge] = []
    symbol_names = set(nodes.keys())
    for sym in symbols:
        try:
            text = Path(sym.file_path).read_text(encoding="utf-8", errors="ignore")
        except FileNotFoundError:
            continue
        for match in CALL_PATTERN.finditer(text):
            target = match.group(1)
            if target in symbol_names and target != sym.name:
                edges.append(PsgEdge(src=sym.name, dst=target, kind="call_static"))
    return PsgGraph(nodes=nodes, edges=edges)
