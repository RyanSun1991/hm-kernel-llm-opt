"""Static callgraph helpers derived from PSG edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .psg import PsgGraph, PsgEdge


@dataclass
class CallGraph:
    edges: List[PsgEdge]
    out_degree: Dict[str, int]


def build_callgraph(psg: PsgGraph) -> CallGraph:
    out_degree: dict[str, int] = {}
    for edge in psg.edges:
        out_degree[edge.src] = out_degree.get(edge.src, 0) + 1
    return CallGraph(edges=psg.edges, out_degree=out_degree)
