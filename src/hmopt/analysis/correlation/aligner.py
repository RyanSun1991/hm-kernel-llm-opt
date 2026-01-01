"""Align runtime hotspots to static PSG nodes."""

from __future__ import annotations

from typing import Iterable, List

from hmopt.analysis.runtime.hotspot import HotspotCandidate
from hmopt.analysis.static.psg import PsgGraph


def align_hotspots_to_psg(
    hotspots: Iterable[HotspotCandidate], psg: PsgGraph
) -> List[HotspotCandidate]:
    aligned: list[HotspotCandidate] = []
    for hs in hotspots:
        node = psg.nodes.get(hs.symbol)
        if node:
            hs.file_path = node.file_path
            hs.line_start = node.line
            hs.line_end = node.line
        aligned.append(hs)
    return aligned
