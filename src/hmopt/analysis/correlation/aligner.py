"""Align runtime hotspots to static PSG nodes."""

from __future__ import annotations

import logging

from typing import Iterable, List

from hmopt.analysis.runtime.hotspot import HotspotCandidate
from hmopt.analysis.static.psg import PsgGraph

logger = logging.getLogger(__name__)


def align_hotspots_to_psg(
    hotspots: Iterable[HotspotCandidate], psg: PsgGraph
) -> List[HotspotCandidate]:
    aligned: list[HotspotCandidate] = []
    hotspots = list(hotspots)
    for hs in hotspots:
        node = psg.nodes.get(hs.symbol)
        if node:
            hs.file_path = node.file_path
            hs.line_start = node.line
            hs.line_end = node.line
        aligned.append(hs)
    logger.info("Aligned hotspots to PSG: aligned=%d", len(aligned))
    return aligned
