"""Rank likely culprit code areas from correlated evidence."""

from __future__ import annotations

from typing import Iterable, List, Mapping

from hmopt.analysis.runtime.hotspot import HotspotCandidate


def rank_correlated(
    hotspots: Iterable[HotspotCandidate],
    metrics: Mapping[str, float] | None = None,
    limit: int = 10,
) -> List[HotspotCandidate]:
    metrics = metrics or {}
    boost = 1.0
    if metrics.get("jank_p95_ms", 0) > 20:
        boost = 1.1
    weighted = [
        HotspotCandidate(
            symbol=h.symbol,
            file_path=h.file_path,
            line_start=h.line_start,
            line_end=h.line_end,
            score=h.score * boost,
            evidence_artifacts=h.evidence_artifacts,
            call_stacks=h.call_stacks,
        )
        for h in hotspots
    ]
    weighted.sort(key=lambda h: h.score, reverse=True)
    return weighted[:limit]
