"""Hotspot detection + ranking."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Iterable, List, Optional

from hmopt.storage.db import models

logger = logging.getLogger(__name__)

@dataclass
class HotspotCandidate:
    symbol: str
    file_path: Optional[str]
    line_start: Optional[int]
    line_end: Optional[int]
    score: float
    evidence_artifacts: list[str] | None = None


def _funcrank_scores(edge_costs: Dict[tuple[str, str], float], damping: float = 0.85) -> Dict[str, float]:
    nodes = set()
    for (caller, callee) in edge_costs:
        nodes.add(caller)
        nodes.add(callee)
    if not nodes:
        return {}
    scores = {n: 1.0 for n in nodes}
    outgoing = {}
    for (caller, _), weight in edge_costs.items():
        outgoing.setdefault(caller, 0.0)
        outgoing[caller] += weight

    for _ in range(10):
        new_scores = {n: (1 - damping) for n in nodes}
        for (caller, callee), weight in edge_costs.items():
            share = weight / outgoing.get(caller, 1.0)
            new_scores[callee] += damping * scores[caller] * share
        scores = new_scores
    return scores


def rank_hotspots(
    hotspot_costs: Dict[str, float],
    edge_costs: Dict[tuple[str, str], float],
    *,
    top_n: int = 10,
) -> List[HotspotCandidate]:
    """Combine baseline cost sort + FuncRank-like propagation."""
    base_scores = dict(hotspot_costs)
    fr_scores = _funcrank_scores(edge_costs)
    combined: dict[str, float] = {}
    for sym in set(base_scores) | set(fr_scores):
        combined[sym] = base_scores.get(sym, 0.0) * 0.6 + fr_scores.get(sym, 0.0) * 0.4
    ordered = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        HotspotCandidate(
            symbol=sym,
            file_path=None,
            line_start=None,
            line_end=None,
            score=score,
            evidence_artifacts=[],
        )
        for sym, score in ordered
    ]


def persist_hotspots(session, run_id: str, hotspots: Iterable[HotspotCandidate]) -> None:
    hotspots = list(hotspots)
    for hs in hotspots:
        session.add(
            models.Hotspot(
                run_id=run_id,
                symbol=hs.symbol,
                file_path=hs.file_path,
                line_start=hs.line_start,
                line_end=hs.line_end,
                score=hs.score,
                evidence_artifact_ids=hs.evidence_artifacts or [],
            )
        )
    session.flush()
    logger.info("Persisted hotspots: run=%s count=%d", run_id, len(hotspots))
