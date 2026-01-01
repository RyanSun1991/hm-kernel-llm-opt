"""Hiperf/perf sample parser."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..metrics import Metric

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    stack: List[str]
    weight: float


@dataclass
class HiperfSummary:
    samples: List[Sample]
    edge_costs: Dict[tuple[str, str], float]
    hotspot_costs: Dict[str, float]

    def to_metrics(self) -> list[Metric]:
        total = sum(self.hotspot_costs.values()) or 1.0
        top = sorted(self.hotspot_costs.items(), key=lambda x: x[1], reverse=True)[:5]
        metrics = [Metric("hiperf_total_weight", total, unit="samples")]
        for idx, (sym, cost) in enumerate(top, 1):
            metrics.append(
                Metric(f"hiperf_top{idx}_weight", cost, unit="samples", tags={"symbol": sym})
            )
        return metrics


def _load_samples(path: Path) -> list[Sample]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("samples", data)
    else:
        # Text format: weight funcA;funcB;funcC
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                weight_str, stack_str = line.split(None, 1)
                entries.append({"weight": float(weight_str), "stack": stack_str.split(";")})
            except ValueError:
                continue

    samples: list[Sample] = []
    for entry in entries:
        stack = entry.get("stack") or entry.get("callstack") or []
        weight = float(entry.get("weight", 0))
        if isinstance(stack, str):
            stack = stack.split(";")
        samples.append(Sample(stack=list(stack), weight=weight))
    return samples


def parse_hiperf(path: Path) -> HiperfSummary:
    samples = _load_samples(path)
    edge_costs: dict[tuple[str, str], float] = {}
    hotspot_costs: dict[str, float] = {}
    for sample in samples:
        for sym in sample.stack:
            hotspot_costs[sym] = hotspot_costs.get(sym, 0.0) + sample.weight
        for caller, callee in zip(sample.stack, sample.stack[1:]):
            key = (caller, callee)
            edge_costs[key] = edge_costs.get(key, 0.0) + sample.weight
    logger.info(
        "Hiperf parsed: file=%s samples=%d hotspots=%d edges=%d",
        path,
        len(samples),
        len(hotspot_costs),
        len(edge_costs),
    )
    return HiperfSummary(samples=samples, edge_costs=edge_costs, hotspot_costs=hotspot_costs)
