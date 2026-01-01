"""Unified metric schema + helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from hmopt.storage.db import models


@dataclass
class Metric:
    metric_name: str
    value: float
    unit: str = ""
    scope: str = "system"
    tags: dict[str, Any] | None = None


def record_metrics(session, run_id: str, metrics: Iterable[Metric]) -> None:
    """Persist metrics to the DB."""
    for metric in metrics:
        session.add(
            models.Metric(
                run_id=run_id,
                scope=metric.scope,
                metric_name=metric.metric_name,
                value=metric.value,
                unit=metric.unit,
                tags_json=metric.tags or {},
            )
        )
    session.flush()


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * q
    lower = int(idx)
    upper = min(lower + 1, len(values) - 1)
    weight = idx - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def compute_delta(
    baseline: dict[str, float], candidate: dict[str, float]
) -> dict[str, float]:
    """Compute deltas between metric maps."""
    delta: dict[str, float] = {}
    keys = set(baseline.keys()) | set(candidate.keys())
    for k in keys:
        base = baseline.get(k, 0.0)
        cand = candidate.get(k, 0.0)
        delta[k] = cand - base
    return delta
