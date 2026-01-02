"""Sysfs node read trace parser.

The parser is intentionally forgiving: it accepts either JSON (``{"events": [...]}`` or
raw lists) or CSV inputs with the following optional columns/fields:

- ``path`` / ``node`` / ``sysfs_path``: sysfs file path being read
- ``ts`` / ``timestamp_us``: timestamp in microseconds
- ``latency_us`` / ``duration_us`` / ``dur``: read latency in microseconds
- ``bytes`` / ``size`` / ``bytes_read``: bytes read
- ``success`` / ``status`` / ``result``: truthy/falsey indicator of read success

Metrics reported:
- read counts (total + errors)
- latency percentiles (p50/p95/p99)
- total bytes read
- top-N nodes by frequency (tagged with ``path``)
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..metrics import Metric, quantile

logger = logging.getLogger(__name__)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return value != 0
    value_str = str(value).strip().lower()
    return value_str in {"1", "true", "ok", "success", "passed"}


@dataclass
class SysfsReadEvent:
    path: str
    timestamp_us: float | None
    latency_us: float
    bytes_read: int
    success: bool


@dataclass
class SysfsTraceSummary:
    events: List[SysfsReadEvent]
    total_reads: int
    error_count: int
    latency_p50: float
    latency_p95: float
    latency_p99: float
    bytes_total: int
    top_nodes: List[tuple[str, int]]

    def to_metrics(self) -> list[Metric]:
        metrics = [
            Metric("sysfs_read_count", self.total_reads, unit="reads"),
            Metric("sysfs_read_errors", self.error_count, unit="reads"),
            Metric("sysfs_read_latency_p50", self.latency_p50, unit="us"),
            Metric("sysfs_read_latency_p95", self.latency_p95, unit="us"),
            Metric("sysfs_read_latency_p99", self.latency_p99, unit="us"),
            Metric("sysfs_read_bytes", self.bytes_total, unit="bytes"),
        ]
        for idx, (path, count) in enumerate(self.top_nodes[:3], start=1):
            metrics.append(
                Metric(
                    f"sysfs_top{idx}_read_count",
                    count,
                    unit="reads",
                    tags={"path": path},
                )
            )
        return metrics


def _load_sysfs_events(path: Path) -> list[SysfsReadEvent]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("events") or data.get("reads") or data
    else:
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    events: list[SysfsReadEvent] = []
    for row in rows:
        sysfs_path = row.get("path") or row.get("node") or row.get("sysfs_path") or ""
        ts_raw = row.get("ts") or row.get("timestamp_us") or row.get("timestamp")
        latency_raw = row.get("latency_us") or row.get("duration_us") or row.get("dur")
        bytes_raw = row.get("bytes") or row.get("size") or row.get("bytes_read") or 0
        success_raw = row.get("success") or row.get("status") or row.get("result")

        try:
            ts = float(ts_raw) if ts_raw not in (None, "", "None") else None
        except (TypeError, ValueError):
            ts = None
        try:
            latency = float(latency_raw) if latency_raw not in (None, "", "None") else 0.0
        except (TypeError, ValueError):
            latency = 0.0
        try:
            bytes_read = int(float(bytes_raw)) if bytes_raw not in (None, "", "None") else 0
        except (TypeError, ValueError):
            bytes_read = 0

        events.append(
            SysfsReadEvent(
                path=str(sysfs_path),
                timestamp_us=ts,
                latency_us=latency,
                bytes_read=bytes_read,
                success=_as_bool(success_raw),
            )
        )
    return events


def parse_sysfs_trace(path: Path) -> SysfsTraceSummary:
    events = _load_sysfs_events(path)
    latencies = [ev.latency_us for ev in events] or [0.0]
    p50 = quantile(latencies, 0.50)
    p95 = quantile(latencies, 0.95)
    p99 = quantile(latencies, 0.99)
    bytes_total = sum(ev.bytes_read for ev in events)
    error_count = len([ev for ev in events if not ev.success])
    counts = Counter(ev.path for ev in events)
    top_nodes = counts.most_common(5)

    logger.info(
        "Sysfs trace parsed: file=%s reads=%d errors=%d p99=%.2f",
        path,
        len(events),
        error_count,
        p99,
    )
    return SysfsTraceSummary(
        events=events,
        total_reads=len(events),
        error_count=error_count,
        latency_p50=p50,
        latency_p95=p95,
        latency_p99=p99,
        bytes_total=bytes_total,
        top_nodes=top_nodes,
    )
