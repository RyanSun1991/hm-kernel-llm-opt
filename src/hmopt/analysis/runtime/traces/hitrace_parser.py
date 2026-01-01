"""Hitrace parser (lightweight)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..metrics import Metric, quantile


@dataclass
class HitraceEvent:
    ts_us: float
    dur_us: float
    pid: int | None
    tid: int | None
    cpu: int | None
    name: str


@dataclass
class HitraceSummary:
    events: List[HitraceEvent]
    sched_latency_p50: float
    sched_latency_p95: float
    sched_latency_p99: float
    cpu_idle_ratio: float

    def to_metrics(self) -> list[Metric]:
        return [
            Metric("sched_latency_p50", self.sched_latency_p50, unit="us"),
            Metric("sched_latency_p95", self.sched_latency_p95, unit="us"),
            Metric("sched_latency_p99", self.sched_latency_p99, unit="us"),
            Metric("cpu_idle_ratio", self.cpu_idle_ratio, unit="ratio"),
        ]


def _load(path: Path) -> list[HitraceEvent]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("events", data)
    else:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    events: list[HitraceEvent] = []
    for row in rows:
        ts = float(row.get("ts") or row.get("timestamp_us") or 0)
        dur = float(row.get("dur") or row.get("duration_us") or 0)
        pid = row.get("pid")
        tid = row.get("tid") or row.get("thread")
        cpu = row.get("cpu")
        name = str(row.get("name") or row.get("event") or "event")
        events.append(
            HitraceEvent(
                ts_us=ts,
                dur_us=dur,
                pid=int(pid) if pid not in (None, "", "None") else None,
                tid=int(tid) if tid not in (None, "", "None") else None,
                cpu=int(cpu) if cpu not in (None, "", "None") else None,
                name=name,
            )
        )
    return events


def parse_hitrace(path: Path) -> HitraceSummary:
    events = _load(path)
    latencies = [ev.dur_us for ev in events if ev.dur_us]
    if not latencies:
        latencies = [0.0]
    p50 = quantile(latencies, 0.50)
    p95 = quantile(latencies, 0.95)
    p99 = quantile(latencies, 0.99)
    idle_time = sum(ev.dur_us for ev in events if ev.name.lower().startswith("idle"))
    total = sum(ev.dur_us for ev in events) or 1.0
    idle_ratio = idle_time / total
    return HitraceSummary(
        events=events,
        sched_latency_p50=p50,
        sched_latency_p95=p95,
        sched_latency_p99=p99,
        cpu_idle_ratio=idle_ratio,
    )
