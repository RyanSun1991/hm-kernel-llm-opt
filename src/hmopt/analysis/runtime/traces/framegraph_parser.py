"""Framegraph parser."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ..metrics import Metric, quantile

logger = logging.getLogger(__name__)


@dataclass
class FrameSample:
    ts_ms: float
    dur_ms: float


@dataclass
class FramegraphResult:
    samples: List[FrameSample]
    fps_avg: float
    frame_drop_rate: float
    jank_p95_ms: float
    jank_windows: List[tuple[float, float]]

    def to_metrics(self) -> list[Metric]:
        return [
            Metric(metric_name="fps_avg", value=self.fps_avg, unit="fps"),
            Metric(metric_name="frame_drop_rate", value=self.frame_drop_rate, unit="ratio"),
            Metric(metric_name="jank_p95_ms", value=self.jank_p95_ms, unit="ms"),
        ]


def _load_frames(path: Path) -> list[FrameSample]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        frames = data.get("frames", data)
        samples = []
        for frame in frames:
            ts = float(frame.get("ts") or frame.get("timestamp_ms") or frame.get("t", 0))
            dur = float(frame.get("dur") or frame.get("duration_ms") or frame.get("duration", 0))
            samples.append(FrameSample(ts_ms=ts, dur_ms=dur))
        return samples

    samples = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row.get("ts") or row.get("timestamp_ms") or row.get("t") or 0)
            dur = float(row.get("dur_ms") or row.get("dur") or row.get("duration_ms") or 0)
            samples.append(FrameSample(ts_ms=ts, dur_ms=dur))
    return samples


def parse_framegraph(path: Path, expected_frame_ms: float = 16.67) -> FramegraphResult:
    samples = _load_frames(path)
    if not samples:
        logger.warning("Framegraph empty: %s", path)
        return FramegraphResult([], 0.0, 0.0, 0.0, [])

    start = min(s.ts_ms for s in samples)
    end = max(s.ts_ms + s.dur_ms for s in samples)
    total_time = max(end - start, 1e-3)
    fps_avg = (len(samples) / total_time) * 1000.0

    jank_threshold = expected_frame_ms * 1.3
    jank_durations = [s.dur_ms for s in samples if s.dur_ms >= jank_threshold]
    jank_p95 = quantile([s.dur_ms for s in samples], 0.95)
    drop_rate = len([s for s in samples if s.dur_ms > expected_frame_ms * 2]) / len(samples)

    # Identify jank windows (contiguous jank frames)
    windows: list[tuple[float, float]] = []
    current_start = None
    current_end = None
    for s in samples:
        if s.dur_ms >= jank_threshold:
            if current_start is None:
                current_start = s.ts_ms
            current_end = s.ts_ms + s.dur_ms
        elif current_start is not None and current_end is not None:
            windows.append((current_start, current_end))
            current_start, current_end = None, None
    if current_start is not None and current_end is not None:
        windows.append((current_start, current_end))

    logger.info(
        "Framegraph parsed: file=%s frames=%d fps=%.2f jank_p95=%.2f drop_rate=%.3f",
        path,
        len(samples),
        fps_avg,
        jank_p95,
        drop_rate,
    )
    return FramegraphResult(
        samples=samples,
        fps_avg=fps_avg,
        frame_drop_rate=drop_rate,
        jank_p95_ms=jank_p95,
        jank_windows=windows,
    )
