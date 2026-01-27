"""Flamegraph parser."""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from ..metrics import Metric, quantile

logger = logging.getLogger(__name__)


@dataclass
class FrameSample:
    ts_ms: float
    dur_ms: float


@dataclass
class FlamegraphResult:
    samples: List[FrameSample]
    fps_avg: float
    frame_drop_rate: float
    jank_p95_ms: float
    jank_windows: List[tuple[float, float]]
    has_frame_timeline: bool = False
    event_count_total: float = 0.0
    event_counts: dict[str, float] = field(default_factory=dict)
    sample_count_total: int = 0
    process_count: int = 0
    thread_count: int = 0
    symbol_counts_raw: dict[str, float] = field(default_factory=dict)
    symbol_counts: dict[str, float] = field(default_factory=dict)
    symbol_counts_per_thread: dict[str, dict[str, float]] = field(default_factory=dict)
    name_maps: Optional["FlamegraphNameMaps"] = None
    process_summaries: dict[str, dict] = field(default_factory=dict)
    thread_summaries: dict[str, dict] = field(default_factory=dict)
    lib_summaries: dict[str, dict] = field(default_factory=dict)
    pcg_nodes: dict[str, dict] = field(default_factory=dict)
    pcg_edges: list[dict] = field(default_factory=list)
    call_stacks: list[dict] = field(default_factory=list)
    source_path: Optional[str] = None

    def to_metrics(self) -> list[Metric]:
        metrics: list[Metric] = []
        if self.has_frame_timeline:
            metrics.extend(
                [
                    Metric(metric_name="fps_avg", value=self.fps_avg, unit="fps"),
                    Metric(metric_name="frame_drop_rate", value=self.frame_drop_rate, unit="ratio"),
                    Metric(metric_name="jank_p95_ms", value=self.jank_p95_ms, unit="ms"),
                ]
            )
        if self.event_count_total:
            metrics.append(
                Metric(metric_name="flamegraph_event_count_total", value=self.event_count_total)
            )
        for name, value in self.event_counts.items():
            metric_name = f"flamegraph_event_{_sanitize_event_name(name)}_total"
            unit = _event_unit(name)
            metrics.append(Metric(metric_name=metric_name, value=value, unit=unit))
            if "instruction" in name.lower():
                metrics.append(Metric(metric_name="instruction_count_total", value=value, unit=unit))
        if self.sample_count_total:
            metrics.append(Metric(metric_name="flamegraph_sample_count_total", value=float(self.sample_count_total)))
        if self.process_count:
            metrics.append(Metric(metric_name="flamegraph_process_count", value=float(self.process_count)))
        if self.thread_count:
            metrics.append(Metric(metric_name="flamegraph_thread_count", value=float(self.thread_count)))
        if self.symbol_counts:
            metrics.append(
                Metric(metric_name="flamegraph_symbol_count", value=float(len(self.symbol_counts)))
            )
        return metrics


@dataclass
class FlamegraphNameMaps:
    process_name_map: dict[str, str] = field(default_factory=dict)
    thread_name_map: dict[str, str] = field(default_factory=dict)
    symbols_file_list: list[str] = field(default_factory=list)
    symbol_map: dict[str, dict] = field(default_factory=dict)

    def resolve_symbol(self, symbol_id: int | str, file_hint: Optional[int] = None) -> str:
        entry = self.symbol_map.get(str(symbol_id)) or self.symbol_map.get(symbol_id)
        symbol = f"symbol_{symbol_id}"
        file_idx = file_hint
        if isinstance(entry, dict):
            symbol = entry.get("symbol", symbol)
            file_idx = entry.get("file", file_idx)
        if file_idx is not None and 0 <= int(file_idx) < len(self.symbols_file_list):
            file_path = self.symbols_file_list[int(file_idx)]
            if file_path and not symbol.startswith(file_path):
                return f"{file_path}:{symbol}"
        return symbol

    def to_dict(self) -> dict:
        return {
            "processNameMap": self.process_name_map,
            "threadNameMap": self.thread_name_map,
            "symbolsFileList": self.symbols_file_list,
            "SymbolMap": self.symbol_map,
        }


def _sanitize_event_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower())
    return cleaned.strip("_") or "event"


def _event_unit(name: str) -> str:
    lower = name.lower()
    if "instruction" in lower:
        return "instructions"
    return "events"


def _normalize_symbol(raw: str) -> str:
    sym = raw.strip()
    if ":" in sym:
        sym = sym.split(":")[-1]
    if "+" in sym:
        sym = sym.split("+")[0]
    sym = sym.strip()
    return sym or raw


def _top_symbols(weights: dict[str, float], top_n: int = 5) -> list[dict]:
    ordered = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"symbol": sym, "weight": weight} for sym, weight in ordered]


def _walk_call_tree(
    node: dict,
    name_maps: FlamegraphNameMaps,
    node_stats: dict[str, dict],
    edge_stats: dict[tuple[str, str, str], float],
    call_stacks: list[dict],
    *,
    parent: Optional[str] = None,
    direction: str = "call",
    current_stack: Optional[List[str]] = None,
    thread_id: Optional[str] = None,
) -> None:
    if current_stack is None:
        current_stack = []
    if not isinstance(node, dict):
        return
    symbol_id = node.get("symbol")
    current: Optional[str] = parent
    new_stack = current_stack

    if symbol_id not in (None, -1):
        raw = name_maps.resolve_symbol(symbol_id)
        norm = _normalize_symbol(raw)
        current = norm
        new_stack = current_stack + [norm]

        stats = node_stats.setdefault(norm, {"self_events": 0.0, "sub_events": 0.0, "label": raw})
        self_events = float(node.get("selfEvents", 0) or 0)
        sub_events = float(node.get("subEvents", 0) or 0)
        stats["self_events"] += self_events
        stats["sub_events"] += sub_events

        if parent:
            key = (parent, norm, direction)
            edge_stats[key] = edge_stats.get(key, 0.0) + sub_events

        children = node.get("callStack", []) or []
        if self_events > 0 or not children:
            call_stacks.append({
                "stack": new_stack,
                "leaf_symbol": norm,
                "total_events": sub_events,
                "self_events": self_events,
                "thread_id": thread_id,
                "direction": direction,
            })

    for child in node.get("callStack", []) or []:
        if isinstance(child, dict):
            _walk_call_tree(
                child,
                name_maps,
                node_stats,
                edge_stats,
                call_stacks,
                parent=current,
                direction=direction,
                current_stack=new_stack,
                thread_id=thread_id,
            )


def _samples_from_payload(frames: list[dict]) -> list[FrameSample]:
    samples = []
    for frame in frames:
        ts = float(frame.get("ts") or frame.get("timestamp_ms") or frame.get("t", 0))
        dur = float(frame.get("dur") or frame.get("duration_ms") or frame.get("duration", 0))
        samples.append(FrameSample(ts_ms=ts, dur_ms=dur))
    return samples


def _load_frames(path: Path) -> list[FrameSample]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        frames = data.get("frames", data)
        if isinstance(frames, list):
            return _samples_from_payload(frames)

    samples = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row.get("ts") or row.get("timestamp_ms") or row.get("t") or 0)
            dur = float(row.get("dur_ms") or row.get("dur") or row.get("duration_ms") or 0)
            samples.append(FrameSample(ts_ms=ts, dur_ms=dur))
    return samples


def _extract_record_data(html_text: str) -> str | None:
    pattern = (
        r"<script[^>]*id=[\"']record_data[\"'][^>]*type=[\"']application/json[\"'][^>]*>"
        r"(.*?)</script>"
    )
    match = re.search(pattern, html_text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _parse_flamegraph_payload(
    data: object, path: Path, expected_frame_ms: float = 16.67
) -> FlamegraphResult:
    if isinstance(data, dict) and ("recordSampleInfo" in data or "processNameMap" in data):
        return _parse_sample_info(data, path)

    frames = data.get("frames", data) if isinstance(data, dict) else data
    samples: list[FrameSample] = []
    if isinstance(frames, list):
        samples = _samples_from_payload(frames)
    return _flamegraph_from_samples(samples, path, expected_frame_ms)


def _flamegraph_from_samples(
    samples: list[FrameSample], path: Path, expected_frame_ms: float
) -> FlamegraphResult:
    if not samples:
        logger.warning("Flamegraph empty: %s", path)
        return FlamegraphResult([], 0.0, 0.0, 0.0, [])

    start = min(s.ts_ms for s in samples)
    end = max(s.ts_ms + s.dur_ms for s in samples)
    total_time = max(end - start, 1e-3)
    fps_avg = (len(samples) / total_time) * 1000.0

    jank_threshold = expected_frame_ms * 1.3
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
        "Flamegraph parsed: file=%s frames=%d fps=%.2f jank_p95=%.2f drop_rate=%.3f",
        path,
        len(samples),
        fps_avg,
        jank_p95,
        drop_rate,
    )
    return FlamegraphResult(
        samples=samples,
        fps_avg=fps_avg,
        frame_drop_rate=drop_rate,
        jank_p95_ms=jank_p95,
        jank_windows=windows,
        has_frame_timeline=True,
        source_path=str(path),
    )


def _parse_flamegraph_html(path: Path, expected_frame_ms: float = 16.67) -> FlamegraphResult | None:
    html_text = path.read_text(encoding="utf-8", errors="ignore")
    payload = _extract_record_data(html_text)
    if not payload:
        logger.warning("Flamegraph HTML missing record_data: %s", path)
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        logger.warning("Flamegraph HTML JSON decode failed: %s error=%s", path, exc)
        return None
    return _parse_flamegraph_payload(data, path, expected_frame_ms)


def parse_flamegraph(path: Path, expected_frame_ms: float = 16.67) -> list[FlamegraphResult]:
    if path.is_dir():
        html_files = sorted(path.rglob("*__sysmgr_hiperfReport.html"))
        results: list[FlamegraphResult] = []
        for html_file in html_files:
            parsed = _parse_flamegraph_html(html_file, expected_frame_ms)
            if parsed:
                results.append(parsed)
        if not results:
            logger.warning("Flamegraph directory had no report HTML: %s", path)
        return results

    if path.suffix.lower() == ".html":
        parsed = _parse_flamegraph_html(path, expected_frame_ms)
        return [parsed] if parsed else []

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [_parse_flamegraph_payload(data, path, expected_frame_ms)]

    samples = _load_frames(path)
    if not samples:
        logger.warning("Flamegraph empty: %s", path)
        return []

    return [_flamegraph_from_samples(samples, path, expected_frame_ms)]


def _parse_sample_info(data: dict, path: Path) -> FlamegraphResult:
    name_maps = FlamegraphNameMaps(
        process_name_map=data.get("processNameMap", {}) or {},
        thread_name_map=data.get("threadNameMap", {}) or {},
        symbols_file_list=data.get("symbolsFileList", [])
        or data.get("symbols_file_list", [])
        or [],
        symbol_map=data.get("SymbolMap", {}) or data.get("symbolMap", {}) or {},
    )
    record_infos = data.get("recordSampleInfo", []) or []

    pids: set[int] = set()
    tids: set[int] = set()
    sample_count_total = 0
    symbol_counts_raw: dict[str, float] = {}
    symbol_counts: dict[str, float] = {}
    symbol_counts_per_thread: dict[str, dict[str, float]] = {}
    process_summaries: dict[str, dict] = {}
    thread_summaries: dict[str, dict] = {}
    lib_summaries: dict[str, dict] = {}

    process_symbol_weights: dict[int, dict[str, float]] = {}
    thread_symbol_weights: dict[int, dict[str, float]] = {}
    lib_symbol_weights: dict[int, dict[str, float]] = {}

    event_count_total = 0.0
    event_counts: dict[str, float] = {}
    node_stats: dict[str, dict] = {}
    edge_stats: dict[tuple[str, str, str], float] = {}
    call_stacks: list[dict] = []
    for info in record_infos:
        event_name = info.get("eventConfigName") or "event"
        count = float(info.get("eventCount", 0) or 0)
        event_count_total += count
        event_counts[event_name] = event_counts.get(event_name, 0.0) + count
        for proc in info.get("processes", []) or []:
            pid = proc.get("pid")
            if pid is not None:
                pids.add(int(pid))
                proc_summary = process_summaries.setdefault(
                    str(pid),
                    {
                        "pid": int(pid),
                        "name": name_maps.process_name_map.get(str(pid), ""),
                        "event_count": 0.0,
                        "sample_count": 0,
                        "thread_count": 0,
                        "top_symbols": [],
                    },
                )
                proc_summary["event_count"] += float(proc.get("eventCount", 0) or 0)
            for thread in proc.get("threads", []) or []:
                tid = thread.get("tid")
                if tid is not None:
                    tids.add(int(tid))
                    thread_summary = thread_summaries.setdefault(
                        str(tid),
                        {
                            "tid": int(tid),
                            "pid": int(pid) if pid is not None else None,
                            "name": name_maps.thread_name_map.get(str(tid), ""),
                            "event_count": 0.0,
                            "sample_count": 0,
                            "top_symbols": [],
                        },
                    )
                    thread_summary["event_count"] += float(thread.get("eventCount", 0) or 0)
                sample_count_total += int(thread.get("sampleCount", 0) or 0)
                if pid is not None:
                    process_summaries[str(pid)]["sample_count"] += int(thread.get("sampleCount", 0) or 0)
                    process_summaries[str(pid)]["thread_count"] += 1
                if tid is not None:
                    thread_summaries[str(tid)]["sample_count"] += int(thread.get("sampleCount", 0) or 0)

                call_order = thread.get("CallOrder")
                thread_id_str = str(tid) if tid is not None else None
                if isinstance(call_order, dict):
                    _walk_call_tree(
                        call_order,
                        name_maps,
                        node_stats,
                        edge_stats,
                        call_stacks,
                        parent=None,
                        direction="call",
                        current_stack=[],
                        thread_id=thread_id_str,
                    )
                called_order = thread.get("CalledOrder")
                if isinstance(called_order, dict):
                    _walk_call_tree(
                        called_order,
                        name_maps,
                        node_stats,
                        edge_stats,
                        call_stacks,
                        parent=None,
                        direction="called",
                        current_stack=[],
                        thread_id=thread_id_str,
                    )
                for lib in thread.get("libs", []) or []:
                    file_id = lib.get("fileId")
                    if file_id is not None:
                        lib_summary = lib_summaries.setdefault(
                            str(file_id),
                            {
                                "file_id": int(file_id),
                                "file_path": name_maps.symbols_file_list[int(file_id)]
                                if 0 <= int(file_id) < len(name_maps.symbols_file_list)
                                else "",
                                "event_count": 0.0,
                                "top_symbols": [],
                            },
                        )
                        lib_summary["event_count"] += float(lib.get("eventCount", 0) or 0)
                    for func in lib.get("functions", []) or []:
                        symbol_id = func.get("symbol")
                        if symbol_id in (None, -1):
                            continue
                        counts = func.get("counts", [])
                        weight = 0.0
                        if isinstance(counts, list) and counts:
                            weight = float(counts[-1])
                        else:
                            weight = float(func.get("eventCount", 0) or 0)
                        symbol_name = name_maps.resolve_symbol(symbol_id, file_id)
                        symbol_counts_raw[symbol_name] = symbol_counts_raw.get(symbol_name, 0.0) + weight
                        normalized = _normalize_symbol(symbol_name)
                        symbol_counts[normalized] = symbol_counts.get(normalized, 0.0) + weight
                        if pid is not None:
                            process_symbol_weights.setdefault(int(pid), {})
                            process_symbol_weights[int(pid)][normalized] = (
                                process_symbol_weights[int(pid)].get(normalized, 0.0) + weight
                            )
                        if tid is not None:
                            thread_symbol_weights.setdefault(int(tid), {})
                            thread_symbol_weights[int(tid)][normalized] = (
                                thread_symbol_weights[int(tid)].get(normalized, 0.0) + weight
                            )
                        if file_id is not None:
                            lib_symbol_weights.setdefault(int(file_id), {})
                            lib_symbol_weights[int(file_id)][normalized] = (
                                lib_symbol_weights[int(file_id)].get(normalized, 0.0) + weight
                            )

    if not event_count_total:
        event_count_total = float(data.get("totalRecordSamples", 0) or 0)

    for pid, weights in process_symbol_weights.items():
        key = str(pid)
        if key in process_summaries:
            process_summaries[key]["top_symbols"] = _top_symbols(weights, top_n=5)
    for tid, weights in thread_symbol_weights.items():
        key = str(tid)
        if key in thread_summaries:
            thread_summaries[key]["top_symbols"] = _top_symbols(weights, top_n=5)
        symbol_counts_per_thread[key] = dict(weights)
    for fid, weights in lib_symbol_weights.items():
        key = str(fid)
        if key in lib_summaries:
            lib_summaries[key]["top_symbols"] = _top_symbols(weights, top_n=5)

    pcg_edges = [
        {"src": src, "dst": dst, "weight": weight, "direction": direction}
        for (src, dst, direction), weight in edge_stats.items()
    ]

    logger.info(
        "Flamegraph sample parsed: file=%s events=%.0f samples=%d processes=%d threads=%d symbols=%d",
        path,
        event_count_total,
        sample_count_total,
        len(pids),
        len(tids),
        len(symbol_counts),
    )
    return FlamegraphResult(
        samples=[],
        fps_avg=0.0,
        frame_drop_rate=0.0,
        jank_p95_ms=0.0,
        jank_windows=[],
        has_frame_timeline=False,
        event_count_total=event_count_total,
        event_counts=event_counts,
        sample_count_total=sample_count_total,
        process_count=len(pids),
        thread_count=len(tids),
        symbol_counts_raw=symbol_counts_raw,
        symbol_counts=symbol_counts,
        symbol_counts_per_thread=symbol_counts_per_thread,
        name_maps=name_maps,
        process_summaries=process_summaries,
        thread_summaries=thread_summaries,
        lib_summaries=lib_summaries,
        pcg_nodes=node_stats,
        pcg_edges=pcg_edges,
        call_stacks=call_stacks,
        source_path=str(path),
    )
