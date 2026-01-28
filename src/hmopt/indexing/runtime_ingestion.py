"""Runtime trace ingestion for LlamaIndex."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.schema import TextNode
from sqlalchemy.orm import Session

from hmopt.storage.db import models

_CALLSTACK_TOP_K = 10


def _read_evidence_text(path: str, max_chars: Optional[int]) -> str:
    try:
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
    except OSError:
        return ""
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return f"{text[:max_chars]}\n\n...[truncated {len(text) - max_chars} chars]"
    return text


def _normalize_metadata_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [str(item) if not isinstance(item, (str, int, float, bool)) else item for item in value]
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def build_runtime_nodes(
    session: Session, run_id: str, max_evidence_chars: Optional[int] = None
) -> List[TextNode]:
    nodes: list[TextNode] = []
    metrics = session.query(models.Metric).filter(models.Metric.run_id == run_id).all()
    hotspots = session.query(models.Hotspot).filter(models.Hotspot.run_id == run_id).all()

    for metric in metrics:
        text = (
            f"metric {metric.metric_name} value {metric.value} "
            f"unit {metric.unit or ''} tags {metric.tags_json}"
        )
        nodes.append(
            TextNode(
                text=text,
                metadata={
                    "type": "runtime_metric",
                    "run_id": run_id,
                    "metric_name": metric.metric_name,
                    "value": _normalize_metadata_value(metric.value),
                    "unit": _normalize_metadata_value(metric.unit),
                    "tags_json": _normalize_metadata_value(metric.tags_json),
                },
            )
        )

    for hotspot in hotspots:
        text = (
            f"hotspot {hotspot.symbol} score {hotspot.score} "
            f"file {hotspot.file_path or ''} lines {hotspot.line_start}-{hotspot.line_end}"
        )
        nodes.append(
            TextNode(
                text=text,
                metadata={
                    "type": "runtime_hotspot",
                    "run_id": run_id,
                    "symbol": hotspot.symbol,
                    "file_path": _normalize_metadata_value(hotspot.file_path),
                    "line_start": _normalize_metadata_value(hotspot.line_start),
                    "line_end": _normalize_metadata_value(hotspot.line_end),
                    "score": _normalize_metadata_value(hotspot.score),
                },
            )
        )

    callstack_artifacts = (
        session.query(models.Artifact)
        .filter(models.Artifact.run_id == run_id, models.Artifact.kind == "flamegraph_call_stacks")
        .all()
    )
    callstack_groups: Dict[Tuple[str, str], List[dict]] = {}
    for artifact in callstack_artifacts:
        try:
            payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for symbol, stacks in payload.items():
            if not isinstance(stacks, list):
                continue
            for entry in stacks:
                if not isinstance(entry, dict):
                    continue
                direction = str(entry.get("direction") or "call")
                callstack_groups.setdefault((symbol, direction), []).append(entry)

    for (symbol, direction), entries in callstack_groups.items():
        entries.sort(key=lambda item: item.get("sub_events", 0.0), reverse=True)
        top_entries = entries[:_CALLSTACK_TOP_K]
        lines = [
            f"callstack symbol {symbol} direction {direction} total_entries {len(entries)}"
        ]
        for entry in top_entries:
            path = entry.get("path") or []
            path_str = " -> ".join(path)
            lines.append(
                f"- sub_events {entry.get('sub_events', 0)} self_events {entry.get('self_events', 0)} "
                f"path {path_str}"
            )
            tids = entry.get("tids") or []
            pids = entry.get("pids") or []
            if tids or pids:
                lines.append(f"  tids {tids} pids {pids}")
        text = "\n".join(lines)
        nodes.append(
            TextNode(
                text=text,
                metadata={
                    "type": "runtime_callstack",
                    "run_id": run_id,
                    "symbol": symbol,
                    "direction": direction,
                    "entry_count": len(entries),
                },
            )
        )

    evidence = (
        session.query(models.Artifact)
        .filter(models.Artifact.run_id == run_id, models.Artifact.kind == "evidence_pack")
        .order_by(models.Artifact.bytes.desc())
        .first()
    )
    if evidence:
        evidence_text = _read_evidence_text(evidence.path, max_evidence_chars)
        if not evidence_text:
            return nodes
        nodes.append(
            TextNode(
                text=f"evidence_pack {evidence_text}",
                metadata={
                    "type": "evidence_pack",
                    "run_id": run_id,
                    "path": _normalize_metadata_value(evidence.path),
                    "max_chars": _normalize_metadata_value(max_evidence_chars),
                },
            )
        )
    return nodes
