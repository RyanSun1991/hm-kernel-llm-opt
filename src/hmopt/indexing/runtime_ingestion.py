"""Runtime trace ingestion for LlamaIndex."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional

from llama_index.core.schema import TextNode
from sqlalchemy.orm import Session

from hmopt.storage.db import models


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

def _load_artifacts_payloads(session: Session, run_id: str, kind: str) -> list[dict | list]:
    artifacts = (
        session.query(models.Artifact)
        .filter(models.Artifact.run_id == run_id, models.Artifact.kind == kind)
        .order_by(models.Artifact.bytes.desc())
        .all()
    )
    payloads: list[dict | list] = []
    for art in artifacts:
        try:
            raw = Path(art.path).read_text(encoding="utf-8")
            payload = json.loads(raw)
            if isinstance(payload, (dict, list)):
                payloads.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return payloads


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
        call_stacks = hotspot.call_stacks_json or []
        call_stack_text = ""
        if call_stacks:
            stack_lines = []
            for idx, stack in enumerate(call_stacks[:5]):
                if isinstance(stack, dict):
                    path = " -> ".join(stack.get("stack", []))
                    events = stack.get("self_events", 0)
                    direction = stack.get("direction", "call")
                    stack_lines.append(
                        f"  [{idx+1}] ({direction}) {path} (events: {events:.0f})"
                    )
            if stack_lines:
                call_stack_text = "\nCall stacks:\n" + "\n".join(stack_lines)

        text = (
            f"hotspot {hotspot.symbol} score {hotspot.score} "
            f"file {hotspot.file_path or ''} lines {hotspot.line_start}-{hotspot.line_end}"
            f"{call_stack_text}"
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
                    "call_stacks": _normalize_metadata_value(call_stacks),
                },
            )
        )


    symbol_counts_payloads = _load_artifacts_payloads(
        session, run_id, "flamegraph_symbol_counts"
    )
    if not symbol_counts_payloads:
        symbol_counts_payloads = _load_artifacts_payloads(
            session, run_id, "flamegraph_symbol_counts_raw"
        )
    symbol_scores: dict[str, float] = {}
    for payload in symbol_counts_payloads:
        if not isinstance(payload, dict):
            continue
        for symbol, score in payload.items():
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            symbol_scores[symbol] = symbol_scores.get(symbol, 0.0) + value

    call_stack_payloads = _load_artifacts_payloads(
        session, run_id, "flamegraph_call_stacks"
    )
    call_stacks: list[dict] = []
    for payload in call_stack_payloads:
        if isinstance(payload, list):
            call_stacks.extend([item for item in payload if isinstance(item, dict)])
    symbol_stacks: dict[str, list[dict]] = defaultdict(list)
    for stack in call_stacks:
        leaf = stack.get("leaf_symbol")
        if not leaf:
            path = stack.get("stack") or []
            if isinstance(path, list) and path:
                leaf = path[-1]
        if leaf:
            symbol_stacks[leaf].append(stack)

    for symbol in sorted(set(symbol_scores) | set(symbol_stacks)):
        score = symbol_scores.get(symbol)
        stacks = symbol_stacks.get(symbol, [])
        stack_lines = []
        for idx, stack in enumerate(stacks[:3]):
            if isinstance(stack, dict):
                path = " -> ".join(stack.get("stack", []))
                events = stack.get("self_events", 0)
                direction = stack.get("direction", "call")
                stack_lines.append(
                    f"  [{idx+1}] ({direction}) {path} (events: {events:.0f})"
                )
        stack_text = "\nCall stacks:\n" + "\n".join(stack_lines) if stack_lines else ""
        score_text = f"{score:.4f}" if isinstance(score, float) else ""
        text = f"symbol {symbol} score {score_text}{stack_text}".strip()
        nodes.append(
            TextNode(
                text=text,
                metadata={
                    "type": "runtime_symbol",
                    "run_id": run_id,
                    "symbol": symbol,
                    "score": _normalize_metadata_value(score),
                    "call_stacks": _normalize_metadata_value(stacks),
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
