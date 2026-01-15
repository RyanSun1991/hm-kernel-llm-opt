"""Runtime trace ingestion for LlamaIndex."""

from __future__ import annotations

from typing import List

from llama_index.core.schema import TextNode
from sqlalchemy.orm import Session

from hmopt.storage.db import models


def build_runtime_nodes(session: Session, run_id: str) -> List[TextNode]:
    nodes: list[TextNode] = []
    metrics = session.query(models.Metric).filter(models.Metric.run_id == run_id).all()
    hotspots = session.query(models.Hotspot).filter(models.Hotspot.run_id == run_id).all()

    for metric in metrics:
        text = f"metric {metric.metric_name} value {metric.value} unit {metric.unit or ''} tags {metric.tags_json}"
        nodes.append(
            TextNode(
                text=text,
                metadata={
                    "type": "runtime_metric",
                    "run_id": run_id,
                    "metric_name": metric.metric_name,
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
                    "file_path": hotspot.file_path,
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
        evidence_text = open(evidence.path, "r", encoding="utf-8").read()
        nodes.append(
            TextNode(
                text=f"evidence_pack {evidence_text}",
                metadata={"type": "evidence_pack", "run_id": run_id},
            )
        )
    return nodes
