"""Format dataset examples from stored artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from sqlalchemy.orm import Session

from hmopt.storage.db import models


def format_example(session: Session, run_id: str) -> Dict:
    evidence = (
        session.query(models.Artifact)
        .filter(models.Artifact.run_id == run_id, models.Artifact.kind == "evidence_pack")
        .order_by(models.Artifact.bytes.desc())
        .first()
    )
    patch = (
        session.query(models.Artifact)
        .filter(models.Artifact.run_id == run_id, models.Artifact.kind == "patch_diff")
        .order_by(models.Artifact.bytes.desc())
        .first()
    )
    evaluation = (
        session.query(models.Evaluation)
        .filter(models.Evaluation.run_id == run_id)
        .order_by(models.Evaluation.id.desc())
        .first()
    )
    evidence_text = Path(evidence.path).read_text(encoding="utf-8") if evidence else ""
    patch_text = Path(patch.path).read_text(encoding="utf-8") if patch else ""
    delta = evaluation.delta_metrics_json if evaluation else {}
    return {
        "run_id": run_id,
        "evidence": json.loads(evidence_text) if evidence_text else {},
        "patch_diff": patch_text,
        "delta_metrics": delta,
    }
