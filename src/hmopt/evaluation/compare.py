"""Compare candidate vs baseline/best."""

from __future__ import annotations

from typing import Dict

from sqlalchemy.orm import Session

from hmopt.analysis.runtime.metrics import compute_delta
from hmopt.storage.db import models


def compare_runs(session: Session, baseline_run_id: str, candidate_run_id: str) -> Dict[str, float]:
    base_metrics = session.query(models.Metric).filter(models.Metric.run_id == baseline_run_id).all()
    cand_metrics = session.query(models.Metric).filter(models.Metric.run_id == candidate_run_id).all()
    base_map = {m.metric_name: m.value for m in base_metrics}
    cand_map = {m.metric_name: m.value for m in cand_metrics}
    return compute_delta(base_map, cand_map)
