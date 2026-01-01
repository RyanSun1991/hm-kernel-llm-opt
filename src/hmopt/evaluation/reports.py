"""Report generation utilities."""

from __future__ import annotations

from sqlalchemy.orm import Session

from hmopt.storage.artifact_store import ArtifactStore
from hmopt.storage.db import models


def render_report(session: Session, run_id: str) -> str:
    run = session.query(models.Run).filter(models.Run.run_id == run_id).one()
    metrics = session.query(models.Metric).filter(models.Metric.run_id == run_id).all()
    hotspots = session.query(models.Hotspot).filter(models.Hotspot.run_id == run_id).all()
    patches = session.query(models.Patch).filter(models.Patch.run_id == run_id).all()
    evaluations = session.query(models.Evaluation).filter(models.Evaluation.run_id == run_id).all()

    lines = [
        f"# HMOPT Report for {run_id}",
        f"Status: {run.status}",
        f"Repo rev: {run.repo_rev or 'n/a'}",
        "## Metrics",
    ]
    for m in metrics:
        lines.append(f"- {m.metric_name}: {m.value} {m.unit or ''}")
    lines.append("## Hotspots")
    for hs in hotspots[:10]:
        lines.append(f"- {hs.symbol} score={hs.score}")
    lines.append("## Patches")
    for p in patches:
        lines.append(f"- Iter {p.iteration}: {p.apply_status} diff={p.diff_artifact_id}")
    lines.append("## Evaluations")
    for ev in evaluations:
        lines.append(
            f"- perf_improved={ev.perf_improved} correctness={ev.correctness_passed} delta={ev.delta_metrics_json}"
        )
    return "\n".join(lines)


def generate_report(session: Session, artifact_store: ArtifactStore, run_id: str) -> models.Artifact:
    text = render_report(session, run_id)
    return artifact_store.store_text(text, kind="report", run_id=run_id, extension=".md", session=session)
