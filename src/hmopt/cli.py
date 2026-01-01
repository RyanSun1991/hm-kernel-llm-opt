"""CLI entrypoint using Typer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from hmopt.core.config import AppConfig
from hmopt.orchestration import run_artifact_analysis, run_pipeline
from hmopt.storage.artifact_store import ArtifactStore
from hmopt.storage.db.engine import init_engine
from hmopt.storage.db import models
from hmopt.storage.db.engine import session_scope
from hmopt.core.config import load_yaml, normalize_raw_config

app = typer.Typer(help="HM-VERIF kernel optimization platform")


def _load_config(path: str) -> AppConfig:
    return AppConfig.from_yaml(path)


@app.command()
def run(config: str = typer.Option("configs/app.yaml", help="Path to config YAML")) -> None:
    # Demo: python -m hmopt.cli run --config configs/app.yaml
    # Purpose: run full optimization pipeline (baseline profile + iterative loop).
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    run_id = run_pipeline(cfg)
    typer.echo(f"Run completed: {run_id}")


@app.command()
def optimize(
    config: str = typer.Option("configs/app.yaml", help="Path to config YAML"),
    iterations: int = typer.Option(2, help="Max iterations"),
) -> None:
    # Demo: python -m hmopt.cli optimize --config configs/app.yaml --iterations 3
    # Purpose: same as run, but override iteration budget.
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    cfg.iterations = iterations
    run_id = run_pipeline(cfg)
    typer.echo(f"Optimization finished. run_id={run_id}")


@app.command()
def ingest_artifact(
    path: str = typer.Argument(..., help="File to store"),
    kind: str = typer.Option("generic", help="Artifact kind"),
    run_id: Optional[str] = typer.Option(None, help="Run ID to attach"),
    config: str = typer.Option("configs/app.yaml", help="Config for storage settings"),
) -> None:
    # Demo: python -m hmopt.cli ingest-artifact outputs/framegraph.json --kind framegraph
    # Purpose: manually stash an artifact into the DB/artifact store.
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    engine = init_engine(cfg.storage.db_url, schema_path=Path("src/hmopt/storage/db/schema.sql"))
    store = ArtifactStore(cfg.storage.artifacts_root)
    with session_scope(engine) as session:
        art = store.store_file(Path(path), kind=kind, run_id=run_id, session=session)
        typer.echo(f"Stored artifact {art.artifact_id} at {art.path}")


@app.command()
def analyze(config: str = typer.Option("configs/app.yaml", help="Config YAML")) -> None:
    # Demo: python -m hmopt.cli analyze --config configs/app.yaml
    # Purpose: run a single-iteration baseline analysis (no extra iterations).
    cfg = _load_config(config)
    cfg.iterations = 1
    run_id = run_pipeline(cfg)
    typer.echo(f"Analysis baseline run_id={run_id}")


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run ID to summarize"),
    config: str = typer.Option("configs/app.yaml", help="Config YAML"),
) -> None:
    # Demo: python -m hmopt.cli report <run_id> --config configs/app.yaml
    # Purpose: fetch status/metrics/hotspots for a finished run.
    cfg = _load_config(config)
    engine = init_engine(cfg.storage.db_url, schema_path=Path("src/hmopt/storage/db/schema.sql"))
    with session_scope(engine) as session:
        run = session.query(models.Run).filter(models.Run.run_id == run_id).one_or_none()
        if not run:
            typer.echo("run not found")
            raise typer.Exit(code=1)
        metrics = session.query(models.Metric).filter(models.Metric.run_id == run_id).all()
        hotspots = session.query(models.Hotspot).filter(models.Hotspot.run_id == run_id).all()
        summary = {
            "run_id": run_id,
            "status": run.status,
            "metrics": {m.metric_name: m.value for m in metrics},
            "hotspots": [h.symbol for h in hotspots],
        }
        typer.echo(json.dumps(summary, indent=2))

@app.command()
def analyze_artifacts(
    artifact: list[str] = typer.Option(
        [],
        "--artifact",
        help="Artifact spec kind:path (e.g., framegraph:outputs/framegraph.json). Repeatable.",
    ),
    config: str = typer.Option("configs/app.yaml", help="Config YAML"),
    repo_path: Optional[str] = typer.Option(None, help="Override repo path for this run"),
    with_patch: bool = typer.Option(True, help="Run Conductor+Coder to suggest patches"),
    with_verify: bool = typer.Option(False, help="Run build/test verification after patch"),
    with_profile: bool = typer.Option(False, help="Re-profile candidate after patch"),
) -> None:
    # Demo: python -m hmopt.cli analyze-artifacts \
    #          --artifact framegraph:outputs/framegraph.json \
    #          --artifact hitrace:outputs/hitrace.json \
    #          --artifact hiperf:outputs/hiperf.json \
    #          --repo-path /path/to/hm-verif-kernel
    # Purpose: ingest existing traces (no live profiling), run analysis -> hotspots/report.
    logging.basicConfig(level=logging.INFO)
    cfg = _load_config(config)
    if repo_path:
        cfg.project.repo_path = repo_path
    artifacts = []
    for spec in artifact:
        if ":" not in spec:
            typer.echo(f"Invalid artifact spec: {spec}")
            raise typer.Exit(code=1)
        kind, path = spec.split(":", 1)
        artifacts.append({"kind": kind, "path": path})
    run_id = run_artifact_analysis(
        cfg,
        artifacts,
        run_conductor=with_patch,
        run_coder=with_patch,
        run_verify=with_verify,
        run_profile=with_profile,
    )
    typer.echo(f"Artifact analysis complete. run_id={run_id}")


if __name__ == "__main__":
    app()
