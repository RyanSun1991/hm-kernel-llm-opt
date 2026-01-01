"""CLI entrypoint using Typer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from hmopt.core.config import AppConfig
from hmopt.orchestration import run_pipeline
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
    cfg = _load_config(config)
    run_id = run_pipeline(cfg)
    typer.echo(f"Run completed: {run_id}")


@app.command()
def optimize(
    config: str = typer.Option("configs/app.yaml", help="Path to config YAML"),
    iterations: int = typer.Option(2, help="Max iterations"),
) -> None:
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
    cfg = _load_config(config)
    engine = init_engine(cfg.storage.db_url, schema_path=Path("src/hmopt/storage/db/schema.sql"))
    store = ArtifactStore(cfg.storage.artifacts_root)
    with session_scope(engine) as session:
        art = store.store_file(Path(path), kind=kind, run_id=run_id, session=session)
        typer.echo(f"Stored artifact {art.artifact_id} at {art.path}")


@app.command()
def analyze(config: str = typer.Option("configs/app.yaml", help="Config YAML")) -> None:
    cfg = _load_config(config)
    cfg.iterations = 1
    run_id = run_pipeline(cfg)
    typer.echo(f"Analysis baseline run_id={run_id}")


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run ID to summarize"),
    config: str = typer.Option("configs/app.yaml", help="Config YAML"),
) -> None:
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


if __name__ == "__main__":
    app()
