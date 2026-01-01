"""FastAPI application."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Annotated, List

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from hmopt.core.config import AppConfig
from hmopt.evaluation.reports import render_report
from hmopt.orchestration import run_pipeline
from hmopt.storage.db import models
from hmopt.storage.db.engine import init_engine, session_scope

CONFIG_PATH = os.getenv("HMOPT_API_CONFIG", "configs/app.yaml")
APP_CONFIG = AppConfig.from_yaml(CONFIG_PATH)
ENGINE = init_engine(APP_CONFIG.storage.db_url, schema_path=Path("src/hmopt/storage/db/schema.sql"))

app = FastAPI(title="HMOPT API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_session() -> Session:
    with session_scope(ENGINE) as session:
        yield session


def _start_pipeline_async(config: AppConfig) -> str:
    def runner() -> None:
        run_pipeline(config)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return config.project.name


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/runs")
def create_run(background_tasks: BackgroundTasks) -> dict:
    cfg = AppConfig.from_yaml(CONFIG_PATH)
    background_tasks.add_task(run_pipeline, cfg)
    return {"status": "submitted", "project": cfg.project.name}


@app.get("/runs")
def list_runs(session: Annotated[Session, Depends(get_session)]) -> List[dict]:
    runs = session.query(models.Run).all()
    return [
        {
            "run_id": r.run_id,
            "status": r.status,
            "created_at": r.created_at,
            "finished_at": r.finished_at,
        }
        for r in runs
    ]


@app.get("/runs/{run_id}")
def get_run(run_id: str, session: Annotated[Session, Depends(get_session)]) -> dict:
    run = session.query(models.Run).filter(models.Run.run_id == run_id).one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "run_id": run.run_id,
        "status": run.status,
        "repo_rev": run.repo_rev,
        "workload_id": run.workload_id,
        "created_at": run.created_at,
        "finished_at": run.finished_at,
    }


@app.get("/runs/{run_id}/metrics")
def get_metrics(run_id: str, session: Annotated[Session, Depends(get_session)]) -> dict:
    metrics = session.query(models.Metric).filter(models.Metric.run_id == run_id).all()
    return {m.metric_name: m.value for m in metrics}


@app.get("/runs/{run_id}/artifacts")
def get_artifacts(run_id: str, session: Annotated[Session, Depends(get_session)]) -> List[dict]:
    artifacts = session.query(models.Artifact).filter(models.Artifact.run_id == run_id).all()
    return [{"artifact_id": a.artifact_id, "kind": a.kind, "path": a.path, "bytes": a.bytes} for a in artifacts]


@app.post("/runs/{run_id}/optimize")
def optimize_existing(run_id: str, background_tasks: BackgroundTasks) -> dict:
    cfg = AppConfig.from_yaml(CONFIG_PATH)
    cfg.iterations = max(cfg.iterations, 1)
    background_tasks.add_task(run_pipeline, cfg)
    return {"status": "submitted", "baseline_run_id": run_id}


@app.get("/runs/{run_id}/report")
def report(run_id: str, session: Annotated[Session, Depends(get_session)]) -> dict:
    run = session.query(models.Run).filter(models.Run.run_id == run_id).one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, "report": render_report(session, run_id)}
