"""Database helper functions."""

from __future__ import annotations

import datetime as dt
from typing import Optional

from hmopt.storage.db import models


def get_run(session, run_id: str) -> models.Run | None:
    return session.query(models.Run).filter(models.Run.run_id == run_id).one_or_none()


def update_run_status(session, run_id: str, status: str) -> None:
    run = get_run(session, run_id)
    if not run:
        return
    run.status = status
    if status in {"succeeded", "failed", "stopped"}:
        run.finished_at = dt.datetime.utcnow()
    session.flush()


def store_agent_message(
    session,
    *,
    run_id: str,
    iteration: int,
    agent_name: str,
    model_id: str,
    prompt_artifact_id: str | None,
    output_artifact_id: str | None,
    summary_artifact_id: str | None = None,
) -> None:
    session.add(
        models.AgentMessage(
            run_id=run_id,
            iteration=iteration,
            agent_name=agent_name,
            model_id=model_id,
            prompt_artifact_id=prompt_artifact_id,
            output_artifact_id=output_artifact_id,
            summary_artifact_id=summary_artifact_id,
        )
    )
    session.flush()
