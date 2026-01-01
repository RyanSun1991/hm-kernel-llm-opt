"""Runtime context object used across the pipeline."""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session, sessionmaker

from hmopt.storage.artifact_store import ArtifactStore
from hmopt.storage.db import models
from hmopt.storage.db.engine import init_engine
from hmopt.storage.vector.embeddings import EmbeddingClient
from hmopt.storage.vector.store import LocalVectorStore

from .config import AppConfig
from .errors import HMOptError


@dataclass
class RunContext:
    config: AppConfig
    engine: object
    session: Session
    artifact_store: ArtifactStore
    vector_store: LocalVectorStore
    embedding_client: EmbeddingClient
    run_id: str
    run_dir: Path

    def close(self) -> None:
        self.session.close()


def build_context(config: AppConfig, *, echo_sql: bool = False) -> RunContext:
    """Initialize DB/FS/embedding services and allocate a run directory."""
    schema_path = Path(__file__).resolve().parent.parent / "storage" / "db" / "schema.sql"
    engine = init_engine(config.storage.db_url, schema_path=schema_path, echo=echo_sql)
    SessionLocal = sessionmaker(bind=engine, future=True)
    session = SessionLocal()

    artifact_store = ArtifactStore(Path(config.storage.artifacts_root))
    embedding_client = EmbeddingClient(
        api_base=config.llm.base_url, api_key=config.llm.api_key, model=config.llm.embedding_model
    )
    vector_store = LocalVectorStore(session)

    run_id = str(uuid.uuid4())
    run_dir = Path(config.storage.run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    return RunContext(
        config=config,
        engine=engine,
        session=session,
        artifact_store=artifact_store,
        vector_store=vector_store,
        embedding_client=embedding_client,
        run_id=run_id,
        run_dir=run_dir,
    )


def register_run(ctx: RunContext, *, workload_id: Optional[str] = None) -> models.Run:
    """Insert a run row and return it."""
    run = models.Run(
        run_id=ctx.run_id,
        workload_id=workload_id or ctx.config.project.workload,
        status="created",
        created_at=dt.datetime.utcnow(),
    )
    ctx.session.add(run)
    ctx.session.commit()
    return run
