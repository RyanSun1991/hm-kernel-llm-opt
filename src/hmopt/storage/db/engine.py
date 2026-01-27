"""SQLite/Postgres engine helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def create_db_engine(db_url: str, echo: bool = False) -> Engine:
    """Create an engine and make sure parent directories exist for sqlite."""
    if db_url.startswith("sqlite:///"):
        db_file = Path(db_url.replace("sqlite:///", "", 1))
        db_file.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(db_url, echo=echo, future=True)


def bootstrap(engine: Engine, schema_path: Path | None = None) -> None:
    """Create tables and optionally run raw schema SQL."""
    Base.metadata.create_all(engine)
    _ensure_hotspots_columns(engine)
    if schema_path and schema_path.exists():
        sql = schema_path.read_text(encoding="utf-8").strip()
        if sql:
            with engine.begin() as conn:
                raw = conn.connection
                raw.executescript(sql)  # type: ignore[attr-defined]


def _ensure_hotspots_columns(engine: Engine) -> None:
    inspector = inspect(engine)
    if "hotspots" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("hotspots")}
    if "call_stacks_json" in existing:
        return
    dialect = engine.dialect.name
    if dialect == "postgresql":
        ddl = "ALTER TABLE hotspots ADD COLUMN call_stacks_json JSON"
    else:
        ddl = "ALTER TABLE hotspots ADD COLUMN call_stacks_json TEXT"
    with engine.begin() as conn:
        conn.execute(text(ddl))


def init_engine(db_url: str, schema_path: Path | None = None, echo: bool = False) -> Engine:
    engine = create_db_engine(db_url, echo=echo)
    bootstrap(engine, schema_path)
    return engine


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Provide a transactional scope around a series of operations."""
    SessionLocal = sessionmaker(bind=engine, future=True)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
