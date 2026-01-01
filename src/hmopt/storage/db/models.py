"""SQLAlchemy ORM models."""

from __future__ import annotations

import datetime as dt
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    run_id = Column(String, primary_key=True)
    parent_run_id = Column(String, nullable=True)
    repo_uri = Column(String, nullable=True)
    repo_rev = Column(String, nullable=True)
    repo_dirty = Column(Boolean, default=False)
    workload_id = Column(String, nullable=True)
    device_id = Column(String, nullable=True)
    toolchain_id = Column(String, nullable=True)
    status = Column(String, default="created")
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)


class Artifact(Base):
    __tablename__ = "artifacts"

    artifact_id = Column(String, primary_key=True)
    run_id = Column(String, nullable=True, index=True)
    kind = Column(String)
    sha256 = Column(String)
    path = Column(String)
    bytes = Column(Integer)
    mime = Column(String, nullable=True)
    metadata_json = Column(JSON, default={})


class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)
    scope = Column(String)
    metric_name = Column(String)
    value = Column(Float)
    unit = Column(String, nullable=True)
    tags_json = Column(JSON, default={})


class Hotspot(Base):
    __tablename__ = "hotspots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)
    symbol = Column(String)
    file_path = Column(String, nullable=True)
    line_start = Column(Integer, nullable=True)
    line_end = Column(Integer, nullable=True)
    score = Column(Float)
    evidence_artifact_ids = Column(JSON, default=[])


class Graph(Base):
    __tablename__ = "graphs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)
    kind = Column(String)  # PSG / PCG
    format = Column(String, default="json")
    payload_artifact_id = Column(String)
    metadata_json = Column(JSON, default={})


class Patch(Base):
    __tablename__ = "patches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)
    iteration = Column(Integer, default=0)
    diff_artifact_id = Column(String)
    apply_status = Column(String, default="pending")  # applied/rejected/failed
    files_changed_json = Column(JSON, default=[])


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)
    baseline_run_id = Column(String, nullable=True)
    delta_metrics_json = Column(JSON, default={})
    correctness_passed = Column(Boolean, default=True)
    perf_improved = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)


class AgentMessage(Base):
    __tablename__ = "agent_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)
    iteration = Column(Integer, default=0)
    agent_name = Column(String)
    model_id = Column(String)
    prompt_artifact_id = Column(String, nullable=True)
    output_artifact_id = Column(String, nullable=True)
    summary_artifact_id = Column(String, nullable=True)


class VectorEmbedding(Base):
    __tablename__ = "vector_embeddings"

    id = Column(String, primary_key=True)
    kind = Column(String)
    ref_id = Column(String)
    run_id = Column(String, nullable=True)
    embedding_json = Column(JSON)
    metadata_json = Column(JSON, default={})
    created_at = Column(DateTime, default=dt.datetime.utcnow)
