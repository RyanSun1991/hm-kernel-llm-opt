"""Runtime indexing pipeline."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode

from hmopt.core.config import AppConfig
from hmopt.llm.models import build_llama_models
from hmopt.storage.embedding_meta import infer_embedding_dimension, persist_embedding_metadata
from hmopt.storage.neo4j import neo4j_index_config, reset_neo4j_vector_index
from hmopt.indexing.paths import index_paths, slugify
from hmopt.indexing.runtime_ingestion import build_runtime_nodes
from hmopt.storage.llamaindex import storage_context
from hmopt.storage.db.engine import init_engine, session_scope

logger = logging.getLogger(__name__)


def build_runtime_index(
    config: AppConfig,
    run_id: str,
    *,
    index_version: Optional[str] = None,
    repo_name: Optional[str] = None,
) -> "IndexPaths":
    from hmopt.indexing.types import IndexPaths  # local import to avoid cycles

    paths: IndexPaths = index_paths(
        config, run_id=run_id, runtime_version=index_version, repo_name=repo_name
    )
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    llm, embed = build_llama_models(config)
    embed_dim = infer_embedding_dimension(embed)
    project_slug = slugify(repo_name or config.project.name)
    neo4j_cfg = neo4j_index_config("runtime", project_slug, paths.runtime_version)
    reset_neo4j_vector_index(
        config,
        embedding_dimension=embed_dim,
        index_name=neo4j_cfg.index_name,
        node_label=neo4j_cfg.node_label,
    )
    engine = init_engine(config.storage.db_url, schema_path=Path("src/hmopt/storage/db/schema.sql"))
    with session_scope(engine) as session:
        nodes = build_runtime_nodes(
            session,
            run_id,
            max_evidence_chars=config.indexing.runtime_evidence_max_chars,
        )
    storage = storage_context(
        config,
        paths.runtime_dir,
        embedding_dimension=embed_dim,
        index_name=neo4j_cfg.index_name,
        node_label=neo4j_cfg.node_label,
    )
    index = VectorStoreIndex(
        nodes,
        storage_context=storage,
        embed_model=embed,
        store_nodes_override=True,
    )
    storage.persist(persist_dir=str(paths.runtime_dir))
    persist_embedding_metadata(
        paths.runtime_dir,
        model=config.llm.embedding_model,
        dimension=embed_dim,
        index_name=neo4j_cfg.index_name,
        node_label=neo4j_cfg.node_label,
        version=paths.runtime_version,
        kind="runtime",
        index_id=index.index_id,
    )
    logger.info("Runtime index built: nodes=%d run_id=%s", len(nodes), run_id)
    return paths


def build_runtime_aggregate_index(
    config: AppConfig,
    run_ids: Sequence[str],
    *,
    group_name: str = "aggregate",
    index_version: Optional[str] = None,
    repo_name: Optional[str] = None,
) -> "IndexPaths":
    from hmopt.indexing.types import IndexPaths  # local import to avoid cycles

    if not run_ids:
        raise ValueError("run_ids is required for runtime aggregate index")
    repo_name_resolved = repo_name or f"aggregate-{group_name}"
    version = index_version or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    paths: IndexPaths = index_paths(
        config,
        repo_name=repo_name_resolved,
        runtime_version=version,
    )
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    llm, embed = build_llama_models(config)
    embed_dim = infer_embedding_dimension(embed)
    project_slug = slugify(repo_name_resolved)
    neo4j_cfg = neo4j_index_config("runtime", project_slug, paths.runtime_version)
    reset_neo4j_vector_index(
        config,
        embedding_dimension=embed_dim,
        index_name=neo4j_cfg.index_name,
        node_label=neo4j_cfg.node_label,
    )
    engine = init_engine(config.storage.db_url, schema_path=Path("src/hmopt/storage/db/schema.sql"))
    nodes: list[TextNode] = []
    with session_scope(engine) as session:
        for run_id in run_ids:
            run_nodes = build_runtime_nodes(
                session,
                run_id,
                max_evidence_chars=config.indexing.runtime_evidence_max_chars,
            )
            for node in run_nodes:
                meta = node.metadata or {}
                meta["aggregate_group"] = repo_name_resolved
                meta["aggregate_version"] = version
                meta["source_run_id"] = run_id
                node.metadata = meta
            nodes.extend(run_nodes)

    storage = storage_context(
        config,
        paths.runtime_dir,
        embedding_dimension=embed_dim,
        index_name=neo4j_cfg.index_name,
        node_label=neo4j_cfg.node_label,
    )
    index = VectorStoreIndex(
        nodes,
        storage_context=storage,
        embed_model=embed,
        store_nodes_override=True,
    )
    storage.persist(persist_dir=str(paths.runtime_dir))
    persist_embedding_metadata(
        paths.runtime_dir,
        model=config.llm.embedding_model,
        dimension=embed_dim,
        index_name=neo4j_cfg.index_name,
        node_label=neo4j_cfg.node_label,
        version=paths.runtime_version,
        kind="runtime_aggregate",
        index_id=index.index_id,
    )
    logger.info(
        "Runtime aggregate index built: nodes=%d runs=%d version=%s",
        len(nodes),
        len(run_ids),
        paths.runtime_version,
    )
    return paths
