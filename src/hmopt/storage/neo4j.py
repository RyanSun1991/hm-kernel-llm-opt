"""Neo4j-specific helpers for indexing pipeline."""

from __future__ import annotations

import logging
import re
from typing import Optional

from hmopt.core.config import AppConfig
from hmopt.indexing.types import Neo4jIndexConfig

logger = logging.getLogger(__name__)

try:
    from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
except Exception:  # pragma: no cover - optional
    Neo4jVectorStore = None


def safe_neo4j_id(value: str, *, max_len: int = 80) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    safe = safe.strip("_")
    if len(safe) > max_len:
        return safe[:max_len]
    return safe or "index"


def neo4j_index_config(kind: str, project_slug: str, version: Optional[str]) -> Neo4jIndexConfig:
    suffix = safe_neo4j_id(version or "latest")
    index_name = safe_neo4j_id(f"{kind}_vector_{project_slug}_{suffix}")
    node_label = safe_neo4j_id(f"{kind.capitalize()}Chunk_{suffix}")
    return Neo4jIndexConfig(index_name=index_name, node_label=node_label)


def reset_neo4j_vector_index(
    config: AppConfig,
    *,
    embedding_dimension: int,
    index_name: str,
    node_label: str,
) -> None:
    if not (config.indexing.neo4j.enabled and Neo4jVectorStore):
        return
    vector_store = Neo4jVectorStore(
        username=config.indexing.neo4j.user,
        password=config.indexing.neo4j.password,
        url=config.indexing.neo4j.uri,
        database=config.indexing.neo4j.database,
        embedding_dimension=embedding_dimension,
        index_name=index_name,
        node_label=node_label,
    )
    try:
        vector_store.database_query(f'DROP INDEX `{index_name}` IF EXISTS')
    except Exception as exc:  # pragma: no cover - best effort cleanup
        logger.warning("Failed to drop Neo4j index %s: %s", index_name, exc)
    try:
        vector_store.database_query(f'MATCH (n:`{node_label}`) DETACH DELETE n')
    except Exception as exc:  # pragma: no cover - best effort cleanup
        logger.warning("Failed to delete Neo4j nodes for %s: %s", node_label, exc)
