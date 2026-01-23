"""Storage utilities for LlamaIndex indexes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import TextNode

from hmopt.core.config import AppConfig
from hmopt.llm.openai_like import OpenAIEmbeddingLike

logger = logging.getLogger(__name__)

try:
    from llama_index.graph_stores.neo4j.neo4j_property_graph import Neo4jPropertyGraphStore
except Exception:  # pragma: no cover - optional
    Neo4jPropertyGraphStore = None

try:
    from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
except Exception:  # pragma: no cover - optional
    Neo4jVectorStore = None


def storage_context(
    config: AppConfig,
    persist_dir: Path,
    *,
    embedding_dimension: Optional[int] = None,
    index_name: str = "vector",
    node_label: str = "Chunk",
) -> StorageContext:
    if config.indexing.neo4j.enabled and Neo4jPropertyGraphStore and Neo4jVectorStore:
        if embedding_dimension is None:
            raise RuntimeError("Embedding dimension is required for Neo4j vector store usage.")
        property_graph_store = Neo4jPropertyGraphStore(
            url=config.indexing.neo4j.uri,
            username=config.indexing.neo4j.user,
            password=config.indexing.neo4j.password,
            database=config.indexing.neo4j.database,
        )
        vector_store = Neo4jVectorStore(
            username=config.indexing.neo4j.user,
            password=config.indexing.neo4j.password,
            url=config.indexing.neo4j.uri,
            database=config.indexing.neo4j.database,
            embedding_dimension=embedding_dimension,
            index_name=index_name,
            node_label=node_label,
        )
        return StorageContext.from_defaults(
            persist_dir=str(persist_dir),
            property_graph_store=property_graph_store,
            vector_store=vector_store,
        )
    if config.indexing.neo4j.enabled:
        logger.warning("Neo4j stores not available; falling back to local storage")
    return StorageContext.from_defaults(persist_dir=str(persist_dir))


def docstore_has_nodes(storage: StorageContext) -> bool:
    docstore = storage.docstore
    docs = getattr(docstore, "docs", None)
    if isinstance(docs, dict):
        return bool(docs)
    try:
        return bool(docstore.get_all_document_hashes())
    except Exception:
        return False


def load_nodes_from_storage(storage: StorageContext) -> list[TextNode]:
    docstore = storage.docstore
    docs = getattr(docstore, "docs", None)
    if isinstance(docs, dict):
        return [doc for doc in docs.values() if isinstance(doc, TextNode)]
    nodes: list[TextNode] = []
    try:
        for doc_id in docstore.get_all_document_hashes():
            doc = docstore.get_document(doc_id)
            if isinstance(doc, TextNode):
                nodes.append(doc)
    except Exception:
        return []
    return nodes


def filter_nodes_by_paths(nodes: list[TextNode], excluded_paths: set[str]) -> list[TextNode]:
    if not excluded_paths:
        return nodes
    filtered: list[TextNode] = []
    for node in nodes:
        path = None
        metadata = getattr(node, "metadata", {}) or {}
        if metadata.get("path"):
            path = str(metadata.get("path"))
        if not path and getattr(node, "ref_doc_id", None):
            path = str(getattr(node, "ref_doc_id"))
        if path and path in excluded_paths:
            continue
        filtered.append(node)
    return filtered


def load_index(
    config: AppConfig,
    persist_dir: Path,
    *,
    embedding_dimension: int,
    index_name: str,
    node_label: str,
    embed_model: OpenAIEmbeddingLike,
) -> VectorStoreIndex:
    storage = storage_context(
        config,
        persist_dir,
        embedding_dimension=embedding_dimension,
        index_name=index_name,
        node_label=node_label,
    )
    if config.indexing.neo4j.enabled and storage.vector_store:
        try:
            return VectorStoreIndex.from_vector_store(
                storage.vector_store,
                storage_context=storage,
                embed_model=embed_model,
            )
        except Exception as exc:
            logger.warning("Failed to load Neo4j vector store directly: %s", exc)
    index_structs = storage.index_store.index_structs()
    if not index_structs:
        raise RuntimeError(f"No indexes found in storage at {persist_dir}.")
    metadata_path = persist_dir / "embedding_meta.json"
    index_id = None
    if metadata_path.exists():
        try:
            import json

            index_id = json.loads(metadata_path.read_text()).get("index_id")
        except Exception:
            index_id = None
    if index_id:
        return load_index_from_storage(storage, index_id=index_id, embed_model=embed_model)
    vector_structs = []
    for struct in index_structs:
        struct_type = struct.get_type()
        struct_type_value = struct_type.value if hasattr(struct_type, "value") else str(struct_type)
        if struct_type_value in ("vector_store", "IndexStructType.VECTOR_STORE"):
            vector_structs.append(struct)
    if len(vector_structs) == 1:
        return load_index_from_storage(
            storage, index_id=vector_structs[0].index_id, embed_model=embed_model
        )
    index_ids = ", ".join(struct.index_id for struct in index_structs)
    raise RuntimeError(
        "Multiple indexes found in storage and no index_id metadata is available. "
        f"Persist dir={persist_dir} index_ids=[{index_ids}]. "
        "Rebuild the index to regenerate metadata or specify index_id explicitly."
    )
