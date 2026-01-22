"""LlamaIndex ingestion, indexing, and query routing."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from llama_index.core import (
    PropertyGraphIndex,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from hmopt.indexing.openai_like import OpenAILike,OpenAIEmbeddingLike

# from hmopt.indexing.llama_embeddings import OpenAICompatEmbedding
from hmopt.core.config import AppConfig
from hmopt.indexing.clangd_client import ClangdConfig as LspClangdConfig
from hmopt.indexing.clangd_indexer import CodeIndex, CodeChunk, index_kernel_code
from hmopt.indexing.runtime_ingestion import build_runtime_nodes
from hmopt.storage.db.engine import init_engine, session_scope


from llama_index.graph_stores.neo4j.neo4j_property_graph import Neo4jPropertyGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

logger = logging.getLogger(__name__)

@dataclass
class IndexPaths:
    base_dir: Path
    code_dir: Path
    runtime_dir: Path

@dataclass
class Neo4jIndexConfig:
    index_name: str
    node_label: str

def _index_paths(config: AppConfig) -> IndexPaths:
    base = Path(config.indexing.persist_dir)
    return IndexPaths(base_dir=base, code_dir=base / "code", runtime_dir=base / "runtime")


def _build_llama_models(config: AppConfig) -> tuple[OpenAILike, OpenAIEmbeddingLike]:
    if not config.llm.api_key:
        raise RuntimeError("LLM API key is required for LlamaIndex indexing")
    if config.llm.api_key:
        os.environ.setdefault("OPENAI_API_KEY", config.llm.api_key)
    if config.llm.base_url:
        os.environ.setdefault("OPENAI_API_BASE", config.llm.base_url)
    llm = OpenAILike(model=config.llm.model,
                    api_key=config.llm.api_key,
                    api_base=config.llm.base_url,
                    timeout = 120,
                    max_tokens=2048,
                    temperature=0)
    embed = OpenAIEmbeddingLike(
        model=config.llm.embedding_model,
        api_base=config.llm.base_url,
        api_key=config.llm.api_key,
    )
    return llm, embed

def _embedding_metadata_path(persist_dir: Path) -> Path:
    return persist_dir / "embedding_meta.json"


def _persist_embedding_metadata(
    persist_dir: Path,
    *,
    model: str,
    dimension: int,
    index_name: str,
    node_label: str,
    index_id: Optional[str] = None,
) -> None:
    metadata_path = _embedding_metadata_path(persist_dir)
    metadata = {
        "model": model,
        "dimension": dimension,
        "index_name": index_name,
        "node_label": node_label,
    }
    if index_id:
        metadata["index_id"] = index_id
    metadata_path.write_text(
        json.dumps(metadata, indent=2)
        + "\n"
    )


def _load_embedding_metadata(persist_dir: Path) -> Optional[dict]:
    metadata_path = _embedding_metadata_path(persist_dir)
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        logger.warning("Failed to parse embedding metadata at %s", metadata_path)
        return None


def _infer_embedding_dimension(embed: OpenAIEmbeddingLike) -> int:
    return len(embed.get_text_embedding("dimension probe"))


def _embedding_dimension_for_query(
    persist_dir: Path, embed: OpenAIEmbeddingLike
) -> int:
    metadata = _load_embedding_metadata(persist_dir)
    if metadata and metadata.get("dimension"):
        return int(metadata["dimension"])
    return _infer_embedding_dimension(embed)

def _ensure_embedding_compat(
    persist_dir: Path,
    *,
    embed_model: str,
    embed_dim: int,
    index_name: str,
    node_label: str,
    index_id: Optional[str] = None,
) -> None:
    metadata = _load_embedding_metadata(persist_dir)
    if not metadata:
        logger.warning("Embedding metadata missing for index at %s", persist_dir)
        return
    stored_dim = metadata.get("dimension")
    stored_model = metadata.get("model")
    stored_index = metadata.get("index_name")
    stored_label = metadata.get("node_label")
    stored_index_id = metadata.get("index_id")
    if stored_dim != embed_dim:
        raise RuntimeError(
            "Embedding dimension mismatch for index at "
            f"{persist_dir}. Stored model={stored_model!r} dim={stored_dim}, "
            f"current model={embed_model!r} dim={embed_dim}. "
            "Rebuild the index or drop the Neo4j vector index before querying."
        )
    if stored_index and stored_index != index_name:
        raise RuntimeError(
            "Neo4j index name mismatch for index at "
            f"{persist_dir}. Stored index={stored_index!r}, current index={index_name!r}. "
            "Rebuild the index to match the current configuration."
        )
    if stored_label and stored_label != node_label:
        raise RuntimeError(
            "Neo4j node label mismatch for index at "
            f"{persist_dir}. Stored label={stored_label!r}, current label={node_label!r}. "
            "Rebuild the index to match the current configuration."
        )
    if stored_index_id and index_id and stored_index_id != index_id:
        raise RuntimeError(
            "Index ID mismatch for index at "
            f"{persist_dir}. Stored index_id={stored_index_id!r}, current index_id={index_id!r}. "
            "Rebuild the index to match the current configuration."
        )



def _storage_context(
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
        # if not for_load:
        #     return StorageContext.from_defaults(property_graph_store=property_graph_store, vector_store=vector_store)
        return StorageContext.from_defaults(
            persist_dir=str(persist_dir),
            property_graph_store=property_graph_store,
            vector_store=vector_store,
        )
    if config.indexing.neo4j.enabled:
        logger.warning("Neo4j stores not available; falling back to local storage")
    # if not for_load:
    #     return StorageContext.from_defaults(property_graph_store=property_graph_store, vector_store=vector_store)
    return StorageContext.from_defaults(persist_dir=str(persist_dir))


def _neo4j_index_config(kind: str) -> Neo4jIndexConfig:
    if kind == "runtime":
        return Neo4jIndexConfig(index_name="runtime_vector", node_label="RuntimeChunk")
    return Neo4jIndexConfig(index_name="code_vector", node_label="CodeChunk")


def _reset_neo4j_vector_index(
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


def _code_header(chunk: CodeChunk) -> str:
    detail = f" {chunk.detail}" if chunk.detail else ""
    return (
        f"// [SYMBOL] {chunk.kind} {chunk.symbol_qualname} "
        f"({chunk.path}:{chunk.start_line}-{chunk.end_line}){detail}"
    )


def _index_to_nodes(index: CodeIndex) -> list[TextNode]:
    nodes: list[TextNode] = []
    for chunk in index.chunks:
        nodes.append(
            TextNode(
                id_=chunk.symbol_id,
                text=f"{_code_header(chunk)}\n{chunk.text}",
                metadata={
                    "type": "code",
                    "symbol_name": chunk.symbol_name,
                    "symbol_qualname": chunk.symbol_qualname,
                    "symbol_id": chunk.symbol_id,
                    "symbol_kind": chunk.kind,
                    "path": str(chunk.path),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "parser": chunk.parser,
                    "container": chunk.container,
                    "detail": chunk.detail,
                },
            )
        )
    for summary in index.file_summaries:
        nodes.append(
            TextNode(
                text=summary.text,
                metadata={"type": "file_summary", "path": str(summary.path)},
            )
        )
    for summary in index.relation_summaries:
        nodes.append(
            TextNode(
                text=summary.text,
                metadata={
                    "type": "symbol_relations",
                    "symbol_id": summary.symbol_id,
                    "symbol_name": summary.symbol_name,
                    "symbol_kind": summary.symbol_kind,
                    "path": summary.path,
                },
            )
        )
    return nodes

def _build_graph_entities(index: CodeIndex) -> tuple[list[EntityNode], list[Relation]]:
    nodes: dict[str, EntityNode] = {}

    def add_node(
        *,
        node_id: str,
        label: str,
        symbol_name: str,
        symbol_kind: str,
        path: Optional[str] = None,
        symbol_qualname: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> None:
        if node_id in nodes:
            return
        properties = {
            "symbol_name": symbol_name,
            "symbol_kind": symbol_kind,
        }
        if symbol_qualname:
            properties["symbol_qualname"] = symbol_qualname
        if path:
            properties["path"] = path
        if start_line is not None:
            properties["start_line"] = start_line
        if end_line is not None:
            properties["end_line"] = end_line
        nodes[node_id] = EntityNode(name=node_id, label=label, properties=properties)

    for chunk in index.chunks:
        add_node(
            node_id=chunk.symbol_id,
            label="symbol",
            symbol_name=chunk.symbol_name,
            symbol_kind=chunk.kind,
            symbol_qualname=chunk.symbol_qualname,
            path=str(chunk.path),
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        )

    relations: list[Relation] = []
    for rel in index.relations:
        add_node(
            node_id=rel.src_id,
            label="symbol",
            symbol_name=rel.src_name,
            symbol_kind=rel.src_kind,
            path=rel.src_path,
        )
        label = "external" if rel.dst_kind == "external" else "symbol"
        add_node(
            node_id=rel.dst_id,
            label=label,
            symbol_name=rel.dst_name,
            symbol_kind=rel.dst_kind,
            path=rel.dst_path,
        )
        relations.append(
            Relation(
                label=rel.kind,
                source_id=rel.src_id,
                target_id=rel.dst_id,
                properties={
                    "src_name": rel.src_name,
                    "dst_name": rel.dst_name,
                    "src_kind": rel.src_kind,
                    "dst_kind": rel.dst_kind,
                    "src_path": rel.src_path,
                    "dst_path": rel.dst_path,
                },
            )
        )

    return list(nodes.values()), relations


def _upsert_clangd_graph(storage: StorageContext, index: CodeIndex, nodes: list[TextNode]) -> None:
    property_graph_store = storage.property_graph_store
    if not property_graph_store:
        logger.warning("Property graph store unavailable; skipping clangd relation upsert")
        return
    property_graph_store.upsert_llama_nodes(nodes)
    entity_nodes, relations = _build_graph_entities(index)
    if entity_nodes:
        property_graph_store.upsert_nodes(entity_nodes)
    if relations:
        property_graph_store.upsert_relations(relations)


def build_kernel_index(config: AppConfig, repo_path: Optional[str] = None) -> IndexPaths:
    paths = _index_paths(config)
    paths.code_dir.mkdir(parents=True, exist_ok=True)
    llm, embed = _build_llama_models(config)
    embed_dim = _infer_embedding_dimension(embed)
    neo4j_cfg = _neo4j_index_config("code")
    _reset_neo4j_vector_index(
        config,
        embedding_dimension=embed_dim,
        index_name=neo4j_cfg.index_name,
        node_label=neo4j_cfg.node_label,
    )

    clangd_cfg = LspClangdConfig(
        binary=config.indexing.clangd.binary,
        compile_commands_dir=config.indexing.clangd.compile_commands_dir,
        extra_args=config.indexing.clangd.extra_args,
        timeout_sec=config.indexing.clangd.timeout_sec,
        symbol_kinds=config.indexing.clangd.symbol_kinds,
        call_hierarchy_enabled=config.indexing.clangd.call_hierarchy_enabled,
        call_hierarchy_max_functions=config.indexing.clangd.call_hierarchy_max_functions,
        call_hierarchy_max_calls=config.indexing.clangd.call_hierarchy_max_calls,
        call_hierarchy_max_depth=config.indexing.clangd.call_hierarchy_max_depth,
        usage_scan_enabled=config.indexing.clangd.usage_scan_enabled,
        usage_scan_max_names=config.indexing.clangd.usage_scan_max_names,
        relation_max_per_symbol=config.indexing.clangd.relation_max_per_symbol,
        file_summary_enabled=config.indexing.clangd.file_summary_enabled,
        relation_summary_enabled=config.indexing.clangd.relation_summary_enabled,
        relation_summary_max_items=config.indexing.clangd.relation_summary_max_items,
    )
    code_index = index_kernel_code(
        Path(repo_path or config.project.repo_path),
        clangd_config=clangd_cfg if config.indexing.clangd.enabled else None,
        max_files=config.indexing.clangd.max_files,
    )
    nodes = _index_to_nodes(code_index)

    if config.indexing.llm_enrich:
        limit = min(config.indexing.llm_enrich_limit, len(nodes))
        for node in nodes[:limit]:
            if node.metadata.get("type") != "code":
                continue
            summary = llm.complete(f"Summarize the following kernel code:\n\n{node.text}")
            node.text = f"// [LLM SUMMARY]\n{summary.text}\n\n{node.text}"

    storage = _storage_context(
        config,
        paths.code_dir,
        embedding_dimension=embed_dim,
        index_name=neo4j_cfg.index_name,
        node_label=neo4j_cfg.node_label,
    )
    vector_index = VectorStoreIndex(nodes, storage_context=storage, embed_model=embed)
    if config.indexing.neo4j.enabled:
        _upsert_clangd_graph(storage, code_index, nodes)
    storage.persist(persist_dir=str(paths.code_dir))
    _persist_embedding_metadata(
        paths.code_dir,
        model=config.llm.embedding_model,
        dimension=embed_dim,
        index_name=neo4j_cfg.index_name,
        node_label=neo4j_cfg.node_label,
        index_id=vector_index.index_id,
    )
    logger.info("Kernel code index built: nodes=%d", len(nodes))
    return paths


def build_runtime_index(config: AppConfig, run_id: str) -> IndexPaths:
    paths = _index_paths(config)
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    llm, embed = _build_llama_models(config)
    embed_dim = _infer_embedding_dimension(embed)
    neo4j_cfg = _neo4j_index_config("runtime")
    _reset_neo4j_vector_index(
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
    storage = _storage_context(
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
    _persist_embedding_metadata(
        paths.runtime_dir,
        model=config.llm.embedding_model,
        dimension=embed_dim,
        index_name=neo4j_cfg.index_name,
        node_label=neo4j_cfg.node_label,
        index_id=index.index_id,
    )
    logger.info("Runtime index built: nodes=%d run_id=%s", len(nodes), run_id)
    return paths


def _load_index(
    config: AppConfig,
    persist_dir: Path,
    *,
    embedding_dimension: int,
    index_name: str,
    node_label: str,
    embed_model: OpenAIEmbedding,
) -> VectorStoreIndex:
    storage = _storage_context(
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
    metadata = _load_embedding_metadata(persist_dir) or {}
    index_id = metadata.get("index_id")
    if index_id:
        return load_index_from_storage(storage, index_id=index_id, embed_model=embed_model)
    index_structs = storage.index_store.index_structs()
    vector_structs = []
    for struct in index_structs:
        struct_type = struct.get_type()
        struct_type_value = struct_type.value if hasattr(struct_type, "value") else str(struct_type)
        if struct_type_value in ("vector_store", "IndexStructType.VECTOR_STORE"):
            vector_structs.append(struct)
    if len(vector_structs) == 1:
        return load_index_from_storage(storage, index_id=vector_structs[0].index_id, embed_model=embed_model)
    if not index_structs:
        raise RuntimeError(f"No indexes found in storage at {persist_dir}.")
    index_ids = ", ".join(struct.index_id for struct in index_structs)
    raise RuntimeError(
        "Multiple indexes found in storage and no index_id metadata is available. "
        f"Persist dir={persist_dir} index_ids=[{index_ids}]. "
        "Rebuild the index to regenerate metadata or specify index_id explicitly."
    )


def _docstore_has_nodes(storage: StorageContext) -> bool:
    docstore = storage.docstore
    docs = getattr(docstore, "docs", None)
    if isinstance(docs, dict):
        return bool(docs)
    try:
        return bool(docstore.get_all_document_hashes())
    except Exception:
        return False


def route_query(config: AppConfig, query: str, mode: str = "auto") -> str:
    paths = _index_paths(config)
    llm, embed = _build_llama_models(config)
    code_index_cfg = _neo4j_index_config("code")
    runtime_index_cfg = _neo4j_index_config("runtime")
    code_embed_dim = _embedding_dimension_for_query(paths.code_dir, embed)
    runtime_embed_dim = _embedding_dimension_for_query(paths.runtime_dir, embed)

    def _format_runtime_sources(source_nodes: list) -> str:
        hotspots = []
        metrics = []
        evidence = []
        for source in source_nodes:
            metadata = getattr(source.node, "metadata", {}) if hasattr(source, "node") else {}
            node_type = metadata.get("type")
            text = getattr(source.node, "text", "")
            if node_type == "runtime_hotspot":
                hotspots.append(
                    {
                        "symbol": metadata.get("symbol"),
                        "score": metadata.get("score"),
                        "file_path": metadata.get("file_path"),
                        "line_start": metadata.get("line_start"),
                        "line_end": metadata.get("line_end"),
                        "text": text,
                    }
                )
            elif node_type == "runtime_metric":
                metrics.append(
                    {
                        "metric_name": metadata.get("metric_name"),
                        "value": metadata.get("value"),
                        "unit": metadata.get("unit"),
                        "text": text,
                    }
                )
            elif node_type == "evidence_pack":
                evidence.append(text)
        lines = []
        if hotspots:
            lines.append("Top runtime hotspots:")
            for item in hotspots:
                lines.append(
                    f"- {item.get('symbol')} score={item.get('score')} "
                    f"path={item.get('file_path')} lines={item.get('line_start')}-{item.get('line_end')}"
                )
        if metrics:
            lines.append("Runtime metrics:")
            for item in metrics:
                lines.append(
                    f"- {item.get('metric_name')} value={item.get('value')} unit={item.get('unit')}"
                )
        if evidence:
            lines.append("Runtime evidence excerpt:")
            lines.extend(evidence[:1])
        return "\n".join(lines).strip()

    def _engine(dir_path: Path):
        index_cfg = runtime_index_cfg if dir_path == paths.runtime_dir else code_index_cfg
        embed_dim = runtime_embed_dim if dir_path == paths.runtime_dir else code_embed_dim
        _ensure_embedding_compat(
            dir_path,
            embed_model=config.llm.embedding_model,
            embed_dim=embed_dim,
            index_name=index_cfg.index_name,
            node_label=index_cfg.node_label,
        )
        index = _load_index(
            config,
            dir_path,
            embedding_dimension=embed_dim,
            index_name=index_cfg.index_name,
            node_label=index_cfg.node_label,
            embed_model=embed,
        )
        top_k = (
            config.indexing.query_runtime_top_k
            if dir_path == paths.runtime_dir
            else config.indexing.query_code_top_k
        )
        return index.as_query_engine(llm=llm, similarity_top_k=top_k, response_mode="compact")

    def _graph_engine(dir_path: Path):
        index_cfg = code_index_cfg
        embed_dim = code_embed_dim
        _ensure_embedding_compat(
            dir_path,
            embed_model=config.llm.embedding_model,
            embed_dim=embed_dim,
            index_name=index_cfg.index_name,
            node_label=index_cfg.node_label,
        )
        storage = _storage_context(
            config,
            dir_path,
            embedding_dimension=embed_dim,
            index_name=index_cfg.index_name,
            node_label=index_cfg.node_label,
        )
        if not storage.property_graph_store:
            return _engine(dir_path)
        if config.indexing.neo4j.enabled and not _docstore_has_nodes(storage):
            logger.warning(
                "Docstore is empty for %s; falling back to Neo4j vector query engine.",
                dir_path,
            )
            return _engine(dir_path)
        graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=storage.property_graph_store,
            vector_store=storage.vector_store,
            llm=llm,
            embed_model=embed,
            embed_kg_nodes=False,
            storage_context=storage,
        )
        return graph_index.as_query_engine(
            llm=llm,
            similarity_top_k=config.indexing.query_graph_top_k,
            response_mode="compact",
        )

    def _runtime_to_code_query() -> str:
        runtime_response = _engine(paths.runtime_dir).query(query)
        combined_query = (
            "Use the runtime findings below to answer the question, and link to relevant code.\n\n"
            f"Runtime findings:\n{runtime_response}\n\n"
            f"Question: {query}"
        )
        return str(_graph_engine(paths.code_dir).query(combined_query))

    if mode == "code":
        return str(_graph_engine(paths.code_dir).query(query))
    if mode == "runtime":
        runtime_response = _engine(paths.runtime_dir).query(query)
        response_text = str(runtime_response).strip()
        if response_text:
            return response_text
        source_nodes = getattr(runtime_response, "source_nodes", []) or []
        fallback = _format_runtime_sources(source_nodes)
        return fallback or "No runtime results available."
    if mode == "graph":
        return str(_graph_engine(paths.code_dir).query(query))
    if mode == "runtime_code":
        return _runtime_to_code_query()

    keywords_runtime = ["perf", "trace", "runtime", "framegraph", "instruction", "hotspot"]
    if any(k in query.lower() for k in keywords_runtime):
        return _runtime_to_code_query()
    return str(_graph_engine(paths.code_dir).query(query))
