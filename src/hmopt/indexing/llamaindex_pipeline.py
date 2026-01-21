"""LlamaIndex ingestion, indexing, and query routing."""

from __future__ import annotations

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

from hmopt.core.config import AppConfig
from hmopt.indexing.clangd_client import ClangdConfig as LspClangdConfig
from hmopt.indexing.clangd_indexer import CodeIndex, CodeChunk, index_kernel_code
from hmopt.indexing.runtime_ingestion import build_runtime_nodes
from hmopt.storage.db.engine import init_engine, session_scope

logger = logging.getLogger(__name__)

try:
    from llama_index.graph_stores.neo4j.neo4j_property_graph import Neo4jPropertyGraphStore
except ImportError:  # pragma: no cover - optional
    Neo4jPropertyGraphStore = None

try:
    from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
except ImportError:  # pragma: no cover - optional
    try:
        from llama_index.vector_stores.neo4j import Neo4jVectorStore
    except ImportError:  # pragma: no cover - optional
        Neo4jVectorStore = None


@dataclass
class IndexPaths:
    base_dir: Path
    code_dir: Path
    runtime_dir: Path


def _index_paths(config: AppConfig) -> IndexPaths:
    base = Path(config.indexing.persist_dir)
    return IndexPaths(base_dir=base, code_dir=base / "code", runtime_dir=base / "runtime")


def _build_llama_models(config: AppConfig) -> tuple[OpenAI, OpenAIEmbedding]:
    if not config.llm.api_key:
        raise RuntimeError("LLM API key is required for LlamaIndex indexing")
    if config.llm.api_key:
        os.environ.setdefault("OPENAI_API_KEY", config.llm.api_key)
    if config.llm.base_url:
        os.environ.setdefault("OPENAI_API_BASE", config.llm.base_url)
    llm = OpenAI(model=config.llm.model, api_key=config.llm.api_key, api_base=config.llm.base_url)
    embed = OpenAIEmbedding(
        model=config.llm.embedding_model, api_key=config.llm.api_key, api_base=config.llm.base_url
    )
    return llm, embed


def _storage_context(config: AppConfig, persist_dir: Path) -> StorageContext:
    if config.indexing.neo4j.enabled and Neo4jPropertyGraphStore and Neo4jVectorStore:
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
        )
        return StorageContext.from_defaults(
            persist_dir=str(persist_dir),
            property_graph_store=property_graph_store,
            vector_store=vector_store,
        )
    if config.indexing.neo4j.enabled:
        logger.warning("Neo4j stores not available; falling back to local storage")
    return StorageContext.from_defaults(persist_dir=str(persist_dir))


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
    graph_store = storage.property_graph_store
    if not graph_store:
        logger.warning("Property graph store unavailable; skipping clangd relation upsert")
        return
    graph_store.upsert_llama_nodes(nodes)
    entity_nodes, relations = _build_graph_entities(index)
    if entity_nodes:
        graph_store.upsert_nodes(entity_nodes)
    if relations:
        graph_store.upsert_relations(relations)


def build_kernel_index(config: AppConfig, repo_path: Optional[str] = None) -> IndexPaths:
    paths = _index_paths(config)
    paths.code_dir.mkdir(parents=True, exist_ok=True)
    llm, embed = _build_llama_models(config)

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
    index = index_kernel_code(
        Path(repo_path or config.project.repo_path),
        clangd_config=clangd_cfg if config.indexing.clangd.enabled else None,
        max_files=config.indexing.clangd.max_files,
    )
    nodes = _index_to_nodes(index)

    if config.indexing.llm_enrich:
        limit = min(config.indexing.llm_enrich_limit, len(nodes))
        for node in nodes[:limit]:
            summary = llm.complete(f"Summarize the following kernel code:\n\n{node.text}")
            node.text = f"// [LLM SUMMARY]\n{summary.text}\n\n{node.text}"

    storage = _storage_context(config, paths.code_dir)
    VectorStoreIndex(nodes, storage_context=storage, embed_model=embed)
    if config.indexing.neo4j.enabled:
        _upsert_clangd_graph(storage, index, nodes)
    storage.persist(persist_dir=str(paths.code_dir))
    logger.info("Kernel code index built: nodes=%d", len(nodes))
    return paths


def build_runtime_index(config: AppConfig, run_id: str) -> IndexPaths:
    paths = _index_paths(config)
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    llm, embed = _build_llama_models(config)
    engine = init_engine(config.storage.db_url, schema_path=Path("src/hmopt/storage/db/schema.sql"))
    with session_scope(engine) as session:
        nodes = build_runtime_nodes(session, run_id)
    storage = _storage_context(config, paths.runtime_dir)
    VectorStoreIndex(nodes, storage_context=storage, embed_model=embed)
    storage.persist(persist_dir=str(paths.runtime_dir))
    logger.info("Runtime index built: nodes=%d run_id=%s", len(nodes), run_id)
    return paths


def _load_index(config: AppConfig, persist_dir: Path) -> VectorStoreIndex:
    storage = _storage_context(config, persist_dir)
    return load_index_from_storage(storage)


def route_query(config: AppConfig, query: str, mode: str = "auto") -> str:
    paths = _index_paths(config)
    llm, embed = _build_llama_models(config)

    def _engine(dir_path: Path):
        index = _load_index(config, dir_path)
        return index.as_query_engine(llm=llm)

    def _graph_engine(dir_path: Path):
        storage = _storage_context(config, dir_path)
        if not storage.property_graph_store:
            return _engine(dir_path)
        graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=storage.property_graph_store,
            vector_store=storage.vector_store,
            llm=llm,
            embed_model=embed,
            embed_kg_nodes=False,
            storage_context=storage,
        )
        return graph_index.as_query_engine(llm=llm)

    if mode == "code":
        return str(_graph_engine(paths.code_dir).query(query))
    if mode == "runtime":
        return str(_engine(paths.runtime_dir).query(query))
    if mode == "graph":
        return str(_graph_engine(paths.code_dir).query(query))

    keywords_runtime = ["perf", "trace", "runtime", "framegraph", "instruction", "hotspot"]
    if any(k in query.lower() for k in keywords_runtime):
        return str(_engine(paths.runtime_dir).query(query))
    return str(_graph_engine(paths.code_dir).query(query))
