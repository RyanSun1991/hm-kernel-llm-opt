"""LlamaIndex ingestion, indexing, and query routing."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Iterable

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
from hmopt.indexing.mcp_agent import MCPToolAgent, MCPToolAgentConfig
from hmopt.storage.db.engine import init_engine, session_scope
from hmopt.storage.db import models


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

def _index_paths(config: AppConfig, *, run_id: Optional[str] = None) -> IndexPaths:
    base = Path(config.indexing.persist_dir)
    runtime_dir = base / "runtime"
    if run_id:
        runtime_dir = runtime_dir / run_id
    return IndexPaths(base_dir=base, code_dir=base / "code", runtime_dir=runtime_dir)


def _load_prompt_file(path: Optional[Path]) -> Optional[str]:
    """Load prompt content from a file.

    Supports:
    - Plain text files (.txt)
    - Markdown files (.md)
    - JSON files (.json) with "prompt" or "content" key
    """
    if not path or not path.exists():
        return None

    try:
        content = path.read_text(encoding="utf-8")

        if path.suffix.lower() == ".json":
            data = json.loads(content)
            if isinstance(data, dict):
                return data.get("prompt") or data.get("content") or str(data)
            return str(data)

        return content.strip()

    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load prompt file %s: %s", path, exc)
        return None


def _load_query_text(query: str) -> str:
    if not query:
        return ""
    candidate = query[1:] if query.startswith("@") else query
    path = Path(candidate)
    if path.exists() and path.is_file():
        content = _load_prompt_file(path)
        if content:
            return content
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Failed to read query file %s: %s", path, exc)
            return query
    return query


def _format_query_with_prompt(
    query: str,
    prompt_template: Optional[str],
    runtime_context: str = "",
    code_context: str = "",
) -> str:
    """Format query using prompt template with context placeholders.

    Template variables:
    - {query}: The user's query
    - {runtime_context}: Runtime findings (hotspots, metrics, call stacks)
    - {code_context}: Relevant code snippets
    """
    if not prompt_template:
        if runtime_context:
            return (
                "Use the runtime findings below to answer the question, and link to relevant code.\n\n"
                f"Runtime findings:\n{runtime_context}\n\n"
                f"Question: {query}"
            )
        return query

    try:
        return prompt_template.format(
            query=query,
            runtime_context=runtime_context or "(No runtime context available)",
            code_context=code_context or "(No code context available)",
        )
    except KeyError as exc:
        logger.warning("Prompt template missing placeholder %s, using default format", exc)
        return query


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
                    max_tokens= 8192,
                    temperature=0)
    embed = OpenAIEmbeddingLike(
        model=config.llm.embedding_model,
        api_base=config.llm.base_url,
        api_key=config.llm.api_key,
    )
    return llm, embed


def _build_mcp_agent(config: AppConfig) -> MCPToolAgent | None:
    mcp_cfg = getattr(config.indexing, "mcp", None)
    if not mcp_cfg or not mcp_cfg.enabled:
        return None
    return MCPToolAgent(
        MCPToolAgentConfig(
            base_url=mcp_cfg.base_url,
            api_key=mcp_cfg.api_key,
            model=mcp_cfg.model,
            timeout_sec=mcp_cfg.timeout_sec,
            tool_name=mcp_cfg.tool_name,
            top_k=mcp_cfg.top_k,
            mcp_base_url=mcp_cfg.mcp_base_url,
            mcp_api_key=mcp_cfg.mcp_api_key,
            graph_tool_name=mcp_cfg.graph_tool_name,
            hotspot_tool_name=mcp_cfg.hotspot_tool_name,
            default_scenario=mcp_cfg.default_scenario,
            graph_depth=mcp_cfg.graph_depth,
            max_snippets=mcp_cfg.max_snippets,
            max_chars=mcp_cfg.max_chars,
            response_format=mcp_cfg.response_format,
        )
    )

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
    try:
        metadata_path.write_text(
            json.dumps(metadata, indent=2)
            + "\n"
        )
    except PermissionError as exc:
        logger.warning("Failed to persist embedding metadata at %s: %s", metadata_path, exc)


def _load_embedding_metadata(persist_dir: Path) -> Optional[dict]:
    metadata_path = _embedding_metadata_path(persist_dir)
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        logger.warning("Failed to parse embedding metadata at %s", metadata_path)
        return None

def _safe_storage_persist(storage: StorageContext, persist_dir: Path) -> bool:
    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
        storage.persist(persist_dir=str(persist_dir))
        return True
    except PermissionError as exc:
        logger.warning("Skipping local index persistence due to permission error at %s: %s", persist_dir, exc)
        return False
    except OSError as exc:
        logger.warning("Skipping local index persistence due to OS error at %s: %s", persist_dir, exc)
        return False


def _symbol_manifest_path(persist_dir: Path) -> Path:
    return persist_dir / "symbol_manifest.json"


def _load_symbol_manifest(persist_dir: Path) -> dict[str, str]:
    path = _symbol_manifest_path(persist_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Failed to parse symbol manifest at %s", path)
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _persist_symbol_manifest(persist_dir: Path, manifest: dict[str, str]) -> None:
    _symbol_manifest_path(persist_dir).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _node_fingerprint(node: TextNode) -> str:
    metadata = dict(node.metadata or {})
    payload = {
        "text": node.text,
        "metadata": metadata,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _delete_code_symbol_nodes(vector_store: Neo4jVectorStore, node_label: str, symbol_ids: Iterable[str]) -> None:
    symbol_ids = [sid for sid in symbol_ids if sid]
    if not symbol_ids:
        return
    vector_store.database_query(
        f"MATCH (n:`{node_label}`) WHERE n.symbol_id IN $symbol_ids DETACH DELETE n",
        {"symbol_ids": symbol_ids},
    )
    vector_store.database_query(
        "MATCH (s:symbol) WHERE s.name IN $symbol_ids DETACH DELETE s",
        {"symbol_ids": symbol_ids},
    )


def _delete_code_summary_nodes(vector_store: Neo4jVectorStore, node_label: str) -> None:
    vector_store.database_query(
        f"MATCH (n:`{node_label}`) WHERE n.type IN ['file_summary', 'symbol_relations'] DETACH DELETE n"
    )


def _prepare_incremental_code_nodes(
    nodes: list[TextNode],
    previous_manifest: dict[str, str],
) -> tuple[list[TextNode], list[str], list[str], list[str], list[TextNode], dict[str, str]]:
    symbol_nodes: list[TextNode] = []
    summary_nodes: list[TextNode] = []
    unchanged_symbol_ids: list[str] = []
    changed_symbol_ids: list[str] = []
    current_manifest: dict[str, str] = {}

    for node in nodes:
        node_type = (node.metadata or {}).get("type")
        if node_type != "code":
            summary_nodes.append(node)
            continue
        symbol_id = str((node.metadata or {}).get("symbol_id") or node.node_id or "")
        if not symbol_id:
            symbol_nodes.append(node)
            continue
        fingerprint = _node_fingerprint(node)
        current_manifest[symbol_id] = fingerprint
        if previous_manifest.get(symbol_id) == fingerprint:
            unchanged_symbol_ids.append(symbol_id)
            continue
        changed_symbol_ids.append(symbol_id)
        symbol_nodes.append(node)

    removed_symbol_ids = [symbol_id for symbol_id in previous_manifest if symbol_id not in current_manifest]
    return symbol_nodes, removed_symbol_ids, unchanged_symbol_ids, changed_symbol_ids, summary_nodes, current_manifest


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
    use_persist_dir = (persist_dir / "docstore.json").exists()
    if config.indexing.neo4j.enabled and Neo4jPropertyGraphStore and Neo4jVectorStore:
        if embedding_dimension is None:
            raise RuntimeError("Embedding dimension is required for Neo4j vector store usage.")
        print(config.indexing.neo4j.uri)
        print(config.indexing.neo4j.user)
        print(config.indexing.neo4j.password)
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
        kwargs = {
            "property_graph_store": property_graph_store,
            "vector_store": vector_store,
        }
        if use_persist_dir:
            kwargs["persist_dir"] = str(persist_dir)
        return StorageContext.from_defaults(**kwargs)
    if config.indexing.neo4j.enabled:
        logger.warning("Neo4j stores not available; falling back to local storage")
    # if not for_load:
    #     return StorageContext.from_defaults(property_graph_store=property_graph_store, vector_store=vector_store)
    if use_persist_dir:
        return StorageContext.from_defaults(persist_dir=str(persist_dir))
    return StorageContext.from_defaults()


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


def _stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized_chunk_text(chunk: CodeChunk, *, max_lines: int = 120) -> str:
    lines = chunk.text.splitlines()
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
    normalized = "\n".join(line.rstrip() for line in lines).strip()
    return normalized


def _code_header(chunk: CodeChunk) -> str:
    detail = f" {chunk.detail}" if chunk.detail else ""
    return (
        f"// [SYMBOL] {chunk.kind} {chunk.symbol_qualname} "
        f"({chunk.path}:{chunk.start_line}-{chunk.end_line}){detail}"
    )

def _index_to_nodes(index: CodeIndex) -> list[TextNode]:
    nodes: list[TextNode] = []
    print("len(index.chunks) : " + str(len(index.chunks)))
    print("len(index.file_summaries) : " + str(len(index.file_summaries)))
    print("len(index.relation_summaries) : " + str(len(index.relation_summaries)))
    for chunk in index.chunks:
        normalized_text = _normalized_chunk_text(chunk)
        body = f"{_code_header(chunk)}\n{normalized_text}"
        nodes.append(
            TextNode(
                id_=chunk.symbol_id,
                text=body,
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
                    "content_hash": _stable_text_hash(body),
                },
            )
        )
    for summary in index.file_summaries:
        file_summary_id = f"file_summary::{summary.path}"
        text = summary.text.strip()
        nodes.append(
            TextNode(
                id_=file_summary_id,
                text=text,
                metadata={
                    "type": "file_summary",
                    "path": str(summary.path),
                    "symbol_id": file_summary_id,
                    "symbol_name": str(summary.path),
                    "symbol_kind": "file_summary",
                    "content_hash": _stable_text_hash(text),
                },
            )
        )
    for summary in index.relation_summaries:
        relation_id = f"symbol_relations::{summary.symbol_id}"
        text = summary.text.strip()
        nodes.append(
            TextNode(
                id_=relation_id,
                text=text,
                metadata={
                    "type": "symbol_relations",
                    "symbol_id": summary.symbol_id,
                    "symbol_name": summary.symbol_name,
                    "symbol_kind": summary.symbol_kind,
                    "path": summary.path,
                    "content_hash": _stable_text_hash(text),
                },
            )
        )
    return nodes


def _filter_unchanged_nodes_by_hash(
    storage: StorageContext,
    nodes: list[TextNode],
    *,
    node_label: str,
    batch_size: int = 500,
) -> tuple[list[TextNode], int]:
    vector_store = getattr(storage, "vector_store", None)
    if not vector_store or not nodes:
        return nodes, 0

    known_hashes: dict[str, str] = {}
    node_ids = [node.node_id for node in nodes if node.node_id]
    for offset in range(0, len(node_ids), batch_size):
        batch = node_ids[offset : offset + batch_size]
        if not batch:
            continue
        rows = vector_store.database_query(
            (
                f"MATCH (c:`{node_label}`) "
                "WHERE c.id IN $ids "
                "RETURN c.id AS id, c.content_hash AS content_hash"
            ),
            params={"ids": batch},
        )
        for row in rows or []:
            row_id = row.get("id")
            if not row_id:
                continue
            known_hashes[str(row_id)] = str(row.get("content_hash") or "")

    upsert_nodes: list[TextNode] = []
    skipped = 0
    for node in nodes:
        node_id = node.node_id
        node_hash = str((getattr(node, "metadata", {}) or {}).get("content_hash") or "")
        if not node_hash:
            upsert_nodes.append(node)
            continue
        if known_hashes.get(node_id) == node_hash:
            skipped += 1
            continue
        upsert_nodes.append(node)
    return upsert_nodes, skipped

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
    neo4j_cfg = _neo4j_index_config("code")
    # _reset_neo4j_vector_index(
    #     config,
    #     embedding_dimension=embed_dim,
    #     index_name=neo4j_cfg.index_name,
    #     node_label=neo4j_cfg.node_label,
    # )

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

    previous_manifest = _load_symbol_manifest(paths.code_dir)
    (
        symbol_nodes_to_upsert,
        removed_symbol_ids,
        unchanged_symbol_ids,
        changed_symbol_ids,
        summary_nodes,
        current_manifest,
    ) = _prepare_incremental_code_nodes(nodes, previous_manifest)

    embed_dim = _infer_embedding_dimension(embed)

    storage = _storage_context(
        config,
        paths.code_dir,
        embedding_dimension=embed_dim,
        index_name=neo4j_cfg.index_name,
        node_label=neo4j_cfg.node_label,
    )
    # vector_index = VectorStoreIndex(nodes, storage_context=storage, embed_model=embed)
    vector_store = storage.vector_store if isinstance(storage.vector_store, Neo4jVectorStore) else None
    if vector_store:
        _delete_code_symbol_nodes(
            vector_store,
            neo4j_cfg.node_label,
            [*changed_symbol_ids, *removed_symbol_ids],
        )
        _delete_code_summary_nodes(vector_store, neo4j_cfg.node_label)

    nodes_to_insert = [*symbol_nodes_to_upsert, *summary_nodes]
    vector_index: VectorStoreIndex | None = None
    if nodes_to_insert:
        # vector_index = VectorStoreIndex(nodes_to_insert, storage_context=storage, embed_model=embed)
        nodes_to_upsert, skipped_unchanged = _filter_unchanged_nodes_by_hash(
            storage,
            nodes,
            node_label=neo4j_cfg.node_label,
        )
        if skipped_unchanged:
            logger.info(
                "Skipping unchanged nodes by content_hash: skipped=%d upsert=%d total=%d",
                skipped_unchanged,
                len(nodes_to_upsert),
                len(nodes),
            )
        if nodes_to_upsert:
            vector_index = VectorStoreIndex(nodes_to_upsert, storage_context=storage, embed_model=embed)
        else:
            vector_index = _load_index(
                config,
                paths.code_dir,
                embedding_dimension=embed_dim,
                index_name=neo4j_cfg.index_name,
                node_label=neo4j_cfg.node_label,
                embed_model=embed,
            )

    affected_symbol_ids = {str((node.metadata or {}).get("symbol_id") or "") for node in symbol_nodes_to_upsert}
    affected_symbol_ids.discard("")

    if config.indexing.neo4j.enabled:
        # _upsert_clangd_graph(storage, code_index, nodes)
        graph_index = CodeIndex(
            chunks=[chunk for chunk in code_index.chunks if chunk.symbol_id in affected_symbol_ids],
            relations=[
                rel
                for rel in code_index.relations
                if rel.src_id in affected_symbol_ids or rel.dst_id in affected_symbol_ids
            ],
            file_summaries=code_index.file_summaries,
            relation_summaries=code_index.relation_summaries,
        )
        _upsert_clangd_graph(storage, graph_index, nodes_to_insert)

    _safe_storage_persist(storage, paths.code_dir)
    _persist_symbol_manifest(paths.code_dir, current_manifest)
    previous_embedding_meta = _load_embedding_metadata(paths.code_dir) or {}
    _persist_embedding_metadata(
        paths.code_dir,
        model=config.llm.embedding_model,
        dimension=embed_dim,
        index_name=neo4j_cfg.index_name,
        node_label=neo4j_cfg.node_label,
        index_id=(vector_index.index_id if vector_index else previous_embedding_meta.get("index_id")),
    )
    logger.info(
        "Kernel code index built incrementally: total=%d inserted=%d unchanged=%d changed=%d removed=%d",
        len(nodes),
        len(nodes_to_insert),
        len(unchanged_symbol_ids),
        len(changed_symbol_ids),
        len(removed_symbol_ids),
    )
    # logger.info("Kernel code index built: nodes=%d", len(nodes))
    return paths


def build_runtime_index(config: AppConfig, run_id: str) -> IndexPaths:
    paths = _index_paths(config, run_id=run_id)
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    llm, embed = _build_llama_models(config)
    embed_dim = _infer_embedding_dimension(embed)
    neo4j_cfg = _neo4j_index_config("runtime")
    # _reset_neo4j_vector_index(
    #     config,
    #     embedding_dimension=embed_dim,
    #     index_name=neo4j_cfg.index_name,
    #     node_label=neo4j_cfg.node_label,
    # )
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
    # storage.persist(persist_dir=str(paths.runtime_dir))
    _safe_storage_persist(storage, paths.runtime_dir)
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


def lookup_code_symbols(config: AppConfig, symbols: Iterable[str]) -> dict[str, dict]:
    """Lookup symbol metadata (path/lines) from the persisted code index."""
    symbol_list = [s for s in symbols if s]
    if not symbol_list:
        return {}
    paths = _index_paths(config)
    if not paths.code_dir.exists():
        logger.warning("Code index not found at %s", paths.code_dir)
        return {}
    targets = set(symbol_list)
    best: dict[str, dict] = {}
    scores: dict[str, int] = {}

    def _update(symbol: str, score: int, metadata: dict) -> None:
        if score < scores.get(symbol, -1):
            return
        scores[symbol] = score
        best[symbol] = {
            "symbol_name": metadata.get("symbol_name"),
            "symbol_qualname": metadata.get("symbol_qualname"),
            "path": metadata.get("path"),
            "start_line": metadata.get("start_line"),
            "end_line": metadata.get("end_line"),
        }

    try:
        storage = StorageContext.from_defaults(persist_dir=str(paths.code_dir))
        docs = getattr(storage.docstore, "docs", {}) or {}
        remaining = set(targets)
        for node in docs.values():
            metadata = getattr(node, "metadata", {}) if node else {}
            if metadata.get("type") != "code":
                continue
            name = metadata.get("symbol_name")
            qual = metadata.get("symbol_qualname")
            if name in remaining:
                _update(name, 2, metadata)
            if qual in remaining:
                _update(qual, 3, metadata)
            if remaining:
                for sym in list(remaining):
                    if not sym or not qual:
                        continue
                    if sym in qual:
                        _update(sym, 1, metadata)
            if len(best) >= len(targets):
                break
    except Exception as exc:
        logger.warning("Failed to scan local code index: %s", exc)

    missing = [sym for sym in targets if sym not in best]
    if missing and config.indexing.neo4j.enabled:
        try:
            code_index_cfg = _neo4j_index_config("code")
            metadata = _load_embedding_metadata(paths.code_dir) or {}
            embed_dim = metadata.get("dimension")
            if embed_dim:
                vector_store = Neo4jVectorStore(
                    username=config.indexing.neo4j.user,
                    password=config.indexing.neo4j.password,
                    url=config.indexing.neo4j.uri,
                    database=config.indexing.neo4j.database,
                    embedding_dimension=int(embed_dim),
                    index_name=code_index_cfg.index_name,
                    node_label=code_index_cfg.node_label,
                )
                cypher = (
                    "MATCH (s:symbol) "
                    "WHERE s.symbol_name IN $symbols OR s.symbol_qualname IN $symbols "
                    "RETURN s.symbol_name as symbol_name, s.symbol_qualname as symbol_qualname, "
                    "s.path as path, s.start_line as start_line, s.end_line as end_line "
                    "LIMIT $limit"
                )
                records = vector_store.database_query(
                    cypher,
                    {"symbols": missing, "limit": max(10, len(missing) * 2)},
                )
                for rec in records or []:
                    name = rec.get("symbol_name")
                    qual = rec.get("symbol_qualname")
                    if name and name in missing:
                        _update(name, 2, rec)
                    if qual and qual in missing:
                        _update(qual, 3, rec)
        except Exception as exc:
            logger.warning("Neo4j symbol lookup failed: %s", exc)

    return best


def route_query(
    config: AppConfig,
    query: str,
    mode: str = "auto",
    prompt_file: Optional[Path] = None,
    run_id: Optional[str] = None,
    focus_symbols: Optional[list[str]] = None,
    hotspot_top_k: Optional[int] = None,
) -> str:
    query_text = _load_query_text(query)
    paths = _index_paths(config, run_id=run_id)
    llm, embed = _build_llama_models(config)
    mcp_agent = _build_mcp_agent(config)
    code_index_cfg = _neo4j_index_config("code")
    runtime_index_cfg = _neo4j_index_config("runtime")
    code_embed_dim = _embedding_dimension_for_query(paths.code_dir, embed)
    runtime_embed_dim = _embedding_dimension_for_query(paths.runtime_dir, embed)

    prompt_path = prompt_file or config.indexing.query_prompt_file
    prompt_template = _load_prompt_file(prompt_path)
    max_code_snippets = 3
    max_code_chars = 1600
    max_runtime_chars = 60000

    max_query_chars = max(0, (llm.metadata.context_window - max(llm.metadata.num_output, 0) - 256) * 4)

    def _truncate_text(text: str, max_chars: int) -> str:
        if max_chars <= 0 or not text:
            return ""
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}\n...[truncated {len(text) - max_chars} chars]"

    def _trim_contexts(runtime_context: str, code_context: str) -> tuple[str, str]:
        if max_query_chars <= 0:
            return "", ""
        if prompt_template:
            base_prompt = _format_query_with_prompt(
                query=query_text,
                prompt_template=prompt_template,
                runtime_context="",
                code_context="",
            )
        else:
            base_prompt = (
                "Use the runtime findings below to answer the question, and link to relevant code.\n\n"
                "Runtime findings:\n\n\n\n"
                f"Question: {query_text}"
            )
        remaining = max_query_chars - len(base_prompt)
        if remaining <= 0:
            return "", ""
        if runtime_context and code_context:
            runtime_budget = int(remaining * 0.6)
            code_budget = remaining - runtime_budget
        elif runtime_context:
            runtime_budget = remaining
            code_budget = 0
        else:
            runtime_budget = 0
            code_budget = remaining
        return _truncate_text(runtime_context, runtime_budget), _truncate_text(code_context, code_budget)

    def _load_evidence_pack_from_path(metadata: Optional[dict]) -> Optional[dict]:
        if not metadata:
            return None
        path = metadata.get("path")
        if not path or not isinstance(path, str):
            return None
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                payload = handle.read().strip()
        except OSError:
            return None
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def _parse_evidence_pack(text: str, *, metadata: Optional[dict] = None) -> Optional[dict]:
        if not text:
            return _load_evidence_pack_from_path(metadata)
        prefix = "evidence_pack"
        payload = text
        if text.startswith(prefix):
            payload = text[len(prefix):].strip()
        if payload:
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                truncated_marker = "\n\n...[truncated"
                if truncated_marker in payload:
                    payload = payload.split(truncated_marker, 1)[0].strip()
                    if payload:
                        try:
                            return json.loads(payload)
                        except json.JSONDecodeError:
                            pass
        return _load_evidence_pack_from_path(metadata)

    def _format_call_stack_line(stack: dict) -> str:
        direction = stack.get("direction", "call")
        frames = stack.get("frames")
        total_events = stack.get("total_events")
        thread_id = stack.get("thread_id")
        if isinstance(frames, list) and frames:
            frame_parts = []
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                symbol = frame.get("symbol")
                if not symbol:
                    continue
                details = []
                self_events = frame.get("self_events")
                if self_events is not None:
                    details.append(f"self={self_events}")
                sub_events = frame.get("sub_events")
                if sub_events is not None:
                    details.append(f"sub={sub_events}")
                frame_thread_id = frame.get("thread_id", thread_id)
                # if frame_thread_id is not None:
                #     details.append(f"thread_id={frame_thread_id}")
                details_text = f" ({', '.join(details)})" if details else ""
                frame_parts.append(f"{symbol}{details_text}")
            frames_text = " -> ".join(frame_parts)
            stack_details = []
            if total_events is not None:
                stack_details.append(f"total_events={total_events}")
            if thread_id is not None:
                stack_details.append(f"thread_id={thread_id}")
            stack_text = f" ({', '.join(stack_details)})" if stack_details else ""
            return f"    ({direction}) {frames_text}{stack_text}"
        path = " -> ".join(stack.get("stack", []))
        details = []
        if stack.get("self_events") is not None:
            details.append(f"self_events={stack.get('self_events')}")
        if total_events is not None:
            details.append(f"total_events={total_events}")
        if thread_id is not None:
            details.append(f"thread_id={thread_id}")
        details_text = f" ({', '.join(details)})" if details else ""
        return f"    ({direction}) {path}{details_text}"

    def _merge_call_stacks(primary: list[dict], secondary: list[dict]) -> list[dict]:
        if not secondary:
            return primary
        merged = list(primary)
        seen: set[tuple[str, tuple[str, ...], str | None]] = set()
        for stack in primary:
            if not isinstance(stack, dict):
                continue
            path = stack.get("stack") or []
            if not isinstance(path, list):
                continue
            key = (stack.get("direction", "call"), tuple(path), stack.get("thread_id"))
            seen.add(key)
        for stack in secondary:
            if not isinstance(stack, dict):
                continue
            path = stack.get("stack") or []
            if not isinstance(path, list):
                continue
            key = (stack.get("direction", "call"), tuple(path), stack.get("thread_id"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(stack)
        return merged


    def _format_evidence_hotspots(payload: dict) -> list[dict]:
        hotspots_payload = payload.get("hotspots", []) if isinstance(payload, dict) else []
        formatted: list[dict] = []
        for item in hotspots_payload:
            if not isinstance(item, dict):
                continue
            formatted.append(
                {
                    "symbol": item.get("symbol"),
                    "score": item.get("score"),
                    "file_path": item.get("file_path"),
                    "line_start": item.get("line_start"),
                    "line_end": item.get("line_end"),
                    "text": "",
                    "call_stacks": item.get("call_stacks", []) or [],
                }
            )
        return formatted

    def _format_evidence_excerpt(payload: dict, focus_symbol: str | None) -> list[str]:
        if not isinstance(payload, dict):
            return []
        hotspots = payload.get("hotspots", []) or []
        if focus_symbol:
            hotspots = [
                item for item in hotspots
                if isinstance(item, dict) and item.get("symbol") == focus_symbol
            ]
        lines: list[str] = []
        for item in hotspots[:3]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {item.get('symbol')} score={item.get('score')} "
                f"path={item.get('file_path')} lines={item.get('line_start')}-{item.get('line_end')}"
            )
            call_stacks = item.get("call_stacks", []) or []
            if call_stacks:
                lines.append("  Call paths:")
                for stack in call_stacks[:3]:
                    if isinstance(stack, dict):
                        lines.append(_format_call_stack_line(stack))
                        # path = " -> ".join(stack.get("stack", []))
                        # direction = stack.get("direction", "call")
                        # lines.append(f"    ({direction}) {path}")
        return lines


    def _filter_runtime_hotspots(items: list[dict], focus_symbol: str | None) -> list[dict]:
        if not focus_symbol:
            return items
        return [item for item in items if item.get("symbol") == focus_symbol]

    def _dedupe_hotspots(items: list[dict]) -> list[dict]:
        by_symbol: dict[str, dict] = {}
        for item in items:
            symbol = item.get("symbol")
            if not symbol:
                continue
            existing = by_symbol.get(symbol)
            if not existing:
                by_symbol[symbol] = item
                continue
            if not existing.get("call_stacks") and item.get("call_stacks"):
                by_symbol[symbol] = item
        return list(by_symbol.values())

    def _collect_runtime_hotspots(source_nodes: list) -> tuple[list[dict], list[dict]]:
        hotspots: list[dict] = []
        evidence_hotspots: list[dict] = []
        for source in source_nodes:
            node = getattr(source, "node", source)
            metadata = getattr(node, "metadata", {}) if node else {}
            node_type = metadata.get("type")
            text = getattr(node, "text", "")
            call_stacks: list = []
            if node_type in ("runtime_hotspot", "runtime_symbol"):
                call_stacks_raw = metadata.get("call_stacks")
                if isinstance(call_stacks_raw, str):
                    try:
                        call_stacks = json.loads(call_stacks_raw)
                    except json.JSONDecodeError:
                        pass
                elif isinstance(call_stacks_raw, list):
                    call_stacks = call_stacks_raw
                hotspots.append(
                    {
                        "symbol": metadata.get("symbol"),
                        "score": metadata.get("score"),
                        "file_path": metadata.get("file_path"),
                        "line_start": metadata.get("line_start"),
                        "line_end": metadata.get("line_end"),
                        "text": text,
                        "call_stacks": call_stacks,
                    }
                )
            elif node_type == "evidence_pack":
                text = getattr(node, "text", "")
                payload = _parse_evidence_pack(text, metadata=metadata)
                if payload:
                    evidence_hotspots.extend(_format_evidence_hotspots(payload))
        return _dedupe_hotspots(hotspots), _dedupe_hotspots(evidence_hotspots)

    def _format_runtime_sources(source_nodes: list, *, focus_symbol: str | None) -> str:
        hotspots, evidence_hotspots = _collect_runtime_hotspots(source_nodes)
        metrics = []
        for source in source_nodes:
            node = getattr(source, "node", source)
            metadata = getattr(node, "metadata", {}) if node else {}
            node_type = metadata.get("type")
            text = getattr(node, "text", "")
            if node_type == "runtime_metric":
                metrics.append(
                    {
                        "metric_name": metadata.get("metric_name"),
                        "value": metadata.get("value"),
                        "unit": metadata.get("unit"),
                        "text": text,
                    }
                )
        lines = []
        if hotspots or evidence_hotspots:
            lines.append("Top runtime hotspots:")
            runtime_items = _filter_runtime_hotspots(hotspots, focus_symbol) or _filter_runtime_hotspots(
                evidence_hotspots, focus_symbol
            )
            if hotspots and evidence_hotspots:
                evidence_map = {
                    item.get("symbol"): item for item in _filter_runtime_hotspots(evidence_hotspots, focus_symbol)
                }
                for item in runtime_items:
                    symbol = item.get("symbol")
                    evidence_item = evidence_map.get(symbol) if symbol else None
                    if evidence_item and evidence_item.get("call_stacks"):
                        item["call_stacks"] = _merge_call_stacks(
                            item.get("call_stacks", []), evidence_item.get("call_stacks", [])
                        )
            for item in runtime_items:
                lines.append(
                    f"- {item.get('symbol')} score={item.get('score')} "
                    f"path={item.get('file_path')} lines={item.get('line_start')}-{item.get('line_end')}"
                )
                call_stacks = item.get("call_stacks", [])
                if call_stacks:
                    lines.append("  Call paths:")

                    sorted_stacks = sorted(
                        [stack for stack in call_stacks if isinstance(stack, dict)],
                        key=lambda s: s.get("total_events") or 0,
                        reverse=True,
                    )
                    top_stacks = sorted_stacks[:20]
                    batch = top_stacks[:10]
                    for stack in batch:
                        if isinstance(stack, str):
                            stack = ast.literal_eval(stack)
                        if isinstance(stack, dict):
                            lines.append(_format_call_stack_line(stack))

                    remaining = len(top_stacks) - len(batch)
                    if remaining > 0:
                        lines.append(f"  ...and {remaining} more call paths (top 20 by total_events)")

                    # for stack in call_stacks:
        if metrics:
            lines.append("Runtime metrics:")
            for item in metrics:
                lines.append(
                    f"- {item.get('metric_name')} value={item.get('value')} unit={item.get('unit')}"
                )
        runtime_text = "\n".join(lines).strip()
        if len(runtime_text) > max_runtime_chars:
            runtime_text = (
                f"{runtime_text[:max_runtime_chars]}\n"
                f"...[truncated {len(runtime_text) - max_runtime_chars} chars]"
            )
        return runtime_text

    def _load_artifacts_payloads(session, run_id: str, kind: str) -> list[dict]:
        artifacts = (
            session.query(models.Artifact)
            .filter(models.Artifact.run_id == run_id, models.Artifact.kind == kind)
            .order_by(models.Artifact.bytes.desc())
            .all()
        )
        payloads: list[dict] = []
        for art in artifacts:
            try:
                raw = Path(art.path).read_text(encoding="utf-8")
                payload = json.loads(raw)
                if isinstance(payload, dict) or isinstance(payload, list):
                    payloads.append(payload)
            except (OSError, json.JSONDecodeError):
                continue
        return payloads

    def _format_flamegraph_symbol_context(
        session, run_id: str, symbol: str
    ) -> str:
        lines: list[str] = []
        if not symbol:
            return ""
        counts_payloads = _load_artifacts_payloads(
            session, run_id, "flamegraph_symbol_counts"
        )
        if not counts_payloads:
            counts_payloads = _load_artifacts_payloads(
                session, run_id, "flamegraph_symbol_counts_raw"
            )
        score = None
        for payload in counts_payloads:
            if isinstance(payload, dict) and symbol in payload:
                try:
                    score = float(payload.get(symbol))
                except (TypeError, ValueError):
                    score = payload.get(symbol)
                break
        if score is not None:
            lines.append(f"Flamegraph symbol stats: {symbol} score={score}")

        call_stack_payloads = _load_artifacts_payloads(
            session, run_id, "flamegraph_call_stacks"
        )
        call_stacks: list[dict] = []
        for payload in call_stack_payloads:
            if isinstance(payload, list):
                call_stacks.extend([item for item in payload if isinstance(item, dict)])
        if call_stacks:
            matching = [
                stack for stack in call_stacks
                if symbol in (stack.get("stack") or [])
                or stack.get("leaf_symbol") == symbol
            ]
            if matching:
                sorted_stacks = sorted(
                    matching,
                    key=lambda s: s.get("total_events") or 0,
                    reverse=True,
                )
                lines.append("Flamegraph call paths:")
                for stack in sorted_stacks[:5]:
                    lines.append(_format_call_stack_line(stack))
                if len(sorted_stacks) > 5:
                    lines.append(f"...and {len(sorted_stacks) - 5} more call paths")
        return "\n".join(lines).strip()

    def _format_code_sources(source_nodes: list, *, include_snippets: bool) -> str:
        if not include_snippets:
            return (
                "Code snippets omitted to reduce context. Use MCP/RAG retrieval for specific symbols."
            )
        snippets = []
        for source in source_nodes:
            node = getattr(source, "node", source)
            text = getattr(node, "text", "")
            if not text:
                continue
            if len(text) > max_code_chars:
                text = f"{text[:max_code_chars]}\n...[truncated {len(text) - max_code_chars} chars]"
            snippets.append(text)
            if len(snippets) >= max_code_snippets:
                break
        if not snippets:
            return ""
        return "Relevant code snippets:\n" + "\n\n".join(snippets)

    def _extract_runtime_symbols(source_nodes: list, *, focus_symbol: str | None) -> list[str]:
        symbols: set[str] = set()
        for source in source_nodes:
            node = getattr(source, "node", source)
            metadata = getattr(node, "metadata", {}) if node else {}
            symbol = metadata.get("symbol")
            if symbol and (not focus_symbol or symbol == focus_symbol):
                symbols.add(symbol)
            call_stacks_raw = metadata.get("call_stacks")
            call_stacks = []
            if isinstance(call_stacks_raw, str):
                try:
                    call_stacks = json.loads(call_stacks_raw)
                except json.JSONDecodeError:
                    pass
            elif isinstance(call_stacks_raw, list):
                call_stacks = call_stacks_raw
            for stack in call_stacks:
                if isinstance(stack, dict):
                    for stack_symbol in stack.get("stack", []):
                        if stack_symbol and (not focus_symbol or stack_symbol == focus_symbol):
                            symbols.add(stack_symbol)
            text = getattr(node, "text", "")
            payload = _parse_evidence_pack(text, metadata=metadata)
            if payload:
                for item in payload.get("hotspots", []) or []:
                    if not isinstance(item, dict):
                        continue
                    hs_symbol = item.get("symbol")
                    if hs_symbol and (not focus_symbol or hs_symbol == focus_symbol):
                        symbols.add(hs_symbol)
                    for stack in item.get("call_stacks", []) or []:
                        if isinstance(stack, dict):
                            for stack_symbol in stack.get("stack", []):
                                if stack_symbol and (not focus_symbol or stack_symbol == focus_symbol):
                                    symbols.add(stack_symbol)
        return list(symbols)

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
        runtime_response_text = ""
        if run_id:
            engine = init_engine(
                config.storage.db_url,
                schema_path=Path("src/hmopt/storage/db/schema.sql"),
            )
            with session_scope(engine) as session:
                source_nodes = build_runtime_nodes(
                    session,
                    run_id,
                    max_evidence_chars=config.indexing.runtime_evidence_max_chars,
                )
        else:
            runtime_response = _engine(paths.runtime_dir).query(query_text)
            runtime_response_text = str(runtime_response).strip()
            source_nodes = getattr(runtime_response, "source_nodes", []) or []

        focus_symbol = config.indexing.hotspot_focus_symbol
        hotspots, evidence_hotspots = _collect_runtime_hotspots(source_nodes)
        runtime_items = hotspots or evidence_hotspots
        if focus_symbols:
            symbol_queue = [s for s in focus_symbols if s]
        elif focus_symbol:
            symbol_queue = [focus_symbol]
        else:
            seen: set[str] = set()
            symbol_queue = []
            top_k = hotspot_top_k or config.indexing.query_hotspot_top_k or config.indexing.hotspot_top_k
            runtime_items_sorted = sorted(
                runtime_items, key=lambda item: item.get("score") or 0, reverse=True
            )
            for item in runtime_items_sorted[: max(1, int(top_k))]:
                symbol = item.get("symbol")
                if symbol and symbol not in seen:
                    seen.add(symbol)
                    symbol_queue.append(symbol)
        if not symbol_queue:
            runtime_context = _format_runtime_sources(source_nodes, focus_symbol=focus_symbol)
            runtime_context, code_context = _trim_contexts(runtime_context, "")
            combined_query = _format_query_with_prompt(
                query=query_text,
                prompt_template=prompt_template,
                runtime_context=runtime_context,
                code_context=code_context,
            ) if prompt_template else (
                "Use the runtime findings below to answer the question, and link to relevant code.\n\n"
                f"Runtime findings:\n{runtime_context}\n\n"
                f"{code_context}\n\n"
                f"Question: {query_text}"
            )
            return str(_graph_engine(paths.code_dir).query(combined_query))

        engine = _graph_engine(paths.code_dir)
        responses: list[str] = []
        code_context_mode = (config.indexing.query_code_context_mode or "snippets").lower()
        for symbol in symbol_queue:
            runtime_context = _format_runtime_sources(source_nodes, focus_symbol=symbol)
            if not runtime_context:
                if run_id:
                    engine_db = init_engine(
                        config.storage.db_url,
                        schema_path=Path("src/hmopt/storage/db/schema.sql"),
                    )
                    with session_scope(engine_db) as session:
                        runtime_context = _format_flamegraph_symbol_context(session, run_id, symbol)
                if not runtime_context:
                    runtime_context = runtime_response_text
            print("runtime_context : ")
            print(runtime_context)
            # code_context = (
            #     "Code snippets omitted to reduce context. Use MCP/RAG retrieval for specific symbols."
            # )
            runtime_symbols = _extract_runtime_symbols(source_nodes, focus_symbol = symbol)
            code_context_parts: list[str] = []
            include_snippets = code_context_mode in ("snippets", "hybrid")
            if include_snippets and runtime_symbols:
                symbol_query = " ".join(runtime_symbols[:30])
                _ensure_embedding_compat(
                    paths.code_dir,
                    embed_model=config.llm.embedding_model,
                    embed_dim=code_embed_dim,
                    index_name=code_index_cfg.index_name,
                    node_label=code_index_cfg.node_label,
                )
                code_index = _load_index(
                    config,
                    paths.code_dir,
                    embedding_dimension=code_embed_dim,
                    index_name=code_index_cfg.index_name,
                    node_label=code_index_cfg.node_label,
                    embed_model=embed,
                )
                code_nodes = code_index.as_retriever(
                    similarity_top_k=config.indexing.query_code_top_k
                ).retrieve(symbol_query)
                snippet_context = _format_code_sources(code_nodes, include_snippets=True)
                if snippet_context:
                    code_context_parts.append(snippet_context)
            if code_context_mode in ("mcp", "hybrid") and mcp_agent:
                symbols_hint = ", ".join(runtime_symbols[:15]) if runtime_symbols else symbol
                mcp_query = (
                    "Fetch kernel code context for hotspot analysis. "
                    "Include implementation snippets (function bodies) for the focus symbol and key related "
                    "symbols. "
                    f"Focus symbol: {symbol}. Related symbols: {symbols_hint}."
                )
                try:
                    mcp_result = mcp_agent.fetch_context(mcp_query)
                    if mcp_result.context:
                        code_context_parts.append(
                            "MCP retrieved context:\n" + mcp_result.context
                        )
                except Exception as exc:
                    logger.warning("MCP retrieval failed: %s", exc)
            print("code_context_parts")
            print(code_context_parts)
            code_context = "\n\n".join(part for part in code_context_parts if part)
            runtime_context, code_context = _trim_contexts(runtime_context, code_context)
            if prompt_template:
                combined_query = _format_query_with_prompt(
                    query=query_text,
                    prompt_template=prompt_template,
                    runtime_context=runtime_context,
                    code_context=code_context,
                )
            else:
                combined_query = (
                    "Use the runtime findings below to answer the question, and link to relevant code.\n\n"
                    f"Runtime findings:\n{runtime_context}\n\n"
                    f"{code_context}\n\n"
                    f"Question: {query_text}"
                )
            response = str(engine.query(combined_query))
            responses.append(f"## Hotspot {symbol}\n{response}".strip())
        return "\n\n".join(responses)

    if mode == "code":
        return str(_graph_engine(paths.code_dir).query(query_text))
    if mode == "runtime":
        runtime_response = _engine(paths.runtime_dir).query(query_text)
        response_text = str(runtime_response).strip()
        if response_text:
            return response_text
        source_nodes = getattr(runtime_response, "source_nodes", []) or []
        fallback = _format_runtime_sources(
            source_nodes, focus_symbol=config.indexing.hotspot_focus_symbol
        )
        return fallback or "No runtime results available."
    if mode == "graph":
        return str(_graph_engine(paths.code_dir).query(query_text))
    if mode == "runtime_code":
        return _runtime_to_code_query()

    if run_id or focus_symbols or hotspot_top_k:
        return _runtime_to_code_query()

    keywords_runtime = ["perf", "trace", "runtime", "framegraph", "instruction", "hotspot"]
    if any(k in query_text.lower() for k in keywords_runtime):
        return _runtime_to_code_query()
    return str(_graph_engine(paths.code_dir).query(query_text))



def retrieve_code_context(
    config: AppConfig,
    query: str,
    *,
    top_k: Optional[int] = None,
    max_snippets: int = 8,
    max_chars: int = 4000,
    output_format: str = "text",
    focus_symbols: Optional[list[str]] = None,
    graph_depth: Optional[int] = None,
    scenario: str = "general",
) -> str | dict[str, Any]:
    """Retrieve forced kernel context (code + graph expansion + rerank)."""
    if not query:
        return ""
    fmt = (output_format or "text").strip().lower()
    scenario_name = (scenario or "general").strip().lower() or "general"
    paths = _index_paths(config)
    _, embed = _build_llama_models(config)
    code_index_cfg = _neo4j_index_config("code")
    code_embed_dim = _embedding_dimension_for_query(paths.code_dir, embed)
    graph_edges: list[dict[str, str | int]] = []
    relation_weights: dict[str, float] = {
        "calls": 0.95,
        "uses_type": 0.35,
        "uses_macro": 0.25,
        "contains": 0.2,
        "related_to": 0.2,
    }
    scenario_relation_weights: dict[str, dict[str, float]] = {
        "implementation": {
            "contains": 0.4,
            "uses_type": 0.45,
        },
        "call_graph": {
            "calls": 1.35,
            "contains": 0.35,
        },
        "impact_analysis": {
            "calls": 1.15,
            "uses_type": 0.6,
            "contains": 0.4,
            "related_to": 0.35,
        },
        "hotspot_debug": {
            "calls": 1.2,
            "uses_type": 0.55,
            "uses_macro": 0.4,
        },
        "patch_planning": {
            "calls": 1.2,
            "uses_type": 0.55,
            "contains": 0.45,
            "related_to": 0.4,
        },
    }
    relation_weights.update(scenario_relation_weights.get(scenario_name, {}))
    graph_relation_bonus: dict[str, float] = {}
    symbol_relation_counts: dict[str, dict[str, int]] = {}
    _ensure_embedding_compat(
        paths.code_dir,
        embed_model=config.llm.embedding_model,
        embed_dim=code_embed_dim,
        index_name=code_index_cfg.index_name,
        node_label=code_index_cfg.node_label,
    )
    index = _load_index(
        config,
        paths.code_dir,
        embedding_dimension=code_embed_dim,
        index_name=code_index_cfg.index_name,
        node_label=code_index_cfg.node_label,
        embed_model=embed,
    )
    retriever = index.as_retriever(similarity_top_k=top_k or config.indexing.query_code_top_k)
    source_nodes = retriever.retrieve(query)

    def _extract_query_symbols(text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
        seen: set[str] = set()
        result: list[str] = []
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            result.append(tok)
        return result[:10]

    candidate_symbols: dict[str, dict] = {}
    vector_scores: dict[str, float] = {}
    for source in source_nodes:
        node = getattr(source, "node", source)
        metadata = getattr(node, "metadata", {}) if node else {}
        symbol_name = metadata.get("symbol_name")
        symbol_qual = metadata.get("symbol_qualname")
        score = getattr(source, "score", 0.0) or 0.0
        for sym in (symbol_name, symbol_qual):
            if not sym:
                continue
            if sym not in candidate_symbols:
                candidate_symbols[sym] = {
                    "symbol": sym,
                    "metadata": metadata,
                    "node": node,
                }
            vector_scores[sym] = max(vector_scores.get(sym, 0.0), float(score))

    selected_focus_symbols: list[str] = []
    if focus_symbols:
        seen_focus: set[str] = set()
        for sym in focus_symbols:
            sym_text = (sym or "").strip()
            if not sym_text or sym_text in seen_focus:
                continue
            seen_focus.add(sym_text)
            selected_focus_symbols.append(sym_text)
    selected_focus_symbols = selected_focus_symbols[:12]

    query_tokens = _extract_query_symbols(query)
    if not selected_focus_symbols:
        for sym in candidate_symbols:
            if sym in query or sym in query_tokens:
                selected_focus_symbols.append(sym)
        if not selected_focus_symbols:
            selected_focus_symbols = list(candidate_symbols.keys())[:3]
        if not selected_focus_symbols and query_tokens:
            selected_focus_symbols = query_tokens[:1]

    depth_map: dict[str, int] = {sym: 0 for sym in selected_focus_symbols}
    used_graph_depth = graph_depth if graph_depth is not None else (config.indexing.query_graph_top_k or 2)
    try:
        used_graph_depth = int(used_graph_depth)
    except (TypeError, ValueError):
        used_graph_depth = 2
    used_graph_depth = max(1, min(4, used_graph_depth))

    if config.indexing.neo4j.enabled and selected_focus_symbols:
        try:
            vector_store = Neo4jVectorStore(
                username=config.indexing.neo4j.user,
                password=config.indexing.neo4j.password,
                url=config.indexing.neo4j.uri,
                database=config.indexing.neo4j.database,
                embedding_dimension=code_embed_dim,
                index_name=code_index_cfg.index_name,
                node_label=code_index_cfg.node_label,
            )
            per_hop_limit = 80 if scenario_name in {"call_graph", "impact_analysis", "patch_planning"} else 50
            frontier = set(selected_focus_symbols)
            visited = set(selected_focus_symbols)
            for depth in range(1, used_graph_depth + 1):
                if not frontier:
                    break
                symbols_param = list(frontier)[:20]
                outgoing = vector_store.database_query(
                    (
                        "MATCH (s:symbol)-[r]->(t) "
                        "WHERE s.symbol_name IN $symbols OR s.symbol_qualname IN $symbols "
                        "RETURN s.symbol_name as src, s.symbol_qualname as src_qual, "
                        "type(r) as rel, t.symbol_name as dst, t.symbol_qualname as dst_qual "
                        "LIMIT $limit"
                    ),
                    {"symbols": symbols_param, "limit": per_hop_limit},
                )
                incoming = vector_store.database_query(
                    (
                        "MATCH (s:symbol)<-[r]-(t) "
                        "WHERE s.symbol_name IN $symbols OR s.symbol_qualname IN $symbols "
                        "RETURN t.symbol_name as src, t.symbol_qualname as src_qual, "
                        "type(r) as rel, s.symbol_name as dst, s.symbol_qualname as dst_qual "
                        "LIMIT $limit"
                    ),
                    {"symbols": symbols_param, "limit": per_hop_limit},
                )
                new_frontier: set[str] = set()
                for rec in (outgoing or []):
                    src = rec.get("src") or rec.get("src_qual")
                    dst = rec.get("dst") or rec.get("dst_qual")
                    if src and dst:
                        rel_name = str(rec.get("rel") or "related_to")
                        graph_edges.append(
                            {"src": src, "dst": dst, "rel": rel_name, "depth": depth}
                        )
                        dst_rel_counts = symbol_relation_counts.setdefault(dst, {})
                        dst_rel_counts[rel_name] = dst_rel_counts.get(rel_name, 0) + 1
                        rel_weight = relation_weights.get(rel_name, 0.2)
                        graph_relation_bonus[dst] = (
                            graph_relation_bonus.get(dst, 0.0) + rel_weight / (1 + depth)
                        )
                        if dst not in visited:
                            visited.add(dst)
                            depth_map[dst] = depth
                            new_frontier.add(dst)
                for rec in (incoming or []):
                    src = rec.get("src") or rec.get("src_qual")
                    dst = rec.get("dst") or rec.get("dst_qual")
                    if src and dst:
                        rel_name = str(rec.get("rel") or "related_to")
                        graph_edges.append(
                            {"src": src, "dst": dst, "rel": rel_name, "depth": depth}
                        )
                        src_rel_counts = symbol_relation_counts.setdefault(src, {})
                        src_rel_counts[rel_name] = src_rel_counts.get(rel_name, 0) + 1
                        rel_weight = relation_weights.get(rel_name, 0.2)
                        graph_relation_bonus[src] = (
                            graph_relation_bonus.get(src, 0.0) + rel_weight / (1 + depth)
                        )
                        if src not in visited:
                            visited.add(src)
                            depth_map[src] = depth
                            new_frontier.add(src)
                frontier = new_frontier
        except Exception as exc:
            logger.warning("Neo4j graph expansion failed: %s", exc)

    target_symbols: set[str] = set(candidate_symbols.keys())
    target_symbols.update(depth_map.keys())
    if not target_symbols:
        return ""

    # Build a lookup for code nodes by symbol name/qualname.
    node_lookup: dict[str, TextNode] = {}
    docstore_nodes: dict[str, TextNode] = {}
    storage = None
    try:
        storage = _storage_context(
            config,
            paths.code_dir,
            embedding_dimension=code_embed_dim,
            index_name=code_index_cfg.index_name,
            node_label=code_index_cfg.node_label,
        )
        docs = getattr(storage.docstore, "docs", {}) or {}
        if isinstance(docs, dict):
            docstore_nodes = docs
        for node in docs.values():
            metadata = getattr(node, "metadata", {}) if node else {}
            if metadata.get("type") != "code":
                continue
            name = metadata.get("symbol_name")
            qual = metadata.get("symbol_qualname")
            if name and name in target_symbols and name not in node_lookup:
                node_lookup[name] = node
            if qual and qual in target_symbols and qual not in node_lookup:
                node_lookup[qual] = node
            if len(node_lookup) >= len(target_symbols):
                break
    except Exception as exc:
        logger.warning("Failed to scan docstore for symbols: %s", exc)

    ranked: list[tuple[str, float]] = []
    for sym in target_symbols:
        base = vector_scores.get(sym, 0.0)
        depth = depth_map.get(sym, 3)
        exact_bonus = 1.5 if sym in selected_focus_symbols else 0.0
        depth_bonus = 0.6 / (1 + depth)
        relation_bonus = graph_relation_bonus.get(sym, 0.0)
        relation_degree = sum(symbol_relation_counts.get(sym, {}).values())
        degree_bonus = min(0.6, 0.08 * relation_degree)
        score = base + exact_bonus + depth_bonus + relation_bonus + degree_bonus
        ranked.append((sym, score))
    ranked.sort(key=lambda item: item[1], reverse=True)

    fallback_metadata = lookup_code_symbols(
        config, [sym for sym in target_symbols if sym not in node_lookup]
    )

    def _load_snippet_from_metadata(symbol: str, metadata: dict | None) -> str:
        if not metadata:
            return ""
        property_graph_store = storage.property_graph_store if storage else None
        if property_graph_store:
            symbol_id = metadata.get("symbol_id")
            if symbol_id:
                try:
                    nodes = property_graph_store.get(ids=[symbol_id])
                except Exception as exc:
                    logger.warning("Neo4j lookup failed for %s: %s", symbol_id, exc)
                    nodes = []
                for node in nodes:
                    text = getattr(node, "text", "")
                    if text:
                        return text
            for key in ("symbol_name", "symbol_qualname"):
                prop_value = metadata.get(key) or symbol
                if not prop_value:
                    continue
                try:
                    nodes = property_graph_store.get(properties={key: prop_value})
                except Exception as exc:
                    logger.warning("Neo4j lookup failed for %s=%s: %s", key, prop_value, exc)
                    nodes = []
                for node in nodes:
                    text = getattr(node, "text", "")
                    if text:
                        return text
        symbol_id = metadata.get("symbol_id")
        if symbol_id and symbol_id in docstore_nodes:
            text = getattr(docstore_nodes[symbol_id], "text", "")
            if text:
                return text
        node = node_lookup.get(symbol)
        if node:
            text = getattr(node, "text", "")
            if text:
                return text
        return ""

    def _metadata_for_symbol(symbol: str) -> dict | None:
        if symbol in fallback_metadata:
            return fallback_metadata.get(symbol)
        entry = candidate_symbols.get(symbol)
        if entry and isinstance(entry.get("metadata"), dict):
            return entry.get("metadata")
        for key, entry in candidate_symbols.items():
            if key == symbol:
                continue
            metadata = entry.get("metadata") if entry else None
            if not isinstance(metadata, dict):
                continue
            if metadata.get("symbol_qualname") == symbol or metadata.get("symbol_name") == symbol:
                return metadata
        for sym_key, metadata in fallback_metadata.items():
            if sym_key == symbol:
                return metadata
            if not isinstance(metadata, dict):
                continue
            if metadata.get("symbol_qualname") == symbol or metadata.get("symbol_name") == symbol:
                return metadata
        return None

    selected: list[str] = []
    for sym in selected_focus_symbols:
        if sym in target_symbols and sym not in selected:
            selected.append(sym)
    for sym, _ in ranked:
        if sym not in selected:
            selected.append(sym)
        if len(selected) >= max_snippets:
            break

    ranked_map = {sym: score for sym, score in ranked}
    selected_entries: list[dict[str, Any]] = []
    snippet_entries: list[dict[str, Any]] = []
    for sym in selected:
        score = ranked_map.get(sym, 0.0)
        depth = depth_map.get(sym)
        vec_score = vector_scores.get(sym)
        metadata = _metadata_for_symbol(sym) or {}
        path = metadata.get("path")
        start_line = metadata.get("start_line")
        end_line = metadata.get("end_line")
        relation_breakdown = symbol_relation_counts.get(sym, {})
        relation_degree = sum(relation_breakdown.values())
        selected_entries.append(
            {
                "symbol": sym,
                "score": round(float(score), 6),
                "vector_score": round(float(vec_score), 6) if vec_score is not None else None,
                "depth": depth,
                "relation_bonus": round(float(graph_relation_bonus.get(sym, 0.0)), 6),
                "graph_degree": relation_degree,
                "relation_breakdown": relation_breakdown,
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
            }
        )
        node = node_lookup.get(sym)
        text = getattr(node, "text", "") if node else ""
        if not text:
            text = _load_snippet_from_metadata(sym, metadata)
        if text:
            truncated = False
            if len(text) > max_chars:
                text = f"{text[:max_chars]}\n...[truncated {len(text) - max_chars} chars]"
                truncated = True
            snippet_entries.append(
                {
                    "symbol": sym,
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "truncated": truncated,
                    "text": text,
                }
            )

    relation_counts: dict[str, int] = {}
    for edge in graph_edges:
        rel_name = str(edge.get("rel") or "related_to")
        relation_counts[rel_name] = relation_counts.get(rel_name, 0) + 1

    payload: dict[str, Any] = {
        "query": query,
        "scenario": scenario_name,
        "retrieval_mode": "vector+neo4j-graph" if config.indexing.neo4j.enabled else "vector",
        "vector_top_k": top_k or config.indexing.query_code_top_k,
        "graph_depth": used_graph_depth,
        "focus_symbols": selected_focus_symbols,
        "ranked_symbols": selected_entries,
        "snippets": snippet_entries,
        "graph_edges": graph_edges[:80],
        "graph_summary": {
            "edge_count": len(graph_edges),
            "relation_counts": relation_counts,
        },
    }

    if fmt in ("json", "payload", "dict"):
        return payload

    lines: list[str] = [
        "[Kernel Index Retrieval]",
        f"Query: {query}",
        f"Retrieval mode: {payload['retrieval_mode']}",
        f"Vector top_k: {payload['vector_top_k']} | Graph depth: {payload['graph_depth']}",
    ]
    if selected_focus_symbols:
        lines.append("Focus symbols: " + ", ".join(selected_focus_symbols))
    lines.append("Reranked symbols (vector + graph):")

    snippet_by_symbol = {item.get("symbol"): item for item in snippet_entries}
    for item in selected_entries:
        sym = item.get("symbol")
        detail = [f"score={item.get('score'):.4f}"]
        if item.get("vector_score") is not None:
            detail.append(f"vector={item.get('vector_score'):.4f}")
        if item.get("relation_bonus"):
            detail.append(f"graph_bonus={item.get('relation_bonus'):.4f}")
        if item.get("graph_degree"):
            detail.append(f"graph_degree={item.get('graph_degree')}")
        if item.get("depth") is not None:
            detail.append(f"depth={item.get('depth')}")
        if item.get("path"):
            detail.append(
                f"path={item.get('path')}:{item.get('start_line')}-{item.get('end_line')}"
            )
        lines.append(f"- {sym} ({', '.join(detail)})")
        snippet_item = snippet_by_symbol.get(sym)
        if snippet_item and snippet_item.get("text"):
            lines.append(str(snippet_item["text"]))

    if graph_edges:
        lines.append("Graph expansion edges:")
        for edge in graph_edges[:80]:
            lines.append(
                f"- {edge['src']} -[{edge['rel']}]-> {edge['dst']} (depth={edge['depth']})"
            )
        if relation_counts:
            rel_summary = ", ".join(
                f"{rel}={count}" for rel, count in sorted(relation_counts.items(), key=lambda x: x[0])
            )
            lines.append("Graph relation summary: " + rel_summary)

    return "\n\n".join(lines).strip()
