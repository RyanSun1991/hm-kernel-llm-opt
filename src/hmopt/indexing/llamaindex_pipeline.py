"""LlamaIndex ingestion, indexing, and query routing."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from llama_index.core import (
    PropertyGraphIndex,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.schema import TextNode
from hmopt.indexing.openai_like import OpenAILike, OpenAIEmbeddingLike

# from hmopt.indexing.llama_embeddings import OpenAICompatEmbedding
from hmopt.core.config import AppConfig
from hmopt.indexing.clangd_client import ClangdConfig as LspClangdConfig
from hmopt.indexing.clangd_indexer import CodeIndex, CodeChunk, index_kernel_code
from hmopt.indexing.runtime_ingestion import build_runtime_nodes
from hmopt.storage.db.engine import init_engine, session_scope
from hmopt.tools.git_tools import get_changed_files, get_repo_state, resolve_git_ref

try:
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, FilterOperator
except Exception:  # pragma: no cover - optional dependency
    MetadataFilters = None
    ExactMatchFilter = None
    FilterOperator = None


try:
    from llama_index.graph_stores.neo4j.neo4j_property_graph import Neo4jPropertyGraphStore
except Exception:  # pragma: no cover - optional
    Neo4jPropertyGraphStore = None

try:
    from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
except Exception:  # pragma: no cover - optional
    Neo4jVectorStore = None

logger = logging.getLogger(__name__)

@dataclass
class IndexPaths:
    base_dir: Path
    code_root: Path
    runtime_root: Path
    code_dir: Path
    runtime_dir: Path
    code_version: Optional[str] = None
    runtime_version: Optional[str] = None

@dataclass
class Neo4jIndexConfig:
    index_name: str
    node_label: str

_VERSION_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(text: str) -> str:
    text = text.strip().lower().replace(" ", "-")
    text = _VERSION_SAFE.sub("-", text)
    return text.strip("-") or "project"


def _sanitize_version(version: str) -> str:
    version = version.strip()
    return _VERSION_SAFE.sub("-", version) or "unknown"


def _index_roots(config: AppConfig) -> tuple[Path, Path, Path]:
    base = Path(config.indexing.persist_dir)
    code_root = Path(config.indexing.code_index_root) if config.indexing.code_index_root else base / "code"
    runtime_root = (
        Path(config.indexing.runtime_index_root)
        if config.indexing.runtime_index_root
        else base / "runtime"
    )
    return base, code_root, runtime_root


def _latest_subdir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_code_version(
    config: AppConfig, repo_path: Path, override: Optional[str] = None
) -> Optional[str]:
    if override:
        return _sanitize_version(override)
    if config.indexing.code_index_version:
        return _sanitize_version(config.indexing.code_index_version)
    scheme = (config.indexing.versioning_scheme or "git_commit").lower()
    if scheme in {"git_commit", "git", "commit"}:
        repo_state = get_repo_state(repo_path)
        commit = repo_state.get("commit")
        if commit:
            version = commit[:12]
            if repo_state.get("dirty") and config.indexing.include_dirty_suffix:
                version = f"{version}-dirty"
            return _sanitize_version(version)
    if scheme in {"timestamp", "time"}:
        return datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return None


def _resolve_runtime_version(
    config: AppConfig, run_id: str, override: Optional[str] = None
) -> Optional[str]:
    if override:
        return _sanitize_version(override)
    if config.indexing.runtime_index_version:
        return _sanitize_version(config.indexing.runtime_index_version)
    return _sanitize_version(run_id)


def _resolve_repo_config(
    config: AppConfig,
    *,
    repo_name: Optional[str] = None,
    repo_path: Optional[str] = None,
) -> tuple[str, Path, Optional[Path]]:
    if repo_path:
        path = Path(repo_path)
        name = repo_name or path.name
        return name, path, None

    repos = getattr(config.project, "repos", []) or []
    if repo_name and repos:
        for entry in repos:
            if entry.name == repo_name:
                return entry.name, Path(entry.repo_path), entry.compile_commands_dir
    if repo_name and not repos:
        return repo_name, Path(config.project.repo_path), None
    if repo_name and repos:
        return repo_name, Path(config.project.repo_path), None
    if repos:
        entry = repos[0]
        return entry.name, Path(entry.repo_path), entry.compile_commands_dir
    return config.project.name, Path(config.project.repo_path), None


def _index_paths(
    config: AppConfig,
    *,
    repo_path: Optional[Path] = None,
    repo_name: Optional[str] = None,
    run_id: Optional[str] = None,
    code_version: Optional[str] = None,
    runtime_version: Optional[str] = None,
) -> IndexPaths:
    base, code_root, runtime_root = _index_roots(config)
    repo_path = repo_path or Path(config.project.repo_path)
    repo_slug = _slugify(repo_name or repo_path.name or config.project.name)

    resolved_code_version = _resolve_code_version(config, repo_path, code_version)
    if resolved_code_version:
        code_dir = code_root / repo_slug / resolved_code_version
    else:
        code_dir = code_root / repo_slug

    resolved_runtime_version = None
    if run_id:
        resolved_runtime_version = _resolve_runtime_version(config, run_id, runtime_version)
    elif runtime_version:
        resolved_runtime_version = _sanitize_version(runtime_version)
    if resolved_runtime_version:
        runtime_dir = runtime_root / repo_slug / resolved_runtime_version
    else:
        runtime_dir = runtime_root / repo_slug

    # Legacy fallback if no versioned dir exists yet.
    if config.indexing.allow_legacy_paths:
        legacy_code = base / "code"
        if not code_dir.exists() and legacy_code.exists():
            code_dir = legacy_code
        legacy_runtime = base / "runtime"
        if not runtime_dir.exists() and legacy_runtime.exists():
            runtime_dir = legacy_runtime

    return IndexPaths(
        base_dir=base,
        code_root=code_root,
        runtime_root=runtime_root,
        code_dir=code_dir,
        runtime_dir=runtime_dir,
        code_version=resolved_code_version,
        runtime_version=resolved_runtime_version,
    )


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
    version: Optional[str] = None,
    kind: Optional[str] = None,
    index_id: Optional[str] = None,
) -> None:
    metadata_path = _embedding_metadata_path(persist_dir)
    metadata = {
        "model": model,
        "dimension": dimension,
        "index_name": index_name,
        "node_label": node_label,
    }
    if version:
        metadata["version"] = version
    if kind:
        metadata["kind"] = kind
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


def _safe_neo4j_id(value: str, *, max_len: int = 80) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    safe = safe.strip("_")
    if len(safe) > max_len:
        return safe[:max_len]
    return safe or "index"


def _neo4j_index_config(kind: str, project_slug: str, version: Optional[str]) -> Neo4jIndexConfig:
    suffix = _safe_neo4j_id(version or "latest")
    index_name = _safe_neo4j_id(f"{kind}_vector_{project_slug}_{suffix}")
    node_label = _safe_neo4j_id(f"{kind.capitalize()}Chunk_{suffix}")
    return Neo4jIndexConfig(index_name=index_name, node_label=node_label)


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
                ref_doc_id=str(chunk.path),
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
                ref_doc_id=str(summary.path),
                text=summary.text,
                metadata={"type": "file_summary", "path": str(summary.path)},
            )
        )
    for summary in index.relation_summaries:
        nodes.append(
            TextNode(
                ref_doc_id=summary.path,
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


def build_kernel_index(
    config: AppConfig,
    repo_path: Optional[str] = None,
    *,
    repo_name: Optional[str] = None,
    compile_commands_dir: Optional[str] = None,
    index_version: Optional[str] = None,
    incremental: bool = False,
    base_ref: Optional[str] = None,
    incremental_mode: Optional[str] = None,
) -> IndexPaths:
    repo_name_resolved, repo, repo_compile = _resolve_repo_config(
        config,
        repo_name=repo_name,
        repo_path=repo_path,
    )
    compile_dir = (
        Path(compile_commands_dir)
        if compile_commands_dir
        else (repo_compile if repo_compile else config.indexing.clangd.compile_commands_dir)
    )
    paths = _index_paths(
        config,
        repo_path=repo,
        repo_name=repo_name_resolved,
        code_version=index_version,
    )
    paths.code_dir.mkdir(parents=True, exist_ok=True)
    llm, embed = _build_llama_models(config)
    embed_dim = _infer_embedding_dimension(embed)
    project_slug = _slugify(repo_name_resolved)
    neo4j_cfg = _neo4j_index_config("code", project_slug, paths.code_version)
    _reset_neo4j_vector_index(
        config,
        embedding_dimension=embed_dim,
        index_name=neo4j_cfg.index_name,
        node_label=neo4j_cfg.node_label,
    )

    clangd_cfg = LspClangdConfig(
        binary=config.indexing.clangd.binary,
        compile_commands_dir=compile_dir,
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

    changed_files: list[Path] = []
    if incremental:
        base_ref = base_ref or config.indexing.incremental_base_ref
        changed_files = get_changed_files(repo, base_ref=base_ref)
        if config.indexing.incremental_max_changed_files:
            changed_files = changed_files[: config.indexing.incremental_max_changed_files]
        logger.info("Incremental indexing: changed_files=%d", len(changed_files))

    code_index = index_kernel_code(
        repo,
        clangd_config=clangd_cfg if config.indexing.clangd.enabled else None,
        max_files=config.indexing.clangd.max_files,
        files=changed_files if incremental and changed_files else None,
    )
    nodes = _index_to_nodes(code_index)

    if config.indexing.llm_enrich:
        limit = min(config.indexing.llm_enrich_limit, len(nodes))
        for node in nodes[:limit]:
            if node.metadata.get("type") != "code":
                continue
            summary = llm.complete(f"Summarize the following kernel code:\n\n{node.text}")
            node.text = f"// [LLM SUMMARY]\n{summary.text}\n\n{node.text}"

    # Incremental rebuild: merge base nodes (if available) with updated files.
    if incremental and changed_files:
        mode = (incremental_mode or config.indexing.incremental_mode or "rebuild").lower()
        base_version = None
        if base_ref:
            resolved = resolve_git_ref(repo, base_ref)
            if resolved:
                base_version = _sanitize_version(resolved[:12])
            else:
                base_version = _sanitize_version(base_ref)
        base_paths = _index_paths(
            config, repo_path=repo, repo_name=repo_name_resolved, code_version=base_version
        )
        base_dir = base_paths.code_dir
        if base_dir.exists() and mode == "merge" and not config.indexing.neo4j.enabled:
            base_index = _load_index(
                config,
                base_dir,
                embedding_dimension=embed_dim,
                index_name=neo4j_cfg.index_name,
                node_label=neo4j_cfg.node_label,
                embed_model=embed,
            )
            for path in changed_files:
                try:
                    base_index.delete_ref_doc(str((repo / path).resolve()))
                except Exception:
                    logger.debug("Failed to delete ref_doc for %s", path)
            base_index.insert_nodes(nodes)
            base_index.storage_context.persist(persist_dir=str(paths.code_dir))
            _persist_embedding_metadata(
                paths.code_dir,
                model=config.llm.embedding_model,
                dimension=embed_dim,
                index_name=neo4j_cfg.index_name,
                node_label=neo4j_cfg.node_label,
                version=paths.code_version,
                kind="code",
                index_id=base_index.index_id,
            )
            logger.info("Kernel code index merged incrementally: nodes=%d", len(nodes))
            return paths
        if base_dir.exists():
            base_storage = _storage_context(
                config,
                base_dir,
                embedding_dimension=embed_dim,
                index_name=neo4j_cfg.index_name,
                node_label=neo4j_cfg.node_label,
            )
            base_nodes = _load_nodes_from_storage(base_storage)
            excluded_paths = {str((repo / p).resolve()) for p in changed_files}
            base_nodes = _filter_nodes_by_paths(base_nodes, excluded_paths)
            nodes = base_nodes + nodes
            logger.info(
                "Incremental rebuild: base_nodes=%d updated_nodes=%d",
                len(base_nodes),
                len(code_index.chunks),
            )

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
        version=paths.code_version,
        kind="code",
        index_id=vector_index.index_id,
    )
    logger.info("Kernel code index built: nodes=%d", len(nodes))
    return paths


def build_runtime_index(
    config: AppConfig,
    run_id: str,
    *,
    index_version: Optional[str] = None,
    repo_name: Optional[str] = None,
) -> IndexPaths:
    paths = _index_paths(
        config, run_id=run_id, runtime_version=index_version, repo_name=repo_name
    )
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    llm, embed = _build_llama_models(config)
    embed_dim = _infer_embedding_dimension(embed)
    project_slug = _slugify(repo_name or config.project.name)
    neo4j_cfg = _neo4j_index_config("runtime", project_slug, paths.runtime_version)
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
) -> IndexPaths:
    if not run_ids:
        raise ValueError("run_ids is required for runtime aggregate index")
    repo_name_resolved = repo_name or f"aggregate-{group_name}"
    version = index_version or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    paths = _index_paths(
        config,
        repo_name=repo_name_resolved,
        runtime_version=version,
    )
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    llm, embed = _build_llama_models(config)
    embed_dim = _infer_embedding_dimension(embed)
    project_slug = _slugify(repo_name_resolved)
    neo4j_cfg = _neo4j_index_config("runtime", project_slug, paths.runtime_version)
    _reset_neo4j_vector_index(
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


def build_kernel_indexes(
    config: AppConfig,
    *,
    repo_names: Optional[Sequence[str]] = None,
    index_version: Optional[str] = None,
    incremental: bool = False,
    base_ref: Optional[str] = None,
    incremental_mode: Optional[str] = None,
) -> list[IndexPaths]:
    repos = list(getattr(config.project, "repos", []) or [])
    if repo_names:
        repos = [r for r in repos if r.name in set(repo_names)]
    if not repos:
        return [
            build_kernel_index(
                config,
                index_version=index_version,
                incremental=incremental,
                base_ref=base_ref,
                incremental_mode=incremental_mode,
            )
        ]
    results: list[IndexPaths] = []
    for repo in repos:
        results.append(
            build_kernel_index(
                config,
                repo_name=repo.name,
                repo_path=repo.repo_path,
                compile_commands_dir=str(repo.compile_commands_dir)
                if repo.compile_commands_dir
                else None,
                index_version=index_version,
                incremental=incremental,
                base_ref=base_ref,
                incremental_mode=incremental_mode,
            )
        )
    return results


def _load_index(
    config: AppConfig,
    persist_dir: Path,
    *,
    embedding_dimension: int,
    index_name: str,
    node_label: str,
    embed_model: OpenAIEmbeddingLike,
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


def _load_nodes_from_storage(storage: StorageContext) -> list[TextNode]:
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


def _filter_nodes_by_paths(nodes: list[TextNode], excluded_paths: set[str]) -> list[TextNode]:
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


def _infer_version_from_dir(dir_path: Path, root: Path) -> Optional[str]:
    try:
        rel = dir_path.relative_to(root)
    except ValueError:
        return None
    if len(rel.parts) >= 2:
        return rel.parts[-1]
    return None


def _select_existing_dir(primary: Path, root: Path) -> Path:
    if primary.exists():
        return primary
    latest = _latest_subdir(root)
    if latest:
        return latest
    return primary


def _extract_runtime_signals(source_nodes: list, *, max_symbols: int, max_paths: int) -> dict:
    hotspots = []
    metrics = []
    evidence = []
    for source in source_nodes:
        node = getattr(source, "node", source)
        metadata = getattr(node, "metadata", {}) if node else {}
        node_type = metadata.get("type")
        text = getattr(node, "text", "")
        if node_type == "runtime_hotspot":
            hotspots.append(
                {
                    "symbol": metadata.get("symbol"),
                    "score": metadata.get("score"),
                    "file_path": metadata.get("file_path"),
                    "line_start": metadata.get("line_start"),
                    "line_end": metadata.get("line_end"),
                }
            )
        elif node_type == "runtime_metric":
            metrics.append(
                {
                    "metric_name": metadata.get("metric_name"),
                    "value": metadata.get("value"),
                    "unit": metadata.get("unit"),
                }
            )
        elif node_type == "evidence_pack" and text:
            evidence.append(text)
    hotspots.sort(key=lambda x: (x.get("score") or 0), reverse=True)
    symbols = [h.get("symbol") for h in hotspots if h.get("symbol")]
    paths = [h.get("file_path") for h in hotspots if h.get("file_path")]
    return {
        "hotspots": hotspots,
        "metrics": metrics,
        "evidence": evidence,
        "symbols": symbols[:max_symbols],
        "paths": paths[:max_paths],
    }


def _format_runtime_summary(runtime_info: dict) -> str:
    lines = []
    hotspots = runtime_info.get("hotspots", [])
    metrics = runtime_info.get("metrics", [])
    evidence = runtime_info.get("evidence", [])
    if hotspots:
        lines.append("Runtime hotspots:")
        for item in hotspots[:5]:
            lines.append(
                f"- {item.get('symbol')} score={item.get('score')} "
                f"path={item.get('file_path')} lines={item.get('line_start')}-{item.get('line_end')}"
            )
    if metrics:
        lines.append("Runtime metrics:")
        for item in metrics[:5]:
            lines.append(
                f"- {item.get('metric_name')} value={item.get('value')} unit={item.get('unit')}"
            )
    if evidence:
        lines.append("Runtime evidence excerpt:")
        lines.append(evidence[0])
    return "\n".join(lines).strip()


def _build_code_filters(symbols: Sequence[str], paths: Sequence[str]):
    if not MetadataFilters or not ExactMatchFilter or not FilterOperator:
        return None
    filters = []
    for symbol in symbols:
        filters.append(ExactMatchFilter(key="symbol_name", value=symbol))
        filters.append(ExactMatchFilter(key="symbol_qualname", value=symbol))
    for path in paths:
        filters.append(ExactMatchFilter(key="path", value=path))
    if not filters:
        return None
    return MetadataFilters(filters=filters, condition=FilterOperator.OR)


def _format_code_candidates(candidates: list, max_chars: int = 4000) -> str:
    texts: list[str] = []
    total = 0
    for item in candidates:
        node = getattr(item, "node", item)
        text = getattr(node, "text", "")
        if not text:
            continue
        if total + len(text) > max_chars:
            text = text[: max_chars - total]
        texts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n\n".join(texts)


def route_query(
    config: AppConfig,
    query: str,
    mode: str = "auto",
    *,
    code_version: Optional[str] = None,
    runtime_version: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    repo_name_resolved, repo, _ = _resolve_repo_config(config, repo_name=repo_name)
    paths = _index_paths(
        config,
        repo_path=repo,
        repo_name=repo_name_resolved,
        run_id=run_id,
        code_version=code_version,
        runtime_version=runtime_version,
    )
    llm, embed = _build_llama_models(config)
    project_slug = _slugify(repo_name_resolved or repo.name or config.project.name)

    code_root = paths.code_root / project_slug
    runtime_root = paths.runtime_root / project_slug
    code_dir = _select_existing_dir(paths.code_dir, code_root)
    runtime_dir = _select_existing_dir(paths.runtime_dir, runtime_root)
    resolved_code_version = paths.code_version or _infer_version_from_dir(code_dir, code_root)
    resolved_runtime_version = paths.runtime_version or _infer_version_from_dir(
        runtime_dir, runtime_root
    )

    code_index_cfg = _neo4j_index_config("code", project_slug, resolved_code_version)
    runtime_index_cfg = _neo4j_index_config("runtime", project_slug, resolved_runtime_version)
    code_embed_dim = _embedding_dimension_for_query(code_dir, embed)
    runtime_embed_dim = _embedding_dimension_for_query(runtime_dir, embed)

    def _engine(dir_path: Path, *, runtime: bool = False):
        index_cfg = runtime_index_cfg if runtime else code_index_cfg
        embed_dim = runtime_embed_dim if runtime else code_embed_dim
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
        top_k = config.indexing.query_runtime_top_k if runtime else config.indexing.query_code_top_k
        return index.as_query_engine(llm=llm, similarity_top_k=top_k, response_mode="compact"), index

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
            engine, _ = _engine(dir_path, runtime=False)
            return engine
        if config.indexing.neo4j.enabled and not _docstore_has_nodes(storage):
            logger.warning(
                "Docstore is empty for %s; falling back to Neo4j vector query engine.",
                dir_path,
            )
            engine, _ = _engine(dir_path, runtime=False)
            return engine
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

    def _structured_runtime_to_code() -> str:
        if not runtime_dir.exists():
            return "Runtime index not found. Build a runtime index for this run_id first."
        runtime_engine, _ = _engine(runtime_dir, runtime=True)
        runtime_response = runtime_engine.query(query)
        source_nodes = getattr(runtime_response, "source_nodes", []) or []
        runtime_info = _extract_runtime_signals(
            source_nodes,
            max_symbols=config.indexing.query_runtime_symbol_top_k,
            max_paths=config.indexing.query_runtime_path_top_k,
        )
        runtime_summary = _format_runtime_summary(runtime_info)
        filters = _build_code_filters(runtime_info.get("symbols", []), runtime_info.get("paths", []))
        _, code_index = _engine(code_dir, runtime=False)
        if filters:
            retriever = code_index.as_retriever(
                similarity_top_k=config.indexing.query_code_filter_top_k, filters=filters
            )
        else:
            retriever = code_index.as_retriever(similarity_top_k=config.indexing.query_code_filter_top_k)
        candidates = retriever.retrieve(query)
        code_context = _format_code_candidates(candidates)
        combined_query = (
            "You are analyzing kernel performance.\n"
            "Step 1: interpret runtime signals.\n"
            "Step 2: use the candidate code snippets to ground the analysis.\n"
            "Step 3: expand with graph relations (callers/callees/types).\n\n"
            f"Runtime summary:\n{runtime_summary}\n\n"
            f"Candidate code snippets:\n{code_context}\n\n"
            f"Question: {query}"
        )
        return str(_graph_engine(code_dir).query(combined_query))

    if mode == "code":
        return str(_graph_engine(code_dir).query(query))
    if mode == "runtime":
        if not runtime_dir.exists():
            return "Runtime index not found. Build a runtime index for this run_id first."
        runtime_engine, _ = _engine(runtime_dir, runtime=True)
        runtime_response = runtime_engine.query(query)
        response_text = str(runtime_response).strip()
        if response_text:
            return response_text
        source_nodes = getattr(runtime_response, "source_nodes", []) or []
        runtime_info = _extract_runtime_signals(
            source_nodes,
            max_symbols=config.indexing.query_runtime_symbol_top_k,
            max_paths=config.indexing.query_runtime_path_top_k,
        )
        fallback = _format_runtime_summary(runtime_info)
        return fallback or "No runtime results available."
    if mode == "graph":
        return str(_graph_engine(code_dir).query(query))
    if mode == "runtime_code":
        return _structured_runtime_to_code()

    keywords_runtime = ["perf", "trace", "runtime", "framegraph", "instruction", "hotspot"]
    if any(k in query.lower() for k in keywords_runtime) and runtime_dir.exists():
        return _structured_runtime_to_code()
    return str(_graph_engine(code_dir).query(query))
