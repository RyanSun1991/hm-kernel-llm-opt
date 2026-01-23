"""Kernel code indexing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from llama_index.core import VectorStoreIndex
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.schema import TextNode

from hmopt.core.config import AppConfig
from hmopt.indexing.clangd_client import ClangdConfig as LspClangdConfig
from hmopt.indexing.clangd_indexer import CodeChunk, CodeIndex, index_kernel_code
from hmopt.llm.models import build_llama_models
from hmopt.storage.embedding_meta import infer_embedding_dimension, persist_embedding_metadata
from hmopt.storage.neo4j import neo4j_index_config, reset_neo4j_vector_index
from hmopt.indexing.paths import index_paths, resolve_repo_config, slugify, sanitize_version
from hmopt.storage.llamaindex import (
    filter_nodes_by_paths,
    load_index,
    load_nodes_from_storage,
    storage_context,
)
from hmopt.tools.git_tools import get_changed_files, resolve_git_ref

logger = logging.getLogger(__name__)


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


def _upsert_clangd_graph(storage, index: CodeIndex, nodes: list[TextNode]) -> None:
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
    repo_name_resolved, repo, repo_compile = resolve_repo_config(
        config,
        repo_name=repo_name,
        repo_path=repo_path,
    )
    compile_dir = (
        Path(compile_commands_dir)
        if compile_commands_dir
        else (repo_compile if repo_compile else config.indexing.clangd.compile_commands_dir)
    )
    paths = index_paths(
        config,
        repo_path=repo,
        repo_name=repo_name_resolved,
        code_version=index_version,
    )
    paths.code_dir.mkdir(parents=True, exist_ok=True)
    llm, embed = build_llama_models(config)
    embed_dim = infer_embedding_dimension(embed)
    project_slug = slugify(repo_name_resolved)
    neo4j_cfg = neo4j_index_config("code", project_slug, paths.code_version)
    reset_neo4j_vector_index(
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

    # Incremental merge for local vector store only.
    if incremental and changed_files:
        mode = (incremental_mode or config.indexing.incremental_mode or "rebuild").lower()
        base_version = None
        if base_ref:
            resolved = resolve_git_ref(repo, base_ref)
            if resolved:
                base_version = sanitize_version(resolved[:12])
            else:
                base_version = sanitize_version(base_ref)
        base_paths = index_paths(
            config, repo_path=repo, repo_name=repo_name_resolved, code_version=base_version
        )
        base_dir = base_paths.code_dir
        if base_dir.exists() and mode == "merge" and not config.indexing.neo4j.enabled:
            base_index = load_index(
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
            persist_embedding_metadata(
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
            base_storage = storage_context(
                config,
                base_dir,
                embedding_dimension=embed_dim,
                index_name=neo4j_cfg.index_name,
                node_label=neo4j_cfg.node_label,
            )
            base_nodes = load_nodes_from_storage(base_storage)
            excluded_paths = {str((repo / p).resolve()) for p in changed_files}
            base_nodes = filter_nodes_by_paths(base_nodes, excluded_paths)
            nodes = base_nodes + nodes
            logger.info(
                "Incremental rebuild: base_nodes=%d updated_nodes=%d",
                len(base_nodes),
                len(code_index.chunks),
            )

    storage = storage_context(
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
    persist_embedding_metadata(
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
