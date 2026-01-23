"""Query routing for code/runtime indexes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from llama_index.core import PropertyGraphIndex

from hmopt.core.config import AppConfig
from hmopt.llm.models import build_llama_models
from hmopt.storage.embedding_meta import (
    embedding_dimension_for_query,
    ensure_embedding_compat,
)
from hmopt.storage.neo4j import neo4j_index_config
from hmopt.indexing.paths import (
    index_paths,
    infer_version_from_dir,
    resolve_repo_config,
    select_existing_dir,
    slugify,
)
from hmopt.storage.llamaindex import docstore_has_nodes, load_index, storage_context
from hmopt.prompting import build_prompt_registry, resolve_prompt_name

logger = logging.getLogger(__name__)

try:
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, FilterOperator
except Exception:  # pragma: no cover - optional dependency
    MetadataFilters = None
    ExactMatchFilter = None
    FilterOperator = None


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
    repo_name: Optional[str] = None,
) -> str:
    repo_name_resolved, repo, _ = resolve_repo_config(config, repo_name=repo_name)
    paths = index_paths(
        config,
        repo_path=repo,
        repo_name=repo_name_resolved,
        run_id=run_id,
        code_version=code_version,
        runtime_version=runtime_version,
    )
    llm, embed = build_llama_models(config)
    project_slug = slugify(repo_name_resolved or repo.name or config.project.name)
    registry = build_prompt_registry(config)

    def _render_prompt(prompt_name: str, **kwargs: object) -> str | None:
        try:
            system_prompt, prompt = registry.render(prompt_name, **kwargs)
        except Exception as exc:
            logger.warning("Query prompt render failed for %s: %s", prompt_name, exc)
            return None
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt

    code_root = paths.code_root / project_slug
    runtime_root = paths.runtime_root / project_slug
    code_dir = select_existing_dir(paths.code_dir, code_root)
    runtime_dir = select_existing_dir(paths.runtime_dir, runtime_root)
    resolved_code_version = paths.code_version or infer_version_from_dir(code_dir, code_root)
    resolved_runtime_version = paths.runtime_version or infer_version_from_dir(runtime_dir, runtime_root)

    code_index_cfg = neo4j_index_config("code", project_slug, resolved_code_version)
    runtime_index_cfg = neo4j_index_config("runtime", project_slug, resolved_runtime_version)
    code_embed_dim = embedding_dimension_for_query(code_dir, embed)
    runtime_embed_dim = embedding_dimension_for_query(runtime_dir, embed)

    def _engine(dir_path: Path, *, runtime: bool = False):
        index_cfg = runtime_index_cfg if runtime else code_index_cfg
        embed_dim = runtime_embed_dim if runtime else code_embed_dim
        ensure_embedding_compat(
            dir_path,
            embed_model=config.llm.embedding_model,
            embed_dim=embed_dim,
            index_name=index_cfg.index_name,
            node_label=index_cfg.node_label,
        )
        index = load_index(
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
        ensure_embedding_compat(
            dir_path,
            embed_model=config.llm.embedding_model,
            embed_dim=embed_dim,
            index_name=index_cfg.index_name,
            node_label=index_cfg.node_label,
        )
        storage = storage_context(
            config,
            dir_path,
            embedding_dimension=embed_dim,
            index_name=index_cfg.index_name,
            node_label=index_cfg.node_label,
        )
        if not storage.property_graph_store:
            engine, _ = _engine(dir_path, runtime=False)
            return engine
        if config.indexing.neo4j.enabled and not docstore_has_nodes(storage):
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
        prompt_name = resolve_prompt_name(config, "query_runtime_code", "query_runtime_code")
        rendered = _render_prompt(
            prompt_name,
            runtime_summary=runtime_summary,
            code_context=code_context,
            query=query,
        )
        combined_query = rendered or (
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
        prompt_name = resolve_prompt_name(config, "query_code", "query_code")
        rendered = _render_prompt(prompt_name, query=query)
        return str(_graph_engine(code_dir).query(rendered or query))
    if mode == "runtime":
        if not runtime_dir.exists():
            return "Runtime index not found. Build a runtime index for this run_id first."
        runtime_engine, _ = _engine(runtime_dir, runtime=True)
        prompt_name = resolve_prompt_name(config, "query_runtime", "query_runtime")
        rendered = _render_prompt(prompt_name, query=query)
        runtime_response = runtime_engine.query(rendered or query)
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
        prompt_name = resolve_prompt_name(config, "query_graph", "query_graph")
        rendered = _render_prompt(prompt_name, query=query)
        return str(_graph_engine(code_dir).query(rendered or query))
    if mode == "runtime_code":
        return _structured_runtime_to_code()

    keywords_runtime = ["perf", "trace", "runtime", "framegraph", "instruction", "hotspot"]
    if any(k in query.lower() for k in keywords_runtime) and runtime_dir.exists():
        return _structured_runtime_to_code()
    prompt_name = resolve_prompt_name(config, "query_graph", "query_graph")
    rendered = _render_prompt(prompt_name, query=query)
    return str(_graph_engine(code_dir).query(rendered or query))
