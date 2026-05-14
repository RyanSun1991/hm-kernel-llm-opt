"""Shared MCP tool service for kernel index retrieval."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, Mapping

from hmopt.core.config import AppConfig
from hmopt.indexing.llamaindex_pipeline import (
    fetch_code_snippets,
    retrieve_call_chain,
    retrieve_code_context,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/app.yaml"
DEFAULT_TOOL_NAME = "kernel_index_code"
DEFAULT_GRAPH_TOOL_NAME = "kernel_symbol_graph"
DEFAULT_HOTSPOT_TOOL_NAME = "kernel_hotspot_context"
DEFAULT_CALL_CHAIN_TOOL_NAME = "kernel_call_chain"
DEFAULT_SNIPPETS_TOOL_NAME = "kernel_get_snippets"
DEFAULT_SERVER_NAME = "hmopt-kernel-index"

_MAX_TOP_K = 64
_MAX_SNIPPETS = 32
_MAX_CHARS = 20000
_MIN_CHARS = 256
_MAX_GRAPH_DEPTH = 4
_MAX_FOCUS_SYMBOLS = 20

_CALL_CHAIN_MAX_DEPTH = 6
_CALL_CHAIN_MAX_PER_HOP = 500
_CALL_CHAIN_MAX_FRONTIER = 200
_CALL_CHAIN_MAX_ROOTS = 32
_SNIPPETS_MAX_BATCH = 64
_SNIPPETS_MAX_PER_CHARS = 20000
_SNIPPETS_MAX_TOTAL_CHARS = 200000
_VALID_EDGE_KINDS = {"calls", "uses_type", "uses_macro"}
_VALID_DIRECTIONS = {"both", "callers", "callees"}

SUPPORTED_SCENARIOS = {
    "general",
    "implementation",
    "call_graph",
    "impact_analysis",
    "hotspot_debug",
    "patch_planning",
}

SCENARIO_PRESETS: dict[str, dict[str, Any]] = {
    "general": {
        "top_k": 10,
        "max_snippets": 8,
        "max_chars": 4000,
        "graph_depth": 2,
        "query_suffix": "",
    },
    "implementation": {
        "top_k": 12,
        "max_snippets": 10,
        "max_chars": 5000,
        "graph_depth": 2,
        "query_suffix": (
            "Prioritize full function bodies, file/line location, and key data structure usage."
        ),
    },
    "call_graph": {
        "top_k": 14,
        "max_snippets": 12,
        "max_chars": 5000,
        "graph_depth": 3,
        "query_suffix": (
            "Prioritize caller/callee relationships, transitive call paths, and impact radius."
        ),
    },
    "impact_analysis": {
        "top_k": 14,
        "max_snippets": 10,
        "max_chars": 4500,
        "graph_depth": 3,
        "query_suffix": (
            "Prioritize upstream/downstream dependencies and symbols likely affected by changes."
        ),
    },
    "hotspot_debug": {
        "top_k": 16,
        "max_snippets": 12,
        "max_chars": 6000,
        "graph_depth": 3,
        "query_suffix": (
            "Prioritize hotspot symbol implementations, expensive call paths, and optimization-relevant dependencies."
        ),
    },
    "patch_planning": {
        "top_k": 16,
        "max_snippets": 14,
        "max_chars": 6500,
        "graph_depth": 4,
        "query_suffix": (
            "Prioritize refactoring-safe boundaries, call chains, and symbols that should be modified together."
        ),
    },
}


def resolve_mcp_config_path(config_path: str | None = None) -> str:
    path = config_path or os.getenv("HMOPT_MCP_CONFIG", DEFAULT_CONFIG_PATH)
    path = path.strip()
    return path or DEFAULT_CONFIG_PATH


@lru_cache(maxsize=8)
def load_app_config(config_path: str) -> AppConfig:
    return AppConfig.from_yaml(config_path)


def get_tool_names() -> dict[str, str]:
    general = os.getenv("HMOPT_MCP_TOOL_NAME", DEFAULT_TOOL_NAME).strip() or DEFAULT_TOOL_NAME
    graph = os.getenv("HMOPT_MCP_GRAPH_TOOL_NAME", DEFAULT_GRAPH_TOOL_NAME).strip() or DEFAULT_GRAPH_TOOL_NAME
    hotspot = (
        os.getenv("HMOPT_MCP_HOTSPOT_TOOL_NAME", DEFAULT_HOTSPOT_TOOL_NAME).strip()
        or DEFAULT_HOTSPOT_TOOL_NAME
    )
    call_chain = (
        os.getenv("HMOPT_MCP_CALL_CHAIN_TOOL_NAME", DEFAULT_CALL_CHAIN_TOOL_NAME).strip()
        or DEFAULT_CALL_CHAIN_TOOL_NAME
    )
    snippets = (
        os.getenv("HMOPT_MCP_SNIPPETS_TOOL_NAME", DEFAULT_SNIPPETS_TOOL_NAME).strip()
        or DEFAULT_SNIPPETS_TOOL_NAME
    )
    return {
        "general": general,
        "graph": graph,
        "hotspot": hotspot,
        "call_chain": call_chain,
        "snippets": snippets,
    }


def _coerce_optional_positive_int(
    value: Any,
    field_name: str,
    *,
    max_value: int,
    min_value: int = 1,
) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}")
    return min(parsed, max_value)


def _coerce_scenario(value: Any, *, default: str = "general") -> str:
    scenario = str(value or default).strip().lower()
    if not scenario:
        scenario = default
    if scenario not in SUPPORTED_SCENARIOS:
        supported = ", ".join(sorted(SUPPORTED_SCENARIOS))
        raise ValueError(f"unsupported scenario: {scenario!r}. supported: {supported}")
    return scenario


def _coerce_response_format(value: Any, *, default: str = "markdown") -> str:
    fmt = str(value or default).strip().lower()
    if fmt in ("text", "md"):
        return "markdown"
    if fmt in ("json", "payload"):
        return "json"
    if fmt == "markdown":
        return fmt
    raise ValueError("response_format must be one of: markdown, json")


def _coerce_symbols(value: Any, field_name: str = "symbols") -> list[str]:
    if value is None:
        return []
    values: list[str]
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        values = [str(item).strip() for item in value]
    else:
        raise ValueError(f"{field_name} must be a list of strings or a comma-separated string")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped[:_MAX_FOCUS_SYMBOLS]


def _coerce_tool_inputs(arguments: Mapping[str, Any], *, scenario_default: str) -> dict[str, Any]:
    scenario = _coerce_scenario(arguments.get("scenario"), default=scenario_default)
    preset = SCENARIO_PRESETS[scenario]

    top_k = _coerce_optional_positive_int(arguments.get("top_k"), "top_k", max_value=_MAX_TOP_K)
    max_snippets = _coerce_optional_positive_int(
        arguments.get("max_snippets"),
        "max_snippets",
        max_value=_MAX_SNIPPETS,
    )
    max_chars = _coerce_optional_positive_int(
        arguments.get("max_chars"),
        "max_chars",
        min_value=_MIN_CHARS,
        max_value=_MAX_CHARS,
    )
    graph_depth = _coerce_optional_positive_int(
        arguments.get("graph_depth"),
        "graph_depth",
        max_value=_MAX_GRAPH_DEPTH,
    )
    symbols = _coerce_symbols(arguments.get("symbols"))
    runtime_hints = str(arguments.get("runtime_hints", "")).strip()
    response_format = _coerce_response_format(arguments.get("response_format"), default="markdown")

    return {
        "scenario": scenario,
        "top_k": top_k or int(preset["top_k"]),
        "max_snippets": max_snippets or int(preset["max_snippets"]),
        "max_chars": max_chars or int(preset["max_chars"]),
        "graph_depth": graph_depth or int(preset["graph_depth"]),
        "symbols": symbols,
        "runtime_hints": runtime_hints,
        "response_format": response_format,
    }


def _build_query_for_scenario(
    *,
    scenario: str,
    query: str,
    symbols: list[str],
    runtime_hints: str = "",
) -> str:
    base_query = query.strip()
    if not base_query and symbols:
        base_query = f"Analyze kernel symbols: {', '.join(symbols)}"
    if not base_query:
        raise ValueError("query is required (or provide symbols)")

    preset = SCENARIO_PRESETS.get(scenario, SCENARIO_PRESETS["general"])
    suffix = str(preset.get("query_suffix") or "").strip()
    parts = [base_query]
    if symbols:
        parts.append("Focus symbols: " + ", ".join(symbols))
    if runtime_hints:
        parts.append("Runtime hints: " + runtime_hints)
    if suffix:
        parts.append(suffix)
    return "\n".join(parts).strip()


def _render_markdown_payload(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    scenario = payload.get("scenario", "general")
    query = payload.get("query", "")
    retrieval_mode = payload.get("retrieval_mode", "")
    lines.append(f"# Kernel Context ({scenario})")
    lines.append("")
    lines.append(f"- Query: {query}")
    lines.append(f"- Retrieval mode: {retrieval_mode}")
    lines.append(f"- Vector top_k: {payload.get('vector_top_k')}")
    lines.append(f"- Graph depth: {payload.get('graph_depth')}")

    focus_symbols = payload.get("focus_symbols") or []
    if focus_symbols:
        lines.append(f"- Focus symbols: {', '.join(str(s) for s in focus_symbols)}")

    ranked = payload.get("ranked_symbols") or []
    if ranked:
        lines.append("")
        lines.append("## Ranked Symbols")
        for item in ranked:
            symbol = item.get("symbol")
            score = item.get("score")
            vector_score = item.get("vector_score")
            relation_bonus = item.get("relation_bonus")
            graph_degree = item.get("graph_degree")
            relation_breakdown = item.get("relation_breakdown") or {}
            depth = item.get("depth")
            path = item.get("path")
            start_line = item.get("start_line")
            end_line = item.get("end_line")
            detail = [f"score={score}"]
            if vector_score is not None:
                detail.append(f"vector={vector_score}")
            if relation_bonus:
                detail.append(f"graph_bonus={relation_bonus}")
            if graph_degree:
                detail.append(f"graph_degree={graph_degree}")
            if relation_breakdown:
                rel_text = ",".join(
                    f"{name}:{count}" for name, count in sorted(relation_breakdown.items(), key=lambda x: x[0])
                )
                detail.append(f"relations={rel_text}")
            if depth is not None:
                detail.append(f"depth={depth}")
            if path:
                detail.append(f"path={path}:{start_line}-{end_line}")
            lines.append(f"- {symbol} ({', '.join(detail)})")

    snippets = payload.get("snippets") or []
    if snippets:
        lines.append("")
        lines.append("## Snippets")
        for item in snippets:
            symbol = item.get("symbol")
            path = item.get("path")
            start_line = item.get("start_line")
            end_line = item.get("end_line")
            location = "unknown"
            if path:
                if start_line is not None and end_line is not None:
                    location = f"{path}:{start_line}-{end_line}"
                else:
                    location = str(path)
            lines.append(f"### {symbol} ({location})")
            lines.append("```c")
            lines.append(str(item.get("text") or ""))
            lines.append("```")

    graph_summary = payload.get("graph_summary") or {}
    edges = payload.get("graph_edges") or []
    if graph_summary:
        lines.append("")
        lines.append("## Graph Summary")
        lines.append(f"- Edge count: {graph_summary.get('edge_count', 0)}")
        relation_counts = graph_summary.get("relation_counts") or {}
        if relation_counts:
            rel_text = ", ".join(
                f"{name}={count}" for name, count in sorted(relation_counts.items(), key=lambda x: x[0])
            )
            lines.append(f"- Relation counts: {rel_text}")

    if edges:
        lines.append("")
        lines.append("## Graph Edges")
        for edge in edges:
            lines.append(
                f"- {edge.get('src')} -[{edge.get('rel')}]-> {edge.get('dst')} (depth={edge.get('depth')})"
            )

    return "\n".join(lines).strip()


def retrieve_kernel_index_context(
    query: str,
    *,
    scenario: str = "general",
    top_k: int = 10,
    max_snippets: int = 8,
    max_chars: int = 4000,
    graph_depth: int = 2,
    symbols: list[str] | None = None,
    runtime_hints: str = "",
    response_format: str = "markdown",
    config_path: str | None = None,
) -> str:
    scenario_name = _coerce_scenario(scenario, default="general")
    symbol_hints = _coerce_symbols(symbols or [])
    fmt = _coerce_response_format(response_format, default="markdown")
    query_text = _build_query_for_scenario(
        scenario=scenario_name,
        query=(query or ""),
        symbols=symbol_hints,
        runtime_hints=runtime_hints,
    )

    config_file = resolve_mcp_config_path(config_path)
    config = load_app_config(config_file)
    payload = retrieve_code_context(
        config,
        query_text,
        top_k=top_k,
        max_snippets=max_snippets,
        max_chars=max_chars,
        output_format="payload",
        focus_symbols=symbol_hints,
        graph_depth=graph_depth,
        scenario=scenario_name,
    )
    if not isinstance(payload, dict):
        # Defensive fallback in case an older retrieve_code_context implementation returns text.
        return str(payload)

    payload["scenario"] = scenario_name
    payload["requested_symbols"] = symbol_hints

    if fmt == "json":
        return json.dumps(payload, indent=2, ensure_ascii=False)
    return _render_markdown_payload(payload)


def _call_general_tool(arguments: Mapping[str, Any], *, config_path: str | None = None) -> str:
    parsed = _coerce_tool_inputs(arguments, scenario_default="general")
    return retrieve_kernel_index_context(
        str(arguments.get("query", "")).strip(),
        scenario=parsed["scenario"],
        top_k=parsed["top_k"],
        max_snippets=parsed["max_snippets"],
        max_chars=parsed["max_chars"],
        graph_depth=parsed["graph_depth"],
        symbols=parsed["symbols"],
        runtime_hints=parsed["runtime_hints"],
        response_format=parsed["response_format"],
        config_path=config_path,
    )


def _call_graph_tool(arguments: Mapping[str, Any], *, config_path: str | None = None) -> str:
    normalized = dict(arguments)
    normalized.pop("scenario", None)
    parsed = _coerce_tool_inputs(normalized, scenario_default="call_graph")
    query = str(arguments.get("query", "")).strip()
    return retrieve_kernel_index_context(
        query,
        scenario="call_graph",
        top_k=parsed["top_k"],
        max_snippets=parsed["max_snippets"],
        max_chars=parsed["max_chars"],
        graph_depth=max(parsed["graph_depth"], 2),
        symbols=parsed["symbols"],
        runtime_hints=parsed["runtime_hints"],
        response_format=parsed["response_format"],
        config_path=config_path,
    )


def _call_hotspot_tool(arguments: Mapping[str, Any], *, config_path: str | None = None) -> str:
    normalized = dict(arguments)
    normalized.pop("scenario", None)
    parsed = _coerce_tool_inputs(normalized, scenario_default="hotspot_debug")
    query = str(arguments.get("query", "")).strip()
    return retrieve_kernel_index_context(
        query,
        scenario="hotspot_debug",
        top_k=max(parsed["top_k"], 12),
        max_snippets=max(parsed["max_snippets"], 10),
        max_chars=max(parsed["max_chars"], 5000),
        graph_depth=max(parsed["graph_depth"], 2),
        symbols=parsed["symbols"],
        runtime_hints=parsed["runtime_hints"],
        response_format=parsed["response_format"],
        config_path=config_path,
    )


def _coerce_direction(value: Any) -> str:
    direction = str(value or "both").strip().lower()
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of: {', '.join(sorted(_VALID_DIRECTIONS))}")
    return direction


def _coerce_edge_kinds(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        items = [str(item).strip() for item in value]
    else:
        raise ValueError("edge_kinds must be a list of strings or comma-separated string")
    cleaned: list[str] = []
    for item in items:
        if not item:
            continue
        if item not in _VALID_EDGE_KINDS:
            raise ValueError(
                f"edge_kinds entries must be one of: {', '.join(sorted(_VALID_EDGE_KINDS))}"
            )
        if item not in cleaned:
            cleaned.append(item)
    return cleaned or None


def call_chain_to_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Kernel Call Chain"]
    roots = payload.get("roots") or []
    lines.append(f"- Roots: {', '.join(roots)}")
    lines.append(f"- Direction: {payload.get('direction')}")
    lines.append(f"- Depth requested: {payload.get('depth_requested')}")
    edge_kinds = payload.get("edge_kinds") or []
    if edge_kinds:
        lines.append(f"- Edge kinds: {', '.join(edge_kinds)}")
    stats = payload.get("stats") or {}
    if stats:
        lines.append(
            "- Stats: "
            f"edges={stats.get('total_edges')} "
            f"nodes={stats.get('total_nodes')} "
            f"depth_reached={stats.get('depth_reached')} "
            f"truncated_at={stats.get('hops_truncated_at') or '[]'} "
            f"per_hop={stats.get('per_hop_limit')} "
            f"frontier_cap={stats.get('frontier_cap')}"
        )

    edges = payload.get("edges") or []
    if edges:
        lines.append("")
        lines.append("## Edges")
        for edge in edges:
            suffix = ""
            cs_path = edge.get("call_site_path")
            cs_line = edge.get("call_site_line")
            if cs_path and cs_line is not None:
                cs_col = edge.get("call_site_col")
                if cs_col is not None:
                    suffix = f"  @{cs_path}:{cs_line}:{cs_col}"
                else:
                    suffix = f"  @{cs_path}:{cs_line}"
                is_write = edge.get("is_write")
                if is_write is True:
                    suffix += " [write]"
            lines.append(
                f"- depth={edge.get('depth')} {edge.get('src')} -[{edge.get('rel')}]-> "
                f"{edge.get('dst')} ({edge.get('direction')}){suffix}"
            )

    nodes = payload.get("nodes") or {}
    if nodes:
        lines.append("")
        lines.append("## Nodes")
        for key, meta in nodes.items():
            path = meta.get("path") or "?"
            start = meta.get("start_line")
            end = meta.get("end_line")
            kind = meta.get("kind") or "?"
            depth_v = meta.get("depth")
            lines.append(
                f"- {key} kind={kind} depth={depth_v} path={path}:{start}-{end}"
            )
    return "\n".join(lines).strip()


def snippets_to_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Kernel Snippets"]
    stats = payload.get("stats") or {}
    if stats:
        lines.append(
            "- Stats: "
            f"returned={stats.get('returned')} "
            f"missing={stats.get('missing')} "
            f"truncated={stats.get('truncated_count')} "
            f"total_chars={stats.get('total_chars')} "
            f"budget_hit={stats.get('budget_hit')}"
        )
    for entry in payload.get("snippets") or []:
        path = entry.get("path") or "?"
        start = entry.get("start_line")
        end = entry.get("end_line")
        lines.append("")
        lines.append(f"## {entry.get('symbol')} ({path}:{start}-{end})")
        lines.append(
            f"- chars={entry.get('char_count')} truncated={entry.get('truncated')}"
        )
        lines.append("```c")
        lines.append(str(entry.get("text") or ""))
        lines.append("```")
    missing = payload.get("missing") or []
    if missing:
        lines.append("")
        lines.append("## Missing")
        for entry in missing:
            lines.append(f"- {entry.get('symbol')} ({entry.get('reason')})")
    return "\n".join(lines).strip()


def retrieve_kernel_call_chain(
    symbols: list[str],
    *,
    direction: str = "both",
    depth: int = 3,
    per_hop_limit: int = 100,
    frontier_cap: int = 50,
    edge_kinds: list[str] | None = None,
    response_format: str = "markdown",
    config_path: str | None = None,
) -> str:
    coerced_symbols = _coerce_symbols(symbols, field_name="symbols")
    if not coerced_symbols:
        raise ValueError("symbols is required and must contain at least one entry")
    coerced_symbols = coerced_symbols[:_CALL_CHAIN_MAX_ROOTS]
    direction_norm = _coerce_direction(direction)
    depth_int = _coerce_optional_positive_int(
        depth, "depth", max_value=_CALL_CHAIN_MAX_DEPTH
    ) or 3
    per_hop = _coerce_optional_positive_int(
        per_hop_limit, "per_hop_limit", max_value=_CALL_CHAIN_MAX_PER_HOP
    ) or 100
    cap = _coerce_optional_positive_int(
        frontier_cap, "frontier_cap", max_value=_CALL_CHAIN_MAX_FRONTIER
    ) or 50
    kinds = _coerce_edge_kinds(edge_kinds)
    fmt = _coerce_response_format(response_format, default="markdown")

    config_file = resolve_mcp_config_path(config_path)
    config = load_app_config(config_file)
    payload = retrieve_call_chain(
        config,
        coerced_symbols,
        direction=direction_norm,
        depth=depth_int,
        per_hop_limit=per_hop,
        frontier_cap=cap,
        edge_kinds=kinds,
        output_format="json",
    )
    if not isinstance(payload, dict):
        return str(payload)

    if fmt == "json":
        return json.dumps(payload, indent=2, ensure_ascii=False)
    return call_chain_to_markdown(payload)


def retrieve_kernel_snippets(
    symbols: list[str],
    *,
    per_symbol_max_chars: int = 4000,
    total_max_chars: int | None = None,
    response_format: str = "markdown",
    config_path: str | None = None,
) -> str:
    coerced = _coerce_symbols(symbols, field_name="symbols")
    if not coerced:
        raise ValueError("symbols is required and must contain at least one entry")
    coerced = coerced[:_SNIPPETS_MAX_BATCH]
    per_max = _coerce_optional_positive_int(
        per_symbol_max_chars,
        "per_symbol_max_chars",
        min_value=_MIN_CHARS,
        max_value=_SNIPPETS_MAX_PER_CHARS,
    ) or 4000
    total_max = _coerce_optional_positive_int(
        total_max_chars,
        "total_max_chars",
        min_value=per_max,
        max_value=_SNIPPETS_MAX_TOTAL_CHARS,
    )
    fmt = _coerce_response_format(response_format, default="markdown")

    config_file = resolve_mcp_config_path(config_path)
    config = load_app_config(config_file)
    payload = fetch_code_snippets(
        config,
        coerced,
        per_symbol_max_chars=per_max,
        total_max_chars=total_max,
        output_format="json",
    )
    if not isinstance(payload, dict):
        return str(payload)

    if fmt == "json":
        return json.dumps(payload, indent=2, ensure_ascii=False)
    return snippets_to_markdown(payload)


def _call_call_chain_tool(arguments: Mapping[str, Any], *, config_path: str | None = None) -> str:
    return retrieve_kernel_call_chain(
        symbols=arguments.get("symbols") or [],
        direction=arguments.get("direction", "both"),
        depth=arguments.get("depth", 3),
        per_hop_limit=arguments.get("per_hop_limit", 100),
        frontier_cap=arguments.get("frontier_cap", 50),
        edge_kinds=arguments.get("edge_kinds"),
        response_format=arguments.get("response_format", "markdown"),
        config_path=config_path,
    )


def _call_snippets_tool(arguments: Mapping[str, Any], *, config_path: str | None = None) -> str:
    return retrieve_kernel_snippets(
        symbols=arguments.get("symbols") or [],
        per_symbol_max_chars=arguments.get("per_symbol_max_chars", 4000),
        total_max_chars=arguments.get("total_max_chars"),
        response_format=arguments.get("response_format", "markdown"),
        config_path=config_path,
    )


def call_tool_by_name(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    config_path: str | None = None,
) -> str:
    names = get_tool_names()
    if tool_name == names["general"]:
        return _call_general_tool(arguments, config_path=config_path)
    if tool_name == names["graph"]:
        return _call_graph_tool(arguments, config_path=config_path)
    if tool_name == names["hotspot"]:
        return _call_hotspot_tool(arguments, config_path=config_path)
    if tool_name == names["call_chain"]:
        return _call_call_chain_tool(arguments, config_path=config_path)
    if tool_name == names["snippets"]:
        return _call_snippets_tool(arguments, config_path=config_path)
    raise ValueError(f"unknown tool: {tool_name}")


def call_kernel_tool(arguments: Mapping[str, Any], *, config_path: str | None = None) -> str:
    """Backward-compatible alias for the original single-tool behavior."""
    return _call_general_tool(arguments, config_path=config_path)


@lru_cache(maxsize=1)
def build_fastmcp_server() -> Any | None:
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError:
        logger.warning(
            "mcp package is not installed; standard MCP protocol endpoint is unavailable. "
            "Install with: pip install 'mcp[cli]'"
        )
        return None

    server_name = os.getenv("HMOPT_MCP_SERVER_NAME", DEFAULT_SERVER_NAME).strip() or DEFAULT_SERVER_NAME

    disable_host_check = os.getenv("HMOPT_MCP_DISABLE_HOST_CHECK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    allowed_hosts_raw = os.getenv("HMOPT_MCP_ALLOWED_HOSTS", "").strip()
    transport_security = None
    if disable_host_check:
        from mcp.server.transport_security import TransportSecuritySettings  # type: ignore

        transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    elif allowed_hosts_raw:
        allowed_hosts = [h.strip() for h in allowed_hosts_raw.split(",") if h.strip()]
        if allowed_hosts:
            from mcp.server.transport_security import TransportSecuritySettings  # type: ignore

            transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=allowed_hosts,
            )

    mcp = FastMCP(
        server_name,
        stateless_http=True,
        json_response=True,
        transport_security=transport_security,
    )
    tool_names = get_tool_names()

    @mcp.tool(
        name=tool_names["general"],
        description=(
            "General kernel retrieval tool. Uses vector + Neo4j graph expansion to return "
            "symbol implementation context, dependencies, and ranked code snippets."
        ),
    )
    def kernel_index_code(
        query: str,
        scenario: str = "general",
        symbols: list[str] | None = None,
        runtime_hints: str = "",
        top_k: int | None = None,
        max_snippets: int = 8,
        max_chars: int = 4000,
        graph_depth: int = 2,
        response_format: str = "markdown",
    ) -> str:
        arguments: dict[str, Any] = {
            "query": query,
            "scenario": scenario,
            "symbols": symbols,
            "runtime_hints": runtime_hints,
            "top_k": top_k,
            "max_snippets": max_snippets,
            "max_chars": max_chars,
            "graph_depth": graph_depth,
            "response_format": response_format,
        }
        return _call_general_tool(arguments)

    @mcp.tool(
        name=tool_names["graph"],
        description=(
            "Call-graph-focused retrieval tool. Prioritizes caller/callee chains and "
            "transitive dependencies for impact analysis."
        ),
    )
    def kernel_symbol_graph(
        symbols: list[str],
        query: str = "",
        top_k: int | None = None,
        graph_depth: int = 3,
        max_snippets: int = 12,
        max_chars: int = 5000,
        response_format: str = "markdown",
    ) -> str:
        arguments: dict[str, Any] = {
            "query": query,
            "symbols": symbols,
            "top_k": top_k,
            "graph_depth": graph_depth,
            "max_snippets": max_snippets,
            "max_chars": max_chars,
            "response_format": response_format,
        }
        return _call_graph_tool(arguments)

    @mcp.tool(
        name=tool_names["hotspot"],
        description=(
            "Hotspot-focused retrieval tool. Prioritizes performance-critical code paths, "
            "related symbol implementations, and optimization context."
        ),
    )
    def kernel_hotspot_context(
        symbols: list[str] | None = None,
        query: str = "",
        runtime_hints: str = "",
        top_k: int | None = None,
        graph_depth: int = 3,
        response_format: str = "markdown",
    ) -> str:
        arguments: dict[str, Any] = {
            "query": query,
            "symbols": symbols,
            "runtime_hints": runtime_hints,
            "top_k": top_k,
            "graph_depth": graph_depth,
            "response_format": response_format,
        }
        return _call_hotspot_tool(arguments)

    @mcp.tool(
        name=tool_names["call_chain"],
        description=(
            "Pure structural call-chain retrieval. Returns caller/callee edges with call-site "
            "file:line and per-node metadata (path, lines, kind) — no code bodies. "
            "Use this first to shape the call graph; then call kernel_get_snippets for selected nodes. "
            "Supports depth up to 6, per-hop and frontier caps configurable."
        ),
    )
    def kernel_call_chain(
        symbols: list[str],
        direction: str = "both",
        depth: int = 3,
        per_hop_limit: int = 100,
        frontier_cap: int = 50,
        edge_kinds: list[str] | None = None,
        response_format: str = "markdown",
    ) -> str:
        arguments: dict[str, Any] = {
            "symbols": symbols,
            "direction": direction,
            "depth": depth,
            "per_hop_limit": per_hop_limit,
            "frontier_cap": frontier_cap,
            "edge_kinds": edge_kinds,
            "response_format": response_format,
        }
        return _call_call_chain_tool(arguments)

    @mcp.tool(
        name=tool_names["snippets"],
        description=(
            "Batch fetch function bodies for an explicit symbol list. Use after kernel_call_chain "
            "to retrieve code for selected nodes. Honors per-symbol and total char budgets; "
            "symbols beyond the budget are reported in `missing` with reason=budget_hit."
        ),
    )
    def kernel_get_snippets(
        symbols: list[str],
        per_symbol_max_chars: int = 4000,
        total_max_chars: int | None = None,
        response_format: str = "markdown",
    ) -> str:
        arguments: dict[str, Any] = {
            "symbols": symbols,
            "per_symbol_max_chars": per_symbol_max_chars,
            "total_max_chars": total_max_chars,
            "response_format": response_format,
        }
        return _call_snippets_tool(arguments)

    return mcp
