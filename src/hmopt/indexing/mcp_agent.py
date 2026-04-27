"""MCP tool agent for hybrid context retrieval (forced + LLM tool calls)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

_SUPPORTED_SCENARIOS = {
    "general",
    "implementation",
    "call_graph",
    "impact_analysis",
    "hotspot_debug",
    "patch_planning",
}


@dataclass(frozen=True)
class MCPToolAgentConfig:
    base_url: str
    api_key: str | None
    model: str
    timeout_sec: int
    tool_name: str
    top_k: int
    mcp_base_url: str
    mcp_api_key: str | None = None
    graph_tool_name: str = "kernel_symbol_graph"
    hotspot_tool_name: str = "kernel_hotspot_context"
    call_chain_tool_name: str = "kernel_call_chain"
    snippets_tool_name: str = "kernel_get_snippets"
    default_scenario: str = "general"
    graph_depth: int = 2
    max_snippets: int = 8
    max_chars: int = 4000
    response_format: str = "markdown"


@dataclass(frozen=True)
class MCPToolAgentResult:
    context: str
    tool_used: bool
    raw_response: dict[str, Any] | None = None


class MCPToolAgent:
    """Hybrid MCP retrieval wrapper (forced first pass + LLM tool calls)."""

    def __init__(self, config: MCPToolAgentConfig) -> None:
        self._config = config
        self._client = httpx.Client(timeout=config.timeout_sec)

    def fetch_context(
        self,
        query: str,
        *,
        symbols: list[str] | None = None,
        runtime_hints: str = "",
        scenario: str | None = None,
    ) -> MCPToolAgentResult:
        normalized_symbols = _normalize_symbols(symbols)
        if not query and not normalized_symbols:
            return MCPToolAgentResult(context="", tool_used=False)

        scenario_name = self._resolve_scenario(
            scenario_override=scenario,
            query=query,
            runtime_hints=runtime_hints,
        )
        initial_tool_name, initial_arguments = self._build_initial_tool_call(
            scenario=scenario_name,
            query=query,
            symbols=normalized_symbols,
            runtime_hints=runtime_hints,
        )
        initial_context = self._call_mcp_tool(initial_tool_name, initial_arguments)
        tool_specs = self._build_tool_specs()

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a kernel retrieval assistant. Prefer MCP tool calls for exact symbol "
                    "implementations, dependency edges, and impact-aware context."
                ),
            },
            {
                "role": "user",
                "content": query
                or ("Analyze kernel symbols: " + ", ".join(normalized_symbols)),
            },
        ]
        if initial_context:
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Initial MCP context:\n{initial_context}",
                }
            )

        tool_outputs: list[dict[str, Any]] = []
        final_text = ""
        max_rounds = 20
        for _ in range(max_rounds):
            response = self._post_chat(messages, tools=tool_specs, tool_choice="auto")
            tool_calls = _extract_tool_calls(response)
            if not tool_calls:
                final_text = _extract_message_content(response).strip()
                break
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [call.get("raw") for call in tool_calls if call.get("raw")],
                    "content": "",
                }
            )
            for call in tool_calls:
                tool_name = str(call.get("name") or "").strip()
                if tool_name not in self._allowed_tool_names():
                    continue
                arguments = self._normalize_tool_arguments(
                    tool_name=tool_name,
                    raw_arguments=call.get("arguments", {}),
                    default_query=query,
                    default_symbols=normalized_symbols,
                    default_runtime_hints=runtime_hints,
                    default_scenario=scenario_name,
                )
                tool_content = self._call_mcp_tool(tool_name, arguments)
                tool_outputs.append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "content": tool_content,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": tool_content,
                    }
                )

        parts = []
        if initial_context:
            parts.append(
                "[MCP initial retrieval]\n"
                f"tool={initial_tool_name}\n"
                f"{initial_context.strip()}"
            )
        if tool_outputs:
            follow_up_blocks = []
            for output in tool_outputs:
                follow_up_blocks.append(
                    f"tool={output['tool']}\n{str(output.get('content') or '').strip()}"
                )
            parts.append("[MCP follow-up retrievals]\n" + "\n\n".join(follow_up_blocks))
        if final_text:
            parts.append("[MCP agent summary]\n" + final_text)
        context = "\n\n".join(parts).strip()
        return MCPToolAgentResult(
            context=context,
            tool_used=True,
            raw_response={
                "scenario": scenario_name,
                "initial_tool": initial_tool_name,
                "initial_arguments": initial_arguments,
                "initial_context": initial_context,
                "tool_outputs": tool_outputs,
            },
        )

    def _post_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        payload = {
            "model": self._config.model,
            "messages": messages,
            **kwargs,
        }
        response = self._client.post(
            f"{self._config.base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    def _build_tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": self._config.tool_name,
                    "description": (
                        "General kernel retrieval using vector + graph context. "
                        "Supports scenario-specific retrieval."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "scenario": {
                                "type": "string",
                                "enum": sorted(_SUPPORTED_SCENARIOS),
                            },
                            "symbols": {"type": "array", "items": {"type": "string"}},
                            "runtime_hints": {"type": "string"},
                            "top_k": {"type": "integer"},
                            "graph_depth": {"type": "integer"},
                            "max_snippets": {"type": "integer"},
                            "max_chars": {"type": "integer"},
                            "response_format": {"type": "string", "enum": ["markdown", "json"]},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": self._config.graph_tool_name,
                    "description": "Graph-focused retrieval for caller/callee dependencies.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbols": {"type": "array", "items": {"type": "string"}},
                            "query": {"type": "string"},
                            "top_k": {"type": "integer"},
                            "graph_depth": {"type": "integer"},
                            "max_snippets": {"type": "integer"},
                            "max_chars": {"type": "integer"},
                            "response_format": {"type": "string", "enum": ["markdown", "json"]},
                        },
                        "required": ["symbols"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": self._config.hotspot_tool_name,
                    "description": "Hotspot-focused retrieval for performance optimization.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbols": {"type": "array", "items": {"type": "string"}},
                            "query": {"type": "string"},
                            "runtime_hints": {"type": "string"},
                            "top_k": {"type": "integer"},
                            "graph_depth": {"type": "integer"},
                            "response_format": {"type": "string", "enum": ["markdown", "json"]},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": self._config.call_chain_tool_name,
                    "description": (
                        "Pure structural call-chain retrieval. Returns caller/callee edges with "
                        "call-site file:line and per-node metadata — no code bodies. Use this first "
                        "to shape the graph; depth up to 6."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbols": {"type": "array", "items": {"type": "string"}},
                            "direction": {"type": "string", "enum": ["both", "callers", "callees"]},
                            "depth": {"type": "integer"},
                            "per_hop_limit": {"type": "integer"},
                            "frontier_cap": {"type": "integer"},
                            "edge_kinds": {"type": "array", "items": {"type": "string"}},
                            "response_format": {"type": "string", "enum": ["markdown", "json"]},
                        },
                        "required": ["symbols"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": self._config.snippets_tool_name,
                    "description": (
                        "Batch fetch function bodies for an explicit symbol list. Use after "
                        "kernel_call_chain to retrieve code for selected nodes. Honors per-symbol "
                        "and total char budgets."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbols": {"type": "array", "items": {"type": "string"}},
                            "per_symbol_max_chars": {"type": "integer"},
                            "total_max_chars": {"type": "integer"},
                            "response_format": {"type": "string", "enum": ["markdown", "json"]},
                        },
                        "required": ["symbols"],
                    },
                },
            },
        ]

    def _allowed_tool_names(self) -> set[str]:
        return {
            self._config.tool_name,
            self._config.graph_tool_name,
            self._config.hotspot_tool_name,
            self._config.call_chain_tool_name,
            self._config.snippets_tool_name,
        }

    def _resolve_scenario(
        self,
        *,
        scenario_override: str | None,
        query: str,
        runtime_hints: str,
    ) -> str:
        if scenario_override:
            normalized_override = scenario_override.strip().lower()
            if normalized_override in _SUPPORTED_SCENARIOS:
                return normalized_override
        inferred = _infer_scenario(query, runtime_hints)
        if inferred:
            return inferred
        default_scenario = (self._config.default_scenario or "general").strip().lower()
        if default_scenario not in _SUPPORTED_SCENARIOS:
            return "general"
        return default_scenario

    def _build_initial_tool_call(
        self,
        *,
        scenario: str,
        query: str,
        symbols: list[str],
        runtime_hints: str,
    ) -> tuple[str, dict[str, Any]]:
        tool_name = _tool_name_for_scenario(
            scenario=scenario,
            general_tool=self._config.tool_name,
            graph_tool=self._config.graph_tool_name,
            hotspot_tool=self._config.hotspot_tool_name,
        )
        arguments = self._normalize_tool_arguments(
            tool_name=tool_name,
            raw_arguments={
                "query": query,
                "scenario": scenario,
                "symbols": symbols,
                "runtime_hints": runtime_hints,
                "top_k": self._config.top_k,
                "graph_depth": self._config.graph_depth,
                "max_snippets": self._config.max_snippets,
                "max_chars": self._config.max_chars,
                "response_format": self._config.response_format,
            },
            default_query=query,
            default_symbols=symbols,
            default_runtime_hints=runtime_hints,
            default_scenario=scenario,
        )
        return tool_name, arguments

    def _normalize_tool_arguments(
        self,
        *,
        tool_name: str,
        raw_arguments: Mapping[str, Any],
        default_query: str,
        default_symbols: list[str],
        default_runtime_hints: str,
        default_scenario: str,
    ) -> dict[str, Any]:
        base_query = str(raw_arguments.get("query", default_query) or "").strip()
        symbols = _normalize_symbols(raw_arguments.get("symbols", default_symbols))
        runtime_hints = str(
            raw_arguments.get("runtime_hints", default_runtime_hints) or ""
        ).strip()
        response_format = str(
            raw_arguments.get("response_format", self._config.response_format) or "markdown"
        ).strip()
        if response_format not in {"markdown", "json"}:
            response_format = "markdown"

        if tool_name == self._config.graph_tool_name:
            if not symbols:
                symbols = _extract_symbol_candidates(base_query)
            arguments: dict[str, Any] = {
                "symbols": symbols,
                "query": base_query,
                "top_k": _coerce_optional_int(raw_arguments.get("top_k"), fallback=self._config.top_k),
                "graph_depth": _coerce_optional_int(
                    raw_arguments.get("graph_depth"),
                    fallback=self._config.graph_depth,
                ),
                "max_snippets": _coerce_optional_int(
                    raw_arguments.get("max_snippets"),
                    fallback=self._config.max_snippets,
                ),
                "max_chars": _coerce_optional_int(
                    raw_arguments.get("max_chars"),
                    fallback=self._config.max_chars,
                ),
                "response_format": response_format,
            }
            return {k: v for k, v in arguments.items() if v is not None}

        if tool_name == self._config.hotspot_tool_name:
            arguments = {
                "symbols": symbols or None,
                "query": base_query,
                "runtime_hints": runtime_hints,
                "top_k": _coerce_optional_int(raw_arguments.get("top_k"), fallback=self._config.top_k),
                "graph_depth": _coerce_optional_int(
                    raw_arguments.get("graph_depth"),
                    fallback=self._config.graph_depth,
                ),
                "response_format": response_format,
            }
            return {k: v for k, v in arguments.items() if v is not None and v != ""}

        if tool_name == self._config.call_chain_tool_name:
            if not symbols:
                symbols = _extract_symbol_candidates(base_query)
            direction_raw = str(raw_arguments.get("direction") or "both").strip().lower()
            if direction_raw not in {"both", "callers", "callees"}:
                direction_raw = "both"
            edge_kinds_raw = raw_arguments.get("edge_kinds")
            if isinstance(edge_kinds_raw, str):
                edge_kinds = [item.strip() for item in edge_kinds_raw.split(",") if item.strip()]
            elif isinstance(edge_kinds_raw, list):
                edge_kinds = [str(item).strip() for item in edge_kinds_raw if str(item).strip()]
            else:
                edge_kinds = None
            arguments = {
                "symbols": symbols,
                "direction": direction_raw,
                "depth": _coerce_optional_int(raw_arguments.get("depth"), fallback=3),
                "per_hop_limit": _coerce_optional_int(
                    raw_arguments.get("per_hop_limit"), fallback=100
                ),
                "frontier_cap": _coerce_optional_int(
                    raw_arguments.get("frontier_cap"), fallback=50
                ),
                "edge_kinds": edge_kinds,
                "response_format": response_format,
            }
            return {k: v for k, v in arguments.items() if v is not None}

        if tool_name == self._config.snippets_tool_name:
            if not symbols:
                symbols = _extract_symbol_candidates(base_query)
            arguments = {
                "symbols": symbols,
                "per_symbol_max_chars": _coerce_optional_int(
                    raw_arguments.get("per_symbol_max_chars"),
                    fallback=self._config.max_chars,
                ),
                "total_max_chars": _coerce_optional_int(
                    raw_arguments.get("total_max_chars"), fallback=None
                ),
                "response_format": response_format,
            }
            return {k: v for k, v in arguments.items() if v is not None}

        scenario = str(raw_arguments.get("scenario", default_scenario) or "general").strip().lower()
        if scenario not in _SUPPORTED_SCENARIOS:
            scenario = default_scenario
        arguments = {
            "query": base_query,
            "scenario": scenario,
            "symbols": symbols or None,
            "runtime_hints": runtime_hints,
            "top_k": _coerce_optional_int(raw_arguments.get("top_k"), fallback=self._config.top_k),
            "graph_depth": _coerce_optional_int(
                raw_arguments.get("graph_depth"),
                fallback=self._config.graph_depth,
            ),
            "max_snippets": _coerce_optional_int(
                raw_arguments.get("max_snippets"),
                fallback=self._config.max_snippets,
            ),
            "max_chars": _coerce_optional_int(
                raw_arguments.get("max_chars"),
                fallback=self._config.max_chars,
            ),
            "response_format": response_format,
        }
        return {k: v for k, v in arguments.items() if v is not None and not (k == "query" and v == "")}

    def _call_mcp_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> str:
        payload = {"tool": tool_name, "arguments": dict(arguments)}
        headers = {"Content-Type": "application/json"}
        if self._config.mcp_api_key:
            headers["Authorization"] = f"Bearer {self._config.mcp_api_key}"
        response = self._client.post(
            f"{self._config.mcp_base_url.rstrip('/')}/tools/call",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return _stringify_result(response.json()).strip()


def _tool_name_for_scenario(
    *,
    scenario: str,
    general_tool: str,
    graph_tool: str,
    hotspot_tool: str,
) -> str:
    if scenario in {"call_graph", "impact_analysis"}:
        return graph_tool
    if scenario == "hotspot_debug":
        return hotspot_tool
    return general_tool


def _infer_scenario(query: str, runtime_hints: str = "") -> str | None:
    text = f"{query} {runtime_hints}".lower()
    if any(token in text for token in ("hotspot", "perf", "latency", "flamegraph", "cpu")):
        return "hotspot_debug"
    if any(token in text for token in ("call graph", "caller", "callee", "dependency chain")):
        return "call_graph"
    if any(token in text for token in ("impact", "upstream", "downstream", "blast radius")):
        return "impact_analysis"
    if any(token in text for token in ("patch", "refactor", "modify together")):
        return "patch_planning"
    if any(token in text for token in ("implementation", "function body", "definition")):
        return "implementation"
    return None


def _coerce_optional_int(value: Any, *, fallback: int | None = None) -> int | None:
    if value is None:
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _normalize_symbols(raw: Any) -> list[str]:
    if raw is None:
        return []
    values: list[str]
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",")]
    elif isinstance(raw, list):
        values = [str(item).strip() for item in raw]
    else:
        return []
    deduped: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped[:20]


def _extract_symbol_candidates(text: str, *, limit: int = 6) -> list[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or "")
    candidates: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        candidates.append(token)
        if len(candidates) >= limit:
            break
    return candidates


def _stringify_result(data: Any) -> str:
    result = data.get("result") if isinstance(data, Mapping) else None
    payload = result if result is not None else data
    if isinstance(payload, Mapping):
        for key in ("content", "text", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        items = payload.get("items")
        if isinstance(items, list):
            return "\n".join(_stringify_result(item) for item in items if item)
    if isinstance(payload, list):
        return "\n".join(_stringify_result(item) for item in payload if item)
    if payload is None:
        return ""
    return str(payload)


def _extract_message_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {}) if isinstance(choices[0], Mapping) else {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _extract_tool_calls(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    choices = response.get("choices", [])
    if not choices:
        return []
    message = choices[0].get("message", {}) if isinstance(choices[0], Mapping) else {}
    tool_calls = message.get("tool_calls") if isinstance(message, Mapping) else None
    if not isinstance(tool_calls, list):
        return []
    parsed_calls = []
    for call in tool_calls:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function", {})
        if not isinstance(function, Mapping):
            continue
        arguments = function.get("arguments")
        parsed_args = _safe_json(arguments)
        parsed_calls.append(
            {
                "id": call.get("id"),
                "name": function.get("name"),
                "arguments": parsed_args or {},
                "raw": call,
            }
        )
    return parsed_calls


def _safe_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
