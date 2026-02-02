"""MCP tool-calling agent for context retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import httpx

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


@dataclass(frozen=True)
class MCPToolAgentResult:
    context: str
    tool_used: bool
    raw_response: dict[str, Any] | None = None


class MCPToolAgent:
    """OpenAI-compatible tool-calling wrapper that delegates to MCP."""

    def __init__(self, config: MCPToolAgentConfig) -> None:
        self._config = config
        self._client = httpx.Client(timeout=config.timeout_sec)

    def fetch_context(self, query: str) -> MCPToolAgentResult:
        if not query:
            return MCPToolAgentResult(context="", tool_used=False)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a retrieval agent. Decide whether to call the MCP tool "
                    "to fetch kernel context. If the tool is not needed, reply with an empty string."
                ),
            },
            {"role": "user", "content": query},
        ]
        tool_spec = {
            "type": "function",
            "function": {
                "name": self._config.tool_name,
                "description": "Fetch kernel index context via MCP.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        }
        response = self._post_chat(messages, tools=[tool_spec], tool_choice="auto")
        tool_calls = _extract_tool_calls(response)
        if not tool_calls:
            content = _extract_message_content(response).strip()
            return MCPToolAgentResult(context=content, tool_used=False, raw_response=response)

        tool_results = []
        for call in tool_calls:
            if call.get("name") != self._config.tool_name:
                continue
            arguments = call.get("arguments", {})
            query_arg = arguments.get("query", query)
            top_k = arguments.get("top_k")
            tool_results.append(
                {
                    "tool_call_id": call.get("id"),
                    "content": self._call_mcp_tool(query_arg, top_k=top_k),
                }
            )

        if not tool_results:
            return MCPToolAgentResult(context="", tool_used=False, raw_response=response)

        messages.append(
            {
                "role": "assistant",
                "tool_calls": [call.get("raw") for call in tool_calls if call.get("raw")],
                "content": "",
            }
        )
        for result in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["content"],
                }
            )

        follow_up = self._post_chat(messages, tools=[tool_spec])
        return MCPToolAgentResult(
            context=_extract_message_content(follow_up).strip(),
            tool_used=True,
            raw_response=follow_up,
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

    def _call_mcp_tool(self, query: str, *, top_k: int | None = None) -> str:
        payload = {
            "tool": self._config.tool_name,
            "arguments": {
                "query": query,
                "top_k": top_k or self._config.top_k,
            },
        }
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
        import json

        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


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
