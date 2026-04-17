"""MCP server for HMOPT kernel index retrieval.

This server exposes:
- Standard MCP streamable-http endpoint (for OpenCode / generic MCP clients)
- Legacy /tools/call endpoint (backward compatible with internal MCPToolAgent)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from hmopt.api.mcp_service import (
    build_fastmcp_server,
    call_tool_by_name,
    get_tool_names,
    resolve_mcp_config_path,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = resolve_mcp_config_path()
TOOL_NAMES = get_tool_names()
MCP_MOUNT_PATH = os.getenv("HMOPT_MCP_MOUNT_PATH", "/mcp").strip() or "/mcp"
MCP_SERVER_API_KEY = os.getenv("HMOPT_MCP_SERVER_API_KEY")


def _normalize_mount_path(path: str) -> str:
    cleaned = path.strip()
    if not cleaned:
        return "/mcp"
    if not cleaned.startswith("/"):
        cleaned = "/" + cleaned
    cleaned = cleaned.rstrip("/") or "/"
    if cleaned == "/":
        return "/mcp"
    return cleaned


MCP_MOUNT_PATH = _normalize_mount_path(MCP_MOUNT_PATH)


class _BearerPathMiddleware:
    """Optional bearer auth middleware for MCP endpoints."""

    def __init__(self, app: Any, *, api_key: str, protected_paths: tuple[str, ...]) -> None:
        self._app = app
        self._api_key = api_key
        self._protected_paths = protected_paths

    def _is_protected(self, path: str) -> bool:
        for protected in self._protected_paths:
            if path == protected or path.startswith(f"{protected}/"):
                return True
        return False

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not self._api_key:
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not self._is_protected(path):
            await self._app(scope, receive, send)
            return

        headers = {
            key.decode("latin1").lower(): value.decode("latin1")
            for key, value in scope.get("headers", [])
        }
        authorization = headers.get("authorization")
        expected = f"Bearer {self._api_key}"
        if authorization != expected:
            response = JSONResponse(status_code=401, content={"detail": "unauthorized"})
            await response(scope, receive, send)
            return

        await self._app(scope, receive, send)


app = FastAPI(title="HMOPT MCP Server", version="0.2.0")

if MCP_SERVER_API_KEY:
    app.add_middleware(
        _BearerPathMiddleware,
        api_key=MCP_SERVER_API_KEY,
        protected_paths=("/tools/call", MCP_MOUNT_PATH),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "config_path": CONFIG_PATH,
        "tool_name": TOOL_NAMES["general"],
        "tool_names": TOOL_NAMES,
        "mcp_mount_path": MCP_MOUNT_PATH,
        "mcp_api_key_required": bool(MCP_SERVER_API_KEY),
        "mcp_protocol_enabled": build_fastmcp_server() is not None,
    }


@app.post("/tools/call")
def call_tool(payload: dict[str, Any]) -> dict[str, Any]:
    tool_name = payload.get("tool")
    arguments = payload.get("arguments") or {}
    if arguments and not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="arguments must be an object")
    if not tool_name:
        raise HTTPException(status_code=400, detail="tool is required")
    if tool_name not in TOOL_NAMES.values():
        raise HTTPException(
            status_code=404,
            detail=f"unknown tool: {tool_name}. available tools: {sorted(TOOL_NAMES.values())}",
        )

    try:
        context = call_tool_by_name(str(tool_name), arguments, config_path=CONFIG_PATH)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Legacy tool call failed")
        raise HTTPException(status_code=500, detail=f"tool execution failed: {exc}") from exc

    return {"result": {"content": context, "tool": tool_name}}


_fast_mcp = build_fastmcp_server()
if _fast_mcp is not None:
    # Mount FastMCP at /mcp (not /mcp/mcp).
    _fast_mcp.settings.streamable_http_path = "/"
    app.mount(MCP_MOUNT_PATH, _fast_mcp.streamable_http_app())
