"""Unified MCP HTTP transport for HMOPT index retrieval and Evolution.

This server exposes:
- Standard MCP streamable-http endpoint (for OpenCode / generic MCP clients)
- Legacy /tools/call endpoint (backward compatible with internal MCPToolAgent)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
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


def _legacy_tool_content(result: Any, tool: Any, *, text_result: bool) -> Any:
    """Adapt FastMCP's structured/content results without changing legacy index text."""
    structured = None
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        content, structured = result
    elif isinstance(result, dict):
        content, structured = [], result
    elif hasattr(result, "structuredContent"):
        content, structured = result.content, result.structuredContent
    else:
        content = result
    if structured is not None:
        schema = getattr(tool, "outputSchema", None) or {}
        if set(schema.get("properties", {})) == {"result"} and set(structured) == {"result"}:
            return structured["result"]
        return structured
    texts = [item.text for item in content if getattr(item, "type", None) == "text"]
    if text_result:
        return "\n".join(texts)
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            return texts[0]
    return [item.model_dump(mode="json") for item in content]


def create_app(
    server: Any | None,
    *,
    config_path: str = CONFIG_PATH,
    tool_names: dict[str, str] | None = None,
    mount_path: str = MCP_MOUNT_PATH,
    api_key: str | None = MCP_SERVER_API_KEY,
) -> FastAPI:
    """Bind HTTP, legacy calls and health to one startup-frozen MCP registry."""
    names = dict(TOOL_NAMES if tool_names is None else tool_names)
    mounted_path = _normalize_mount_path(mount_path)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # Mounted Starlette/FastAPI applications do not run their own lifespan.
        # Starting this manager in the parent is required before any MCP request.
        if server is None:
            yield
        else:
            async with server.session_manager.run():
                yield

    application = FastAPI(title="HMOPT MCP Server", version="0.2.0", lifespan=lifespan)
    application.state.mcp_server = server
    binding = getattr(server, "evolution_approval_binding", None)
    if binding and binding[1] is not None:
        from hmopt.api.evolution_approval import approval_router

        application.include_router(approval_router(*binding))
    if api_key:
        application.add_middleware(
            _BearerPathMiddleware,
            api_key=api_key,
            protected_paths=("/tools/call", mounted_path),
        )

    @application.get("/health")
    async def health() -> dict[str, Any]:
        tools = await server.list_tools() if server is not None else []
        return {
            "status": "ok",
            "config_path": config_path,
            "tool_name": names["general"],
            "tool_names": names,
            "tools": [tool.name for tool in tools] if server is not None else list(names.values()),
            "mcp_mount_path": mounted_path,
            "mcp_api_key_required": bool(api_key),
            "mcp_protocol_enabled": server is not None,
        }

    @application.post("/tools/call")
    async def call_tool(payload: dict[str, Any]) -> dict[str, Any]:
        tool_name = payload.get("tool")
        arguments = payload.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise HTTPException(status_code=400, detail="arguments must be an object")
        if not isinstance(tool_name, str) or not tool_name:
            raise HTTPException(status_code=400, detail="tool must be a nonempty string")
        registered = await server.list_tools() if server is not None else []
        available = (
            [tool.name for tool in registered] if server is not None else list(names.values())
        )
        if tool_name not in available:
            raise HTTPException(
                status_code=404,
                detail=f"unknown tool: {tool_name}. available tools: {sorted(available)}",
            )
        try:
            if server is None:
                # Preserve the optional-SDK legacy fallback for index-only installs.
                context = call_tool_by_name(tool_name, arguments, config_path=config_path)
            else:
                from mcp.server.fastmcp.exceptions import ToolError

                try:
                    result = await server.call_tool(tool_name, arguments)
                except ToolError as exc:
                    if isinstance(exc.__cause__, ValueError):
                        # Unwrap a tool's existing validation error, not a type check.
                        raise ValueError(str(exc)) from exc  # noqa: TRY004
                    raise
                tool = next(tool for tool in registered if tool.name == tool_name)
                context = _legacy_tool_content(
                    result, tool, text_result=tool_name in names.values()
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Legacy tool call failed")
            raise HTTPException(status_code=500, detail=f"tool execution failed: {exc}") from exc
        return {"result": {"content": context, "tool": tool_name}}

    if server is not None:
        # Mount FastMCP at /mcp (not /mcp/mcp). This also creates the manager
        # before the parent lifespan enters it.
        server.settings.streamable_http_path = "/"
        application.mount(mounted_path, server.streamable_http_app())
    return application


_fast_mcp = build_fastmcp_server()
app = create_app(_fast_mcp)
