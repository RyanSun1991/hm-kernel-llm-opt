"""
HTTP MCP server for Sequential Thinking (streamable-http).

Pattern aligns with:
src/hmopt/api/mcp_server.py
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from hmopt.api.seq_mcp_service import build_seq_fastmcp_server, call_seq_tool

from contextlib import asynccontextmanager
import anyio


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


MCP_MOUNT_PATH = _normalize_mount_path(os.getenv("HMOPT_SEQ_MCP_MOUNT_PATH", "/mcp"))
MCP_SERVER_API_KEY = os.getenv("HMOPT_SEQ_MCP_SERVER_API_KEY")
SEQ_TOOL_NAME = os.getenv("HMOPT_SEQ_MCP_TOOL_NAME", "sequential_thinking").strip() or "sequential_thinking"


class _BearerPathMiddleware:
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


app = FastAPI(title="HMOPT Sequential Thinking MCP Server", version="0.1.0")

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
        "tool_name": SEQ_TOOL_NAME,
        "mcp_mount_path": MCP_MOUNT_PATH,
        "mcp_api_key_required": bool(MCP_SERVER_API_KEY),
    }


@app.post("/tools/call")
def tools_call(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Optional legacy-style endpoint (like your kernel server).
    Useful for quick curl testing. MCP clients should use /mcp.
    """
    tool = payload.get("tool")
    arguments = payload.get("arguments") or {}
    if not tool:
        raise HTTPException(status_code=400, detail="tool is required")
    if tool != SEQ_TOOL_NAME:
        raise HTTPException(status_code=404, detail=f"unknown tool: {tool}. expected: {SEQ_TOOL_NAME}")
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="arguments must be an object")

    result = call_seq_tool(arguments)
    return {"result": {"content": result, "tool": tool}}


_fast_mcp = build_seq_fastmcp_server()
# Mount FastMCP at /mcp (not /mcp/mcp) — same trick as your kernel server
_fast_mcp.settings.streamable_http_path = "/"
app.mount(MCP_MOUNT_PATH, _fast_mcp.streamable_http_app())

def _bind_task_group(mcp_server, tg) -> None:
    mgr = getattr(mcp_server, "session_manager", None)
    if mgr is None:
        return
    if hasattr(mgr, "task_group"):
        mgr.task_group = tg
    elif hasattr(mgr, "_task_group"):
        mgr._task_group = tg
    elif hasattr(mgr, "set_task_group"):
        mgr.set_task_group(tg)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with anyio.create_task_group() as tg:
        _bind_task_group(_fast_mcp, tg)
        yield

app = FastAPI(lifespan=lifespan)
# app.router.redirect_slashes = False

app.mount("/mcp", _fast_mcp.streamable_http_app())
