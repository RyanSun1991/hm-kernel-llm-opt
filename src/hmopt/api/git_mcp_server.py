"""HTTP MCP server for Git toolset (streamable-http)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import FastAPI

from hmopt.api.git_mcp_service import build_git_fastmcp_server, get_default_repository


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


def _bind_task_group(mcp_server: Any, tg: anyio.abc.TaskGroup) -> None:
    mgr = getattr(mcp_server, "session_manager", None)
    if mgr is None:
        return
    if hasattr(mgr, "task_group"):
        mgr.task_group = tg
    elif hasattr(mgr, "_task_group"):
        mgr._task_group = tg
    elif hasattr(mgr, "set_task_group"):
        mgr.set_task_group(tg)


MCP_MOUNT_PATH = _normalize_mount_path(os.getenv("HMOPT_GIT_MCP_MOUNT_PATH", "/mcp"))
_fast_mcp = build_git_fastmcp_server()


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with anyio.create_task_group() as tg:
        if _fast_mcp is not None:
            _bind_task_group(_fast_mcp, tg)
        yield


app = FastAPI(
    title="HMOPT Git MCP Server",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mcp_mount_path": MCP_MOUNT_PATH,
        "mcp_protocol_enabled": _fast_mcp is not None,
        "default_repository": get_default_repository(),
    }


if _fast_mcp is not None:
    # Let FastMCP own the protocol path (e.g. /mcp) and mount it at root.
    # This avoids double-prefix/trailing-slash mismatches across FastMCP versions.
    _fast_mcp.settings.streamable_http_path = MCP_MOUNT_PATH
    app.mount("/", _fast_mcp.streamable_http_app())
