"""HTTP MCP server for the team Skill Hub bridge (streamable-http).

Mirrors git_mcp_server.py. Exposes skillhub_resolve / skillhub_sediment /
skillhub_status so OpenCode agents in a kernel repo reach the hub read/write
path over MCP instead of the `hmopt` CLI.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import FastAPI

from hmopt.api.skillhub_mcp_service import build_skillhub_fastmcp_server


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


MCP_MOUNT_PATH = _normalize_mount_path(os.getenv("HMOPT_SKILLHUB_MCP_MOUNT_PATH", "/mcp"))
_fast_mcp = build_skillhub_fastmcp_server()


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with anyio.create_task_group() as tg:
        if _fast_mcp is not None:
            _bind_task_group(_fast_mcp, tg)
        yield


app = FastAPI(
    title="HMOPT Skill-Hub MCP Server",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mcp_mount_path": MCP_MOUNT_PATH,
        "mcp_protocol_enabled": _fast_mcp is not None,
    }


if _fast_mcp is not None:
    _fast_mcp.settings.streamable_http_path = MCP_MOUNT_PATH
    app.mount("/", _fast_mcp.streamable_http_app())
