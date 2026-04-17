"""HTTP MCP server for kernel build toolset (streamable-http)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException

from hmopt.api.build_mcp_service import (
    build_build_fastmcp_server,
    get_build_task_status,
    trigger_hione_build,
    trigger_hione_build_async,
    trigger_hione_sign,
)


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


MCP_MOUNT_PATH = _normalize_mount_path(os.getenv("HMOPT_BUILD_MCP_MOUNT_PATH", "/mcp"))
_fast_mcp = build_build_fastmcp_server()
_SUPPORTED_TOOLS = {"kernel_build_trigger", "kernel_build_trigger_async", "kernel_build_status", "kernel_sign_trigger"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with anyio.create_task_group() as tg:
        if _fast_mcp is not None:
            _bind_task_group(_fast_mcp, tg)
        yield


app = FastAPI(
    title="HMOPT Build MCP Server",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mcp_mount_path": MCP_MOUNT_PATH,
        "mcp_protocol_enabled": _fast_mcp is not None,
        "mode": os.getenv("HMOPT_BUILD_MCP_MODE", "exec"),
        "runner_container": os.getenv("HMOPT_BUILD_MCP_RUNNER_CONTAINER", ""),
        "runner_image": os.getenv("HMOPT_BUILD_MCP_RUNNER_IMAGE", ""),
        "legacy_tools": sorted(_SUPPORTED_TOOLS),
    }


@app.post("/tools/call")
def call_tool(payload: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(payload.get("tool") or "").strip()
    arguments = payload.get("arguments") or {}
    if not tool_name:
        raise HTTPException(status_code=400, detail="tool is required")
    if tool_name not in _SUPPORTED_TOOLS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown tool: {tool_name}. available tools: {sorted(_SUPPORTED_TOOLS)}",
        )
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="arguments must be an object")

    try:
        if tool_name == "kernel_build_trigger":
            context = trigger_hione_build(**arguments)
        elif tool_name == "kernel_build_trigger_async":
            context = trigger_hione_build_async(**arguments)
        elif tool_name == "kernel_build_status":
            context = get_build_task_status(**arguments)
        else:
            context = trigger_hione_sign(**arguments)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid arguments: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"tool execution failed: {exc}") from exc

    return {"result": {"tool": tool_name, "content": context}}


if _fast_mcp is not None:
    _fast_mcp.settings.streamable_http_path = MCP_MOUNT_PATH
    app.mount("/", _fast_mcp.streamable_http_app())
