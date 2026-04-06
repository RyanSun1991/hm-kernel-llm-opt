"""HTTP MCP server for device flash toolset (streamable-http)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException

from hmopt.api.flash_mcp_service import (
    build_flash_fastmcp_server,
    flash_and_boot,
    flash_and_boot_async,
    flash_device,
    get_flash_task_status,
    list_fastboot_devices,
    reboot_device,
    relay_health_check,
    wait_for_device_boot,
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


MCP_MOUNT_PATH = _normalize_mount_path(os.getenv("HMOPT_FLASH_MCP_MOUNT_PATH", "/mcp"))
_fast_mcp = build_flash_fastmcp_server()
_SUPPORTED_TOOLS = {
    "flash_device",
    "flash_and_boot",
    "flash_and_boot_async",
    "flash_status",
    "device_reboot",
    "device_wait_boot",
    "relay_health",
    "list_devices",
}

_TOOL_DISPATCH = {
    "flash_device": lambda args: flash_device(**args),
    "flash_and_boot": lambda args: flash_and_boot(**args),
    "flash_and_boot_async": lambda args: flash_and_boot_async(**args),
    "flash_status": lambda args: get_flash_task_status(**args),
    "device_reboot": lambda args: reboot_device(**args),
    "device_wait_boot": lambda args: wait_for_device_boot(**args),
    "relay_health": lambda _args: relay_health_check(),
    "list_devices": lambda _args: list_fastboot_devices(),
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with anyio.create_task_group() as tg:
        if _fast_mcp is not None:
            _bind_task_group(_fast_mcp, tg)
        yield


app = FastAPI(
    title="HMOPT Flash MCP Server",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mcp_mount_path": MCP_MOUNT_PATH,
        "mcp_protocol_enabled": _fast_mcp is not None,
        "relay_url": os.getenv("HMOPT_FLASH_RELAY_URL", ""),
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

    dispatch = _TOOL_DISPATCH.get(tool_name)
    if dispatch is None:
        raise HTTPException(status_code=400, detail=f"no dispatch for tool: {tool_name}")

    try:
        context = dispatch(arguments)
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
