"""HTTP MCP server for phone auto-test toolset (streamable-http)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException

from hmopt.api.auto_test_mcp_service import (
    DEFAULT_TOOL_NAME,
    auto_test_relay_health_check,
    build_auto_test_fastmcp_server,
    compare_reports,
    compare_reports_async,
    get_async_task_status,
    run_instruction_test,
    run_instruction_test_async,
    run_phone_test,
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


MCP_MOUNT_PATH = _normalize_mount_path(os.getenv("HMOPT_AUTO_TEST_MCP_MOUNT_PATH", "/mcp"))
_fast_mcp = build_auto_test_fastmcp_server()
_TOOL_NAME = os.getenv("HMOPT_AUTO_TEST_TOOL_NAME", DEFAULT_TOOL_NAME).strip() or DEFAULT_TOOL_NAME


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with anyio.create_task_group() as tg:
        if _fast_mcp is not None:
            _bind_task_group(_fast_mcp, tg)
        yield


app = FastAPI(
    title="HMOPT Auto-Test MCP Server",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mcp_mount_path": MCP_MOUNT_PATH,
        "mcp_protocol_enabled": _fast_mcp is not None,
        "tool_name": _TOOL_NAME,
        "instruction_test_tools": sorted(_TOOL_DISPATCH.keys()),
        "hdc_bin": os.getenv("HMOPT_AUTO_TEST_HDC_BIN", "hdc"),
        "default_result_dir": os.getenv("HMOPT_AUTO_TEST_RESULT_DIR", "outputs/phone_test_results"),
        "windows_relay_url": os.getenv("HMOPT_AUTO_TEST_RELAY_URL") or os.getenv("HMOPT_FLASH_RELAY_URL") or "",
        "windows_test_dir": os.getenv("HMOPT_AUTO_TEST_WINDOWS_TEST_DIR", r"D:\modelCase_OH_single"),
    }


_TOOL_DISPATCH: dict[str, Any] = {
    "run_instruction_test": run_instruction_test,
    "run_instruction_test_async": run_instruction_test_async,
    "instruction_test_status": get_async_task_status,
    "async_task_status": get_async_task_status,
    "compare_reports": compare_reports,
    "compare_reports_async": compare_reports_async,
    "auto_test_relay_health": auto_test_relay_health_check,
}


@app.post("/tools/call")
def call_tool(payload: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(payload.get("tool") or "").strip()
    arguments = payload.get("arguments") or {}
    if not tool_name:
        raise HTTPException(status_code=400, detail="tool is required")
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="arguments must be an object")

    if tool_name == _TOOL_NAME:
        handler: Any = run_phone_test
    elif tool_name in _TOOL_DISPATCH:
        handler = _TOOL_DISPATCH[tool_name]
    else:
        available = sorted({_TOOL_NAME, *_TOOL_DISPATCH.keys()})
        raise HTTPException(
            status_code=400,
            detail=f"unknown tool: {tool_name}. available tools: {available}",
        )

    try:
        context = handler(**arguments)
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
