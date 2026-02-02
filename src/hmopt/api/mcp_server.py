"""MCP server implementation for tool calls."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException

from hmopt.core.config import AppConfig
from hmopt.indexing.llamaindex_pipeline import retrieve_code_context


CONFIG_PATH = os.getenv("HMOPT_MCP_CONFIG", "configs/app.yaml")
APP_CONFIG = AppConfig.from_yaml(CONFIG_PATH)

app = FastAPI(title="HMOPT MCP Server", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/tools/call")
def call_tool(payload: dict[str, Any]) -> dict[str, Any]:
    tool_name = payload.get("tool")
    arguments = payload.get("arguments") or {}
    if not tool_name:
        raise HTTPException(status_code=400, detail="tool is required")
    if tool_name != "kernel_index_code":
        raise HTTPException(status_code=404, detail=f"unknown tool: {tool_name}")
    query = arguments.get("query", "")
    top_k = arguments.get("top_k")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    context = retrieve_code_context(APP_CONFIG, query, top_k=top_k)
    return {"result": {"content": context}}
