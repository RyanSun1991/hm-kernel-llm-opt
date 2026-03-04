#!/usr/bin/env bash
set -euo pipefail

HOST="${HMOPT_BUILD_MCP_HOST:-0.0.0.0}"
PORT="${HMOPT_BUILD_MCP_PORT:-7335}"

uvicorn hmopt.api.build_mcp_server:app --host "$HOST" --port "$PORT"
