#!/usr/bin/env bash
set -euo pipefail

HOST="${HMOPT_GIT_MCP_HOST:-0.0.0.0}"
PORT="${HMOPT_GIT_MCP_PORT:-7334}"

uvicorn hmopt.api.git_mcp_server:app --host "$HOST" --port "$PORT"
