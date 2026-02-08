#!/usr/bin/env bash
set -euo pipefail

HOST="${HMOPT_MCP_HOST:-0.0.0.0}"
PORT="${HMOPT_MCP_PORT:-7331}"

uvicorn hmopt.api.mcp_server:app --host "$HOST" --port "$PORT"
