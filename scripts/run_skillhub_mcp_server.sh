#!/usr/bin/env bash
set -euo pipefail

HOST="${HMOPT_SKILLHUB_MCP_HOST:-0.0.0.0}"
PORT="${HMOPT_SKILLHUB_MCP_PORT:-7338}"

uvicorn hmopt.api.skillhub_mcp_server:app --host "$HOST" --port "$PORT"
