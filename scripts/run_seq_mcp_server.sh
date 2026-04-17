#!/usr/bin/env bash
set -euo pipefail

HOST="${HMOPT_SEQ_MCP_HOST:-0.0.0.0}"
PORT="${HMOPT_SEQ_MCP_PORT:-7333}"

uvicorn hmopt.api.seq_mcp_server:app --host "$HOST" --port "$PORT"
