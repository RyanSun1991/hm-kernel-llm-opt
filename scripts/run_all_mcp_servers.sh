#!/usr/bin/env bash
set -euo pipefail

MCP_HOST="${HMOPT_MCP_HOST:-0.0.0.0}"
MCP_PORT="${HMOPT_MCP_PORT:-7331}"
SEQ_MCP_HOST="${HMOPT_SEQ_MCP_HOST:-0.0.0.0}"
SEQ_MCP_PORT="${HMOPT_SEQ_MCP_PORT:-7333}"

cleanup() {
  jobs -pr | xargs -r kill
}
trap cleanup EXIT INT TERM

uvicorn hmopt.api.mcp_server:app --host "$MCP_HOST" --port "$MCP_PORT" &
uvicorn hmopt.api.seq_mcp_server:app --host "$SEQ_MCP_HOST" --port "$SEQ_MCP_PORT" &

wait -n
