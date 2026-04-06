#!/usr/bin/env bash
set -euo pipefail

MCP_HOST="${HMOPT_MCP_HOST:-0.0.0.0}"
MCP_PORT="${HMOPT_MCP_PORT:-7331}"
SEQ_MCP_HOST="${HMOPT_SEQ_MCP_HOST:-0.0.0.0}"
SEQ_MCP_PORT="${HMOPT_SEQ_MCP_PORT:-7333}"
AUTO_TEST_MCP_HOST="${HMOPT_AUTO_TEST_MCP_HOST:-0.0.0.0}"
AUTO_TEST_MCP_PORT="${HMOPT_AUTO_TEST_MCP_PORT:-7336}"
FLASH_MCP_HOST="${HMOPT_FLASH_MCP_HOST:-0.0.0.0}"
FLASH_MCP_PORT="${HMOPT_FLASH_MCP_PORT:-7337}"

cleanup() {
  jobs -pr | xargs -r kill
}
trap cleanup EXIT INT TERM

uvicorn hmopt.api.mcp_server:app --host "$MCP_HOST" --port "$MCP_PORT" &
uvicorn hmopt.api.seq_mcp_server:app --host "$SEQ_MCP_HOST" --port "$SEQ_MCP_PORT" &
uvicorn hmopt.api.auto_test_mcp_server:app --host "$AUTO_TEST_MCP_HOST" --port "$AUTO_TEST_MCP_PORT" &
uvicorn hmopt.api.flash_mcp_server:app --host "$FLASH_MCP_HOST" --port "$FLASH_MCP_PORT" &

wait -n
