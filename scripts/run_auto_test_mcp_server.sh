#!/usr/bin/env bash
set -euo pipefail

HOST="${HMOPT_AUTO_TEST_MCP_HOST:-0.0.0.0}"
PORT="${HMOPT_AUTO_TEST_MCP_PORT:-7336}"

uvicorn hmopt.api.auto_test_mcp_server:app --host "$HOST" --port "$PORT"
