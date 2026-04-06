#!/usr/bin/env bash
set -euo pipefail

HOST="${HMOPT_FLASH_MCP_HOST:-0.0.0.0}"
PORT="${HMOPT_FLASH_MCP_PORT:-7337}"

uvicorn hmopt.api.flash_mcp_server:app --host "$HOST" --port "$PORT"
