#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entrypoint.
# Avoid inherited HMOPT_BUILD_MCP_* env from shell affecting local mock test.
unset HMOPT_BUILD_MCP_HOST HMOPT_BUILD_MCP_PORT HMOPT_BUILD_MCP_MODE \
  HMOPT_BUILD_MCP_RUNNER_CONTAINER HMOPT_BUILD_MCP_RUNNER_IMAGE \
  HMOPT_BUILD_MCP_PROJECT_PATH HMOPT_BUILD_MCP_PROJECT_MOUNT \
  HMOPT_BUILD_MCP_SIGN_WORKSPACE HMOPT_BUILD_MCP_DOCKER_BIN HMOPT_BUILD_MCP_WORKDIR

exec bash scripts/test_build_mcp_local.sh
