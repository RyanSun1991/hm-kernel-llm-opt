"""Compatibility import; MCP registration belongs to hmopt.api.evolution_mcp_service.

Production deployments use hmopt.api.mcp_stdio or hmopt.api.mcp_server. The old
Evolution-only CLI entrypoint continues to use the same tool registration.
"""

from hmopt.api.evolution_mcp_service import build_evolution_fastmcp_server as build_server

__all__ = ["build_server"]
