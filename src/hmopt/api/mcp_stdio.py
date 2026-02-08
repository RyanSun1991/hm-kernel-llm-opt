"""Stdio MCP entrypoint for OpenCode local MCP integration."""

from __future__ import annotations

from hmopt.api.mcp_service import build_fastmcp_server


def main() -> None:
    server = build_fastmcp_server()
    if server is None:
        raise RuntimeError(
            "mcp package is not installed. Install with: pip install 'mcp[cli]'"
        )
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
