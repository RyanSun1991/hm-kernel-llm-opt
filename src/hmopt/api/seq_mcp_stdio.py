"""Stdio MCP entrypoint for Sequential Thinking."""

from __future__ import annotations

from hmopt.api.seq_mcp_service import build_seq_fastmcp_server


def main() -> None:
    server = build_seq_fastmcp_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
