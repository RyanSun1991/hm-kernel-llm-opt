"""Shared MCP transport contract for the platform's registered tool groups."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


class StrictFastMCP(FastMCP):
    """Reject unknown Evolution arguments while preserving legacy indexing coercion."""

    async def list_tools(self):
        registered = await super().list_tools()
        for tool in registered:
            if tool.name.startswith("evolution_"):
                tool.inputSchema = {**tool.inputSchema, "additionalProperties": False}
        return registered

    async def call_tool(self, name, arguments):
        # FastMCP otherwise drops unknown fields before invoking the function.
        # Both HTTP and stdio must reject an ignored path/recovery override.
        if name.startswith("evolution_"):
            for tool in await self.list_tools():
                if tool.name == name:
                    unknown = set(arguments) - set(tool.inputSchema.get("properties", {}))
                    if unknown:
                        raise ValueError("Unexpected tool arguments: " + ", ".join(sorted(unknown)))
                    break
        return await super().call_tool(name, arguments)
