"""Minimal MCP demo: let the LLM call MCP tool and print the result.

Usage:
  export HMOPT_MCP_DEMO_QUERY="memcpy"
  export HMOPT_MCP_SERVER_URL="http://localhost:7331"
  export HMOPT_LLM_BASE_URL="http://<llm-host>:<port>/v1"
  export HMOPT_LLM_API_KEY="..."
  python examples/mcp_demo.py
"""

from __future__ import annotations

import os
from dataclasses import asdict

from hmopt.core.config import AppConfig
from hmopt.indexing.mcp_agent import MCPToolAgent, MCPToolAgentConfig


def main() -> None:
    config = AppConfig.from_yaml(os.getenv("HMOPT_DEMO_CONFIG", "configs/app.yaml"))
    query = os.getenv("HMOPT_MCP_DEMO_QUERY", "memcpy")
    mcp_url = "http://10.123.104.98:7331"#os.getenv("HMOPT_MCP_SERVER_URL", config.indexing.mcp.base_url)
    llm_base_url = os.getenv("HMOPT_LLM_BASE_URL", config.llm.base_url)
    llm_api_key = os.getenv("HMOPT_LLM_API_KEY", config.llm.api_key or "")

    agent_cfg = MCPToolAgentConfig(
        base_url=llm_base_url,
        api_key=llm_api_key or None,
        model=config.llm.model,
        timeout_sec=30,
        tool_name="kernel_index_code",
        top_k=10,
        mcp_base_url=mcp_url,
    )
    agent = MCPToolAgent(agent_cfg)
    result = agent.fetch_context(query)

    print("== MCP demo ==")
    print(f"query: {query}")
    print(f"tool_used: {result.tool_used}")
    print(f"agent_config: {asdict(agent_cfg)}")
    print("context:")
    print(result.context or "<empty>")


if __name__ == "__main__":
    main()
