"""
Sequential Thinking MCP tool service (UltraThink engine).

UltraThink source (MIT):
https://github.com/husniadil/ultrathink
"""

from __future__ import annotations

import os
from threading import RLock
from typing import Annotated, Any

from pydantic import Field

from hmopt.sequential_thinking.models.assumption import Assumption
from hmopt.sequential_thinking.models.thought import ThoughtRequest


DEFAULT_SEQ_SERVER_NAME = "hmopt-sequential-thinking"
DEFAULT_SEQ_TOOL_NAME = "sequential_thinking"

# Single in-memory engine instance (holds sessions)
# UltraThinkService is NOT thread-safe by itself (shared dict), so we guard it with a lock.
_SEQ_LOCK = RLock()
_SEQ_SERVICE: Any | None = None


def _get_tool_name() -> str:
    name = os.getenv("HMOPT_SEQ_MCP_TOOL_NAME", DEFAULT_SEQ_TOOL_NAME).strip()
    return name or DEFAULT_SEQ_TOOL_NAME


def _get_server_name() -> str:
    name = os.getenv("HMOPT_SEQ_MCP_SERVER_NAME", DEFAULT_SEQ_SERVER_NAME).strip()
    return name or DEFAULT_SEQ_SERVER_NAME


def _get_service():
    global _SEQ_SERVICE
    if _SEQ_SERVICE is None:
        # Optional: allow hmopt-prefixed env to control UltraThink logging without modifying their file
        if "HMOPT_DISABLE_THOUGHT_LOGGING" in os.environ and "DISABLE_THOUGHT_LOGGING" not in os.environ:
            os.environ["DISABLE_THOUGHT_LOGGING"] = os.environ["HMOPT_DISABLE_THOUGHT_LOGGING"]

        from hmopt.sequential_thinking.services.thinking_service import UltraThinkService

        _SEQ_SERVICE = UltraThinkService()
    return _SEQ_SERVICE


def call_seq_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Legacy-friendly callable: arguments -> ThoughtResponse (dict).
    """
    req = ThoughtRequest(**arguments)  # pydantic validates + parses json-string fields into lists
    svc = _get_service()
    with _SEQ_LOCK:
        resp = svc.process_thought(req)
    return resp.model_dump()


def build_seq_fastmcp_server() -> Any:
    """
    Build a standalone MCP server exposing exactly one tool:
    - name: HMOPT_SEQ_MCP_TOOL_NAME (default: sequential_thinking)
    """
    from mcp.server.fastmcp import FastMCP  # uses your repo dependency mcp[cli]

    mcp = FastMCP(_get_server_name(), stateless_http=True, json_response=True)
    tool_name = _get_tool_name()

    @mcp.tool(
        name=tool_name,
        description=(
            "Sequential thinking tool (sessions, revisions, branching, confidence, assumptions). "
            "Engine adapted from husniadil/ultrathink (MIT)."
        ),
    )
    def sequential_thinking(
        thought: Annotated[str, Field(min_length=1, description="Your current thinking step")],
        total_thoughts: Annotated[int, Field(ge=1, description="Estimated total thoughts needed")],
        next_thought_needed: Annotated[
            bool | None,
            Field(None, description="Auto: (thought_number < total_thoughts) if omitted"),
        ] = None,
        thought_number: Annotated[
            int | None,
            Field(None, ge=1, description="Auto-assigned sequentially if omitted"),
        ] = None,
        session_id: Annotated[
            str | None,
            Field(None, description="None=create new; provide to continue/resume session"),
        ] = None,
        is_revision: Annotated[bool | None, Field(None, description="Revises previous thinking")] = None,
        revises_thought: Annotated[int | None, Field(None, ge=1, description="Which thought is revised")] = None,
        branch_from_thought: Annotated[int | None, Field(None, ge=1, description="Branch point thought number")] = None,
        branch_id: Annotated[str | None, Field(None, description="Branch identifier")] = None,
        needs_more_thoughts: Annotated[bool | None, Field(None, description="If more thoughts are needed")] = None,
        confidence: Annotated[float | None, Field(None, ge=0.0, le=1.0, description="0.0-1.0")] = None,
        uncertainty_notes: Annotated[str | None, Field(None, description="Optional uncertainty notes")] = None,
        outcome: Annotated[str | None, Field(None, description="Optional outcome")] = None,
        assumptions: Annotated[
            list[Assumption] | str | None,
            Field(None, description="List[Assumption] or JSON string"),
        ] = None,
        depends_on_assumptions: Annotated[
            list[str] | str | None,
            Field(None, description="List[str] or JSON string"),
        ] = None,
        invalidates_assumptions: Annotated[
            list[str] | str | None,
            Field(None, description="List[str] or JSON string"),
        ] = None,
    ) -> dict[str, Any]:
        args = {
            "thought": thought,
            "total_thoughts": total_thoughts,
            "next_thought_needed": next_thought_needed,
            "thought_number": thought_number,
            "session_id": session_id,
            "is_revision": is_revision,
            "revises_thought": revises_thought,
            "branch_from_thought": branch_from_thought,
            "branch_id": branch_id,
            "needs_more_thoughts": needs_more_thoughts,
            "confidence": confidence,
            "uncertainty_notes": uncertainty_notes,
            "outcome": outcome,
            "assumptions": assumptions,
            "depends_on_assumptions": depends_on_assumptions,
            "invalidates_assumptions": invalidates_assumptions,
        }
        return call_seq_tool(args)

    return mcp
