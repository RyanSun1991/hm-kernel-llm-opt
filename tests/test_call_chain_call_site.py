"""Phase 5 tests: call-chain output and rendering carry call_site fields.

Both `retrieve_call_chain` and `call_chain_to_markdown` live in heavy
modules (`llamaindex_pipeline.py` requires llama_index + neo4j, and
`mcp_service.py` requires FastAPI + MCP). To keep these tests fast and
self-contained, we extract the relevant Phase-5 pieces via AST and exec
them in an isolated namespace — verifying the actual production source
without paying for the surrounding module's import chain.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SRC = REPO_ROOT / "src" / "hmopt" / "indexing" / "llamaindex_pipeline.py"
MCP_SRC = REPO_ROOT / "src" / "hmopt" / "api" / "mcp_service.py"


def _extract_function(source: Path, name: str) -> Callable:
    """Parse `source` and exec just the function body of `name`.

    The function is exec'd in a namespace that provides `Any` from typing;
    everything else the function depends on must already live inside the
    function body (which the Phase 5 helpers happen to satisfy — they only
    touch built-in dict/str/list APIs).
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns: dict[str, Any] = {"Any": Any}
            mod = ast.Module(body=[node], type_ignores=[])
            exec(compile(mod, str(source), "exec"), ns)
            return ns[name]
    raise RuntimeError(f"function {name!r} not found in {source}")


# ---------------------------------------------------------------------------
# _call_site_fields — pure dict-to-dict translator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def call_site_fields():
    return _extract_function(PIPELINE_SRC, "_call_site_fields")


def test_call_site_fields_complete_scip_row(call_site_fields):
    """All scip-clang call-site fields propagate to the edge dict."""
    row = {
        "src": "caller",
        "dst": "helper",
        "rel": "calls",
        "cs_path": "kernel/foo.c",
        "cs_line": 42,
        "cs_col": 8,
        "cs_syntax_kind": 15,  # IdentifierFunction
        "cs_is_write": False,
        "cs_backend_origin": "scip-clang",
    }
    out = call_site_fields(row)
    assert out == {
        "call_site_path": "kernel/foo.c",
        "call_site_line": 42,
        "call_site_col": 8,
        "syntax_kind": 15,
        "is_write": False,
        "backend_origin": "scip-clang",
    }


def test_call_site_fields_clangd_row_yields_empty(call_site_fields):
    """clangd-origin rows return null for every cs_* alias; helper drops them."""
    row = {
        "src": "caller",
        "dst": "helper",
        "rel": "calls",
        "cs_path": None,
        "cs_line": None,
        "cs_col": None,
        "cs_syntax_kind": None,
        "cs_is_write": None,
        "cs_backend_origin": None,
    }
    assert call_site_fields(row) == {}


def test_call_site_fields_partial_row(call_site_fields):
    """Some scip-clang occurrences carry path+line but not col (e.g.
    multi-line references). Only fields actually present should appear."""
    row = {
        "cs_path": "kernel/bar.c",
        "cs_line": 10,
        "cs_col": None,
        "cs_syntax_kind": 19,  # IdentifierType
        "cs_is_write": None,
        "cs_backend_origin": "scip-clang",
    }
    out = call_site_fields(row)
    assert out == {
        "call_site_path": "kernel/bar.c",
        "call_site_line": 10,
        "syntax_kind": 19,
        "backend_origin": "scip-clang",
    }


def test_call_site_fields_handles_zero_values(call_site_fields):
    """Zero is a valid call_site_col (column 0); must not be dropped as if it
    were None."""
    row = {
        "cs_path": "foo.c",
        "cs_line": 1,
        "cs_col": 0,
        "cs_is_write": False,  # explicit False — also must not be dropped
    }
    out = call_site_fields(row)
    assert out["call_site_col"] == 0
    assert out["is_write"] is False


# ---------------------------------------------------------------------------
# _format_call_chain_text — text formatter (Phase 5a output side)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def format_text():
    return _extract_function(PIPELINE_SRC, "_format_call_chain_text")


def test_format_text_includes_call_site_suffix(format_text):
    payload = {
        "roots": ["caller"],
        "direction": "callees",
        "depth_requested": 2,
        "edge_kinds": ["calls"],
        "edges": [
            {
                "src": "caller",
                "dst": "helper",
                "rel": "calls",
                "depth": 1,
                "direction": "callee",
                "call_site_path": "foo.c",
                "call_site_line": 42,
                "call_site_col": 8,
            }
        ],
        "nodes": {},
        "stats": {"total_edges": 1, "total_nodes": 0, "depth_reached": 1, "hops_truncated_at": []},
    }
    out = format_text(payload)
    assert "caller -[calls]-> helper" in out
    assert "@foo.c:42:8" in out


def test_format_text_omits_call_site_when_clangd_origin(format_text):
    """clangd-origin edges don't carry call_site_path → no @-suffix appears."""
    payload = {
        "roots": ["caller"],
        "direction": "callees",
        "depth_requested": 1,
        "edge_kinds": ["calls"],
        "edges": [
            {
                "src": "caller",
                "dst": "helper",
                "rel": "calls",
                "depth": 1,
                "direction": "callee",
            }
        ],
        "nodes": {},
        "stats": {"total_edges": 1, "total_nodes": 0, "depth_reached": 1, "hops_truncated_at": []},
    }
    out = format_text(payload)
    assert "caller -[calls]-> helper" in out
    assert "@" not in out.split("Edges:")[1]


def test_format_text_line_only_when_col_missing(format_text):
    payload = {
        "roots": ["caller"],
        "direction": "callees",
        "depth_requested": 1,
        "edge_kinds": ["calls"],
        "edges": [
            {
                "src": "caller",
                "dst": "helper",
                "rel": "calls",
                "depth": 1,
                "direction": "callee",
                "call_site_path": "bar.c",
                "call_site_line": 7,
                # no call_site_col
            }
        ],
        "nodes": {},
        "stats": {"total_edges": 1, "total_nodes": 0, "depth_reached": 1, "hops_truncated_at": []},
    }
    out = format_text(payload)
    assert "@bar.c:7" in out
    # No trailing :col when col is absent.
    assert "@bar.c:7:" not in out


# ---------------------------------------------------------------------------
# call_chain_to_markdown — MCP-side markdown formatter (Phase 5b)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def to_markdown():
    return _extract_function(MCP_SRC, "call_chain_to_markdown")


def test_markdown_includes_call_site_and_write_flag(to_markdown):
    payload = {
        "roots": ["caller"],
        "direction": "callees",
        "depth_requested": 1,
        "edge_kinds": ["calls", "uses_type"],
        "edges": [
            {
                "src": "caller",
                "dst": "shared_var",
                "rel": "uses_type",
                "depth": 1,
                "direction": "callee",
                "call_site_path": "foo.c",
                "call_site_line": 99,
                "call_site_col": 4,
                "is_write": True,
            }
        ],
        "nodes": {},
        "stats": {"total_edges": 1, "total_nodes": 0, "depth_reached": 1, "hops_truncated_at": [], "per_hop_limit": 100, "frontier_cap": 50},
    }
    md = to_markdown(payload)
    assert "@foo.c:99:4" in md
    assert "[write]" in md


def test_markdown_omits_call_site_for_clangd_origin(to_markdown):
    payload = {
        "roots": ["caller"],
        "direction": "callees",
        "depth_requested": 1,
        "edge_kinds": ["calls"],
        "edges": [
            {
                "src": "caller",
                "dst": "helper",
                "rel": "calls",
                "depth": 1,
                "direction": "callee",
            }
        ],
        "nodes": {},
        "stats": {"total_edges": 1, "total_nodes": 0, "depth_reached": 1, "hops_truncated_at": [], "per_hop_limit": 100, "frontier_cap": 50},
    }
    md = to_markdown(payload)
    # The edge renders cleanly, no call-site suffix or write flag.
    assert "caller -[calls]-> helper" in md
    assert "@" not in md.split("## Edges")[1]
    assert "[write]" not in md
