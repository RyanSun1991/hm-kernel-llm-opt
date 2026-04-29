"""Direct in-process test for the kernel MCP tools registered in
``hmopt.api.mcp_service``.

Tools covered (all 5 registered FastMCP tools as of commit 428987d)
-------------------------------------------------------------------
Bundled (one-shot vector + graph + rerank):
* ``kernel_index_code``      — general code/snippet retrieval.
* ``kernel_symbol_graph``    — call-graph oriented bundled retrieval.
* ``kernel_hotspot_context`` — hotspot / performance bundled retrieval.

Two-step protocol (decoupled, lets the agent budget tokens explicitly):
* ``kernel_call_chain``      — pure structural BFS over the call graph;
                               returns edges + per-node metadata, no bodies.
* ``kernel_get_snippets``    — batch fetch function bodies for an explicit
                               symbol list, with per-symbol / total char
                               budgets and explicit ``missing[]`` reporting.

This file calls each tool in-process via the underlying helpers in
``hmopt.api.mcp_service`` so no FastMCP HTTP server has to be running. It can
be executed two ways:

* ``pytest tests/test_kernel_mcp_tools.py -s`` — runs as integration tests
  (skipped automatically if the kernel index / Neo4j is not available).
* ``python tests/test_kernel_mcp_tools.py`` — runs as a CLI demo and prints
  the full tool return content for inspection.

Both modes honour ``HMOPT_MCP_CONFIG`` (path to an ``app.yaml``); if unset the
default ``configs/app.yaml`` is used.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hmopt.api.mcp_service import (  # noqa: E402
    _call_call_chain_tool,
    _call_general_tool,
    _call_graph_tool,
    _call_hotspot_tool,
    _call_snippets_tool,
    call_tool_by_name,
    get_tool_names,
    load_app_config,
    resolve_mcp_config_path,
)


DEFAULT_QUERY = os.getenv(
    "HMOPT_TEST_QUERY",
    "Locate the entry point for hyperhold reclaim and explain its hot path",
)
DEFAULT_SYMBOLS = [
    s.strip()
    for s in os.getenv("HMOPT_TEST_SYMBOLS", "hyperhold_reclaim,zswapd_reclaim").split(",")
    if s.strip()
]
DEFAULT_RUNTIME_HINTS = os.getenv(
    "HMOPT_TEST_RUNTIME_HINTS",
    "perf shows ~35% time in hyperhold_reclaim under memory pressure",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_path() -> str:
    return resolve_mcp_config_path(os.getenv("HMOPT_MCP_CONFIG"))


def _index_is_available() -> bool:
    """Quickly sanity-check that the LlamaIndex / Neo4j kernel index can load.

    Returns True iff the configured persist_dir exists and at least one of the
    expected sub-indexes has been written. Used to xfail/skip pytest cases on
    machines where the index has not been built yet.
    """
    try:
        cfg = load_app_config(_config_path())
    except Exception:
        return False
    persist = Path(cfg.indexing.persist_dir).expanduser()
    if not persist.exists():
        return False
    code_dir = persist / "code"
    runtime_dir = persist / "runtime"
    return code_dir.exists() or runtime_dir.exists()


def _pretty(label: str, content: str) -> None:
    bar = "=" * 80
    print(f"\n{bar}\n[{label}]\n{bar}")
    print(content)
    print(bar)


# ---------------------------------------------------------------------------
# Pytest-style smoke tests (skipped if the index is not built)
# ---------------------------------------------------------------------------

INDEX_AVAILABLE = _index_is_available()
SKIP_REASON = (
    "kernel index not built (set HMOPT_MCP_CONFIG to a valid app.yaml whose "
    "indexing.persist_dir contains a 'code' or 'runtime' sub-index)"
)


@pytest.mark.skipif(not INDEX_AVAILABLE, reason=SKIP_REASON)
def test_kernel_index_code_general() -> None:
    """kernel_index_code: general bundled retrieval, JSON output."""
    out = _call_general_tool(
        {
            "query": DEFAULT_QUERY,
            "scenario": "general",
            "symbols": DEFAULT_SYMBOLS,
            "top_k": 6,
            "max_snippets": 4,
            "max_chars": 2500,
            "graph_depth": 2,
            "response_format": "json",
        }
    )
    payload = json.loads(out)
    assert isinstance(payload, dict)
    assert payload.get("scenario") == "general"
    assert "ranked_symbols" in payload or "snippets" in payload


@pytest.mark.skipif(not INDEX_AVAILABLE, reason=SKIP_REASON)
def test_kernel_symbol_graph_bundled() -> None:
    """kernel_symbol_graph: call-graph focused bundled retrieval."""
    if not DEFAULT_SYMBOLS:
        pytest.skip("HMOPT_TEST_SYMBOLS is empty")
    out = _call_graph_tool(
        {
            "query": "trace caller/callee chain",
            "symbols": DEFAULT_SYMBOLS,
            "graph_depth": 3,
            "max_snippets": 6,
            "max_chars": 3000,
            "response_format": "json",
        }
    )
    payload = json.loads(out)
    assert payload.get("scenario") == "call_graph"
    assert "graph_edges" in payload or "graph_summary" in payload


@pytest.mark.skipif(not INDEX_AVAILABLE, reason=SKIP_REASON)
def test_kernel_hotspot_context() -> None:
    """kernel_hotspot_context: hotspot/perf focused bundled retrieval."""
    out = _call_hotspot_tool(
        {
            "query": DEFAULT_QUERY,
            "symbols": DEFAULT_SYMBOLS,
            "runtime_hints": DEFAULT_RUNTIME_HINTS,
            "graph_depth": 3,
            "response_format": "json",
        }
    )
    payload = json.loads(out)
    assert payload.get("scenario") == "hotspot_debug"


@pytest.mark.skipif(not INDEX_AVAILABLE, reason=SKIP_REASON)
def test_kernel_call_chain() -> None:
    """kernel_call_chain: pure structural BFS, no bodies."""
    if not DEFAULT_SYMBOLS:
        pytest.skip("HMOPT_TEST_SYMBOLS is empty")
    out = _call_call_chain_tool(
        {
            "symbols": DEFAULT_SYMBOLS,
            "direction": "both",
            "depth": 3,
            "per_hop_limit": 50,
            "frontier_cap": 30,
            "edge_kinds": ["calls"],
            "response_format": "json",
        }
    )
    payload = json.loads(out)
    # call-chain payload always carries roots / direction / stats / edges / nodes
    assert payload.get("direction") == "both"
    assert payload.get("depth_requested") == 3
    assert "edges" in payload
    assert "nodes" in payload
    stats = payload.get("stats") or {}
    assert "total_edges" in stats
    assert "total_nodes" in stats
    assert "depth_reached" in stats


@pytest.mark.skipif(not INDEX_AVAILABLE, reason=SKIP_REASON)
def test_kernel_get_snippets() -> None:
    """kernel_get_snippets: batch body fetch with budgets + missing reporting."""
    if not DEFAULT_SYMBOLS:
        pytest.skip("HMOPT_TEST_SYMBOLS is empty")
    out = _call_snippets_tool(
        {
            "symbols": DEFAULT_SYMBOLS,
            "per_symbol_max_chars": 3000,
            "total_max_chars": 12000,
            "response_format": "json",
        }
    )
    payload = json.loads(out)
    assert "snippets" in payload
    assert "missing" in payload
    stats = payload.get("stats") or {}
    assert "returned" in stats
    assert "missing" in stats
    assert "total_chars" in stats


@pytest.mark.skipif(not INDEX_AVAILABLE, reason=SKIP_REASON)
def test_two_step_protocol_chain_then_snippets() -> None:
    """Realistic flow: walk the call chain first, then fetch bodies for top hits."""
    if not DEFAULT_SYMBOLS:
        pytest.skip("HMOPT_TEST_SYMBOLS is empty")
    chain_out = _call_call_chain_tool(
        {
            "symbols": DEFAULT_SYMBOLS,
            "direction": "callees",
            "depth": 2,
            "per_hop_limit": 30,
            "frontier_cap": 20,
            "response_format": "json",
        }
    )
    chain = json.loads(chain_out)
    nodes = list((chain.get("nodes") or {}).keys())
    if not nodes:
        pytest.skip("call_chain returned no nodes — kernel index may be empty for these symbols")
    targets = nodes[:5]

    snippets_out = _call_snippets_tool(
        {
            "symbols": targets,
            "per_symbol_max_chars": 2000,
            "total_max_chars": 8000,
            "response_format": "json",
        }
    )
    payload = json.loads(snippets_out)
    returned = {entry.get("symbol") for entry in payload.get("snippets") or []}
    missing = {entry.get("symbol") for entry in payload.get("missing") or []}
    # every requested symbol must be accounted for in either snippets or missing
    assert set(targets).issubset(returned | missing)


@pytest.mark.skipif(not INDEX_AVAILABLE, reason=SKIP_REASON)
def test_call_tool_by_name_dispatch() -> None:
    """``call_tool_by_name`` should route to the right helper for each name."""
    names = get_tool_names()
    assert {"general", "graph", "hotspot", "call_chain", "snippets"}.issubset(names)
    out = call_tool_by_name(
        names["general"],
        {"query": DEFAULT_QUERY, "top_k": 4, "max_snippets": 3, "response_format": "json"},
    )
    assert json.loads(out).get("scenario") == "general"


# ---------------------------------------------------------------------------
# Argument-validation tests (do not require a live index)
# ---------------------------------------------------------------------------

def test_call_chain_requires_symbols() -> None:
    with pytest.raises(ValueError):
        _call_call_chain_tool({"symbols": [], "response_format": "json"})


def test_call_chain_rejects_invalid_direction() -> None:
    with pytest.raises(ValueError):
        _call_call_chain_tool(
            {"symbols": ["foo"], "direction": "sideways", "response_format": "json"}
        )


def test_call_chain_rejects_invalid_edge_kind() -> None:
    with pytest.raises(ValueError):
        _call_call_chain_tool(
            {"symbols": ["foo"], "edge_kinds": ["bogus_kind"], "response_format": "json"}
        )


def test_get_snippets_requires_symbols() -> None:
    with pytest.raises(ValueError):
        _call_snippets_tool({"symbols": [], "response_format": "json"})


# ---------------------------------------------------------------------------
# CLI demo entry-point — prints the actual returned content for inspection
# ---------------------------------------------------------------------------

def _run_cli_demo(response_format: str = "markdown") -> int:
    print(f"# config_path = {_config_path()}")
    print(f"# tool_names  = {get_tool_names()}")
    print(f"# index_built = {INDEX_AVAILABLE}")
    if not INDEX_AVAILABLE:
        print("# WARNING: kernel index does not appear to be built. The calls")
        print("#   below will likely raise. Run 'hmopt index-kernel ...' first")
        print("#   and/or set HMOPT_MCP_CONFIG to a valid app.yaml.")

    seed_symbols = DEFAULT_SYMBOLS or ["hyperhold_reclaim"]

    cases: list[tuple[str, dict[str, Any], Any]] = [
        (
            "kernel_index_code (general bundled)",
            {
                "query": DEFAULT_QUERY,
                "scenario": "general",
                "symbols": DEFAULT_SYMBOLS,
                "top_k": 8,
                "max_snippets": 6,
                "max_chars": 3500,
                "graph_depth": 2,
                "response_format": response_format,
            },
            _call_general_tool,
        ),
        (
            "kernel_symbol_graph (bundled call-graph)",
            {
                "query": "Show caller/callee chain",
                "symbols": seed_symbols,
                "graph_depth": 3,
                "max_snippets": 8,
                "max_chars": 4000,
                "response_format": response_format,
            },
            _call_graph_tool,
        ),
        (
            "kernel_hotspot_context",
            {
                "query": DEFAULT_QUERY,
                "symbols": DEFAULT_SYMBOLS,
                "runtime_hints": DEFAULT_RUNTIME_HINTS,
                "graph_depth": 3,
                "response_format": response_format,
            },
            _call_hotspot_tool,
        ),
        (
            "kernel_call_chain (structural BFS, no bodies)",
            {
                "symbols": seed_symbols,
                "direction": "both",
                "depth": 3,
                "per_hop_limit": 80,
                "frontier_cap": 40,
                "edge_kinds": ["calls"],
                "response_format": response_format,
            },
            _call_call_chain_tool,
        ),
        (
            "kernel_get_snippets (batch body fetch)",
            {
                "symbols": seed_symbols,
                "per_symbol_max_chars": 3000,
                "total_max_chars": 12000,
                "response_format": response_format,
            },
            _call_snippets_tool,
        ),
    ]

    rc = 0
    for label, args, fn in cases:
        try:
            result = fn(args)
            _pretty(label, result)
        except Exception as exc:  # noqa: BLE001
            rc = 1
            _pretty(label, f"ERROR: {exc!r}\n{traceback.format_exc()}")
    return rc


if __name__ == "__main__":
    fmt = "json" if "--json" in sys.argv else "markdown"
    sys.exit(_run_cli_demo(response_format=fmt))
