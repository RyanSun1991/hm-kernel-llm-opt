"""Direct-Python probe for the kernel index MCP tools.

Calls the three FastMCP tools defined in ``src/hmopt/api/mcp_service.py``
WITHOUT going through the MCP / HTTP layer.  Use this to test argument
shapes, see the response structure (both markdown and JSON), and debug
the underlying retrieval pipeline.

Existing tools in this repo (per ``configs/app.yaml`` and
``mcp_service.py``):

  1. ``kernel_index_code``      — general retrieval (vector + graph)
  2. ``kernel_symbol_graph``    — call-graph-focused retrieval
  3. ``kernel_hotspot_context`` — hotspot / perf-focused retrieval

(There is no ``kernel_call_chain`` or ``kernel_get_snippets`` in this
repo.  ``kernel_symbol_graph`` is the call-graph tool;
``kernel_index_code`` is the snippet-returning tool.)

Usage
-----

    # default config: configs/app.yaml
    PYTHONPATH=src python scripts/probe_kernel_index_tools.py

    # custom config
    PYTHONPATH=src python scripts/probe_kernel_index_tools.py \
        --config configs/app.yaml \
        --query "How does try_to_free_pages decide which zone to scan?" \
        --symbol try_to_free_pages \
        --tool all                 # all | index_code | symbol_graph | hotspot_context

    # dump JSON instead of markdown
    PYTHONPATH=src python scripts/probe_kernel_index_tools.py --format json

Prerequisites
-------------

The tools call ``hmopt.indexing.llamaindex_pipeline.retrieve_code_context``,
which needs:

  - a built LlamaIndex (``data/llamaindex/`` or wherever
    ``indexing.persist_dir`` points)
  - optionally a reachable Neo4j (``indexing.neo4j.uri``) for graph
    expansion — without it the tools still work, just without the graph
    component
  - the kernel repo path in ``project.repo_path`` (used for snippet
    file:line resolution)

If your environment doesn't have the index built, the script still
demonstrates argument shapes and surfaces the pipeline error verbatim
so you can see exactly what's missing.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any, Mapping


# Make the package importable when running from a checkout (avoid relying on
# `pip install -e .` so the script is usable from a fresh clone).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Sample arguments — edit these to probe your own queries / symbols
# ---------------------------------------------------------------------------

# Each block matches one tool's parameter signature.  The script will splice
# in --query / --symbol overrides from argparse on top of these.

INDEX_CODE_DEFAULT = {
    # ``query`` is the natural-language search prompt.  It is augmented per
    # scenario with a built-in suffix (see SCENARIO_PRESETS in mcp_service.py)
    # and with the focus_symbols and runtime_hints below.
    "query": "How does the page reclaim path decide which LRU list to scan first?",
    # ``scenario`` selects a preset of (top_k, max_snippets, max_chars,
    # graph_depth, query_suffix).  Allowed:
    #   general | implementation | call_graph | impact_analysis |
    #   hotspot_debug | patch_planning
    "scenario": "implementation",
    # ``symbols`` (optional, max 20) — focus symbol hints; deduped, prepended
    # to the query as "Focus symbols: ...".  Pass [] or None to skip.
    "symbols": ["try_to_free_pages", "shrink_node"],
    # ``runtime_hints`` (optional) — free text appended to the query, e.g.
    # "hot under memory pressure on phone X".  Empty string skips it.
    "runtime_hints": "",
    # ``top_k`` (1..64, optional) — vector retrieval top-K.  None → preset.
    "top_k": None,
    # ``max_snippets`` (1..32) — cap on returned snippets in the payload.
    "max_snippets": 8,
    # ``max_chars`` (256..20000) — character budget for the rendered output.
    "max_chars": 4000,
    # ``graph_depth`` (1..4) — Neo4j graph expansion radius around focus
    # symbols.  Bigger = wider impact view, slower, more noise.
    "graph_depth": 2,
    # ``response_format`` — "markdown" | "json".  JSON returns the raw
    # payload dict serialized; markdown returns the human-readable render.
    "response_format": "markdown",
}

SYMBOL_GRAPH_DEFAULT = {
    # ``symbols`` is REQUIRED for kernel_symbol_graph (it's the entry of the
    # call-graph walk).  Other args mirror kernel_index_code minus
    # ``scenario`` (the tool forces scenario="call_graph" internally).
    "symbols": ["try_to_free_pages"],
    "query": "trace caller and callee chains, expose impact radius",
    "top_k": None,
    "graph_depth": 3,           # 1..4; 3 is a good default for call-graph use
    "max_snippets": 12,
    "max_chars": 5000,
    "response_format": "markdown",
}

HOTSPOT_CONTEXT_DEFAULT = {
    # ``symbols`` (optional but strongly recommended) — the hot symbols you
    # already know from runtime evidence.
    "symbols": ["shrink_node", "shrink_lruvec"],
    # ``runtime_hints`` is the headline parameter here — paste the hiperf /
    # flamegraph / hitrace observation that motivated the lookup.
    "runtime_hints": (
        "hiperf shows shrink_node took 18% of CPU samples on the reclaim "
        "fast path under sustained 80% memory pressure"
    ),
    "query": "what makes this hot path expensive and what nearby code "
             "should I look at when planning an optimization?",
    "top_k": None,
    "graph_depth": 3,
    "response_format": "markdown",
}


# ---------------------------------------------------------------------------
# Pretty-printer helpers
# ---------------------------------------------------------------------------

def _banner(text: str, char: str = "=") -> str:
    line = char * 78
    return f"\n{line}\n{text}\n{line}"


def _format_args(args: Mapping[str, Any]) -> str:
    return json.dumps(args, indent=2, ensure_ascii=False, default=str)


def _print_response(name: str, args: Mapping[str, Any], result: str) -> None:
    print(_banner(f"{name}  →  RESPONSE"))
    print(f"--- arguments ({len(_format_args(args))} chars) ---")
    print(_format_args(args))
    print(f"--- response ({len(result)} chars) ---")
    print(result)
    print(_banner(f"{name}  →  END", char="-"))


# ---------------------------------------------------------------------------
# Tool wrappers — call the underlying mcp_service helpers directly
# ---------------------------------------------------------------------------

def call_kernel_index_code(args: dict[str, Any], config_path: str) -> str:
    """Call ``kernel_index_code`` (the general retrieval tool).

    Internally goes through ``mcp_service._call_general_tool`` which is the
    same code path the FastMCP-decorated function uses.
    """
    from hmopt.api.mcp_service import _call_general_tool
    return _call_general_tool(args, config_path=config_path)


def call_kernel_symbol_graph(args: dict[str, Any], config_path: str) -> str:
    """Call ``kernel_symbol_graph`` (the call-graph tool)."""
    from hmopt.api.mcp_service import _call_graph_tool
    return _call_graph_tool(args, config_path=config_path)


def call_kernel_hotspot_context(args: dict[str, Any], config_path: str) -> str:
    """Call ``kernel_hotspot_context`` (the hotspot tool)."""
    from hmopt.api.mcp_service import _call_hotspot_tool
    return _call_hotspot_tool(args, config_path=config_path)


# Alternative dispatch: by name, mirroring how the MCP server routes calls.
# Useful when you're testing tool-name resolution end-to-end.
def call_by_name(tool_name: str, args: dict[str, Any], config_path: str) -> str:
    from hmopt.api.mcp_service import call_tool_by_name
    return call_tool_by_name(tool_name, args, config_path=config_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _override(defaults: dict[str, Any], cli_args: argparse.Namespace) -> dict[str, Any]:
    """Apply --query, --symbol, --format CLI overrides on top of defaults."""
    out = dict(defaults)
    if cli_args.query:
        out["query"] = cli_args.query
    if cli_args.symbol:
        out["symbols"] = cli_args.symbol
    if cli_args.runtime_hints is not None:
        out["runtime_hints"] = cli_args.runtime_hints
    if cli_args.format:
        out["response_format"] = cli_args.format
    if cli_args.scenario and "scenario" in out:
        out["scenario"] = cli_args.scenario
    if cli_args.top_k is not None:
        out["top_k"] = cli_args.top_k
    if cli_args.graph_depth is not None:
        out["graph_depth"] = cli_args.graph_depth
    return out


def _run_one(label: str, fn, defaults: dict[str, Any], cli_args: argparse.Namespace,
             config_path: str) -> None:
    final_args = _override(defaults, cli_args)
    print(_banner(f"{label}  →  CALL"))
    print(_format_args(final_args))
    try:
        result = fn(final_args, config_path)
    except Exception:  # noqa: BLE001 — we want every failure visible
        print(_banner(f"{label}  →  RAISED"))
        traceback.print_exc()
        return
    _print_response(label, final_args, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Direct-Python probe for the three kernel index MCP tools.

            Tools:
              kernel_index_code      — general retrieval, returns ranked code snippets +
                                       file:line + graph-derived hints
              kernel_symbol_graph    — same engine, call-graph scenario forced; expects a
                                       focus-symbol list as the entry point
              kernel_hotspot_context — same engine, hotspot scenario forced; expects
                                       runtime_hints + focus symbols
            """),
    )
    parser.add_argument("--config", default="configs/app.yaml",
                        help="path to AppConfig YAML (default: configs/app.yaml)")
    parser.add_argument("--tool", choices=["all", "index_code", "symbol_graph",
                                            "hotspot_context"],
                        default="all",
                        help="which tool to probe (default: all)")
    parser.add_argument("--query", default=None,
                        help="override the query string for whichever tool(s) run")
    parser.add_argument("--symbol", action="append", default=None,
                        help="focus symbol; repeat for multiple "
                             "(e.g. --symbol foo --symbol bar)")
    parser.add_argument("--runtime-hints", default=None,
                        help="override runtime_hints (used by hotspot_context)")
    parser.add_argument("--scenario", default=None,
                        choices=["general", "implementation", "call_graph",
                                 "impact_analysis", "hotspot_debug", "patch_planning"],
                        help="override scenario (kernel_index_code only — graph and "
                             "hotspot tools force their own scenario)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="override top_k (1..64)")
    parser.add_argument("--graph-depth", type=int, default=None,
                        help="override graph_depth (1..4)")
    parser.add_argument("--format", choices=["markdown", "json"], default=None,
                        help="response_format (default: markdown for human reading; "
                             "use json to see the raw payload structure)")
    parser.add_argument("--also-json", action="store_true",
                        help="run each selected tool a second time with "
                             "response_format=json so you see both renderings")
    args = parser.parse_args(argv)

    # Resolve config path relative to repo root if it's not absolute.
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str(_REPO_ROOT / config_path)
    if not Path(config_path).is_file():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 2

    # Validate import early — surfaces missing deps before any tool call.
    try:
        from hmopt.api.mcp_service import get_tool_names
    except ImportError as exc:
        print(f"ERROR: hmopt import failed: {exc}", file=sys.stderr)
        print("Hint: PYTHONPATH=src or `pip install -e .`", file=sys.stderr)
        return 3

    names = get_tool_names()
    print(_banner("Configured tool names"))
    print(_format_args(names))
    print(f"Config: {config_path}")

    selected = (
        ["index_code", "symbol_graph", "hotspot_context"] if args.tool == "all"
        else [args.tool]
    )

    runners = {
        "index_code":      ("kernel_index_code",      call_kernel_index_code,      INDEX_CODE_DEFAULT),
        "symbol_graph":    ("kernel_symbol_graph",    call_kernel_symbol_graph,    SYMBOL_GRAPH_DEFAULT),
        "hotspot_context": ("kernel_hotspot_context", call_kernel_hotspot_context, HOTSPOT_CONTEXT_DEFAULT),
    }

    for key in selected:
        label, fn, defaults = runners[key]
        _run_one(label, fn, defaults, args, config_path)
        if args.also_json:
            json_args = argparse.Namespace(**vars(args), )
            json_args.format = "json"
            _run_one(f"{label} [json]", fn, defaults, json_args, config_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
