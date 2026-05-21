"""Tests for backend-origin filtering and scip-clang field surfacing
across the MCP-facing retrieval helpers.

Targets the small helpers (`_normalize_backend_filter`,
`_backend_filter_for_relation`, `_backend_filter_for_node`,
`_call_site_fields`) plus the two markdown renderers
(`_render_markdown_payload` Graph Edges block, `snippets_to_markdown`).
All are pure functions; AST-extracting them lets us exercise the
production source without spinning up llama_index + Neo4j + FastMCP.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SRC = REPO_ROOT / "src" / "hmopt" / "indexing" / "llamaindex_pipeline.py"
MCP_SRC = REPO_ROOT / "src" / "hmopt" / "api" / "mcp_service.py"


def _extract_functions(source: Path, names: list[str], extra_ns: dict[str, Any] | None = None) -> dict[str, Callable]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    wanted = set(names)
    bodies = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    missing = wanted - {b.name for b in bodies}
    if missing:
        raise RuntimeError(f"functions {missing!r} not found in {source}")
    ns: dict[str, Any] = {
        "Any": Any,
        "Optional": __import__("typing").Optional,
        "Mapping": __import__("typing").Mapping,
        "Iterable": __import__("typing").Iterable,
        "re": re,
        "json": json,
    }
    if extra_ns:
        ns.update(extra_ns)
    mod = ast.Module(body=bodies, type_ignores=[])
    exec(compile(mod, str(source), "exec"), ns)
    return {name: ns[name] for name in names}


# ---------------------------------------------------------------------------
# _normalize_backend_filter — the gatekeeper
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def normalize():
    extracted = _extract_functions(PIPELINE_SRC, ["_normalize_backend_filter"], extra_ns={
        "_VALID_BACKEND_FILTERS": {"clangd", "scip-clang"},
    })
    return extracted["_normalize_backend_filter"]


def test_normalize_returns_none_for_disabled_values(normalize):
    """None, '', 'any', 'all', '*' all disable filtering. Callers can
    short-circuit on None instead of constructing empty Cypher clauses."""
    for value in (None, "", "any", "ALL", "All", "*"):
        assert normalize(value) is None


def test_normalize_lowercases_and_strips(normalize):
    assert normalize(" SCIP-CLANG ") == "scip-clang"
    assert normalize("Clangd") == "clangd"


def test_normalize_rejects_unknown_value(normalize):
    """A typo at the MCP boundary surfaces as ValueError, not silent
    empty result — easier to debug downstream."""
    with pytest.raises(ValueError, match="backend must be one of"):
        normalize("scip-clangd")  # common typo, the binary is scip-clang
    with pytest.raises(ValueError):
        normalize("c")


# ---------------------------------------------------------------------------
# _backend_filter_for_relation — Cypher fragment builder
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rel_filter():
    extracted = _extract_functions(
        PIPELINE_SRC,
        ["_normalize_backend_filter", "_backend_filter_for_relation"],
        extra_ns={"_VALID_BACKEND_FILTERS": {"clangd", "scip-clang"}},
    )
    return extracted["_backend_filter_for_relation"]


def test_relation_filter_disabled_returns_empty(rel_filter):
    """No filter requested → empty clause, empty params. Caller's
    `{symbols: ..., **backend_params}` spread still works (no-op)."""
    clause, params = rel_filter(None, "r")
    assert clause == ""
    assert params == {}
    clause, params = rel_filter("any", "r")
    assert clause == ""
    assert params == {}


def test_relation_filter_scip_clang_exact_match(rel_filter):
    clause, params = rel_filter("scip-clang", "r")
    assert clause == " AND r.backend_origin = $backend_origin"
    assert params == {"backend_origin": "scip-clang"}


def test_relation_filter_clangd_includes_legacy_null(rel_filter):
    """`clangd` filter must include legacy NULL backend_origin values
    so users migrating from a pre-schema index don't silently lose data."""
    clause, params = rel_filter("clangd", "r")
    assert "r.backend_origin IS NULL" in clause
    assert "r.backend_origin = $backend_origin" in clause
    assert params == {"backend_origin": "clangd"}


def test_relation_filter_respects_alias(rel_filter):
    """The alias parameter lets callers compose the clause against an
    arbitrary Cypher variable name."""
    clause, _ = rel_filter("scip-clang", "edge42")
    assert "edge42.backend_origin = $backend_origin" in clause


# ---------------------------------------------------------------------------
# _backend_filter_for_node — same shape but on a node property
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def node_filter():
    extracted = _extract_functions(
        PIPELINE_SRC,
        ["_normalize_backend_filter", "_backend_filter_for_node"],
        extra_ns={"_VALID_BACKEND_FILTERS": {"clangd", "scip-clang"}},
    )
    return extracted["_backend_filter_for_node"]


def test_node_filter_uses_backend_origin_by_default(node_filter):
    clause, _ = node_filter("scip-clang", "n")
    assert "n.backend_origin = $backend_origin" in clause


def test_node_filter_supports_parser_property(node_filter):
    """`:CodeChunk` (vector store) nodes use `parser` instead of
    `backend_origin` — the helper takes a property_name override so
    both can share the same Cypher composition logic."""
    clause, _ = node_filter("scip-clang", "c", property_name="parser")
    assert "c.parser = $backend_origin" in clause


# ---------------------------------------------------------------------------
# _call_site_fields — strip nulls from Cypher rows
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cs_fields():
    extracted = _extract_functions(PIPELINE_SRC, ["_call_site_fields"])
    return extracted["_call_site_fields"]


def test_call_site_fields_strips_nulls(cs_fields):
    """When scip-clang's enrichments aren't populated (clangd-origin
    edge), Cypher returns NULL for those columns — the helper must
    drop them so they don't appear in the markdown output."""
    rec = {
        "cs_path": None,
        "cs_line": None,
        "cs_col": None,
        "cs_syntax_kind": None,
        "cs_is_write": None,
        "cs_backend_origin": None,
    }
    assert cs_fields(rec) == {}


def test_call_site_fields_carries_scip_values_through(cs_fields):
    rec = {
        "cs_path": "sysmgr/sched/core.c",
        "cs_line": 142,
        "cs_col": 18,
        "cs_syntax_kind": 15,
        "cs_is_write": True,
        "cs_backend_origin": "scip-clang",
    }
    result = cs_fields(rec)
    assert result["call_site_path"] == "sysmgr/sched/core.c"
    assert result["call_site_line"] == 142
    assert result["call_site_col"] == 18
    assert result["syntax_kind"] == 15
    assert result["is_write"] is True
    assert result["backend_origin"] == "scip-clang"


# ---------------------------------------------------------------------------
# _render_markdown_payload — Graph Edges section with call-site suffix
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def render_markdown():
    extracted = _extract_functions(MCP_SRC, ["_render_markdown_payload"])
    return extracted["_render_markdown_payload"]


def test_render_markdown_clangd_edge_has_no_suffix(render_markdown):
    """An edge with no call_site_path renders exactly like before the
    extension — preserving output stability for clangd-only users."""
    payload = {
        "scenario": "general",
        "query": "test",
        "graph_edges": [
            {"src": "funcA", "dst": "funcB", "rel": "calls", "depth": 1},
        ],
    }
    md = render_markdown(payload)
    # The line should look like: "- funcA -[calls]-> funcB (depth=1)"
    # with NO " @file:line[:col]" or " [write]" suffix.
    edge_line = next(l for l in md.splitlines() if "funcA" in l and "funcB" in l)
    assert "@" not in edge_line
    assert "[write]" not in edge_line


def test_render_markdown_scip_edge_shows_call_site(render_markdown):
    payload = {
        "scenario": "general",
        "query": "test",
        "graph_edges": [
            {
                "src": "caller",
                "dst": "callee",
                "rel": "calls",
                "depth": 1,
                "call_site_path": "sysmgr/sched/core.c",
                "call_site_line": 142,
                "call_site_col": 18,
                "is_write": False,
                "backend_origin": "scip-clang",
            },
        ],
    }
    md = render_markdown(payload)
    assert "@sysmgr/sched/core.c:142:18" in md
    # is_write is False so no [write] tag should appear
    assert "[write]" not in md


def test_render_markdown_write_edge_gets_write_tag(render_markdown):
    payload = {
        "scenario": "general",
        "query": "",
        "graph_edges": [
            {
                "src": "kthread",
                "dst": "task_struct.flags",
                "rel": "uses_field",
                "depth": 2,
                "call_site_path": "sysmgr/sched/wait.c",
                "call_site_line": 91,
                "call_site_col": 7,
                "is_write": True,
                "backend_origin": "scip-clang",
            }
        ],
    }
    md = render_markdown(payload)
    assert "@sysmgr/sched/wait.c:91:7 [write]" in md


def test_render_markdown_line_only_when_col_missing(render_markdown):
    """If scip-clang didn't report a column (older proto, or specific
    occurrence shape), still surface path:line instead of nothing."""
    payload = {
        "scenario": "general",
        "query": "",
        "graph_edges": [
            {
                "src": "a",
                "dst": "b",
                "rel": "calls",
                "depth": 1,
                "call_site_path": "x.c",
                "call_site_line": 10,
                # call_site_col absent
            }
        ],
    }
    md = render_markdown(payload)
    assert "@x.c:10" in md
    # Ensure we didn't render a trailing colon or "None" sentinel
    edge_line = next(l for l in md.splitlines() if "@x.c" in l)
    assert "@x.c:10:None" not in edge_line
    assert edge_line.rstrip().endswith("@x.c:10")


# ---------------------------------------------------------------------------
# snippets_to_markdown — signature + documentation rendering
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def render_snippets():
    extracted = _extract_functions(MCP_SRC, ["snippets_to_markdown"])
    return extracted["snippets_to_markdown"]


def test_snippets_clangd_only_no_signature_no_doc(render_snippets):
    payload = {
        "snippets": [
            {
                "symbol": "foo",
                "path": "/a/foo.c",
                "start_line": 10,
                "end_line": 20,
                "char_count": 300,
                "truncated": False,
                "text": "int foo(void) { return 0; }",
            }
        ],
        "stats": {"returned": 1, "missing": 0, "truncated_count": 0, "total_chars": 300, "budget_hit": False},
    }
    md = render_snippets(payload)
    assert "## foo" in md
    assert "int foo(void)" in md
    # No scip-only sections show up for a clangd-shape entry
    assert "Signature:" not in md
    assert "Documentation:" not in md
    assert "backend=" not in md


def test_snippets_scip_clang_includes_signature_and_doc(render_snippets):
    payload = {
        "snippets": [
            {
                "symbol": "wake_up_process",
                "path": "/k/sched/core.c",
                "start_line": 100,
                "end_line": 150,
                "char_count": 500,
                "truncated": False,
                "text": "int wake_up_process(struct task_struct *p) { ... }",
                "backend_origin": "scip-clang",
                "signature": "int wake_up_process(struct task_struct *p)",
                "documentation": [
                    "/**\n * Wake up a sleeping task.\n * @p: task to wake.\n */",
                ],
            }
        ],
        "stats": {"returned": 1, "missing": 0, "truncated_count": 0, "total_chars": 500, "budget_hit": False},
    }
    md = render_snippets(payload)
    assert "backend=scip-clang" in md
    assert "Signature: `int wake_up_process(struct task_struct *p)`" in md
    # Doc block is fenced and prefixed by `  ` so it renders as nested commentary
    assert "Documentation:" in md
    assert "Wake up a sleeping task." in md


# ---------------------------------------------------------------------------
# _resolve_effective_backend — MCP layer's app.yaml fallback
#
# Default semantics: when caller doesn't pass `backend` explicitly,
# fall back to `config.indexing.backend` so MCP queries surface data
# from whichever backend the deployment is currently using. Explicit
# overrides (including "any" / "all" to disable filtering entirely)
# bypass the fallback.
# ---------------------------------------------------------------------------


class _FakeIndexing:
    def __init__(self, backend):
        self.backend = backend


class _FakeConfig:
    def __init__(self, backend):
        self.indexing = _FakeIndexing(backend)


@pytest.fixture(scope="module")
def resolve_effective():
    extracted = _extract_functions(MCP_SRC, ["_resolve_effective_backend"])
    return extracted["_resolve_effective_backend"]


def test_explicit_request_wins_over_config(resolve_effective):
    """If the MCP caller passed `backend="clangd"`, the config's
    `backend: scip-clang` MUST NOT override it. Users running A/B
    comparisons rely on the explicit value being respected."""
    cfg = _FakeConfig("scip-clang")
    assert resolve_effective(cfg, "clangd") == "clangd"


def test_falls_back_to_config_when_caller_omits(resolve_effective):
    """The headline behavior: omit `backend` → use config's backend.
    No more typing `backend="scip-clang"` on every MCP call when the
    deployment is already configured that way."""
    cfg = _FakeConfig("scip-clang")
    assert resolve_effective(cfg, None) == "scip-clang"


def test_falls_back_to_config_when_caller_passes_empty_string(resolve_effective):
    """Empty / whitespace-only string is the same as omitted."""
    cfg = _FakeConfig("clangd")
    assert resolve_effective(cfg, "") == "clangd"
    assert resolve_effective(cfg, "   ") == "clangd"


def test_explicit_any_disables_filtering_even_when_config_set(resolve_effective):
    """The opt-out escape hatch: when a graph contains both backends'
    data, passing "any" / "all" / "*" lets the caller query everything
    regardless of what app.yaml configured. _normalize_backend_filter
    further downstream treats these as no-filter."""
    cfg = _FakeConfig("scip-clang")
    for opt_out in ("any", "all", "*", "ALL", "AnY"):
        assert resolve_effective(cfg, opt_out) == opt_out


def test_no_filter_when_config_missing_indexing_backend(resolve_effective):
    """If app.yaml has no backend setting and the caller didn't pass
    one either, no filter applies — preserves the original
    behavior of single-backend deployments that never knew about
    backend_origin tags."""
    cfg = _FakeConfig(None)
    assert resolve_effective(cfg, None) is None


def test_no_filter_when_config_object_missing_indexing_attr(resolve_effective):
    """Tolerate config objects that don't expose `.indexing` (unit-test
    stubs, partial configs). Returns None → downstream treats as no
    filter, so no exception leaks into the MCP response."""
    class _CfgNoIndexing:
        pass
    assert resolve_effective(_CfgNoIndexing(), None) is None


def test_snippets_scip_clang_documentation_multiline_preserves_layout(render_snippets):
    payload = {
        "snippets": [
            {
                "symbol": "fn",
                "path": "/x.c",
                "start_line": 1,
                "end_line": 5,
                "char_count": 100,
                "truncated": False,
                "text": "...",
                "documentation": ["line one\nline two", "another block"],
            }
        ],
        "stats": {"returned": 1, "missing": 0, "truncated_count": 0, "total_chars": 100, "budget_hit": False},
    }
    md = render_snippets(payload)
    assert "line one" in md
    assert "line two" in md
    assert "another block" in md
