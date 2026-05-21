"""Tests for `_prepare_incremental_code_nodes` — scoped removal + accumulated manifest.

The function lives inside `hmopt.indexing.llamaindex_pipeline`, a module that
on import pulls in llama_index + neo4j + several MCP / storage subsystems.
To keep these tests fast and self-contained, we extract `_prepare_incremental_code_nodes`
and the two helpers it calls (`_node_fingerprint`, `_path_from_symbol_id`) via AST
and exec them in a hermetic namespace.

These tests guard against two related regressions:

1. **Cross-repo indexing must accumulate, not replace.** Running
   `hmopt index-kernel --repo-path A/` and then `--repo-path B/`
   used to compute the entire A manifest as "removed" and delete every
   A node + edge from Neo4j. The scoped-removal logic now only marks a
   symbol as removed when (a) it's missing from the current manifest
   AND (b) its source file path is in the set of paths the current
   run touched.

2. **The persisted manifest must be the accumulated state.** Previously
   the function returned just this run's symbols, which then overwrote
   the on-disk manifest — losing prior runs' fingerprints and
   defeating incremental re-indexing.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SRC = REPO_ROOT / "src" / "hmopt" / "indexing" / "llamaindex_pipeline.py"


def _extract_functions(source: Path, names: list[str]) -> dict[str, Callable]:
    """Parse `source` and exec only the named top-level functions in a
    hermetic namespace. The functions we extract only touch built-in
    APIs (dict, set, list, hashlib, json, str) plus typing.Optional —
    no need to import the heavy parts of the module."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    wanted = set(names)
    bodies: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            bodies.append(node)
    if {b.name for b in bodies} != wanted:
        missing = wanted - {b.name for b in bodies}
        raise RuntimeError(f"functions {missing!r} not found in {source}")

    import re as _re

    ns: dict[str, Any] = {
        "Optional": __import__("typing").Optional,
        "Any": Any,
        "hashlib": hashlib,
        "json": json,
        "re": _re,
        "Iterable": __import__("typing").Iterable,
        # Module-level helper compiled regex used by _path_from_symbol_id;
        # AST extraction only pulls function bodies, so module-level
        # constants must be re-supplied to the exec namespace.
        "_SYMBOL_ID_SUFFIX_RE": _re.compile(r":(\d+):([a-z_][a-z_0-9]*)$"),
        # _node_fingerprint expects something with .text and .metadata; we
        # pass FakeNode instances at call time, so no need to import TextNode.
    }
    mod = ast.Module(body=bodies, type_ignores=[])
    exec(compile(mod, str(source), "exec"), ns)
    return {name: ns[name] for name in names}


@pytest.fixture(scope="module")
def fns():
    extracted = _extract_functions(
        PIPELINE_SRC,
        ["_path_from_symbol_id", "_node_fingerprint", "_prepare_incremental_code_nodes"],
    )
    return extracted


@dataclass
class FakeNode:
    """Stand-in for llama_index TextNode — only the attributes
    `_prepare_incremental_code_nodes` and `_node_fingerprint` actually read."""

    metadata: dict = field(default_factory=dict)
    text: str = ""
    node_id: str = ""


def _make_node(*, path: str, qualname: str, line: int, kind: str, text: str = "body") -> FakeNode:
    sid = f"{path}:{qualname}:{line}:{kind}"
    return FakeNode(
        metadata={"type": "code", "symbol_id": sid},
        text=text,
        node_id=sid,
    )


# ---------------------------------------------------------------------------
# _path_from_symbol_id — pure string surgery
# ---------------------------------------------------------------------------


def test_path_from_symbol_id_basic_linux_path(fns):
    assert fns["_path_from_symbol_id"](
        "/home/repo/src/foo.c:do_work:42:function"
    ) == "/home/repo/src/foo.c"


def test_path_from_symbol_id_cxx_qualname_with_double_colon(fns):
    """C++ qualnames use `::`; the path-extracting rsplit must not be
    fooled by them. The expected path is everything LEFT of the third
    `:` counted from the right."""
    # symbol_id = "/repo/x.cpp:Foo::Bar::baz:120:method"
    assert fns["_path_from_symbol_id"](
        "/repo/x.cpp:Foo::Bar::baz:120:method"
    ) == "/repo/x.cpp"


def test_path_from_symbol_id_relative_path(fns):
    assert fns["_path_from_symbol_id"](
        "sysmgr/activation/actv_bind.c:bind_actv:88:function"
    ) == "sysmgr/activation/actv_bind.c"


def test_path_from_symbol_id_malformed_returns_none(fns):
    """No colons / too few colons → return None so callers can treat the
    id as "out of scope" rather than guessing."""
    assert fns["_path_from_symbol_id"]("not-a-symbol-id") is None
    assert fns["_path_from_symbol_id"]("only:two:fields") is None


# ---------------------------------------------------------------------------
# _prepare_incremental_code_nodes — cross-repo accumulation
# ---------------------------------------------------------------------------


def test_first_run_no_previous_manifest_marks_everything_changed(fns):
    nodes = [
        _make_node(path="/A/foo.c", qualname="foo", line=1, kind="function"),
        _make_node(path="/A/bar.c", qualname="bar", line=10, kind="function"),
    ]
    (
        symbol_nodes,
        removed,
        unchanged,
        changed,
        summary_nodes,
        manifest,
    ) = fns["_prepare_incremental_code_nodes"](nodes, {})

    assert len(symbol_nodes) == 2
    assert removed == []
    assert unchanged == []
    assert len(changed) == 2
    assert summary_nodes == []
    assert set(manifest.keys()) == {
        "/A/foo.c:foo:1:function",
        "/A/bar.c:bar:10:function",
    }


def test_indexing_second_repo_preserves_first_repos_symbols(fns):
    """The headline scenario: A indexed first, B indexed second.
    B's run must NOT mark A's symbols as removed, and the persisted
    manifest must contain BOTH A's and B's symbols."""
    # Set up the manifest as if /A/ was indexed previously.
    a_node = _make_node(path="/A/foo.c", qualname="foo", line=1, kind="function")
    previous_manifest = {a_node.node_id: fns["_node_fingerprint"](a_node)}

    # Now /B/ is being indexed — completely different paths.
    b_nodes = [
        _make_node(path="/B/bar.c", qualname="bar", line=5, kind="function"),
        _make_node(path="/B/baz.c", qualname="baz", line=12, kind="function"),
    ]
    (
        symbol_nodes,
        removed,
        unchanged,
        changed,
        _,
        manifest,
    ) = fns["_prepare_incremental_code_nodes"](b_nodes, previous_manifest)

    # /A/foo.c's symbol is NOT considered removed — its path was never
    # touched by this run, so it stays preserved both in Neo4j (because
    # `_delete_code_symbol_nodes` won't be called for it) and in the
    # persisted manifest.
    assert removed == []
    assert set(manifest.keys()) == {
        "/A/foo.c:foo:1:function",  # preserved from previous_manifest
        "/B/bar.c:bar:5:function",  # newly added this run
        "/B/baz.c:baz:12:function",
    }


def test_reindexing_same_path_with_deleted_symbol_marks_it_removed(fns):
    """The "you re-indexed the same file but a function disappeared"
    case — that function MUST be marked removed so the corresponding
    Neo4j nodes get cleaned up."""
    foo = _make_node(path="/A/foo.c", qualname="foo", line=1, kind="function")
    bar = _make_node(path="/A/foo.c", qualname="bar", line=10, kind="function")
    previous_manifest = {
        foo.node_id: fns["_node_fingerprint"](foo),
        bar.node_id: fns["_node_fingerprint"](bar),
    }

    # Re-index /A/foo.c; `bar` was deleted from the source.
    (
        _,
        removed,
        _,
        _,
        _,
        manifest,
    ) = fns["_prepare_incremental_code_nodes"]([foo], previous_manifest)

    assert removed == ["/A/foo.c:bar:10:function"]
    # bar dropped from manifest, foo retained.
    assert set(manifest.keys()) == {"/A/foo.c:foo:1:function"}


def test_reindexing_with_renamed_symbol_keeps_repo_data_intact(fns):
    """Mixed case: re-index repo A where bar.c was modified (one fn
    renamed) but baz.c in repo B is completely untouched and must
    survive."""
    foo_v1 = _make_node(path="/A/foo.c", qualname="old_name", line=1, kind="function")
    bar_v1 = _make_node(path="/A/bar.c", qualname="stable_fn", line=5, kind="function")
    b_baz = _make_node(path="/B/baz.c", qualname="baz", line=12, kind="function")
    previous_manifest = {
        foo_v1.node_id: fns["_node_fingerprint"](foo_v1),
        bar_v1.node_id: fns["_node_fingerprint"](bar_v1),
        b_baz.node_id: fns["_node_fingerprint"](b_baz),
    }

    # Re-index /A/: old_name was renamed to new_name (different symbol_id)
    foo_v2 = _make_node(path="/A/foo.c", qualname="new_name", line=1, kind="function")
    bar_unchanged = _make_node(path="/A/bar.c", qualname="stable_fn", line=5, kind="function")

    (
        symbol_nodes,
        removed,
        unchanged,
        changed,
        _,
        manifest,
    ) = fns["_prepare_incremental_code_nodes"](
        [foo_v2, bar_unchanged], previous_manifest
    )

    # Within the touched paths (/A/foo.c, /A/bar.c):
    #   - old_name disappeared from /A/foo.c → removed
    #   - new_name appeared in /A/foo.c → changed (will upsert)
    #   - stable_fn in /A/bar.c → unchanged
    assert removed == ["/A/foo.c:old_name:1:function"]
    assert changed == ["/A/foo.c:new_name:1:function"]
    assert unchanged == ["/A/bar.c:stable_fn:5:function"]
    # /B/baz.c was never touched → preserved.
    assert "/B/baz.c:baz:12:function" in manifest
    # Final manifest reflects all three sources of truth.
    assert set(manifest.keys()) == {
        "/A/foo.c:new_name:1:function",
        "/A/bar.c:stable_fn:5:function",
        "/B/baz.c:baz:12:function",
    }


def test_summary_nodes_dont_affect_removal_logic(fns):
    """file_summary / symbol_relations nodes don't have 'type:code' and
    must be routed to the summary bucket without polluting either
    current_manifest or removed_symbol_ids."""
    code = _make_node(path="/A/foo.c", qualname="foo", line=1, kind="function")
    summary = FakeNode(
        metadata={"type": "file_summary", "path": "/A/foo.c"},
        text="file summary: ...",
        node_id="file_summary::/A/foo.c",
    )
    previous_manifest = {
        "/A/old.c:old:1:function": "somehash",  # not touched, must be preserved
    }

    (
        symbol_nodes,
        removed,
        _,
        _,
        summary_nodes,
        manifest,
    ) = fns["_prepare_incremental_code_nodes"]([code, summary], previous_manifest)

    assert len(symbol_nodes) == 1
    assert len(summary_nodes) == 1
    assert removed == []  # /A/old.c never touched by this run
    assert "/A/old.c:old:1:function" in manifest


def test_corrupt_symbol_id_in_previous_manifest_stays_preserved(fns):
    """If the persisted manifest somehow contains a malformed
    symbol_id (from a buggy older release, or hand-edited), don't
    delete the corresponding Neo4j node just because we can't parse
    its path. Leave it as-is — better to keep orphaned data than to
    drop valid data."""
    new_node = _make_node(path="/A/foo.c", qualname="foo", line=1, kind="function")
    previous_manifest = {
        "malformed-no-colons-here": "stalehash",
        "/A/old_but_present.c:old:1:function": "anotherhash",
    }

    (
        _,
        removed,
        _,
        _,
        _,
        manifest,
    ) = fns["_prepare_incremental_code_nodes"]([new_node], previous_manifest)

    assert "malformed-no-colons-here" not in removed
    assert "/A/old_but_present.c:old:1:function" not in removed
    assert "malformed-no-colons-here" in manifest
