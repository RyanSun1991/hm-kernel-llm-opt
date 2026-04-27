"""Tests for retrieve_call_chain — pure structural call-chain retrieval."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("llama_index.core")
pytest.importorskip("llama_index.embeddings.openai")
pytest.importorskip("llama_index.llms.openai")
pytest.importorskip("llama_index.graph_stores.neo4j")
pytest.importorskip("llama_index.vector_stores.neo4jvector")

# Workaround for an unrelated pre-existing import in flamegraph_parser.py that resolves
# `..preprocess`. If the real module is absent we stub it; if it exists, we leave it alone.
if "hmopt.analysis.runtime.preprocess" not in sys.modules:
    try:
        import hmopt.analysis.runtime.preprocess  # noqa: F401
    except ModuleNotFoundError:
        _stub = types.ModuleType("hmopt.analysis.runtime.preprocess")
        _stub.preprocess_flamegraph_dir = lambda *a, **k: []  # type: ignore[attr-defined]
        _stub.preprocess_flamegraph_file = lambda *a, **k: (None, None)  # type: ignore[attr-defined]
        sys.modules["hmopt.analysis.runtime.preprocess"] = _stub

try:
    from hmopt.indexing import llamaindex_pipeline as pipeline  # noqa: E402
except ModuleNotFoundError as exc:
    pytest.skip(f"llamaindex_pipeline import failed: {exc}", allow_module_level=True)


def _make_config(*, neo4j_enabled: bool = True) -> Any:
    return SimpleNamespace(
        indexing=SimpleNamespace(
            neo4j=SimpleNamespace(
                enabled=neo4j_enabled,
                uri="bolt://localhost:7687",
                user="neo4j",
                password="x",
                database="neo4j",
            )
        )
    )


def _make_row(
    src: str,
    dst: str,
    *,
    rel: str = "calls",
    dst_path: str | None = None,
    dst_start: int | None = None,
    dst_end: int | None = None,
    dst_kind: str | None = "function",
    src_path: str | None = None,
    src_start: int | None = None,
    src_end: int | None = None,
    src_kind: str | None = "function",
    call_site_path: str | None = None,
    call_site_line: int | None = None,
) -> dict[str, Any]:
    return {
        "src": src,
        "src_qual": src,
        "rel": rel,
        "dst": dst,
        "dst_qual": dst,
        "dst_path": dst_path,
        "dst_start": dst_start,
        "dst_end": dst_end,
        "dst_kind": dst_kind,
        "src_path": src_path,
        "src_start": src_start,
        "src_end": src_end,
        "src_kind": src_kind,
        "call_site_path": call_site_path,
        "call_site_line": call_site_line,
    }


class _StubVectorStore:
    """Records queries; replays canned rows keyed by (direction, depth-layer)."""

    def __init__(self, edges: dict[str, list[dict[str, Any]]]):
        self.edges = edges  # symbol -> outgoing rows (used for both directions in stub)
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def database_query(self, cypher: str, params: dict[str, Any]):
        self.queries.append((cypher, dict(params)))
        kinds = set(params.get("kinds") or [])
        is_outgoing = "->" in cypher and "<-" not in cypher
        rows: list[dict[str, Any]] = []
        symbols_param = params.get("symbols") or []
        for sym in symbols_param:
            for row in self.edges.get(sym, []):
                if kinds and row.get("rel") not in kinds:
                    continue
                if is_outgoing:
                    rows.append(row)
                else:
                    flipped = dict(row)
                    flipped["src_path"] = row.get("dst_path")
                    flipped["src_start"] = row.get("dst_start")
                    flipped["src_end"] = row.get("dst_end")
                    flipped["src_kind"] = row.get("dst_kind")
                    rows.append(flipped)
        limit = int(params.get("limit") or 0) or len(rows)
        return rows[:limit]


def _patch_common(monkeypatch, vector_store):
    monkeypatch.setattr(
        pipeline,
        "_index_paths",
        lambda *a, **k: SimpleNamespace(code_dir="dummy/code"),
    )
    monkeypatch.setattr(
        pipeline,
        "_neo4j_index_config",
        lambda kind: SimpleNamespace(index_name="x", node_label="symbol"),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_embedding_metadata",
        lambda persist_dir: {"dimension": 16},
    )
    monkeypatch.setattr(
        pipeline,
        "Neo4jVectorStore",
        lambda **kwargs: vector_store,
    )
    monkeypatch.setattr(
        pipeline,
        "lookup_code_symbols",
        lambda config, symbols: {},
    )


def test_call_chain_returns_empty_when_neo4j_disabled():
    cfg = _make_config(neo4j_enabled=False)
    out = pipeline.retrieve_call_chain(cfg, ["foo"], output_format="json")
    assert out["edges"] == []
    assert out["nodes"] == {}
    assert out["stats"]["neo4j_disabled"] is True


def test_call_chain_returns_empty_for_no_symbols():
    cfg = _make_config(neo4j_enabled=False)
    out = pipeline.retrieve_call_chain(cfg, [], output_format="json")
    assert out["edges"] == []
    assert out["nodes"] == {}


def test_call_chain_callees_only_traverses_outgoing(monkeypatch):
    edges = {
        "root": [_make_row("root", "child_a", dst_path="/k/a.c", dst_start=10, dst_end=20)],
        "child_a": [_make_row("child_a", "leaf_x", dst_path="/k/x.c", dst_start=1, dst_end=5)],
    }
    store = _StubVectorStore(edges)
    cfg = _make_config(neo4j_enabled=True)
    _patch_common(monkeypatch, store)

    out = pipeline.retrieve_call_chain(
        cfg,
        ["root"],
        direction="callees",
        depth=2,
        per_hop_limit=10,
        output_format="json",
    )

    rels = {(e["src"], e["dst"], e["direction"]) for e in out["edges"]}
    assert ("root", "child_a", "callee") in rels
    assert ("child_a", "leaf_x", "callee") in rels
    assert all(e["direction"] == "callee" for e in out["edges"])
    assert out["stats"]["depth_reached"] == 2
    assert all("<-" not in q[0] for q in store.queries)


def test_call_chain_callers_only_traverses_incoming(monkeypatch):
    edges = {
        "target": [_make_row("caller_a", "target", src_path="/k/a.c")],
    }
    store = _StubVectorStore(edges)
    cfg = _make_config(neo4j_enabled=True)
    _patch_common(monkeypatch, store)

    out = pipeline.retrieve_call_chain(
        cfg,
        ["target"],
        direction="callers",
        depth=1,
        output_format="json",
    )

    assert all(e["direction"] == "caller" for e in out["edges"])
    assert any(q[0].count("<-") for q in store.queries)
    assert all("->" not in q[0].split("RETURN")[0] or "<-" in q[0] for q in store.queries)


def test_call_chain_edge_kinds_filter(monkeypatch):
    edges = {
        "root": [
            _make_row("root", "fn_a", rel="calls"),
            _make_row("root", "type_b", rel="uses_type"),
        ]
    }
    store = _StubVectorStore(edges)
    cfg = _make_config(neo4j_enabled=True)
    _patch_common(monkeypatch, store)

    out = pipeline.retrieve_call_chain(
        cfg,
        ["root"],
        direction="callees",
        depth=1,
        edge_kinds=["calls"],
        output_format="json",
    )
    assert {e["dst"] for e in out["edges"]} == {"fn_a"}
    for _cypher, params in store.queries:
        assert params["kinds"] == ["calls"]


def test_call_chain_frontier_cap_truncates(monkeypatch):
    edges = {f"root_{i}": [_make_row(f"root_{i}", f"child_{i}")] for i in range(60)}
    store = _StubVectorStore(edges)
    cfg = _make_config(neo4j_enabled=True)
    _patch_common(monkeypatch, store)

    roots = list(edges.keys())
    out = pipeline.retrieve_call_chain(
        cfg,
        roots,
        direction="callees",
        depth=1,
        frontier_cap=10,
        per_hop_limit=200,
        output_format="json",
    )
    # frontier_cap=10 means only 10 of 60 roots get expanded at depth 1.
    assert len(out["edges"]) == 10
    assert 1 in out["stats"]["hops_truncated_at"]


def test_call_chain_carries_call_site_metadata(monkeypatch):
    edges = {
        "root": [
            _make_row(
                "root",
                "child",
                dst_path="/k/c.c",
                dst_start=42,
                dst_end=99,
                call_site_path="/k/root.c",
                call_site_line=123,
            )
        ]
    }
    store = _StubVectorStore(edges)
    cfg = _make_config(neo4j_enabled=True)
    _patch_common(monkeypatch, store)

    out = pipeline.retrieve_call_chain(
        cfg, ["root"], direction="callees", depth=1, output_format="json"
    )
    edge = out["edges"][0]
    assert edge["call_site_path"] == "/k/root.c"
    assert edge["call_site_line"] == 123
    node = out["nodes"]["child"]
    assert node["path"] == "/k/c.c"
    assert node["start_line"] == 42
    assert node["end_line"] == 99
    assert node["depth"] == 1


def test_call_chain_depth_clamped_to_six(monkeypatch):
    edges: dict[str, list[dict[str, Any]]] = {}
    for i in range(10):
        src = f"n{i}"
        dst = f"n{i + 1}"
        edges[src] = [_make_row(src, dst)]
    store = _StubVectorStore(edges)
    cfg = _make_config(neo4j_enabled=True)
    _patch_common(monkeypatch, store)

    out = pipeline.retrieve_call_chain(
        cfg,
        ["n0"],
        direction="callees",
        depth=99,  # should clamp to 6
        per_hop_limit=10,
        output_format="json",
    )
    assert out["depth_requested"] == 6
    assert out["stats"]["depth_reached"] == 6
    assert len(out["edges"]) == 6
