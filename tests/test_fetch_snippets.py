"""Tests for fetch_code_snippets — batch body retrieval."""

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


def _make_config() -> Any:
    return SimpleNamespace(
        indexing=SimpleNamespace(
            neo4j=SimpleNamespace(
                enabled=False,
                uri="bolt://localhost:7687",
                user="neo4j",
                password="x",
                database="neo4j",
            )
        )
    )


class _FakeNode:
    def __init__(self, text: str, **metadata: Any):
        self.text = text
        self.metadata = {"type": "code", **metadata}


def _patch_storage(monkeypatch, docstore_nodes: dict[str, _FakeNode]):
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

    fake_storage = SimpleNamespace(
        docstore=SimpleNamespace(docs=docstore_nodes),
        property_graph_store=None,
    )
    monkeypatch.setattr(
        pipeline,
        "_storage_context",
        lambda *a, **k: fake_storage,
    )
    monkeypatch.setattr(
        pipeline,
        "lookup_code_symbols",
        lambda config, symbols: {},
    )


def test_fetch_snippets_empty_input_returns_empty():
    cfg = _make_config()
    out = pipeline.fetch_code_snippets(cfg, [], output_format="json")
    assert out["snippets"] == []
    assert out["missing"] == []


def test_fetch_snippets_returns_bodies(monkeypatch):
    docstore = {
        "id-foo": _FakeNode(
            "void foo(){}",
            symbol_name="foo",
            symbol_qualname="foo",
            symbol_id="id-foo",
            path="/k/foo.c",
            start_line=10,
            end_line=12,
        ),
        "id-bar": _FakeNode(
            "void bar(){}",
            symbol_name="bar",
            symbol_qualname="bar",
            symbol_id="id-bar",
            path="/k/bar.c",
            start_line=20,
            end_line=25,
        ),
    }
    _patch_storage(monkeypatch, docstore)
    cfg = _make_config()

    out = pipeline.fetch_code_snippets(cfg, ["foo", "bar"], output_format="json")
    assert {s["symbol"] for s in out["snippets"]} == {"foo", "bar"}
    foo = next(s for s in out["snippets"] if s["symbol"] == "foo")
    assert foo["path"] == "/k/foo.c"
    assert foo["truncated"] is False
    assert foo["text"] == "void foo(){}"
    assert out["missing"] == []
    assert out["stats"]["returned"] == 2


def test_fetch_snippets_per_symbol_truncation(monkeypatch):
    body = "x" * 20000
    docstore = {
        "id-big": _FakeNode(
            body,
            symbol_name="big",
            symbol_qualname="big",
            symbol_id="id-big",
            path="/k/big.c",
            start_line=1,
            end_line=500,
        )
    }
    _patch_storage(monkeypatch, docstore)
    cfg = _make_config()

    out = pipeline.fetch_code_snippets(
        cfg, ["big"], per_symbol_max_chars=500, output_format="json"
    )
    snippet = out["snippets"][0]
    assert snippet["truncated"] is True
    assert snippet["text"].startswith("x" * 500)
    assert "[truncated" in snippet["text"]
    assert out["stats"]["truncated_count"] == 1


def test_fetch_snippets_missing_symbol(monkeypatch):
    _patch_storage(monkeypatch, docstore_nodes={})
    cfg = _make_config()
    out = pipeline.fetch_code_snippets(cfg, ["nope"], output_format="json")
    assert out["snippets"] == []
    assert out["missing"] == [{"symbol": "nope", "reason": "not_found"}]


def test_fetch_snippets_total_budget_marks_overflow(monkeypatch):
    docstore = {
        f"id-{i}": _FakeNode(
            "y" * 1000,
            symbol_name=f"sym{i}",
            symbol_qualname=f"sym{i}",
            symbol_id=f"id-{i}",
            path=f"/k/{i}.c",
            start_line=1,
            end_line=10,
        )
        for i in range(5)
    }
    _patch_storage(monkeypatch, docstore)
    cfg = _make_config()

    out = pipeline.fetch_code_snippets(
        cfg,
        [f"sym{i}" for i in range(5)],
        per_symbol_max_chars=2000,
        total_max_chars=2500,
        output_format="json",
    )
    # 1000-char snippets, budget 2500 → first 2 fit, the remaining 3 are missing as budget_hit.
    assert out["stats"]["budget_hit"] is True
    assert out["stats"]["returned"] == 2
    reasons = {m["reason"] for m in out["missing"]}
    assert reasons == {"budget_hit"}
    assert len(out["missing"]) == 3
