"""Tests for skillhub.records + skillhub.retrieval (Phase 1 read path)."""
from __future__ import annotations

from pathlib import Path


from hmopt.skillhub.embeddings import HashingEmbedder, cosine, tokenize
from hmopt.skillhub.records import Record, load_records, maturity_at_least
from hmopt.skillhub.retrieval import HybridRetriever, ScopeFilter, extract_entities

REPO = Path(__file__).resolve().parent.parent
HUB_KNOWLEDGE = REPO / "hm-skill-hub" / "knowledge"


# ---- records ---------------------------------------------------------------

def test_load_hub_records():
    recs = load_records(HUB_KNOWLEDGE)
    by_id = {r.id: r for r in recs}
    assert {"A001", "H001", "B001", "V001", "F001", "L001"} <= set(by_id)
    f001 = by_id["F001"]
    assert f001.kind == "memory_item"
    assert f001.target_slug == "mm-vmscan-c-shrink-node"
    assert f001.level == "function"
    assert f001.maturity == "L2"
    assert f001.subsystem == "mm-reclaim"


def test_global_lesson_normalization():
    recs = {r.id: r for r in load_records(HUB_KNOWLEDGE)}
    h001 = recs["H001"]
    assert h001.kind == "global_lesson"
    assert h001.level == "global"
    assert h001.maturity == "L2"  # confidence: observed -> L2


def test_bad_plan_global_scope():
    recs = {r.id: r for r in load_records(HUB_KNOWLEDGE)}
    b001 = recs["B001"]
    assert b001.kind == "bad_plan"
    assert b001.level == "global"
    assert b001.mechanism == "inline-callee"


def test_maturity_at_least():
    assert maturity_at_least("L2", "L2")
    assert maturity_at_least("L3", "L2")
    assert not maturity_at_least("L1", "L2")


# ---- embeddings ------------------------------------------------------------

def test_tokenize_expands_symbols():
    toks = tokenize("mm/vmscan.c::shrink_node")
    assert "shrink_node" in toks and "shrink" in toks and "node" in toks


def test_hashing_embedder_shared_tokens_have_positive_cosine():
    e = HashingEmbedder()
    a = e.embed("hoist loop invariant shrink_node")
    b = e.embed("shrink_node loop invariant hoist")
    c = e.embed("completely unrelated batch coalescing workqueue")
    assert cosine(a, b) > cosine(a, c)
    assert cosine(a, b) > 0.5


# ---- retrieval -------------------------------------------------------------

def _retriever() -> HybridRetriever:
    return HybridRetriever(load_records(HUB_KNOWLEDGE))


def test_target_anchored_retrieval_finds_F001():
    r = _retriever()
    hits = r.retrieve(
        "mm/vmscan.c::shrink_node sc->priority hoist",
        scope=ScopeFilter(subsystem="mm-reclaim", target_slug="mm-vmscan-c-shrink-node"),
        k=5,
    )
    ids = [h.record.id for h in hits]
    assert "F001" in ids


def test_scalar_filter_excludes_out_of_scope_targets():
    # a workqueue query scoped to wq must not pull the mm target fact F001
    r = _retriever()
    hits = r.retrieve(
        "kworker inline process_one_work",
        scope=ScopeFilter(subsystem="workqueue-threadpool"),
        k=5,
    )
    ids = [h.record.id for h in hits]
    assert "F001" not in ids
    # B001 (global) is still allowed through scope (broadly applicable)
    assert "B001" in ids


def test_maturity_floor_filters_low_maturity():
    recs = load_records(HUB_KNOWLEDGE)
    recs.append(
        Record(
            id="F900",
            kind="memory_item",
            path="x",
            fields={"title": "draft shrink_node note", "maturity": "L0", "status": "active",
                    "scope": {"level": "function", "target_slug": "mm-vmscan-c-shrink-node"}},
            body="shrink_node draft",
        )
    )
    r = HybridRetriever(recs)
    hits = r.retrieve("shrink_node", maturity_floor="L2", k=10)
    assert "F900" not in [h.record.id for h in hits]
    hits_all = r.retrieve("shrink_node", maturity_floor="L0", k=10)
    assert "F900" in [h.record.id for h in hits_all]


def test_status_filter_excludes_superseded():
    recs = load_records(HUB_KNOWLEDGE)
    recs.append(
        Record(id="H900", kind="global_lesson", path="x",
                fields={"lesson": "old hoist advice", "status": "superseded", "confidence": "observed"},
                body="hoist old"))
    r = HybridRetriever(recs)
    assert "H900" not in [h.record.id for h in r.retrieve("hoist", k=10)]


def test_hybrid_beats_single_arms_on_symbol_query():
    """Plan P1-10 ④: for a symbol-name query, hybrid recall >= each single arm.

    Build a corpus where the wanted record shares the symbol token (favors BM25)
    and a distractor shares paraphrase tokens (favors vectors); hybrid should
    rank the symbol match at top at least as reliably as either arm alone.
    """
    recs = [
        Record(id="T1", kind="memory_item", path="x",
                fields={"title": "shrink_node reclaim loop invariant", "maturity": "L2",
                        "scope": {"level": "function", "target_slug": "t"}},
                body="in shrink_node hoist the invariant out of the reclaim loop"),
        Record(id="T2", kind="memory_item", path="x",
                fields={"title": "generic loop hoisting wisdom", "maturity": "L2",
                        "scope": {"level": "function", "target_slug": "t"}},
                body="move invariant computation out of hot loops in general"),
    ]
    r = HybridRetriever(recs)
    q = "shrink_node"
    top_hybrid = r.retrieve(q, mode="hybrid", k=1)[0].record.id
    top_bm25 = r.retrieve(q, mode="bm25", k=1)[0].record.id
    assert top_hybrid == "T1"  # symbol match wins under hybrid
    assert top_bm25 == "T1"
    # hybrid is not worse than bm25 here, and (unlike a hypothetical vector-only
    # paraphrase bias) keeps the exact symbol match on top.


def test_extract_entities():
    ents = extract_entities("hoist sc->priority in shrink_node reclaim")
    assert "shrink_node" in ents


def test_relevance_floor_drops_zero_signal_records():
    # review F5: a record with no lexical (bm25) match and only stopword-level
    # cosine must NOT be mounted (context pollution §12.1 warns about).
    recs = [
        Record(id="REL", kind="memory_item", path="x",
               fields={"title": "shrink_node hoist invariant", "maturity": "L2",
                       "scope": {"level": "function", "target_slug": "t"}},
               body="in shrink_node hoist the loop-invariant read out of the reclaim loop"),
        Record(id="NOISE", kind="memory_item", path="x",
               fields={"title": "workqueue flush batch coalescing latency", "maturity": "L2",
                       "scope": {"level": "function", "target_slug": "t"}},
               body="coalesce small writes into one batch round-trip on the hyperhold path"),
    ]
    r = HybridRetriever(recs)
    ids = [h.record.id for h in r.retrieve("shrink_node hoist invariant", k=5)]
    assert "REL" in ids and "NOISE" not in ids


def test_score_weighting_breaks_ties():
    recs = [
        Record(id="S1", kind="memory_item", path="x",
                fields={"title": "hoist invariant", "maturity": "L2", "score": 5.0,
                        "scope": {"level": "global"}}, body="hoist invariant"),
        Record(id="S2", kind="memory_item", path="x",
                fields={"title": "hoist invariant", "maturity": "L2", "score": -5.0,
                        "scope": {"level": "global"}}, body="hoist invariant"),
    ]
    r = HybridRetriever(recs)
    hits = r.retrieve("hoist invariant", k=2)
    assert hits[0].record.id == "S1"  # higher promotion score ranks first
