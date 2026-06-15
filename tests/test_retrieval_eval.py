"""Retrieval eval hard gate (design §12, plan P1-10)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from hmopt.skillhub.retrieval_eval import load_baseline, load_queries, run_eval

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "hm-skill-hub" / "eval" / "retrieval"


def test_three_query_types_min_eight_each():
    qs = load_queries(EVAL_DIR / "queries.yaml")
    counts = Counter(q["type"] for q in qs)
    for t in ("target-anchored", "mechanism-anchored", "free-form"):
        assert counts[t] >= 8, f"{t} has {counts[t]} queries (< 8)"


def test_must_hit_recall_at5_absolute_gate():
    r = run_eval(EVAL_DIR, k=5)
    # absolute line (P1-10 ③): must-hit recall@5 >= 0.8
    assert r.must_recall >= 0.8, f"must_recall@5 {r.must_recall:.3f} < 0.8 ({r.per_type})"


def test_lenient_recall_reasonable():
    r = run_eval(EVAL_DIR, k=5)
    assert r.lenient_recall >= 0.8


def test_per_type_recall_all_pass():
    r = run_eval(EVAL_DIR, k=5)
    for t, v in r.per_type.items():
        assert v >= 0.8, f"{t} recall {v:.3f} < 0.8"


def test_symbol_ablation_hybrid_not_worse_than_single_arms():
    # P1-10 ④: hybrid >= each single arm on symbol-name queries.
    r = run_eval(EVAL_DIR)
    ab = r.ablation
    assert ab["hybrid"] >= ab["bm25"] - 1e-9
    assert ab["hybrid"] >= ab["vector"] - 1e-9


def test_symbol_ablation_demonstrates_bm25_rescue():
    # the headline of §12.1: pure vectors dilute on bare symbol names; BM25 rescues.
    r = run_eval(EVAL_DIR)
    ab = r.ablation
    assert ab["vector"] < 1.0, "expected pure-vector to miss some bare-symbol queries"
    assert ab["hybrid"] > ab["vector"], "hybrid should strictly beat pure vector here"


def test_not_worse_than_recorded_baseline():
    # early regression gate (P1-10 ③): not worse than the last recorded green.
    base = load_baseline(EVAL_DIR)
    assert base is not None, "baseline.json missing"
    r = run_eval(EVAL_DIR)
    assert r.must_recall >= base["must_recall"] - 1e-9, (
        f"regression: must_recall {r.must_recall:.3f} < baseline {base['must_recall']:.3f}"
    )
    assert r.ablation["hybrid"] >= base["ablation"]["hybrid"] - 1e-9
