"""Tests for the local curator PoC (design §10.1, plan P1-8 hard metrics)."""
from __future__ import annotations

from hmopt.memory.curator_benchmark import build_cases, run_benchmark
from hmopt.memory.local_curator import (
    NO_DELETE_KINDS,
    CuratorItem,
    apply_relation,
    classify_pair,
)


def _it(idx: str, **kw) -> CuratorItem:
    return CuratorItem(id=idx, **kw)


# ---- P1-8 acceptance criteria ----------------------------------------------

def test_benchmark_covers_seven_routes_min_five_each():
    cases = build_cases()
    from collections import Counter

    counts = Counter(c.expected for c in cases)
    for route in ("duplicate", "contradiction", "temporal", "conditional",
                  "selector", "evidence", "novel", "subsumption"):
        assert counts[route] >= 5, f"{route} has only {counts[route]} cases"
    assert len(cases) >= 40


def test_seven_way_accuracy_at_least_085():
    m = run_benchmark()
    assert m.accuracy >= 0.85, f"accuracy {m.accuracy:.3f} < 0.85 ({m.per_route})"


def test_temporal_conditional_false_delete_is_zero():
    m = run_benchmark()
    assert m.false_delete_total > 0
    assert m.false_delete_count == 0, f"false-delete rate {m.false_delete_rate:.3f} (must be ~0)"


def test_latency_under_3s_per_item():
    m = run_benchmark()
    assert m.avg_latency_s <= 3.0


def test_local_mode_never_emits_subsumption():
    m = run_benchmark()
    assert m.local_subsumption_emitted == 0


# ---- apply discipline (CRDT: never physical-delete) ------------------------

def test_apply_never_deletes_for_no_delete_kinds():
    nb = _it("N", target_slug="t", mechanism="m", applies_when="cond A")
    for kind in NO_DELETE_KINDS:
        from hmopt.memory.local_curator import Relation

        res = apply_relation(Relation(kind=kind, target_id="N"), _it("I"), nb)
        assert res.deleted == []


def test_apply_never_deletes_for_any_relation_kind():
    # CRDT invariant across EVERY relation (incl. contradiction/duplicate/
    # subsumption/novel), both evidence directions — not just NO_DELETE_KINDS.
    from hmopt.memory.local_curator import Relation

    nb = _it("N", target_slug="t", mechanism="m", applies_when="c", delta_pct=-0.5)
    inc = _it("I", target_slug="t", mechanism="m", applies_when="c", delta_pct=-0.4)
    for kind in ("duplicate", "contradiction", "temporal", "conditional",
                 "subsumption", "selector", "evidence", "novel"):
        for stronger in (True, False):
            res = apply_relation(Relation(kind=kind, target_id="N", stronger=stronger), inc, nb)
            assert res.deleted == [], f"{kind} (stronger={stronger}) produced a delete"


def test_conditional_coexists_without_superseding():
    inc = _it("I", target_slug="t", mechanism="m", applies_when="trip > 100", delta_pct=-0.4)
    nb = _it("N", target_slug="t", mechanism="m", applies_when="trip < 10", delta_pct=0.1)
    rel = classify_pair(inc, nb)
    assert rel.kind == "conditional"
    res = apply_relation(rel, inc, nb)
    assert res.added and not res.superseded and not res.deleted


def test_temporal_supersedes_old_but_keeps_auditable():
    inc = _it("I", target_slug="t", mechanism="m", applies_when="c", delta_pct=0.3, kernel_version=6.6)
    nb = _it("N", target_slug="t", mechanism="m", applies_when="c", delta_pct=-0.5, kernel_version=6.1)
    rel = classify_pair(inc, nb)
    assert rel.kind == "temporal"
    res = apply_relation(rel, inc, nb)
    assert res.added and res.superseded == ["N"] and res.deleted == []


def test_contradiction_stronger_supersedes_weaker():
    inc = _it("I", target_slug="t", mechanism="m", applies_when="c", delta_pct=0.7,
              maturity="L3", confirmations=4)
    nb = _it("N", target_slug="t", mechanism="m", applies_when="c", delta_pct=-0.5,
             maturity="L1", confirmations=1)
    rel = classify_pair(inc, nb)
    assert rel.kind == "contradiction" and rel.stronger
    res = apply_relation(rel, inc, nb)
    assert res.superseded == ["N"] and res.deleted == []


def test_contradiction_weaker_drops_incoming_keeps_old():
    inc = _it("I", target_slug="t", mechanism="m", applies_when="c", delta_pct=0.2,
              maturity="L1", confirmations=1)
    nb = _it("N", target_slug="t", mechanism="m", applies_when="c", delta_pct=-0.6,
             maturity="L3", confirmations=5)
    rel = classify_pair(inc, nb)
    assert rel.kind == "contradiction" and not rel.stronger
    res = apply_relation(rel, inc, nb)
    assert not res.added and not res.superseded and not res.deleted  # old untouched, no delete


def test_subsumption_central_links_and_signals_promotion():
    inc = _it("I", target_slug="t", mechanism="m", text="hoist x", granularity="specific")
    nb = _it("N", subsystem="mm", mechanism="m", text="generalize hoist", granularity="general")
    rel = classify_pair(inc, nb, mode="central")
    assert rel.kind == "subsumption"
    res = apply_relation(rel, inc, nb)
    assert res.promotion_signal and res.linked == ["N"] and res.deleted == []


def test_selector_drift_keeps_knowledge():
    inc = _it("I", target_slug="t", mechanism="m", symbol="shrink_node",
              invalidation="rebase moved offset")
    nb = _it("N", target_slug="t", mechanism="m", symbol="shrink_node")
    rel = classify_pair(inc, nb)
    assert rel.kind == "selector"
    res = apply_relation(rel, inc, nb)
    assert not res.deleted and not res.superseded
