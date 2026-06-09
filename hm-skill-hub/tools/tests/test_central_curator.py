"""Tests for the Phase 2 central Curator tools (engine A §10.1.b / §11.5).

Runnable via pytest or standalone:
    cd hm-skill-hub && python -m pytest tools/tests/test_central_curator.py -q
    cd hm-skill-hub && python tools/tests/test_central_curator.py
"""
from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
HUB = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import central_curate  # type: ignore
import conflict_resolve  # type: ignore
import dedup  # type: ignore
import promotion_detector  # type: ignore
import subsumption  # type: ignore
from hub_records import HubRecord, load_hub_knowledge, record_from_candidate  # type: ignore
from similarity import alias_hit, jaccard, text_similarity  # type: ignore


def _rec(rid, schema="memory_item", body="", **fields) -> HubRecord:
    return HubRecord(id=rid, schema=schema, path="x", fields=fields, body=body)


def _specific(rid="F010", target="mm-vmscan-c-shrink-node", contrib="alice", delta=-0.8) -> HubRecord:
    return _rec(rid, type="fact", mechanism="hoist-invariant",
               scope={"level": "function", "target_slug": target},
               evidence={"delta_pct": delta, "compare_level": "function", "confirmations": 2},
               contributor=contrib, maturity="L2", status="active",
               body=f"in {target} hoist sc->priority out of the per-page reclaim loop")


def _general(rid="G010", contrib="bob") -> HubRecord:
    return _rec(rid, type="pattern", mechanism="hoist-invariant",
               scope={"level": "subsystem", "subsystem": "mm-reclaim"},
               contributor=contrib, maturity="L2", status="active",
               body="in mm reclaim hot loops, hoist loop-invariant state out of the loop")


# ---- similarity ------------------------------------------------------------

def test_similarity_primitives():
    assert text_similarity("hoist loop invariant", "hoist loop invariant") > 0.9
    assert jaccard("a b c", "x y z") == 0.0
    assert alias_hit(["hoist-invariant"], ["licm", "hoist-invariant"])


# ---- dedup three-state -----------------------------------------------------

def test_dedup_merge_on_near_duplicate():
    ex = _rec("F100", type="fact", mechanism="hoist-invariant",
              scope={"level": "function", "target_slug": "t"}, status="active",
              body="hoist the invariant read out of the shrink loop")
    inc = _rec("F900", type="fact", mechanism="hoist-invariant",
               scope={"level": "function", "target_slug": "t"}, status="active",
               body="hoist the invariant read out of the shrink loop")
    v = dedup.classify_one(inc, [ex])
    assert v.verdict == "merge" and v.match_id == "F100"


def test_dedup_conflict_on_opposite_conclusion():
    ex = _rec("F100", type="fact", mechanism="inline-callee",
              scope={"level": "function", "target_slug": "t"}, status="active",
              evidence={"delta_pct": -0.5, "compare_level": "function"},
              body="inlining process_one_work helps here")
    inc = _rec("F900", type="fact", mechanism="inline-callee",
               scope={"level": "function", "target_slug": "t"}, status="active",
               evidence={"delta_pct": 0.6, "compare_level": "function"},
               body="inlining process_one_work helps here")
    v = dedup.classify_one(inc, [ex])
    assert v.verdict == "conflict"


def test_dedup_new_on_unrelated():
    ex = _rec("F100", type="fact", mechanism="inline-callee",
              scope={"level": "function", "target_slug": "t1"}, status="active",
              body="inline this callee")
    inc = _rec("F900", type="fact", mechanism="batch-coalesce",
               scope={"level": "function", "target_slug": "t2"}, status="active",
               body="coalesce these writes")
    assert dedup.classify_one(inc, [ex]).verdict == "new"


# ---- conflict_resolve: double-time, never delete ---------------------------

def test_conflict_resolve_supersedes_when_stronger_never_deletes():
    existing = _rec("F100", type="fact", mechanism="inline-callee",
                    scope={"level": "function", "target_slug": "t"}, status="active",
                    maturity="L1", evidence={"delta_pct": -0.2, "compare_level": "function",
                                             "confirmations": 1}, body="inline helps")
    incoming = _rec("F900", type="fact", mechanism="inline-callee",
                    scope={"level": "function", "target_slug": "t"}, status="active",
                    maturity="L3", evidence={"delta_pct": 0.9, "compare_level": "function",
                                             "confirmations": 4}, body="inline regresses")
    res = conflict_resolve.resolve(incoming, existing)
    assert res.decision == "supersede"
    assert res.loser_fields["status"] == "superseded"
    assert "valid_until" in res.loser_fields
    assert res.loser_fields["superseded_by"] == ["F900"]
    assert res.winner_fields["supersedes"] == ["F100"]
    # CRDT: nothing is deleted — the loser record still exists, just tombstoned


def test_conflict_resolve_drops_when_weaker():
    existing = _rec("F100", mechanism="inline-callee", type="fact", maturity="L3",
                    scope={"level": "function", "target_slug": "t"}, status="active",
                    evidence={"confirmations": 4}, body="inline helps")
    incoming = _rec("F900", mechanism="inline-callee", type="fact", maturity="L0",
                    scope={"level": "function", "target_slug": "t"}, status="active",
                    body="inline regresses")
    res = conflict_resolve.resolve(incoming, existing)
    assert res.decision == "drop"


# ---- subsumption (AC: §10.1.b mock) ----------------------------------------

def test_subsumption_is_not_dup_or_contradiction():
    sp, gn = _specific(), _general()
    # subsumption judge orients general over specific
    assert subsumption.judge_subsumption(sp, gn) == "b_subsumes_a"
    # and dedup does NOT call this a merge or conflict (different scope)
    assert dedup.classify_one(sp, [gn]).verdict == "new"


def test_subsumption_build_links_keeps_specific_as_evidence():
    sp, gn = _specific(), _general()
    g_fields, s_fields = subsumption.build_links(dict(gn.fields), dict(sp.fields), gn, sp)
    assert sp.id in g_fields["subsumes"]
    assert gn.id in s_fields["subsumed_by"]
    # specific is added as a source of the general, NOT absorbed/removed
    assert {"kind": "doc", "ref": sp.id} in g_fields["source"]


def test_subsumption_emits_only_with_two_distinct_instances():
    gn = _general()
    sp1 = _specific(rid="F010", target="mm-vmscan-c-shrink-node")
    sp2 = _specific(rid="F011", target="mm-page_alloc-rmqueue")
    one = subsumption.detect_in_set([sp1, gn])
    assert subsumption.should_emit_promotion(gn, one) is False  # single instance -> link only
    two = subsumption.detect_in_set([sp1, sp2, gn])
    assert subsumption.should_emit_promotion(gn, two) is True   # >= 2 distinct -> emit


def test_subsumption_quiet_on_real_hub_except_h001_f001():
    links = subsumption.detect_in_set([r for r in load_hub_knowledge() if r.is_claim])
    pairs = {(x.general_id, x.specific_id) for x in links}
    assert pairs == {("H001", "F001")}  # exactly the designed link, no false positives


# ---- promotion detector (P2-8, both paths) ---------------------------------

def test_promotion_cluster_path():
    recs = [
        _rec("B100", schema="bad_plan", mechanism="inline-callee",
             applies_to={"subsystems": ["*"]}, status="active",
             rejected_by="alice", evidence=[{"kind": "bench", "ref": "a"},
                                            {"kind": "review", "ref": "b"}]),
        _rec("B101", schema="bad_plan", mechanism="inline-callee",
             applies_to={"subsystems": ["*"]}, status="active",
             rejected_by="bob", evidence=[{"kind": "bench", "ref": "c"},
                                          {"kind": "review", "ref": "d"}]),
    ]
    cands = [c for c in promotion_detector.detect_promotions(recs) if c.kind == "cluster"]
    assert cands and cands[0].mechanism == "inline-callee"
    assert set(cands[0].contributors) == {"alice", "bob"}


def test_promotion_subsumption_path_preserves_evidence():
    gn = _general()
    gn.fields["subsumes"] = ["F010", "F011"]  # fed by P2-9 (>= 2 distinct)
    sp1 = _specific(rid="F010", target="t-a")
    sp2 = _specific(rid="F011", target="t-b")
    cands = [c for c in promotion_detector.detect_promotions([gn, sp1, sp2])
             if c.kind == "subsumption"]
    assert cands
    # the subsumed instances are carried as evidence (NOT deleted/absorbed)
    assert set(cands[0].evidence_ids) == {"F010", "F011"}


def test_promotion_single_instance_does_not_promote():
    gn = _general()
    gn.fields["subsumes"] = ["F010"]  # only one
    sp1 = _specific(rid="F010", target="t-a")
    cands = [c for c in promotion_detector.detect_promotions([gn, sp1])
             if c.kind == "subsumption"]
    assert not cands  # single instance -> no promotion (anti over-generalization)


# ---- central_curate orchestrator -------------------------------------------

def test_central_curate_batch_routes_and_signals():
    hub = [_general(), _specific(rid="F010", target="t-a")]
    incoming = [
        {"schema": "memory_item", "record": {
            "id": "F011", "type": "fact", "mechanism": "hoist-invariant",
            "scope": {"level": "function", "target_slug": "t-b"}, "status": "active",
            "maturity": "L2", "contributor": "carol",
            "body": "in t-b hoist sc->priority out of the per-page reclaim loop"}},
        {"schema": "memory_item", "record": {
            "id": "F999", "type": "fact", "mechanism": "batch-coalesce",
            "scope": {"level": "function", "target_slug": "t-z"}, "status": "active",
            "maturity": "L2", "contributor": "dave",
            "body": "coalesce small writes into one batch round-trip"}},
    ]
    report = central_curate.curate_batch(incoming, hub)
    kinds = {d.incoming_id: d.kind for d in report.decisions}
    assert kinds["F011"] == "subsumption"   # generalized by G010
    assert kinds["F999"] == "add"           # unrelated novel
    # F010 (in hub) + F011 (now subsumed) => >= 2 distinct instances => signal
    assert "G010" in report.promotion_signals


def test_central_curate_real_hub_smoke():
    # an incoming dup of an existing global lesson should be flagged merge/subsumption
    inc = [{"schema": "memory_item", "record": {
        "id": "F500", "type": "fact", "mechanism": "hoist-invariant",
        "scope": {"level": "function", "target_slug": "mm-vmscan-c-shrink-node"},
        "status": "active", "maturity": "L2", "contributor": "eve",
        "body": "hoist sc->priority read out of the shrink_node reclaim loop"}}]
    report = central_curate.curate_batch(inc, load_hub_knowledge())
    assert report.decisions and report.decisions[0].incoming_id == "F500"


# ---- standalone runner -----------------------------------------------------

def _run_standalone() -> int:
    g = dict(globals())
    tests = sorted((n, f) for n, f in g.items() if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
