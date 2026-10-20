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
import lint  # type: ignore
import promotion_detector  # type: ignore
import subsumption  # type: ignore
import yaml  # type: ignore
from hub_records import HubRecord, load_hub_knowledge  # type: ignore
from parse_memory import parse_frontmatter  # type: ignore
from similarity import alias_hit, jaccard, text_similarity  # type: ignore


def _rec(rid, schema="memory_item", body="", **fields) -> HubRecord:
    return HubRecord(id=rid, schema=schema, path="x", fields=fields, body=body)


def _write_record(tmp, rel: str, fm: dict, body: str = "body") -> "Path":
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    text = "---\n" + yaml.safe_dump(fm, sort_keys=False, allow_unicode=True) + "---\n\n" + body + "\n"
    p.write_text(text, encoding="utf-8")
    return p


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


def test_dedup_relates_facts_without_a_mechanism_field():
    # review B1: memory_item / global_lesson schemas have NO mechanism/alias
    # field, so the old alias-only relation gate made every duplicate/conflict
    # classify as "new". Schema-valid facts (no mechanism) at the same target
    # must still merge / conflict on shared scope + lexical overlap.
    ex = _rec("F100", type="fact",
              scope={"level": "function", "target_slug": "t"}, status="active",
              applies_when="hot loop",
              evidence={"delta_pct": -0.8, "compare_level": "function"},
              body="hoist the repeated sc->priority read out of the per-page reclaim loop")
    dup = _rec("F900", type="fact",
               scope={"level": "function", "target_slug": "t"}, status="active",
               applies_when="hot loop",
               body="hoist the repeated sc->priority read out of the per-page reclaim loop",
               evidence={"delta_pct": -0.8, "compare_level": "function"})
    con = _rec("F901", type="fact",
               scope={"level": "function", "target_slug": "t"}, status="active",
               applies_when="hot loop",
               body="hoist the repeated sc->priority read out of the per-page reclaim loop",
               evidence={"delta_pct": 0.8, "compare_level": "function"})
    assert "mechanism" not in ex.fields  # schema-valid: no mechanism field
    assert dedup.classify_one(dup, [ex]).verdict == "merge"
    assert dedup.classify_one(con, [ex]).verdict == "conflict"


def test_dedup_different_target_same_mechanism_is_new_not_merge():
    # review #2: near-identical prose but different target_slug => distinct
    # per-target instances, must be `new` (not collapsed to `merge`).
    ex = _rec("F100", type="fact", mechanism="hoist-invariant",
              scope={"level": "function", "target_slug": "t-a"}, status="active",
              body="hoist the loop-invariant read out of the reclaim loop")
    inc = _rec("F900", type="fact", mechanism="hoist-invariant",
               scope={"level": "function", "target_slug": "t-b"}, status="active",
               body="hoist the loop-invariant read out of the reclaim loop")
    assert dedup.classify_one(inc, [ex]).verdict == "new"


def test_polarity_do_avoid_is_negative():
    # review #6: "do: avoid X" is a negative despite the do: prefix
    assert _rec("X", do_or_dont="do: avoid spending an iteration on cold paths").polarity() == -1
    assert _rec("Y", do_or_dont="do: hoist the invariant").polarity() == 1


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


def test_superseded_loser_lints_clean_for_all_schema_families(tmp_path):
    """Review #1: a superseded loser must still pass lint. memory_item carries
    valid_until; global_lesson / bad_plan do NOT (additionalProperties:false)."""
    # memory_item
    wm = _write_record(tmp_path, "knowledge/targets/t/facts/F101-w.md",
                       {"id": "F101", "type": "fact", "title": "w",
                        "scope": {"level": "function", "target_slug": "t"},
                        "source": [{"kind": "doc", "ref": "x"}], "maturity": "L3",
                        "status": "active", "created_at": "2026-01-01T00:00:00Z",
                        "evidence": {"confirmations": 4}})
    lm = _write_record(tmp_path, "knowledge/targets/t/facts/F100-l.md",
                       {"id": "F100", "type": "fact", "title": "l",
                        "scope": {"level": "function", "target_slug": "t"},
                        "source": [{"kind": "doc", "ref": "y"}], "maturity": "L1",
                        "status": "active", "created_at": "2026-01-01T00:00:00Z"})
    assert conflict_resolve.apply_to_files(wm, lm).decision == "supersede"
    assert "valid_until" in lm.read_text(encoding="utf-8")
    assert lint.lint_record_file(lm) == []
    # review P0-2: the forward edge is persisted too — winner.supersedes=[loser]
    wm_fm, _ = parse_frontmatter(wm.read_text(encoding="utf-8"))
    assert wm_fm.get("supersedes") == ["F100"]
    assert lint.lint_record_file(wm) == []   # winner still lints with supersedes[]
    # global_lesson — must NOT get valid_until, must still lint
    wg = _write_record(tmp_path, "knowledge/global/heuristics/H101-w.md",
                       {"id": "H101", "lesson": "w", "kind": "heuristic", "applies_when": "c",
                        "do_or_dont": "do: x", "tags": ["a"], "evidence": [{"kind": "doc", "ref": "x"}],
                        "confidence": "confirmed", "added_on": "2026-01-01", "added_by": "m",
                        "status": "active"})
    lg = _write_record(tmp_path, "knowledge/global/heuristics/H100-l.md",
                       {"id": "H100", "lesson": "l", "kind": "heuristic", "applies_when": "c",
                        "do_or_dont": "do: y", "tags": ["a"], "evidence": [{"kind": "doc", "ref": "y"}],
                        "confidence": "tentative", "added_on": "2026-01-01", "added_by": "m",
                        "status": "active"})
    assert conflict_resolve.apply_to_files(wg, lg).decision == "supersede"
    assert "valid_until" not in lg.read_text(encoding="utf-8")
    assert lint.lint_record_file(lg) == []
    # bad_plan — same: superseded_by scalar, no valid_until
    wb = _write_record(tmp_path, "knowledge/global/bad_plans/B101-w.md",
                       {"id": "B101", "mechanism": "inline-callee", "target_pattern": "x",
                        "scope": "function", "applies_to": {"subsystems": ["*"]}, "reason": "w",
                        "evidence": [{"kind": "bench", "ref": "x"}, {"kind": "review", "ref": "y"},
                                     {"kind": "commit", "ref": "z"}],
                        "rejected_on": "2026-01-01", "rejected_by": "m", "status": "active"})
    lb = _write_record(tmp_path, "knowledge/global/bad_plans/B100-l.md",
                       {"id": "B100", "mechanism": "inline-callee", "target_pattern": "x",
                        "scope": "function", "applies_to": {"subsystems": ["*"]}, "reason": "l",
                        "evidence": [{"kind": "bench", "ref": "x"}], "rejected_on": "2026-01-01",
                        "rejected_by": "m", "status": "active"})
    assert conflict_resolve.apply_to_files(wb, lb).decision == "supersede"
    assert "valid_until" not in lb.read_text(encoding="utf-8")
    assert lint.lint_record_file(lb) == []


# ---- subsumption (AC: §10.1.b mock) ----------------------------------------

def test_subsumption_is_not_dup_or_contradiction():
    sp, gn = _specific(), _general()
    # subsumption judge orients general over specific
    assert subsumption.judge_subsumption(sp, gn) == "b_subsumes_a"
    # and dedup does NOT call this a merge or conflict (different scope)
    assert dedup.classify_one(sp, [gn]).verdict == "new"


def test_subsumption_polarity_guard_blocks_contradiction():
    # review #3: a bad_plan ("don't inline", -1) must NOT subsume a positive fact
    # ("inlining helped", +1) — that's a contradiction, deferred to dedup/conflict.
    bad = _rec("B100", schema="bad_plan", mechanism="inline-callee",
               applies_to={"subsystems": ["*"]}, status="active", rejected_by="x",
               reason="blanket inline regresses i-cache",
               body="blanket inline of kworker entries regresses i-cache")
    good = _rec("F100", type="fact", mechanism="inline-callee",
                scope={"level": "function", "target_slug": "t"}, status="active",
                evidence={"delta_pct": -0.5, "compare_level": "function"},
                body="inlining this kworker entry helped on this target")
    assert subsumption.judge_subsumption(good, bad) == "none"
    # in a batch the opposite-polarity pair must not be routed to subsumption
    report = central_curate.curate_batch(
        [{"schema": "memory_item", "record": {"id": "F100", **good.fields, "body": good.body}}],
        [bad])
    assert report.decisions[0].kind != "subsumption"


def test_subsumption_anonymous_instances_do_not_fake_promotion():
    # review #10: two subsumed records with no target_slug + no contributor collapse
    # to a single distinct key -> cannot fabricate a >=2-instance promotion.
    gn = _general()
    a = _rec("F010", type="fact", mechanism="hoist-invariant",
             scope={"level": "function"}, body="hoist invariant out of loop A")
    b = _rec("F011", type="fact", mechanism="hoist-invariant",
             scope={"level": "function"}, body="hoist invariant out of loop B")
    links = subsumption.detect_in_set([a, b, gn])
    assert subsumption.should_emit_promotion(gn, links) is False


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
    claims = [r for r in load_hub_knowledge() if r.is_claim]
    links = subsumption.detect_in_set(claims)
    pairs = {(x.general_id, x.specific_id) for x in links}
    assert pairs == {("H001", "F001")}  # exactly the designed link, no false positives
    # H001 (heuristic) carries no `mechanism`, so this link rests on a text-overlap
    # margin over the toy embedder. Assert the margin explicitly so threshold
    # erosion (e.g. rewording H001/F001) surfaces in CI instead of silently flipping.
    recs = {r.id: r for r in claims}
    sim_link = text_similarity(recs["H001"].text(), recs["F001"].text())
    nearest_non_link = max(
        text_similarity(recs[g].text(), recs["F001"].text()) for g in ("A001", "B001", "V001"))
    assert sim_link >= 0.30 and sim_link - nearest_non_link >= 0.03


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


def test_materialization_rejects_candidate_path_traversal(tmp_path):
    candidates = [
        {
            "schema": "memory_item",
            "record": {
                "id": "F900",
                "type": "fact",
                "title": "unsafe",
                "body": "body",
                "scope": {"level": "function", "target_slug": "../../escape"},
                "source": [{"kind": "doc", "ref": "x"}],
                "maturity": "L1",
                "status": "active",
                "created_at": "2026-07-30T00:00:00+00:00",
            },
        }
    ]
    report = central_curate.CurationReport(decisions=[central_curate.Decision("F900", "add")])
    actions = central_curate.apply_additions(
        report, candidates, [], tmp_path / "knowledge", write=True
    )
    assert "cannot derive path" in actions[0][3]
    assert not (tmp_path / "escape").exists()


def test_materialization_accepts_project_level_architectural_fact(tmp_path):
    candidates = [
        {
            "schema": "memory_item",
            "record": {
                "id": "F900",
                "type": "fact",
                "title": "project fact",
                "body": "body",
                "scope": {"level": "architectural", "target_slug": "my-project"},
                "source": [{"kind": "doc", "ref": "journal:alice/my-project/J-x"}],
                "maturity": "L1",
                "status": "active",
                "created_at": "2026-07-30T00:00:00+00:00",
            },
        }
    ]
    report = central_curate.CurationReport(decisions=[central_curate.Decision("F900", "add")])
    actions = central_curate.apply_additions(
        report, candidates, [], tmp_path / "knowledge", write=True
    )
    assert actions[0][3] == "WROTE"
    assert actions[0][2].startswith("targets/my-project/facts/")


# ---- standalone runner -----------------------------------------------------

def _run_standalone() -> int:
    import inspect
    import tempfile

    g = dict(globals())
    tests = sorted((n, f) for n, f in g.items() if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in tests:
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
