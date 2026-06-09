"""Tests for the Phase 3 SkillOpt engine (engine B, design §10.2 / §9 / §13).

Runnable via pytest or standalone:
    cd hm-skill-hub && python -m pytest tools/tests/test_skillopt.py -q
    cd hm-skill-hub && python tools/tests/test_skillopt.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
HUB = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import eval_gate  # type: ignore
import skill_optimizer as so  # type: ignore
from pareto import ParetoFrontier, dominates  # type: ignore
from run_evals import ProxyScorer, evaluate_skill_dir, load_skill, load_suite, run_evals  # type: ignore

SKILL = HUB / "skills" / "core" / "instruction-count-first"
SUITE = HUB / "eval" / "task_suites" / "core_optimization_suite"


# ---- run_evals / proxy scorer ----------------------------------------------

def test_seed_skill_scores_partial():
    card = evaluate_skill_dir(SKILL, SUITE)
    assert card.n == 9
    assert 0.6 < card.pass_rate < 0.8   # 6/9 (wq + hh pass, mm fails)
    assert card.passed["wq-pool"] and card.passed["hh-write"]
    assert not card.passed["mm-shrink-node"]


def test_proxy_scorer_mechanism_alias_bonus():
    suite = load_suite(SUITE)
    case = next(c for c in suite.cases if c.id == "mm-shrink-node")
    sc = ProxyScorer()
    assert sc.score("nothing relevant here", case) < 0.3
    # uses a registry ALIAS of hoist-invariant -> mechanism bonus fires
    assert sc.score("apply loop-invariant-code-motion: hoist, loop-invariant, re-measure", case) >= 0.9


def test_scorecard_strictly_better_and_regression():
    suite = load_suite(SUITE)
    base = run_evals("s", "1", "lock-split per-shard contention batch-coalesce round-trip", suite)
    better = run_evals("s", "1",
                       "lock-split per-shard contention batch-coalesce round-trip "
                       "hoist loop-invariant re-measure hoist-invariant", suite)
    assert better.strictly_better_than(base)
    assert better.regression_rate_vs(base) == 0.0
    assert not base.strictly_better_than(better)


def test_safety_property_aggregate_up_but_one_instance_regresses_is_rejected():
    """THE re-feed safety property: an edit that raises pass_rate AND mean but
    regresses even one instance must be REJECTED (monotone gate, design §10.2)."""
    suite = load_suite(SUITE)
    base = run_evals("s", "1", "lock-split per-shard contention batch-coalesce round-trip", suite)
    # adds hoist (mm now passes) but drops 'contention' (wq 1.0 -> 0.8, still >= 0.6)
    cand = run_evals("s", "1",
                     "lock-split per-shard batch-coalesce round-trip "
                     "hoist loop-invariant re-measure hoist-invariant", suite)
    assert cand.pass_rate > base.pass_rate          # aggregate improved
    assert cand.mean_score > base.mean_score
    assert cand.regression_rate_vs(base) > 0.0      # but an instance regressed
    assert not cand.strictly_better_than(base)      # => rejected


def test_empty_expected_terms_does_not_autopass():
    from run_evals import Case  # type: ignore
    c = Case(id="x", target="t", expected_mechanism="hoist-invariant", expected_terms=[])
    assert ProxyScorer().score("apply hoist-invariant guidance", c) < 0.6  # mech-only = 0.4


def test_avoid_term_substring_of_expected_not_penalized():
    from run_evals import Case  # type: ignore
    # avoid 'coalesce' must NOT fire on the expected term 'batch-coalesce' (word boundary)
    c = Case(id="x", target="t", expected_mechanism="batch-coalesce",
             expected_terms=["batch-coalesce"], avoid_terms=["coalesce"])
    assert ProxyScorer().score("apply batch-coalesce here", c) >= 0.9


# ---- pareto ----------------------------------------------------------------

def test_pareto_dominance():
    assert dominates({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 0.5})
    assert not dominates({"a": 1.0, "b": 0.5}, {"a": 0.5, "b": 1.0})  # complementary


def test_pareto_keeps_complementary_drops_dominated():
    pf = ParetoFrontier()
    assert pf.add("A", {"x": 1.0, "y": 0.0})
    assert pf.add("B", {"x": 0.0, "y": 1.0})    # complementary to A -> both kept
    assert set(pf.ids()) == {"A", "B"}
    assert pf.add("C", {"x": 1.0, "y": 1.0})    # dominates both -> A, B dropped
    assert pf.ids() == ["C"]
    assert not pf.add("D", {"x": 0.5, "y": 0.5})  # dominated by C -> not kept


# ---- optimizer (DoD: bounded edit -> gate -> accept) -----------------------

def _tmp_skill(tmp_path: Path) -> Path:
    dst = tmp_path / "skill"
    shutil.copytree(SKILL, dst)
    # start from a clean slate (no prior bad_edits / candidate carryover)
    for p in (dst / "bad_edits.jsonl",):
        if p.exists():
            p.unlink()
    return dst


def test_propose_edit_targets_the_failing_mechanism():
    _, _, text = load_skill(SKILL)
    suite = load_suite(SUITE)
    card = run_evals("s", "1", text, suite)
    edits = so.propose_edits(text, card, suite, bad_edits=set(), max_edits=1)
    assert len(edits) == 1
    assert edits[0].mechanism == "hoist-invariant"   # heaviest failing group


def test_optimize_improves_under_gate_no_regression(tmp_path: Path):
    dst = _tmp_skill(tmp_path)
    res = so.optimize(dst, SUITE, apply=True)
    assert res.baseline.pass_rate < 1.0
    assert res.final.pass_rate == 1.0                # all 9 pass after the hoist edit
    assert res.final.strictly_better_than(res.baseline)
    assert res.final.regression_rate_vs(res.baseline) == 0.0
    assert len(res.accepted) == 1 and res.accepted[0].mechanism == "hoist-invariant"
    # --apply wrote the artifacts
    assert "hoist" in (dst / "best_skill.md").read_text(encoding="utf-8").lower()
    assert (dst / "scorecards").glob("*.json")
    assert (dst / "candidates" / "pareto.json").exists()


def test_bad_edits_buffer_skips_known_bad(tmp_path: Path):
    _, _, text = load_skill(SKILL)
    suite = load_suite(SUITE)
    card = run_evals("s", "1", text, suite)
    edit = so.propose_edits(text, card, suite, bad_edits=set())[0]
    # with its signature already buffered, the same edit is NOT re-proposed (P3-7)
    assert so.propose_edits(text, card, suite, bad_edits={edit.signature()}) == []


def test_regressing_edit_fails_gate_and_is_buffered(tmp_path: Path):
    dst = _tmp_skill(tmp_path)
    _, _, text = load_skill(dst)
    suite = load_suite(SUITE)
    base = run_evals("s", "1", text, suite)
    bad = so.Edit(op="add", payload="- blanket-inline everything always", mechanism="inline-callee")
    cand_text = so.apply_edit(text, bad)
    cand = run_evals("s", "1", cand_text, suite)
    assert not cand.strictly_better_than(base)        # wq-worker regresses (avoid_term)
    so.append_bad_edit(dst, bad)
    assert bad.signature() in so.load_bad_edits(dst)  # buffered -> skipped next time


# ---- eval-gate -------------------------------------------------------------

def test_eval_gate_passes_on_committed_seed():
    ok, msg = eval_gate.check_skill(SKILL)
    assert ok, msg


def test_committed_card_prefers_version_match(tmp_path: Path):
    d = _tmp_skill(tmp_path)
    sc = d / "scorecards"
    sc.mkdir(exist_ok=True)
    # decoy card with a lexically-larger version; SKILL.md version is 0.1.0
    (sc / "instruction-count-first__0.9.0.json").write_text(
        json.dumps({"skill": "instruction-count-first", "version": "0.9.0", "suite": "s",
                    "per_instance": {}, "pass_rate": 0.0, "mean_score": 0.0}), encoding="utf-8")
    card = eval_gate._committed_card(d)
    assert card is not None and card.version == "0.1.0"  # version-matched, not lexical-max


def test_committed_card_semver_max_when_no_version_match(tmp_path: Path):
    d = _tmp_skill(tmp_path)
    sc = d / "scorecards"
    (sc / "instruction-count-first__0.1.0.json").unlink()  # drop the version-matched card
    for v, pr in [("0.9.0", 0.5), ("0.10.0", 0.8)]:
        (sc / f"instruction-count-first__{v}.json").write_text(
            json.dumps({"skill": "x", "version": v, "suite": "s",
                        "per_instance": {}, "pass_rate": pr, "mean_score": pr}), encoding="utf-8")
    card = eval_gate._committed_card(d)
    assert card.version == "0.10.0"  # parsed semver (0.10.0 > 0.9.0), not lexical


def test_eval_gate_flags_regression(tmp_path: Path):
    dst = _tmp_skill(tmp_path)
    # committed scorecard is the seed (0.67); break best_skill so wq cases regress
    bs = dst / "best_skill.md"
    bs.write_text(bs.read_text(encoding="utf-8").replace("lock-split", "X").replace("per-shard", "Y"),
                  encoding="utf-8")
    ok, msg = eval_gate.check_skill(dst)
    assert not ok and "REGRESSION" in msg


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
