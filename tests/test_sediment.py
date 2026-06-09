"""Tests for the sediment write path (Tier 0->1, design §8, plan P1-1/P1-2/P1-9)."""
from __future__ import annotations

import json
from pathlib import Path

from hmopt.sediment import extractors as ex
from hmopt.sediment.pipeline import bundle_staging, sediment_run
from hmopt.sediment.readers import read_run
from hmopt.sediment.salience import llm_salience_pass
from hmopt.sediment.validate import validate_candidate

REPO = Path(__file__).resolve().parent.parent


class _FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply

    def chat(self, messages):  # noqa: ANN001 - mimic hmopt LLMClient.chat
        return self.reply


# ---- extractors emit schema-valid candidates -------------------------------

def test_bench_to_fact_is_schema_valid():
    b = ex.BenchResult(mechanism="hoist-invariant", target="mm/vmscan.c::shrink_node",
                       delta_pct=-0.8, compare_level="function",
                       validation_path="bench/x.md", subsystem="mm-reclaim")
    cand = ex.bench_to_fact(b, run_id="r1", contributor="me")
    assert cand["schema"] == "memory_item"
    assert validate_candidate(cand, REPO) == []
    assert cand["record"]["maturity"] == "L1"
    assert any(s["kind"] == "run_id" for s in cand["record"]["source"])


def test_review_to_anti_pattern_and_bad_plan_valid():
    v = ex.ReviewVerdict(decision="reject", mechanism="inline-callee",
                         target_pattern="kworker entry functions on cache-constrained devices",
                         reason="blows up i-cache (+1.4% miss)", scope="function",
                         subsystems=["*"], platforms=["phone-X"], review_ref="reviews/r.md")
    ap = ex.review_to_anti_pattern(v, run_id="r1", contributor="me")
    bp = ex.review_to_bad_plan(v, run_id="r1")
    assert validate_candidate(ap, REPO) == []
    assert validate_candidate(bp, REPO) == []
    assert ap["record"]["confidence"] == "tentative"
    assert bp["record"]["mechanism"] == "inline-callee"


def test_ledger_to_idea_landed_valid():
    c = ex.LedgerChange(mechanism="hoist-invariant", target_slug="mm-vmscan-c-shrink-node",
                        scope="mm/vmscan.c :: shrink_node", status="landed",
                        rationale="removed redundant load", delta_pct=-0.8,
                        compare_level="function", validation_path="bench/x.md")
    cand = ex.ledger_to_idea(c, run_id="r1")
    assert validate_candidate(cand, REPO) == []
    rec = cand["record"]
    assert rec["status"] == "landed" and rec["delta_pct"] == -0.8 and rec["validation_path"]


def test_ledger_to_idea_deferred_has_reopen_trigger():
    c = ex.LedgerChange(mechanism="batch-coalesce", target_slug="t", scope="s",
                        status="deferred", rationale="needs more data")
    cand = ex.ledger_to_idea(c, run_id="r1")
    assert validate_candidate(cand, REPO) == []
    assert cand["record"]["reopen_trigger"]


def test_validate_catches_broken_record():
    bad = {"schema": "memory_item", "record": {"id": "F901", "type": "fact"}}  # missing required
    errs = validate_candidate(bad, REPO)
    assert errs and any("required" in e for e in errs)


# ---- salience pass ---------------------------------------------------------

def test_salience_pass_parses_llm_json():
    reply = (
        'Here you go: [{"lesson": "check bad_plans before proposing",'
        '"applies_when": "before planning", "do_or_dont": "do: dedup first",'
        '"tags": ["planning"]}]'
    )
    cands = llm_salience_pass(["some design notes"], _FakeLLM(reply), run_id="r1")
    assert len(cands) == 1
    assert cands[0]["schema"] == "global_lesson"
    assert cands[0]["record"]["confidence"] == "tentative"
    assert validate_candidate(cands[0], REPO) == []


def test_salience_pass_offline_is_noop():
    assert llm_salience_pass(["notes"], None, run_id="r1") == []
    # unparseable reply -> nothing, never raises
    assert llm_salience_pass(["notes"], _FakeLLM("no json here"), run_id="r1") == []


# ---- end-to-end sediment_run ----------------------------------------------

def _write_manifest(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "bench": [
            {"mechanism": "hoist-invariant", "target": "mm/vmscan.c::shrink_node",
             "delta_pct": -0.8, "compare_level": "function",
             "validation_path": "bench/mm.md", "verdict": "landed", "subsystem": "mm-reclaim"}
        ],
        "reviews": [
            {"decision": "reject", "mechanism": "inline-callee",
             "target_pattern": "kworker entries on cache-constrained devices",
             "reason": "i-cache blowup", "scope": "function", "subsystems": ["*"]}
        ],
        "ledger": [
            {"mechanism": "hoist-invariant", "target_slug": "mm-vmscan-c-shrink-node",
             "scope": "shrink_node", "status": "landed", "rationale": "load removed",
             "delta_pct": -0.8, "compare_level": "function", "validation_path": "bench/mm.md"}
        ],
        "free_text": ["design summary: prefer hoist before inline"],
    }
    (run_dir / "sediment_input.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_read_run_from_manifest(tmp_path: Path):
    run = tmp_path / "r_2026_0601_mm1"
    _write_manifest(run)
    arts = read_run(run)
    assert len(arts.bench) == 1 and len(arts.reviews) == 1 and len(arts.ledger) == 1
    assert arts.free_text


def test_sediment_run_writes_valid_jsonl(tmp_path: Path):
    run = tmp_path / "r_2026_0601_mm1"
    _write_manifest(run)
    out = tmp_path / "staging"
    res = sediment_run(run, out_dir=out, hub_root=REPO, no_llm=True)
    # bench(landed)->fact, review(reject)->anti_pattern + bad_plan, ledger->idea = 4
    assert res.n_valid == 4
    assert not res.invalid
    schemas = sorted(c["schema"] for c in res.candidates)
    assert schemas == ["bad_plan", "global_lesson", "idea", "memory_item"]
    lines = Path(res.out_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    for line in lines:
        assert validate_candidate(json.loads(line), REPO) == []


def test_sediment_run_with_llm_adds_salience(tmp_path: Path):
    run = tmp_path / "r1"
    _write_manifest(run)
    reply = '[{"lesson": "dedup before planning", "applies_when": "planning", "do_or_dont": "do: x", "tags": ["plan"]}]'
    res = sediment_run(run, out_dir=tmp_path / "s", hub_root=REPO, llm=_FakeLLM(reply))
    assert res.n_valid == 5  # 4 deterministic + 1 salience heuristic
    assert any(c["record"]["id"].startswith("H95") for c in res.candidates)


def test_no_llm_flag_skips_salience(tmp_path: Path):
    run = tmp_path / "r1"
    _write_manifest(run)
    reply = '[{"lesson": "x", "applies_when": "y", "do_or_dont": "do: z", "tags": ["t"]}]'
    res = sediment_run(run, out_dir=tmp_path / "s", hub_root=REPO, llm=_FakeLLM(reply), no_llm=True)
    assert res.n_valid == 4  # salience skipped


def test_bundle_staging_collates(tmp_path: Path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "run1.jsonl").write_text('{"schema":"idea","record":{}}\n', encoding="utf-8")
    (staging / "run2.jsonl").write_text('{"schema":"idea","record":{}}\n{"schema":"idea","record":{}}\n',
                                        encoding="utf-8")
    out, n = bundle_staging(staging, tmp_path / "bundle.jsonl")
    assert n == 3 and out.exists()


def test_markdown_bench_fallback(tmp_path: Path):
    # no manifest -> best-effort markdown bench scan
    run = tmp_path / "r_md"
    (run / "bench").mkdir(parents=True)
    (run / "bench" / "mm_vmscan_shrink.md").write_text(
        "# validation\nmechanism: hoist-invariant\ndelta_pct: -0.8\ncompare_level: function\n",
        encoding="utf-8",
    )
    arts = read_run(run)
    assert len(arts.bench) == 1 and arts.bench[0].mechanism == "hoist-invariant"
