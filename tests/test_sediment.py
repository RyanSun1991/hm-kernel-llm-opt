from __future__ import annotations

import json
from pathlib import Path

from hmopt.sediment import bundle_staging, extract_run, write_jsonl


def test_extract_run_collects_bench_review_and_idea(tmp_path: Path) -> None:
    run = tmp_path / "r1"
    run.mkdir()
    (run / "bench_report.json").write_text(json.dumps({"target": "mm", "delta_pct": -1.2}), encoding="utf-8")
    (run / "plan_review.md").write_text("# Reject cold plan\n\nRegression risk.", encoding="utf-8")
    (run / "idea_ledger.md").write_text("### L001 — Hoist first\n- **status**: approved\n- **note**: keep\n", encoding="utf-8")

    records = extract_run(run)

    assert {record.type for record in records} == {"fact", "anti_pattern", "playbook_step"}
    assert all(record.maturity == "L1" for record in records)


def test_write_and_bundle_jsonl(tmp_path: Path) -> None:
    run = tmp_path / "r1"
    run.mkdir()
    (run / "bench_report.json").write_text(json.dumps({"target": "mm", "pass_rate": 1.0}), encoding="utf-8")
    records = extract_run(run)
    staging = tmp_path / "staging"
    first = staging / "r1.jsonl"
    assert write_jsonl(records, first) == 1
    assert bundle_staging(staging, staging / "bundle.jsonl") == 1
