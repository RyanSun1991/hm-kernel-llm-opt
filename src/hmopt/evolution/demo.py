"""Disposable, explicitly synthetic example of the complete gated workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .mining import Hotspot, Matcher, Pattern, mine_git_history
from .service import EvolutionService, GateError, Plan, Review
from .store import digest
from .validation import ABReport


def run_demo(output: Path, *, git_bin: str = "git") -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = output / "example-repo"
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.check_output(
            [git_bin, "-C", str(repo), *args], text=True, encoding="utf-8", stderr=subprocess.PIPE
        ).strip()

    git("init", "--quiet")
    git("config", "user.name", "HMOPT synthetic example")
    git("config", "user.email", "example@invalid.local")
    git("config", "core.hooksPath", str(output / "no-hooks"))
    git("config", "core.autocrlf", "false")

    def commit(message: str) -> str:
        git("add", "--all")
        git("-c", "commit.gpgsign=false", "commit", "--quiet", "-m", message)
        return git("rev-parse", "HEAD")

    before = "def total(values):\n    return sum([item * 2 for item in values])\n"
    after = "def total(values):\n    return sum(item * 2 for item in values)\n"
    (repo / "history_example.py").write_text(before, encoding="utf-8")
    commit("example: add historical allocation path")
    (repo / "history_example.py").write_text(after, encoding="utf-8")
    historical_revision = commit("perf: avoid a temporary list allocation in sum")
    (repo / "target.py").write_text(before, encoding="utf-8")
    base = commit("example: add a similar candidate path")

    service = EvolutionService(output / "state", git_bin=git_bin)
    changes = mine_git_history(repo, git_bin=git_bin)
    mining = service.ingest_history(changes)
    source_id = next(
        change.source_id for change in changes if change.revision == historical_revision
    )
    pattern = Pattern(
        pattern_id="example-avoid-temporary-list",
        version=1,
        title="Avoid materializing sum input",
        kind="optimization",
        problem="A temporary list is constructed only to be immediately consumed.",
        diagnosis="Inspect iterator semantics, side effects and exception timing before substitution.",
        remedy="Use a generator only where evaluation order and visible behavior remain valid.",
        matcher=Matcher(file_globs=["**/*.py"], all_of=["sum([item * 2 for item in values])"]),
        primary_metric="allocation_bytes",
        direction="minimize",
        unit="bytes",
        preconditions=["Pure integer expression and a finite re-iterable input in this example."],
        risks=["Iterator lifetime and exception timing require independent review."],
        source_ids=[source_id],
    )
    service.import_pattern(pattern)
    (output / "curated_pattern.json").write_text(
        pattern.model_dump_json(indent=2), encoding="utf-8"
    )
    service.activate_pattern(
        f"{pattern.pattern_id}@1",
        actor="example-curator",
        note="Curated synthetic fixture: only the demonstrated pure integer reduction.",
    )
    scan = service.scan(
        repo,
        owners={"**/*.py": "example-owner"},
        hotspots=[Hotspot(path="target.py", symbol="total", weight=0.9, revision=base)],
    )
    row = scan["candidates"][0]
    candidate_id = row["id"]
    blocked = []
    try:
        service.handoff(candidate_id)
    except GateError:
        blocked.append("implementation_handoff_before_owner_confirmation")

    def transition(action: str, actor: str, payload: dict) -> dict:
        nonlocal row
        row = service.transition(
            candidate_id,
            action,
            actor=actor,
            expected_version=row["version"],
            request_id=f"example-{action}",
            payload=payload,
        )
        return row

    transition(
        "confirm", "example-owner", {"note": "Approve this synthetic example candidate for design."}
    )
    policy = {
        "metrics": [
            {
                "name": "allocation_bytes",
                "unit": "bytes",
                "direction": "minimize",
                "primary": True,
                "min_improvement_pct": 5.0,
                "max_regression_pct": 0.0,
            },
            {
                "name": "latency_ms",
                "unit": "ms",
                "direction": "minimize",
                "primary": False,
                "min_improvement_pct": 0.0,
                "max_regression_pct": 1.0,
            },
        ],
        "minimum_pairs": 3,
        "device_id": "SYNTHETIC-NO-DEVICE",
        "workload_id": "example-integer-sum",
        "workload_config_sha256": digest({"fixture": "fixed integer input"}),
        "environment_sha256": digest({"environment": "synthetic example"}),
    }
    plan = Plan(
        candidate_id=candidate_id,
        base_revision=base,
        author="example-architect",
        hypothesis="Remove temporary list materialization while retaining pure integer summation.",
        bottleneck="memory",
        metric_rationale="Measure allocation bytes with latency as a guardrail.",
        allowed_paths=["target.py"],
        validation=policy,
    )
    plan_data = plan.model_dump(mode="json")
    review = Review(
        candidate_id=candidate_id,
        subject_digest=digest(plan_data),
        author="example-plan-reviewer",
        decision="approve",
        rationale="The bounded fixture has pure inputs and a measurable allocation hypothesis.",
    )
    (output / "approved_plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    (output / "plan_review_payload.json").write_text(
        json.dumps({"plan": plan_data, "review": review.model_dump(mode="json")}, indent=2),
        encoding="utf-8",
    )
    transition(
        "approve_plan",
        "example-plan-reviewer",
        {"plan": plan_data, "review": review.model_dump(mode="json")},
    )
    packet = service.handoff(candidate_id)
    (output / "implementer_handoff.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # This edit affects only the newly created example repository, after its demo approvals.
    (repo / "target.py").write_text(after, encoding="utf-8")
    impl = commit("perf: remove candidate temporary allocation (synthetic example)")
    transition("record_implementation", "example-implementer", {"revision": impl})
    review = Review(
        candidate_id=candidate_id,
        subject_digest=row["data"]["implementation_digest"],
        author="example-code-reviewer",
        decision="approve",
        rationale="The example diff is within scope and preserves pure integer reduction behavior.",
    )
    (output / "code_review.json").write_text(review.model_dump_json(indent=2), encoding="utf-8")
    transition("approve_code", "example-code-reviewer", review.model_dump(mode="json"))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from target import total; assert total([])==0; assert total([1,2,3])==12",
        ],
        cwd=repo,
        check=True,
        timeout=30,
    )

    def arm(revision: str, image_label: str, allocation: float) -> dict:
        return {
            "repo_revision": revision,
            "image_sha256": digest(image_label),
            "hardware": False,
            **{
                key: policy[key]
                for key in (
                    "device_id",
                    "workload_id",
                    "workload_config_sha256",
                    "environment_sha256",
                )
            },
            "measurements": [
                {
                    "pair_id": f"case-{i}",
                    "metrics": {"allocation_bytes": allocation, "latency_ms": 10.0},
                }
                for i in range(3)
            ],
        }

    report = ABReport(
        candidate_id=candidate_id,
        implementation_revision=impl,
        baseline=arm(base, "synthetic baseline image", 1000.0),
        candidate=arm(impl, "synthetic candidate image", 800.0),
        functional_passed=True,
        metrics=policy["metrics"],
        minimum_pairs=3,
    )
    (output / "synthetic_ab_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    row = service.validate(
        candidate_id,
        report,
        actor="example-validator",
        expected_version=row["version"],
        request_id="example-validate",
        allow_synthetic=True,
    )
    skill = service.store.list("skill")[0]
    try:
        service.promote(
            skill["id"],
            tier="staging",
            actor="example-curator",
            expected_version=skill["version"],
            request_id="example-forbidden-promotion",
            note="Attempt to promote the synthetic example must be blocked.",
        )
    except GateError:
        blocked.append("synthetic_result_promotion")
    summary = {
        "simulation": True,
        "hardware_executed": False,
        "output": str(output),
        "candidate_id": candidate_id,
        "mining": mining,
        "final_state": row,
        "knowledge": skill,
        "blocked_gates": blocked,
        "audit_events": len(service.store.audit(candidate_id)),
        "note": "The numbers and reviewer identities are fixtures, not measured kernel benefits or real approvals.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
