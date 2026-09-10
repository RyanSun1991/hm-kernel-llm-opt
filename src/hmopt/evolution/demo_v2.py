"""Disposable source-to-journal integration demo with explicitly synthetic IC data.

This exercises local adapters and gates. It neither starts an agent nor measures
a device, merges a change, publishes knowledge, or touches an existing checkout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .learning import export_bundle, quality_report
from .mining import Hotspot, Matcher, Pattern
from .reports import ICManifest, convert_ic_report
from .service import EvolutionService, GateError, Plan, Review
from .sources import distill_sources, import_workspace
from .store import digest
from .validation import ABReport
from .workflow import apply_review_sheet, dispatch_candidate, export_review_sheet


def run_demo_v2(output: Path, *, git_bin: str = "git") -> dict:
    """Create a new fixture workspace and run the integrated v2 local protocol."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = output / "example-repo"
    repo.mkdir()
    hooks = output / "no-hooks"
    hooks.mkdir()
    # All writes below target the newly created fixture. Inherited Git location,
    # configuration and index overrides must not redirect them to a real repo.
    git_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    git_env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_TERMINAL_PROMPT="0",
        GIT_TEMPLATE_DIR=str(hooks),
    )

    def git(*args: str) -> str:
        result = subprocess.run(
            [git_bin, "-c", f"core.hooksPath={hooks}", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            env=git_env,
        )
        return result.stdout.strip()

    def write_json(path: Path, data: dict) -> None:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")

    git("init", "--quiet")
    git("config", "user.name", "HMOPT synthetic v2 fixture")
    git("config", "user.email", "synthetic-v2@invalid.local")
    git("config", "core.hooksPath", str(hooks))
    git("config", "core.autocrlf", "false")

    def commit(message: str) -> str:
        git("add", "--all")
        git("-c", "commit.gpgsign=false", "commit", "--quiet", "-m", message)
        return git("rev-parse", "HEAD")

    before = "def total(values):\n    return sum([item * 2 for item in values])\n"
    after = "def total(values):\n    return sum(item * 2 for item in values)\n"
    (repo / "target.py").write_text(before, encoding="utf-8")
    review_path = repo / ".opencode/reviews/fixture-allocation.md"
    review_path.parent.mkdir(parents=True)
    source_text = (
        "# Synthetic prior review; not measured kernel evidence\n\n"
        "Optimization clue in target.py: `sum([item * 2 for item in values])`.\n"
        "A temporary list might be avoided for pure integer reduction. Verify evaluation, "
        "iterator lifetime and exception behavior before applying this idea.\n"
        "This document supplies no measured IC improvement and grants no approval.\n"
        "decision_reason: unknown\n"
    )
    review_path.write_text(source_text, encoding="utf-8")
    base = commit("fixture: add a source review and a candidate implementation")

    service = EvolutionService(output / "state", git_bin=git_bin)
    imported = import_workspace(service, repo, "synthetic-v2-repository", actor="fixture-reader")
    if imported["errors"] or len(imported["imported"]) != 1:
        raise RuntimeError("The synthetic source fixture did not import exactly once")
    source_id = imported["imported"][0]
    distilled = distill_sources(service, actor="fixture-distiller")
    pattern = Pattern(
        pattern_id="fixture-v2-integer-reduction",
        title="Synthetic pure integer reduction candidate",
        kind="optimization",
        problem="A list is materialized for immediate consumption by a pure reduction.",
        diagnosis="Check input purity, evaluation order and iterator lifetime for the selected file.",
        remedy="Evaluate a generator substitution and require independently reviewed evidence.",
        matcher=Matcher(file_globs=["**/*.py"], all_of=["sum([item * 2 for item in values])"]),
        primary_metric="instructions",
        direction="minimize",
        unit="count",
        preconditions=["Only the fixture's finite integer input and pure expression are in scope."],
        risks=[
            "This fixture does not establish any real instruction-count or latency improvement."
        ],
        source_ids=[source_id],
    )
    pattern_key = f"{pattern.pattern_id}@1"
    service.import_pattern(pattern)
    service.activate_pattern(
        pattern_key,
        actor="fixture-curator",
        note="Activate only the bounded synthetic example; no production evidence is claimed.",
    )
    write_json(output / "curated_pattern.json", pattern.model_dump(mode="json"))
    scan = service.scan(
        repo,
        owners={"**/*.py": "fixture-owner"},
        hotspots=[Hotspot(path="target.py", symbol="total", weight=0.9, revision=base)],
    )
    if len(scan["candidates"]) != 1:
        raise RuntimeError("The synthetic rule must produce exactly one candidate")
    row = scan["candidates"][0]
    candidate_id = row["id"]
    blocked: list[str] = []
    try:
        dispatch_candidate(
            service,
            candidate_id,
            output / "dispatches",
            actor="fixture-operator",
            request_id="v2-unconfirmed",
        )
    except GateError:
        blocked.append("dispatch_before_owner_confirmation")
    else:
        raise RuntimeError("Unconfirmed fixture dispatch was incorrectly allowed")

    sheet = export_review_sheet(service, output / "owner-review", owner="fixture-owner")
    sheet_path = Path(sheet["json_path"])
    decisions = json.loads(sheet_path.read_text(encoding="utf-8"))
    decisions["items"][0].update(
        decision="confirm",
        note="Fixture owner confirms only this disposable example for design.",
    )
    # The generated owner sheet deliberately permits editing only these fields.
    sheet_path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    owner_result = apply_review_sheet(service, sheet_path, actor="fixture-owner")
    if owner_result["errors"] or owner_result["applied"] != 1:
        raise RuntimeError("Fixture owner review did not apply")
    row = service.store.read("candidate", candidate_id)
    dispatches = [
        dispatch_candidate(
            service,
            candidate_id,
            output / "dispatches",
            actor="fixture-operator",
            request_id="v2-architect",
        )
    ]

    def transition(action: str, actor: str, payload: dict) -> dict:
        nonlocal row
        row = service.transition(
            candidate_id,
            action,
            actor=actor,
            expected_version=row["version"],
            request_id="v2-" + action,
            payload=payload,
        )
        return row

    policy = {
        "metrics": [
            {
                "name": "instructions",
                "unit": "count",
                "direction": "minimize",
                "primary": True,
                "min_improvement_pct": 5.0,
                "max_regression_pct": 0.0,
            }
        ],
        "minimum_pairs": 3,
        "device_id": "SYNTHETIC-NO-DEVICE",
        "workload_id": "fixture-integer-reduction",
        "workload_config_sha256": digest({"fixture": "fixed integer input"}),
        "environment_sha256": digest({"fixture": "synthetic IC protocol demonstration"}),
    }
    plan = Plan(
        candidate_id=candidate_id,
        base_revision=base,
        author="fixture-architect",
        hypothesis="Exercise a bounded implementation and the IC acceptance protocol using synthetic data.",
        bottleneck="unknown",
        metric_rationale="Freeze instruction counts for this adapter demo; real bottleneck attribution remains unknown.",
        allowed_paths=["target.py"],
        validation=policy,
    )
    plan_data = plan.model_dump(mode="json")
    plan_review = Review(
        candidate_id=candidate_id,
        subject_digest=digest(plan_data),
        author="fixture-plan-reviewer",
        decision="approve",
        rationale="Independent fixture review approves only the disposable file and synthetic acceptance exercise.",
    )
    write_json(output / "approved_plan.json", plan_data)
    write_json(output / "plan_review.json", plan_review.model_dump(mode="json"))
    transition(
        "approve_plan",
        "fixture-plan-reviewer",
        {"plan": plan_data, "review": plan_review.model_dump(mode="json")},
    )
    dispatches.append(
        dispatch_candidate(
            service,
            candidate_id,
            output / "dispatches",
            actor="fixture-operator",
            request_id="v2-implementer",
        )
    )

    (repo / "target.py").write_text(after, encoding="utf-8")
    implementation = commit("fixture: exercise reviewed generator substitution")
    transition("record_implementation", "fixture-implementer", {"revision": implementation})
    dispatches.append(
        dispatch_candidate(
            service,
            candidate_id,
            output / "dispatches",
            actor="fixture-operator",
            request_id="v2-reviewer",
        )
    )
    code_review = Review(
        candidate_id=candidate_id,
        subject_digest=row["data"]["implementation_digest"],
        author="fixture-code-reviewer",
        decision="approve",
        rationale="The fixture diff stays within target.py and pure integer behavior is checked separately.",
    )
    write_json(output / "code_review.json", code_review.model_dump(mode="json"))
    transition("approve_code", "fixture-code-reviewer", code_review.model_dump(mode="json"))
    dispatches.append(
        dispatch_candidate(
            service,
            candidate_id,
            output / "dispatches",
            actor="fixture-operator",
            request_id="v2-validator",
        )
    )
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import runpy; total=runpy.run_path('target.py')['total']; assert total([])==0; assert total([1,2,3])==12",
        ],
        cwd=repo,
        check=True,
        timeout=30,
        capture_output=True,
    )

    baseline_dir = output / "synthetic-ic/baseline"
    candidate_dir = output / "synthetic-ic/candidate"
    baseline_dir.mkdir(parents=True)
    candidate_dir.mkdir()
    write_json(baseline_dir / "counts.json", {"simulation": True, "counts": [1000, 1000, 1000]})
    write_json(candidate_dir / "counts.json", {"simulation": True, "counts": [900, 900, 900]})
    compare = {
        "success": True,
        "level": "total",
        "target": {},
        "simulation": True,
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "aggregate": {"note": "Synthetic aggregate text is not an acceptance verdict."},
        "reports": [
            {
                "case": "synthetic-total",
                "round": index,
                "step": 0,
                "baseline": 1000,
                "candidate": 900,
                "baseline_found": True,
                "candidate_found": True,
            }
            for index in range(3)
        ],
    }

    def arm(revision: str, label: str) -> dict:
        return {
            "repo_revision": revision,
            "image_sha256": digest({"synthetic_image": label}),
            "hardware": False,
            "measurements": [],
            **{
                key: policy[key]
                for key in (
                    "device_id",
                    "workload_id",
                    "workload_config_sha256",
                    "environment_sha256",
                )
            },
        }

    manifest = ICManifest(
        candidate_id=candidate_id,
        plan_digest=row["data"]["plan_digest"],
        implementation_digest=row["data"]["implementation_digest"],
        compare_sha256=digest(compare),
        metric_name="instructions",
        level="total",
        target={},
        baseline_dir=str(baseline_dir),
        candidate_dir=str(candidate_dir),
        baseline=arm(base, "baseline"),
        candidate=arm(implementation, "candidate"),
        functional_passed=True,
    )
    converted = convert_ic_report(service, candidate_id, compare, manifest)
    write_json(output / "synthetic_ic_compare.json", compare)
    write_json(output / "synthetic_ic_manifest.json", manifest.model_dump(mode="json"))
    write_json(output / "synthetic_ab_report.json", converted["report"])
    row = service.validate(
        candidate_id,
        ABReport.model_validate(converted["report"]),
        actor="fixture-validator",
        expected_version=row["version"],
        request_id="v2-synthetic-validation",
        allow_synthetic=True,
    )
    skills = service.store.list("skill")
    skill = next(item for item in skills if item["data"].get("candidate_id") == candidate_id)
    try:
        service.promote(
            skill["id"],
            tier="staging",
            actor="fixture-curator",
            expected_version=skill["version"],
            request_id="v2-prohibited-promotion",
            note="Synthetic demo results must never become promotion-eligible.",
        )
    except GateError:
        blocked.append("synthetic_skill_promotion")
    else:
        raise RuntimeError("Synthetic fixture skill was incorrectly promoted")
    try:
        export_bundle(service, output / "prohibited-bundle", "fixture-curator", [skill["id"]])
    except ValueError:
        blocked.append("synthetic_bundle_export")
    else:
        raise RuntimeError("Synthetic fixture bundle was incorrectly exported")
    quality = quality_report(service, pattern_key)
    write_json(output / "quality.json", quality)
    if review_path.read_text(encoding="utf-8") != source_text:
        raise RuntimeError("The fixture source review changed during implementation")
    summary = {
        "simulation": True,
        "hardware_executed": False,
        "agent_started": False,
        "merged": False,
        "published": False,
        "output": str(output),
        "repo": str(repo),
        "candidate_id": candidate_id,
        "pattern_key": pattern_key,
        "source_id": source_id,
        "source_import": imported,
        "source_distillation": distilled,
        "owner_review": owner_result,
        "review_sheet": sheet,
        "dispatches": dispatches,
        "source_evidence": converted["source_evidence"],
        "functional_smoke_passed": True,
        "final_state": row,
        "knowledge": skill,
        "quality": quality,
        "blocked_gates": blocked,
        "audit_events": len(service.store.audit(candidate_id)),
        "note": "IC samples, image hashes and actor identities are fixtures. Only the local Python functional smoke was executed; no real kernel benefit or production approval is claimed.",
    }
    write_json(output / "summary.json", summary)
    return summary
