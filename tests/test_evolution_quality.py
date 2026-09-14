"""Quality/overlay/export checks using local stores and declared fixture evidence."""

import hashlib
import json
from copy import deepcopy

import pytest
from test_evolution_service import Scenario
from test_evolution_service import git_bin as shared_git_bin

from hmopt.evolution.correctness import CorrectnessPolicy, CorrectnessReport
from hmopt.evolution.learning import export_bundle, quality_report, set_pattern_overlay
from hmopt.evolution.mining import Hotspot
from hmopt.evolution.reports import ICManifest, convert_ic_report
from hmopt.evolution.service import EvolutionService
from hmopt.evolution.store import ConflictError, digest
from hmopt.evolution.validation import ABReport

git_bin = shared_git_bin


@pytest.fixture
def scenario(tmp_path, git_bin):
    return Scenario(tmp_path, git_bin)


@pytest.fixture
def reviewed(scenario):
    return scenario.ready()


@pytest.fixture
def curated(reviewed):
    reviewed.validate()
    reviewed.promote()
    return reviewed


def summary(scenario):
    return quality_report(scenario.service, scenario.pattern_key)["patterns"][scenario.pattern_key]


def seed_snapshot(scenario, candidate_id, **changes):
    """Persist a minimal candidate snapshot for fact-aggregation edge cases."""
    data = deepcopy(scenario.row["data"])
    data["candidate"]["candidate_id"] = candidate_id
    data.update(changes)
    with scenario.service.store.transaction() as db:
        return scenario.service.store.put(db, "candidate", candidate_id, data)


def update_record(service, kind, record_id, change):
    with service.store.transaction() as db:
        row = service.store.get(db, kind, record_id)
        data = deepcopy(row["data"])
        change(data)
        return service.store.put(db, kind, record_id, data, row["version"])


def test_quality_uses_distinct_owner_facts_not_events_or_stage_labels(scenario):
    s = scenario
    s.confirm()
    seed_snapshot(
        s, "rejected-candidate", rejection={"actor": "owner", "note": "Rejected after review."}
    )
    seed_snapshot(s, "undecided-candidate", owner_decision=None, stage="validated")
    with s.service.store.transaction() as db:
        for _ in range(5):
            s.service.store.event(db, s.candidate_id, "confirm", "owner", {})
    result = summary(s)
    assert result["candidates"] == 3
    assert result["owner_confirmed"] == 1
    assert result["owner_rejected"] == 1
    assert result["owner_undecided"] == 1
    assert result["owner_decisions"] == 2
    assert result["acceptance_rate"] == 0.5
    assert result["true_precision"] is None
    assert result["validated_candidates"] == 0


def test_quality_never_invents_precision_or_an_acceptance_denominator(scenario):
    result = summary(scenario)
    assert result["acceptance_rate"] is None
    assert result["true_precision"] is None
    assert result["latest_reports"]["none"] == 1


def test_quality_separates_attempts_latest_report_and_distinct_validated_candidates(reviewed):
    s = reviewed
    s.validate(ABReport.model_validate(s.report_data(values=(99.5,) * 3)))
    s.transition(
        "retry_validation", "owner", {"note": "Repeat measurements under the approved policy."}
    )
    # Moving an attempt into history must not erase the latest known outcome.
    waiting = summary(s)
    assert waiting["latest_reports"]["inconclusive"] == 1
    s.validate()
    result = summary(s)
    assert result["attempts"]["inconclusive"] == 1
    assert result["attempts"]["pass"] == 1
    assert result["latest_reports"]["inconclusive"] == 0
    assert result["latest_reports"]["pass"] == 1
    assert result["validated_candidates"] == 1
    assert result["distinct_report_evidence"] == 2
    assert result["attempts_by_execution"]["performance_hardware"]["pass"] == 1


@pytest.mark.parametrize("hardware,allow_synthetic", [(False, True), (True, True), (False, False)])
def test_quality_excludes_simulated_measurements(reviewed, hardware, allow_synthetic):
    s = reviewed
    s.validate(
        ABReport.model_validate(s.report_data(hardware=hardware)), allow_synthetic=allow_synthetic
    )
    result = summary(s)
    assert result["attempts"]["simulation"] == 1
    assert result["attempts"]["pass"] == 0
    assert result["validated_candidates"] == 0


@pytest.mark.parametrize(
    "corruption",
    [
        "missing-evidence",
        "digest",
        "verdict",
        "wrong-candidate",
        "simulation-metadata",
        "execution-metadata",
    ],
)
def test_quality_excludes_invalid_reports_instead_of_trusting_cached_verdicts(reviewed, corruption):
    s = reviewed
    s.validate()
    if corruption == "missing-evidence":
        update_record(
            s.service,
            "candidate",
            s.candidate_id,
            lambda data: data["validation"].update(report_evidence="f" * 64),
        )
    elif corruption == "digest":
        with s.service.store.transaction() as db:
            sha = s.row["data"]["validation"]["report_evidence"]
            db.execute("UPDATE evidence SET content=? WHERE sha256=?", ('{"tampered":true}', sha))
    elif corruption == "verdict":
        update_record(
            s.service,
            "candidate",
            s.candidate_id,
            lambda data: data["validation"]["result"].update(verdict="fail"),
        )
    elif corruption == "wrong-candidate":
        update_record(
            s.service,
            "candidate",
            s.candidate_id,
            lambda data: data["candidate"].update(candidate_id="another-candidate"),
        )
    elif corruption == "simulation-metadata":
        update_record(
            s.service,
            "candidate",
            s.candidate_id,
            lambda data: data["validation"].update(simulation="false"),
        )
    else:
        update_record(
            s.service,
            "candidate",
            s.candidate_id,
            lambda data: data["validation"].update(execution_kind="local"),
        )
    result = summary(s)
    assert result["latest_reports"]["invalid"] == 1
    assert result["attempts"]["pass"] == 0
    assert result["validated_candidates"] == 0


def test_quality_snapshot_is_deterministic_and_survives_restart(scenario):
    first = quality_report(scenario.service)
    assert quality_report(scenario.service) == first
    restarted = EvolutionService(scenario.service.store.root, git_bin=scenario.git_bin)
    assert quality_report(restarted) == first
    scenario.confirm()
    assert quality_report(restarted)["quality_sha256"] != first["quality_sha256"]


def test_quality_reads_candidates_after_the_first_thousand(scenario):
    s = scenario
    with s.service.store.transaction() as db:
        for index in range(1001):
            data = {
                "candidate": {**s.row["data"]["candidate"], "candidate_id": f"extra-{index}"},
                "stage": "discovered",
            }
            if index == 1000:
                data["owner_decision"] = {"actor": "owner", "note": "Confirmed late record."}
            s.service.store.put(db, "candidate", f"extra-{index}", data)
    result = summary(s)
    assert result["candidates"] == 1002
    assert result["owner_confirmed"] == 1


def overlay(s, **overrides):
    params = {
        "pattern_key": s.pattern_key,
        "factor": 0.5,
        "state": "probation",
        "actor": "curator",
        "note": "Apply a conservative ranking overlay pending more representative evidence.",
        "request_id": "overlay-one",
    }
    params.update(overrides)
    return set_pattern_overlay(s.service, **params)


def test_overlay_is_absolute_versioned_and_freezes_quality_provenance(scenario):
    s = scenario
    original_pattern = s.service.store.read("pattern", s.pattern_key)
    first = overlay(s)
    frozen = deepcopy(first["data"]["quality"])
    s.confirm()
    assert first["data"]["quality"] == frozen
    assert quality_report(s.service, s.pattern_key) != frozen
    second = overlay(s, factor=0.4, request_id="overlay-two", expected_version=first["version"])
    assert second["data"]["factor"] == 0.4
    assert second["data"]["factor"] != 0.5 * 0.4
    assert second["version"] == first["version"] + 1
    assert second["data"]["quality_sha256"] == second["data"]["quality"]["quality_sha256"]
    assert s.service.store.read("pattern", s.pattern_key) == original_pattern


def test_overlay_replays_exact_request_and_rejects_conflicting_or_stale_writes(scenario):
    s = scenario
    first = overlay(s)
    assert overlay(s) == first
    assert (
        len(
            [
                e
                for e in s.service.store.audit(s.pattern_key)
                if e["action"] == "set_pattern_overlay"
            ]
        )
        == 1
    )
    with pytest.raises(ConflictError):
        overlay(s, factor=0.7)
    with pytest.raises(ConflictError):
        overlay(s, request_id="missing-version")
    overlay(s, request_id="second", expected_version=first["version"], factor=0.8, state="active")
    with pytest.raises(ConflictError):
        overlay(s, request_id="stale", expected_version=first["version"])


@pytest.mark.parametrize(
    "overrides",
    [
        {"factor": float("nan")},
        {"factor": float("inf")},
        {"factor": -0.1},
        {"factor": 1.1},
        {"factor": True},
        {"factor": "0.5"},
        {"factor": 10**400},
        {"state": "retired"},
        {"actor": " "},
        {"note": "short"},
        {"request_id": " "},
        {"expected_version": True},
    ],
)
def test_overlay_rejects_invalid_controls_without_retiring_a_pattern(scenario, overrides):
    with pytest.raises(ValueError):
        overlay(scenario, **overrides)
    assert scenario.service.store.list("overlay") == []
    assert (
        scenario.service.store.read("pattern", scenario.pattern_key)["data"]["status"] == "active"
    )


def test_zero_sample_overlay_is_explicit_and_does_not_auto_retire(scenario):
    before = scenario.service.store.read("pattern", scenario.pattern_key)
    result = overlay(scenario, factor=0.0)
    assert result["data"]["quality"]["patterns"][scenario.pattern_key]["owner_decisions"] == 0
    assert scenario.service.store.read("pattern", scenario.pattern_key) == before


def test_export_is_a_sanitized_review_manifest_not_an_installable_or_published_skill(
    curated, tmp_path
):
    s = curated
    output = tmp_path / "bundle"
    result = export_bundle(s.service, output, actor="private-curator@example.invalid")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    proposals = [
        json.loads(line)
        for line in (output / "proposals.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["schema_version"] == 1
    assert manifest["native_hub_package"] is False
    assert manifest["published"] is False
    assert manifest["merged"] is False
    assert manifest["status"] == "exported"
    assert manifest["proposal_count"] == len(proposals) == 1
    assert proposals[0]["validation_status"] == "validated"
    assert proposals[0]["merge_status"] == "not_asserted"
    assert proposals[0]["publication_status"] == "not_published"
    assert proposals[0]["validation_kind"] == "performance"
    assert proposals[0]["execution_kind"] == "hardware"
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "proposals.jsonl",
        "review_checklist.md",
    }
    for filename, expected in manifest["files"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
    assert s.service.store.read("bundle", result["id"])["data"]["status"] == "exported"
    text = "\n".join(path.read_text(encoding="utf-8") for path in output.iterdir())
    for secret in (
        s.candidate_id,
        s.pattern_key,
        s.path,
        str(s.repo),
        "fixture-device",
        "private-curator@example.invalid",
        "Remove repeated lookup instructions",
    ):
        assert secret not in text
    assert "not a native Hub package" in text
    assert "shareable instructions" in text


def test_export_does_not_copy_free_text_pattern_metric_recipe_or_secrets(curated, tmp_path):
    s = curated
    secret = "NEVER_EXPORT_SECRET_7c39"
    update_record(
        s.service,
        "skill",
        s.skill()["id"],
        lambda data: data.update(
            recipe=secret + " raw source",
            curation_note=secret + " operator note",
            target="/private/" + secret,
            additional_free_text=secret,
        ),
    )
    output = tmp_path / "sanitized"
    export_bundle(s.service, output, actor=secret + "@private.invalid")
    assert all(secret not in path.read_text(encoding="utf-8") for path in output.iterdir())


def test_export_requires_a_new_output_directory_and_preserves_existing_contents(curated, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "user-file.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_bundle(curated.service, output, actor="curator")
    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert curated.service.store.list("bundle") == []


@pytest.mark.parametrize("which", ["journal", "captured", "simulation"])
def test_export_rejects_unapproved_or_simulated_knowledge(reviewed, tmp_path, which):
    s = reviewed
    if which == "captured":
        skill = s.service.capture(
            signal="effective_recipe",
            recipe="Promising but unverified local observation.",
            actor="expert",
            source_kind="candidate",
            source_id=s.candidate_id,
        )
    else:
        s.validate(allow_synthetic=which == "simulation")
        skill = s.skill()
    with pytest.raises(ValueError):
        export_bundle(
            s.service, tmp_path / "not-exported", actor="curator", skill_ids=[skill["id"]]
        )
    assert not (tmp_path / "not-exported").exists()
    assert s.service.store.list("bundle") == []


@pytest.mark.parametrize("reference", ["journal", "report", "patch"])
def test_export_digest_verifies_all_exported_evidence_references(curated, tmp_path, reference):
    s = curated
    hashes = {
        "journal": s.skill()["data"]["evidence"],
        "report": s.row["data"]["validation"]["report_evidence"],
        "patch": s.row["data"]["implementation"]["patch_evidence"],
    }
    with s.service.store.transaction() as db:
        db.execute(
            "UPDATE evidence SET content=? WHERE sha256=?", ('{"changed":true}', hashes[reference])
        )
    with pytest.raises(ValueError):
        export_bundle(s.service, tmp_path / "corrupt", actor="curator")
    assert not (tmp_path / "corrupt").exists()
    assert s.service.store.list("bundle") == []


@pytest.fixture
def converted_curated(reviewed):
    """Convert declared fixture IC data so validation freezes its source provenance."""
    s = reviewed
    original = s.report_data()
    compare = {
        "success": True,
        "level": "total",
        "target": {},
        "baseline_dir": "fixture-baseline",
        "candidate_dir": "fixture-candidate",
        "reports": [
            {
                "case": "lookup",
                "round": index,
                "step": 0,
                "baseline": 100,
                "candidate": 90,
                "baseline_found": True,
                "candidate_found": True,
            }
            for index in range(3)
        ],
    }
    manifest = ICManifest(
        candidate_id=s.candidate_id,
        plan_digest=s.row["data"]["plan_digest"],
        implementation_digest=s.row["data"]["implementation_digest"],
        compare_sha256=digest(compare),
        metric_name="instructions",
        level="total",
        target={},
        baseline_dir=compare["baseline_dir"],
        candidate_dir=compare["candidate_dir"],
        baseline={**original["baseline"], "measurements": []},
        candidate={**original["candidate"], "measurements": []},
        functional_passed=True,
    )
    converted = convert_ic_report(s.service, s.candidate_id, compare, manifest)
    s.validate(ABReport.model_validate(converted["report"]))
    assert s.row["data"]["validation"]["source_evidence"] == [converted["source_evidence"]]
    s.promote()
    return s


@pytest.mark.parametrize("damage", ["deleted", "changed"])
def test_quality_and_export_reject_damaged_frozen_ic_sources(converted_curated, tmp_path, damage):
    s = converted_curated
    source = s.row["data"]["validation"]["source_evidence"][0]
    assert summary(s)["validated_candidates"] == 1
    with s.service.store.transaction() as db:
        if damage == "deleted":
            db.execute("DELETE FROM evidence WHERE sha256=?", (source,))
        else:
            db.execute("UPDATE evidence SET content=? WHERE sha256=?", ('{"changed":true}', source))
    result = summary(s)
    assert result["latest_reports"]["invalid"] == 1
    assert result["validated_candidates"] == 0
    with pytest.raises(ValueError):
        export_bundle(s.service, tmp_path / "damaged-ic", actor="curator")
    assert not (tmp_path / "damaged-ic").exists()
    assert s.service.store.list("bundle") == []


def test_export_includes_frozen_ic_hashes_and_ignores_later_conversion_index(
    converted_curated, tmp_path
):
    s = converted_curated
    frozen = s.row["data"]["validation"]["source_evidence"]
    report_digest = s.row["data"]["validation"]["report_evidence"]
    before = quality_report(s.service)
    # Later equivalent collection declarations must neither invalidate nor expand this snapshot.
    update_record(
        s.service,
        "report_conversion",
        report_digest,
        lambda data: data["source_evidence"].append("0" * 64),
    )
    assert quality_report(s.service) == before
    output = tmp_path / "frozen-ic"
    export_bundle(s.service, output, actor="curator")
    proposal = json.loads((output / "proposals.jsonl").read_text(encoding="utf-8"))
    assert set(frozen) <= set(proposal["evidence_sha256"])
    assert "0" * 64 not in proposal["evidence_sha256"]


def test_export_rejects_candidate_provenance_stripped_from_curated_snapshot(
    converted_curated, tmp_path
):
    s = converted_curated
    update_record(
        s.service,
        "candidate",
        s.candidate_id,
        lambda data: data["validation"].update(source_evidence=[]),
    )
    with pytest.raises(ValueError, match="source provenance"):
        export_bundle(s.service, tmp_path / "stripped-ic", actor="curator")
    assert not (tmp_path / "stripped-ic").exists()


@pytest.mark.parametrize("invalid", [None, "a" * 64, ["short"], ["a" * 64] * 33])
def test_quality_rejects_malformed_frozen_source_reference_lists(curated, invalid):
    update_record(
        curated.service,
        "candidate",
        curated.candidate_id,
        lambda data: data["validation"].update(source_evidence=invalid),
    )
    assert summary(curated)["latest_reports"]["invalid"] == 1


def test_legacy_report_without_source_field_remains_eligible(curated, tmp_path):
    s = curated
    update_record(
        s.service,
        "candidate",
        s.candidate_id,
        lambda data: data["validation"].pop("source_evidence", None),
    )
    assert summary(s)["validated_candidates"] == 1
    export_bundle(s.service, tmp_path / "legacy-report", actor="curator")


def test_export_respects_an_explicit_selection_and_rejects_duplicates(curated, tmp_path):
    s = curated
    skill_id = s.skill()["id"]
    with pytest.raises(ValueError):
        export_bundle(
            s.service, tmp_path / "duplicates", actor="curator", skill_ids=[skill_id, skill_id]
        )
    result = export_bundle(s.service, tmp_path / "selected", actor="curator", skill_ids=[skill_id])
    assert result["data"]["proposal_count"] == 1


def test_export_removes_only_its_new_partial_files_on_failed_write(curated, tmp_path, monkeypatch):
    from pathlib import Path

    original_open = Path.open

    def fail_manifest(path, *args, **kwargs):
        if path.name == "manifest.json" and args and args[0] == "x":
            raise OSError("simulated local disk failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_manifest)
    output = tmp_path / "partial"
    with pytest.raises(OSError, match="disk failure"):
        export_bundle(curated.service, output, actor="curator")
    assert not output.exists()
    assert curated.service.store.list("bundle") == []


def local_correctness_scenario(s):
    """Return a real-declared local correctness fixture, with no device or network I/O."""
    s.confirm()
    payload = s.plan_payload()
    policy = CorrectnessPolicy(
        kind="correctness",
        required_checks=["regression-case", "sanity"],
        reproduction_checks=["regression-case"],
        execution_kind="local",
        device_id="local-test-executor",
        workload_id="unit-regression",
        workload_config_sha256="c" * 64,
        environment_sha256="d" * 64,
    )
    payload["plan"]["validation"] = policy.model_dump(mode="json")
    payload["review"] = s.review(digest(payload["plan"]), "plan-reviewer")
    s.transition("approve_plan", "plan-reviewer", payload)
    s.implement()
    s.approve_code()
    common = {
        field: getattr(policy, field)
        for field in ("device_id", "workload_id", "workload_config_sha256", "environment_sha256")
    }
    report = CorrectnessReport(
        kind="correctness",
        candidate_id=s.candidate_id,
        implementation_revision=s.row["data"]["implementation"]["revision"],
        policy=policy,
        baseline={
            **common,
            "repo_revision": s.base,
            "artifact_sha256": "a" * 64,
            "checks": {"regression-case": {"outcome": "fail", "evidence_sha256": "e" * 64}},
        },
        candidate={
            **common,
            "repo_revision": s.row["data"]["implementation"]["revision"],
            "artifact_sha256": "b" * 64,
            "checks": {
                name: {"outcome": "pass", "evidence_sha256": "f" * 64}
                for name in policy.required_checks
            },
        },
        execution_kind="local",
        functional_passed=True,
    )
    s.row = s.service.validate(
        s.candidate_id,
        report,
        actor="validator",
        expected_version=s.row["version"],
        request_id=s.request_id("correctness"),
    )
    return s


def test_local_correctness_is_counted_separately_from_hardware_performance(scenario, tmp_path):
    s = local_correctness_scenario(scenario)
    assert s.row["data"]["validation"]["result"]["hardware_verified"] is False
    result = summary(s)
    assert result["validated_candidates"] == 1
    assert result["attempts_by_execution"]["correctness_local"]["pass"] == 1
    assert result["attempts_by_execution"]["performance_hardware"]["pass"] == 0
    s.promote()
    output = tmp_path / "correctness-bundle"
    export_bundle(s.service, output, actor="curator")
    proposal = json.loads((output / "proposals.jsonl").read_text(encoding="utf-8"))
    assert proposal["validation_kind"] == "correctness"
    assert proposal["execution_kind"] == "local"
    assert proposal["matched_pairs"] == 0
    assert proposal["check_count"] == 2


def test_sensitive_metric_name_is_represented_only_by_a_count(scenario, tmp_path):
    s = scenario
    secret_name = "PRIVATE_METRIC_SECRET_79ae"
    s.confirm()
    payload = s.plan_payload()
    payload["plan"]["validation"]["metrics"][0]["name"] = secret_name
    payload["review"] = s.review(digest(payload["plan"]), "plan-reviewer")
    s.transition("approve_plan", "plan-reviewer", payload)
    s.implement()
    s.approve_code()
    report = s.report_data()
    for arm in ("baseline", "candidate"):
        for measurement in report[arm]["measurements"]:
            measurement["metrics"] = {secret_name: measurement["metrics"]["instructions"]}
    s.validate(ABReport.model_validate(report))
    s.promote()
    output = tmp_path / "private-metric"
    export_bundle(s.service, output, actor="curator")
    assert all(secret_name not in path.read_text(encoding="utf-8") for path in output.iterdir())
    proposal = json.loads((output / "proposals.jsonl").read_text(encoding="utf-8"))
    assert proposal["metric_count"] == 1


def test_scan_applies_overlay_to_original_score_and_probation_lane(scenario):
    s = scenario
    arguments = {
        "owners": {"**/*.c": "owner"},
        "hotspots": [Hotspot(path=s.path, weight=0.9, revision=s.base)],
    }
    baseline = s.service.scan(s.repo, **arguments)["candidates"][0]["data"]["candidate"]
    first = overlay(s, factor=0.5, state="active")
    once = s.service.scan(s.repo, **arguments)["candidates"][0]["data"]["candidate"]
    assert once["score"] == pytest.approx(baseline["score"] * 0.5)
    overlay(s, factor=0.4, state="probation", expected_version=first["version"], request_id="next")
    twice = s.service.scan(s.repo, **arguments)["candidates"][0]["data"]["candidate"]
    assert twice["score"] == pytest.approx(baseline["score"] * 0.4)
    assert twice["lane"] == "workbench"


def test_export_rejects_validly_hashed_snapshot_of_a_different_approved_plan(curated, tmp_path):
    s = curated
    skill = s.skill()
    snapshot = s.service.store.read_evidence(skill["data"]["evidence"])
    snapshot["plan_digest"] = "f" * 64
    with s.service.store.transaction() as db:
        replacement = s.service.store.evidence(db, snapshot)
    update_record(s.service, "skill", skill["id"], lambda data: data.update(evidence=replacement))
    with pytest.raises(ValueError, match="snapshot"):
        export_bundle(s.service, tmp_path / "mismatched-snapshot", actor="curator")
    assert not (tmp_path / "mismatched-snapshot").exists()
