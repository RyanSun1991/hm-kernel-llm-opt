import pytest
from pydantic import ValidationError
from test_evolution_service import Scenario
from test_evolution_service import git_bin as shared_git_bin

from hmopt.evolution.correctness import (
    CorrectnessPolicy,
    CorrectnessReport,
    evaluate_correctness,
)
from hmopt.evolution.service import GateError
from hmopt.evolution.store import digest

git_bin = shared_git_bin


def policy():
    return {
        "kind": "correctness",
        "required_checks": ["reproducer", "regressions"],
        "reproduction_checks": ["reproducer"],
        "execution_kind": "local",
        "device_id": "fixture-host",
        "workload_id": "correctness-tests",
        "workload_config_sha256": "a" * 64,
        "environment_sha256": "b" * 64,
    }


def report_data(candidate="fixture", base="1" * 40, feature="2" * 40):
    common = {
        key: value
        for key, value in policy().items()
        if key in ("device_id", "workload_id", "workload_config_sha256", "environment_sha256")
    }

    def arm(revision, outcome):
        return {
            **common,
            "repo_revision": revision,
            "artifact_sha256": digest(revision),
            "checks": {
                "reproducer": {"outcome": outcome, "evidence_sha256": digest([revision, outcome])},
                "regressions": {
                    "outcome": "pass",
                    "evidence_sha256": digest([revision, "regressions"]),
                },
            },
        }

    return {
        "kind": "correctness",
        "candidate_id": candidate,
        "implementation_revision": feature,
        "policy": policy(),
        "baseline": arm(base, "fail"),
        "candidate": arm(feature, "pass"),
        "execution_kind": "local",
        "functional_passed": True,
    }


def ready(tmp_path, git_bin):
    s = Scenario(tmp_path, git_bin)
    s.confirm()
    payload = s.plan_payload()
    payload["plan"]["validation"] = policy()
    payload["plan"]["bottleneck"] = "reliability"
    payload["review"] = s.review(digest(payload["plan"]), "plan-reviewer")
    s.transition("approve_plan", "plan-reviewer", payload)
    s.implement()
    s.approve_code()
    return s


def test_correctness_does_not_need_fabricated_improvement():
    result = evaluate_correctness(CorrectnessReport.model_validate(report_data()))
    assert result.verdict == "pass"
    assert result.metrics == {}
    assert result.hardware_verified is False


@pytest.mark.parametrize(
    "arm,check,outcome,verdict",
    [
        ("baseline", "reproducer", "pass", "inconclusive"),
        ("baseline", "reproducer", "skipped", "inconclusive"),
        ("candidate", "reproducer", "fail", "fail"),
        ("candidate", "regressions", "fail", "fail"),
        ("candidate", "regressions", "skipped", "inconclusive"),
    ],
)
def test_reproducer_and_regression_requirements(arm, check, outcome, verdict):
    value = report_data()
    value[arm]["checks"][check]["outcome"] = outcome
    assert evaluate_correctness(CorrectnessReport.model_validate(value)).verdict == verdict


@pytest.mark.parametrize(
    "field", ["device_id", "workload_id", "workload_config_sha256", "environment_sha256"]
)
def test_environment_mismatch_inconclusive(field):
    value = report_data()
    value["candidate"][field] = "c" * 64
    assert evaluate_correctness(CorrectnessReport.model_validate(value)).verdict == "inconclusive"


def test_missing_check_and_unchanged_artifact_are_not_passes():
    value = report_data()
    del value["candidate"]["checks"]["regressions"]
    value["candidate"]["artifact_sha256"] = value["baseline"]["artifact_sha256"]
    result = evaluate_correctness(CorrectnessReport.model_validate(value))
    assert result.verdict == "inconclusive"
    assert len(result.reasons) == 2


def test_simulation_requires_explicit_mode():
    value = report_data()
    value["execution_kind"] = "simulation"
    report = CorrectnessReport.model_validate(value)
    assert evaluate_correctness(report).verdict == "inconclusive"
    assert evaluate_correctness(report, allow_synthetic=True).verdict == "pass"


def test_hash_case_does_not_make_identical_artifact_distinct():
    value = report_data()
    value["baseline"]["artifact_sha256"] = "ab" * 32
    value["candidate"]["artifact_sha256"] = "AB" * 32
    result = evaluate_correctness(CorrectnessReport.model_validate(value))
    assert result.verdict == "inconclusive"
    assert any("artifacts must differ" in reason for reason in result.reasons)


def test_revalidate_mutated_checks():
    report = CorrectnessReport.model_validate(report_data())
    report.candidate.checks["bad"] = {"outcome": "pass", "evidence_sha256": "no"}
    with pytest.raises(ValidationError):
        evaluate_correctness(report)


@pytest.mark.parametrize(
    "change",
    [
        {"required_checks": []},
        {"required_checks": ["reproducer", "reproducer"]},
        {"reproduction_checks": ["unapproved"]},
        {"reproduction_checks": []},
    ],
)
def test_bad_policy_rejected(change):
    with pytest.raises(ValidationError):
        CorrectnessPolicy.model_validate({**policy(), **change})


def test_correctness_uses_review_chain_and_can_be_curated(tmp_path, git_bin):
    s = ready(tmp_path, git_bin)
    report = CorrectnessReport.model_validate(
        report_data(s.candidate_id, s.base, s.row["data"]["implementation"]["revision"])
    )
    row = s.service.validate(
        s.candidate_id,
        report,
        actor="validator",
        expected_version=s.row["version"],
        request_id="correctness-check",
    )
    assert row["data"]["stage"] == "validated"
    assert row["data"]["validation"]["kind"] == "correctness"
    skill = s.service.store.list("skill")[0]
    curated = s.service.promote(
        skill["id"],
        tier="staging",
        actor="curator",
        expected_version=1,
        request_id="curate",
        note="Independent review of reproducer and regression evidence.",
    )
    assert curated["data"]["tier"] == "staging"


@pytest.mark.parametrize("simulation", [False, True])
def test_unapproved_policy_cannot_be_changed(tmp_path, git_bin, simulation):
    s = ready(tmp_path, git_bin)
    value = report_data(s.candidate_id, s.base, s.row["data"]["implementation"]["revision"])
    value["policy"]["required_checks"] = ["reproducer"]
    with pytest.raises(GateError, match="policy"):
        s.service.validate(
            s.candidate_id,
            CorrectnessReport.model_validate(value),
            actor="validator",
            expected_version=s.row["version"],
            request_id="changed",
            allow_synthetic=simulation,
        )


def test_correctness_cannot_replace_approved_performance_policy(tmp_path, git_bin):
    s = Scenario(tmp_path, git_bin).ready()
    value = report_data(s.candidate_id, s.base, s.row["data"]["implementation"]["revision"])
    with pytest.raises(GateError, match="kind"):
        s.service.validate(
            s.candidate_id,
            CorrectnessReport.model_validate(value),
            actor="validator",
            expected_version=s.row["version"],
            request_id="wrong-kind",
        )


def test_simulated_correctness_never_promotes(tmp_path, git_bin):
    s = ready(tmp_path, git_bin)
    value = report_data(s.candidate_id, s.base, s.row["data"]["implementation"]["revision"])
    value["execution_kind"] = "simulation"
    row = s.service.validate(
        s.candidate_id,
        CorrectnessReport.model_validate(value),
        actor="validator",
        expected_version=s.row["version"],
        request_id="simulation",
        allow_synthetic=True,
    )
    assert row["data"]["stage"] == "code_approved"
    assert not s.service.store.list("skill")[0]["data"]["eligible_for_promotion"]
