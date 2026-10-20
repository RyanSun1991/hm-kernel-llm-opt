"""Real adapter subprocesses and workflow gates; fixture reports do not claim hardware gains."""

import json
import sys

import pytest
from test_evolution_correctness import ready, report_data
from test_evolution_mining import GIT

from hmopt.evolution.correctness import CorrectnessReport
from hmopt.evolution.experiments import prepare_experiment, retire_experiment, run_experiment
from hmopt.evolution.production import ValidationRunnerConfig
from hmopt.evolution.service import GateError
from hmopt.evolution.store import ConflictError


def adapter(tmp_path, report, *, exit_code=0):
    source = tmp_path / "adapter.py"
    source.write_text(
        "import json, os\nfrom pathlib import Path\n"
        "request = json.loads(Path(os.environ['HMOPT_EXPERIMENT_REQUEST']).read_text(encoding='utf-8'))\n"
        "assert request['candidate']['data']['stage'] == 'code_approved'\n"
        f"report = json.loads({json.dumps(json.dumps(report))})\n"
        "Path(os.environ['HMOPT_EXPERIMENT_RESULT']).write_text(json.dumps({'report': report}), encoding='utf-8')\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
        newline="\n",
    )
    return ValidationRunnerConfig(
        command=[sys.executable, str(source)],
        cwd=str(tmp_path),
        resource_id="fixture-device",
        timeout_seconds=30,
    )


def test_adapter_subprocess_validates_and_replays_without_reexecution(tmp_path, monkeypatch):
    s = ready(tmp_path, GIT)
    report = report_data(s.candidate_id, s.base, s.row["data"]["implementation"]["revision"])
    config = adapter(tmp_path, report)
    inherited = {
        "HMOPT_BUILD_MCP_PROJECT_PATH": str(tmp_path / "existing-business"),
        "HMOPT_FLASH_WINDOWS_IMAGE_DIR": "existing-images",
        "HMOPT_AUTO_TEST_TARGET": "fixture-device-do-not-connect",
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)
    source = tmp_path / "adapter.py"
    source.write_text(
        "import os\n"
        + f"assert {{name: os.environ.get(name) for name in {list(inherited)!r}}} == {inherited!r}\n"
        + source.read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )
    job = prepare_experiment(
        s.service, s.candidate_id, config, actor="validator", request_id="prepare"
    )
    assert (
        prepare_experiment(
            s.service, s.candidate_id, config, actor="validator", request_id="prepare"
        )
        == job
    )
    result = run_experiment(s.service, job["id"], config)
    assert result["data"]["status"] == "complete"
    assert result["data"]["validation"]["data"]["stage"] == "validated"
    assert run_experiment(s.service, job["id"], config) == result
    assert not s.service.store.list("experiment_claim")
    assert not s.service.store.list("experiment_candidate_claim")
    assert s.service.store.read_evidence(result["data"]["output_sha256"])["exit_code"] == 0


def test_adapter_failure_keeps_resource_until_explicit_stopped_acknowledgement(tmp_path):
    s = ready(tmp_path, GIT)
    config = adapter(tmp_path, {}, exit_code=2)
    job = prepare_experiment(
        s.service, s.candidate_id, config, actor="validator", request_id="prepare"
    )
    with pytest.raises(ValueError, match="exited with code 2"):
        run_experiment(s.service, job["id"], config)
    assert s.service.store.read("candidate", s.candidate_id)["data"]["stage"] == "code_approved"
    row = s.service.store.read("experiment", job["id"])
    assert row["data"]["status"] == "attention"
    assert s.service.store.list("experiment_claim")
    alternate = config.model_copy(update={"resource_id": "another-device"})
    second = prepare_experiment(
        s.service, s.candidate_id, alternate, actor="validator", request_id="second-device"
    )
    with pytest.raises(ConflictError, match="Candidate already"):
        run_experiment(s.service, second["id"], alternate)
    with pytest.raises(ValueError, match="Confirm"):
        retire_experiment(
            s.service,
            row["id"],
            actor="operator",
            expected_version=row["version"],
            execution_stopped=False,
        )
    retire_experiment(
        s.service,
        row["id"],
        actor="operator",
        expected_version=row["version"],
        execution_stopped=True,
    )
    assert not s.service.store.list("experiment_claim")
    assert not s.service.store.list("experiment_candidate_claim")


def test_changed_candidate_cannot_run_prepared_adapter(tmp_path):
    s = ready(tmp_path, GIT)
    config = adapter(tmp_path, {})
    job = prepare_experiment(
        s.service, s.candidate_id, config, actor="validator", request_id="prepare"
    )
    s.transition("reject", "owner", {"note": "Owner withdraws this candidate before validation."})
    with pytest.raises(ConflictError, match="changed"):
        run_experiment(s.service, job["id"], config)
    assert not s.service.store.list("experiment_claim")


def test_exit_success_does_not_override_failing_report(tmp_path):
    s = ready(tmp_path, GIT)
    report = report_data(s.candidate_id, s.base, s.row["data"]["implementation"]["revision"])
    report["candidate"]["checks"]["regressions"]["outcome"] = "fail"
    config = adapter(tmp_path, report)
    job = prepare_experiment(
        s.service, s.candidate_id, config, actor="validator", request_id="prepare"
    )
    result = run_experiment(s.service, job["id"], config)
    assert result["data"]["validation"]["data"]["validation"]["result"]["verdict"] == "fail"
    assert s.service.store.read("candidate", s.candidate_id)["data"]["stage"] == "code_approved"


def test_workspace_report_must_bind_all_dependencies(tmp_path):
    s = ready(tmp_path, GIT)
    report = report_data(s.candidate_id, s.base, s.row["data"]["implementation"]["revision"])
    with s.service.store.transaction() as db:
        sha = s.service.store.evidence(
            db,
            {
                "projects": [{"repo_id": "target", "revision": s.base}],
                "dependencies": [{"repo_id": "dependency", "revision": "d" * 40}],
            },
        )
        row = s.service.store.get(db, "candidate", s.candidate_id)
        row["data"]["candidate"].update(repo_id="target", workspace_manifest_sha256=sha)
        s.row = s.service.store.put(db, "candidate", s.candidate_id, row["data"], row["version"])
    with pytest.raises(GateError, match="project revisions"):
        s.service.validate(
            s.candidate_id,
            CorrectnessReport.model_validate(report),
            actor="validator",
            expected_version=s.row["version"],
            request_id="missing",
        )
    report["baseline"]["project_revisions"] = {"target": s.base, "dependency": "d" * 40}
    report["candidate"]["project_revisions"] = {
        "target": report["implementation_revision"],
        "dependency": "e" * 40,
    }
    with pytest.raises(GateError, match="project revisions"):
        s.service.validate(
            s.candidate_id,
            CorrectnessReport.model_validate(report),
            actor="validator",
            expected_version=s.row["version"],
            request_id="drift",
        )
    report["candidate"]["project_revisions"]["dependency"] = "d" * 40
    result = s.service.validate(
        s.candidate_id,
        CorrectnessReport.model_validate(report),
        actor="validator",
        expected_version=s.row["version"],
        request_id="bound",
    )
    assert result["data"]["stage"] == "validated"


def test_multi_git_scan_plan_reviews_and_adapter_share_one_manifest(tmp_path):
    from test_evolution_correctness import policy
    from test_evolution_service import Scenario

    from hmopt.evolution.store import digest
    from hmopt.evolution.workspace import identities

    s = Scenario(tmp_path, GIT)
    dependency = Scenario(tmp_path, GIT, name="dependency")
    config = {
        "workspace_id": "business",
        "root": str(tmp_path),
        "projects": {
            "target": {"path": s.repo.name, "owners": {"**": "owner"}},
            "build": {"path": dependency.repo.name, "owners": {"**": "build-owner"}},
        },
        "selected": ["target"],
        "dependencies": ["build"],
    }
    s.service.source_workspace = config
    s.service.repo_identities = identities(config)
    s.row = s.service.scan(s.repo, owners={"**": "owner"})["candidates"][0]
    s.candidate_id = s.row["id"]
    manifest_sha = s.row["data"]["candidate"]["workspace_manifest_sha256"]
    s.confirm()
    payload = s.plan_payload()
    payload["plan"].update(validation=policy(), bottleneck="reliability")
    payload["review"] = s.review(digest(payload["plan"]), "plan-reviewer")
    with pytest.raises(GateError, match="Plan must bind"):
        s.transition("approve_plan", "plan-reviewer", payload)
    payload["plan"]["workspace_manifest_sha256"] = manifest_sha
    payload["review"] = s.review(digest(payload["plan"]), "plan-reviewer")
    s.transition("approve_plan", "plan-reviewer", payload)
    s.implement()
    s.approve_code()
    report = report_data(s.candidate_id, s.base, s.row["data"]["implementation"]["revision"])
    manifest = s.service.store.read_evidence(manifest_sha)
    baseline = {
        p["repo_id"]: p["revision"] for p in manifest["projects"] + manifest["dependencies"]
    }
    report["baseline"]["project_revisions"] = baseline
    report["candidate"]["project_revisions"] = {
        **baseline,
        s.row["data"]["candidate"]["repo_id"]: report["implementation_revision"],
    }
    adapter_config = adapter(tmp_path, report)
    experiment = prepare_experiment(
        s.service, s.candidate_id, adapter_config, actor="validator", request_id="multi-git"
    )
    frozen = s.service.store.read_evidence(experiment["data"]["request_sha256"])
    assert frozen["workspace_manifest"] == manifest
    assert (
        run_experiment(s.service, experiment["id"], adapter_config)["data"]["validation"]["data"][
            "stage"
        ]
        == "validated"
    )
    assert dependency.git("rev-parse", "HEAD").strip() == dependency.base


def test_retired_adapter_completion_cannot_validate_candidate(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    s = ready(tmp_path, GIT)
    report = report_data(s.candidate_id, s.base, s.row["data"]["implementation"]["revision"])
    config = adapter(tmp_path, report)
    job = prepare_experiment(
        s.service, s.candidate_id, config, actor="validator", request_id="retire-during-run"
    )

    def delayed_result(*args, **kwargs):
        current = s.service.store.read("experiment", job["id"])
        retire_experiment(
            s.service,
            job["id"],
            actor="operator",
            expected_version=current["version"],
            execution_stopped=True,
        )
        Path(kwargs["env"]["HMOPT_EXPERIMENT_RESULT"]).write_text(
            json.dumps({"report": report}), encoding="utf-8", newline="\n"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("hmopt.evolution.experiments.subprocess.run", delayed_result)
    with pytest.raises(ConflictError, match="retired or superseded"):
        run_experiment(s.service, job["id"], config)
    assert s.service.store.read("candidate", s.candidate_id)["data"]["stage"] == "code_approved"
    assert s.service.store.read("experiment", job["id"])["data"]["status"] == "retired"
