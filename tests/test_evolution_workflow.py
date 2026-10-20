"""Workflow adapter tests use temporary Git repositories, never agents or devices."""

import json
import shutil
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as SchemaError
from test_evolution_service import Scenario

from hmopt.evolution import workflow
from hmopt.evolution.service import GateError
from hmopt.evolution.store import ConflictError, digest
from hmopt.opencode.pipeline import resume_pipeline_session


@pytest.fixture
def git_bin():
    binary = shutil.which("git")
    bundled = (
        Path.home()
        / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe"
    )
    if binary:
        return binary
    if bundled.is_file():
        return str(bundled)
    pytest.skip("These integration tests require local Git")


@pytest.fixture
def scenario(tmp_path, git_bin):
    return Scenario(tmp_path, git_bin)


def edit_sheet(exported, **updates):
    path = Path(exported["json_path"])
    sheet = json.loads(path.read_text(encoding="utf-8"))
    sheet["items"][0].update(updates)
    path.write_text(json.dumps(sheet), encoding="utf-8")
    return path


def dispatch(scenario, tmp_path, **kwargs):
    return workflow.dispatch_candidate(
        scenario.service,
        scenario.candidate_id,
        kwargs.pop("output", tmp_path / "staged"),
        actor=kwargs.pop("actor", "operator"),
        request_id=kwargs.pop("request_id", "stage-one"),
        **kwargs,
    )


def test_review_sheet_exports_evidence_and_applies_once(scenario, tmp_path):
    exported = workflow.export_review_sheet(scenario.service, tmp_path / "review", owner="owner")
    assert exported["items"] == 1
    assert "Preconditions" in Path(exported["markdown_path"]).read_text(encoding="utf-8")
    path = edit_sheet(exported, decision="confirm", note="The owner verified scope and priority.")
    first = workflow.apply_review_sheet(scenario.service, path, actor="owner")
    second = workflow.apply_review_sheet(scenario.service, path, actor="owner")
    assert first == second
    assert first["applied"] == 1
    assert first["results"][0]["applied_stage"] == "confirmed"
    actions = [event["action"] for event in scenario.service.store.audit(scenario.candidate_id)]
    assert actions.count("confirm") == 1


def test_review_sheet_only_assigned_owner_can_apply(scenario, tmp_path):
    exported = workflow.export_review_sheet(scenario.service, tmp_path / "review")
    path = edit_sheet(exported, decision="confirm", note="The owner verified scope and priority.")
    result = workflow.apply_review_sheet(scenario.service, path, actor="impostor")
    assert result["errors"] == 1
    assert (
        scenario.service.store.read("candidate", scenario.candidate_id)["data"]["stage"]
        == "discovered"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner", "impostor"),
        ("repo_revision", "f" * 40),
        ("source_sha256", "f" * 64),
    ],
)
def test_review_sheet_rejects_edited_context(scenario, tmp_path, field, value):
    exported = workflow.export_review_sheet(scenario.service, tmp_path / "review")
    path = edit_sheet(exported, decision="confirm", note="The owner verified scope and priority.")
    sheet = json.loads(path.read_text())
    sheet["items"][0]["context"]["candidate"][field] = value
    path.write_text(json.dumps(sheet))
    result = workflow.apply_review_sheet(scenario.service, path, actor="owner")
    assert result["errors"] == 1
    assert "context changed" in result["results"][0]["error"]


def test_review_sheet_stale_version_and_different_retry_are_conflicts(scenario, tmp_path):
    exported = workflow.export_review_sheet(scenario.service, tmp_path / "review")
    path = edit_sheet(exported, decision="confirm", note="The owner verified scope and priority.")
    workflow.apply_review_sheet(scenario.service, path, actor="owner")
    path = edit_sheet(
        exported, decision="reject", note="Changing an already applied owner decision."
    )
    result = workflow.apply_review_sheet(scenario.service, path, actor="owner")
    assert result["results"][0]["status"] == "conflict"


def test_review_sheet_reports_concurrent_decision(scenario, tmp_path):
    exported = workflow.export_review_sheet(scenario.service, tmp_path / "review")
    path = edit_sheet(exported, decision="confirm", note="The owner verified scope and priority.")
    scenario.confirm()
    result = workflow.apply_review_sheet(scenario.service, path, actor="owner")
    assert result["results"][0]["status"] == "conflict"


def test_review_sheet_isolates_item_failure_and_skip(scenario, tmp_path, git_bin):
    second = Scenario(tmp_path, git_bin, service=scenario.service, name="two")
    exported = workflow.export_review_sheet(scenario.service, tmp_path / "review")
    path = Path(exported["json_path"])
    sheet = json.loads(path.read_text())
    sheet["items"][0].update(decision="confirm", note="short")
    sheet["items"][1].update(decision="reject", note="Owner deferred this candidate due to scope.")
    path.write_text(json.dumps(sheet))
    result = workflow.apply_review_sheet(scenario.service, path, actor="owner")
    assert result["errors"] == 1 and result["applied"] == 1
    assert (
        second.service.store.read("candidate", second.candidate_id)["data"]["stage"] == "rejected"
    )
    # A blank item is intentionally left undecided.
    sheet["items"][0].update(decision="", note="")
    path.write_text(json.dumps(sheet))
    assert (
        workflow.apply_review_sheet(scenario.service, path, actor="owner")["results"][0]["status"]
        == "skipped"
    )


def test_review_sheet_strict_shape_and_size(scenario, tmp_path, monkeypatch):
    exported = workflow.export_review_sheet(scenario.service, tmp_path / "review")
    path = edit_sheet(exported, force=True)
    assert workflow.apply_review_sheet(scenario.service, path, actor="owner")["errors"] == 1
    monkeypatch.setattr(workflow, "MAX_SHEET_BYTES", 100)
    with pytest.raises(GateError, match="exceeds"):
        workflow.apply_review_sheet(scenario.service, path, actor="owner")


def test_review_export_filters_before_limiting(scenario, tmp_path, git_bin):
    scenario.confirm()
    second = Scenario(tmp_path, git_bin, service=scenario.service, name="two")
    exported = workflow.export_review_sheet(scenario.service, tmp_path / "review", limit=1)
    sheet = json.loads(Path(exported["json_path"]).read_text())
    assert sheet["items"][0]["candidate_id"] == second.candidate_id
    empty = workflow.export_review_sheet(scenario.service, tmp_path / "review", owner="unassigned")
    assert empty["items"] == 0


def test_review_export_never_overwrites_edited_decisions(scenario, tmp_path):
    exported = workflow.export_review_sheet(scenario.service, tmp_path / "review")
    path = edit_sheet(exported, decision="confirm", note="Persist this unsent owner decision.")
    content = path.read_bytes()
    with pytest.raises(ConflictError, match="refusing overwrite"):
        workflow.export_review_sheet(scenario.service, tmp_path / "review")
    assert path.read_bytes() == content


def test_dispatch_requires_owner_gate(scenario, tmp_path):
    with pytest.raises(GateError, match="handoff"):
        dispatch(scenario, tmp_path)
    assert not scenario.service.store.list("dispatch")


def test_dispatch_is_compatible_with_explicit_pipeline_resume(scenario, tmp_path):
    scenario.confirm()
    sentinel = scenario.repo / ".opencode/state/current_task.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("do not replace")
    staged = dispatch(scenario, tmp_path)
    resumed = resume_pipeline_session(repo_root=scenario.repo, state_path=staged["state_path"])
    assert resumed["task"]["status"] == "staged"
    assert resumed["task"]["active_agent"] == "researcher"
    assert resumed["task"]["evolution"]["source_changes_allowed"] is False
    assert resumed["task"]["evolution"]["automatic_execution"] is False
    assert "Verify the proposed mechanism" in resumed["prompt_text"]
    assert sentinel.read_text() == "do not replace"
    assert dispatch(scenario, tmp_path) == staged


def test_dispatch_preserves_approved_policy_and_scope(scenario, tmp_path):
    scenario.confirm()
    scenario.approve_plan()
    staged = dispatch(scenario, tmp_path)
    task = json.loads(Path(staged["state_path"]).read_text())
    assert task["active_agent"] == "implementer"
    assert task["evolution"]["source_changes_allowed"] is True
    assert task["evolution"]["allowed_paths"] == [scenario.path]
    assert task["evolution"]["validation_policy"] == scenario.row["data"]["plan"]["validation"]
    assert task["approved_plan"] == scenario.row["data"]["plan_digest"]


def test_dispatch_correctness_policy_has_no_fabricated_metric(scenario, tmp_path):
    scenario.confirm()
    payload = scenario.plan_payload()
    policy = {
        "kind": "correctness",
        "required_checks": ["reproducer", "regression"],
        "reproduction_checks": ["reproducer"],
        "execution_kind": "local",
        "device_id": "local-fixture",
        "workload_id": "reproduction-suite",
        "workload_config_sha256": "c" * 64,
        "environment_sha256": "d" * 64,
    }
    payload["plan"]["validation"] = policy
    payload["review"] = scenario.review(digest(payload["plan"]), "plan-reviewer")
    scenario.transition("approve_plan", "plan-reviewer", payload)
    scenario.implement()
    scenario.approve_code()
    staged = dispatch(scenario, tmp_path)
    task = json.loads(Path(staged["state_path"]).read_text())
    assert task["evolution"]["validation_policy"] == policy
    assert task["primary_goal"] == "correctness checks: reproducer, regression"
    assert "correctness-report.json" in [artifact["kind"] for artifact in task["artifacts"]]
    assert "ab-report.json" not in [artifact["kind"] for artifact in task["artifacts"]]
    assert "do not invent a performance gain" in Path(staged["prompt_path"]).read_text()


def test_review_invalid_decision_is_isolated_from_other_rows(scenario, tmp_path, git_bin):
    second = Scenario(tmp_path, git_bin, service=scenario.service, name="two")
    exported = workflow.export_review_sheet(scenario.service, tmp_path / "review")
    path = Path(exported["json_path"])
    sheet = json.loads(path.read_text())
    sheet["items"][0].update(decision="force-implement", note="Invalid bypass must be rejected.")
    sheet["items"][1].update(decision="confirm", note="Owner confirmed evidence and priority.")
    path.write_text(json.dumps(sheet))
    result = workflow.apply_review_sheet(scenario.service, path, actor="owner")
    assert result["errors"] == 1 and result["applied"] == 1
    assert (
        second.service.store.read("candidate", second.candidate_id)["data"]["stage"] == "confirmed"
    )


@pytest.mark.parametrize("role", ["reviewer", "validator"])
def test_dispatch_stage_specific_review_and_validation(scenario, tmp_path, role):
    scenario.confirm()
    scenario.approve_plan()
    scenario.implement()
    if role == "validator":
        scenario.approve_code()
    staged = dispatch(scenario, tmp_path)
    handoff = json.loads(Path(staged["handoff_path"]).read_text())
    assert staged["role"] == role
    assert handoff["source_changes_allowed"] is False
    assert handoff["implementation_digest"] == scenario.row["data"]["implementation_digest"]


def test_dispatch_reservation_survives_crash_and_repairs_missing_files(
    scenario, tmp_path, monkeypatch
):
    scenario.confirm()
    original = workflow._exclusive_artifact
    calls = 0

    def interrupted(path, content):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        return original(path, content)

    monkeypatch.setattr(workflow, "_exclusive_artifact", interrupted)
    with pytest.raises(OSError, match="injected"):
        dispatch(scenario, tmp_path)
    record = scenario.service.store.list("dispatch")[0]
    assert record["data"]["status"] == "reserved"
    assert (Path(record["data"]["directory"]) / "handoff.json").is_file()
    assert not (Path(record["data"]["directory"]) / "task.json").exists()
    monkeypatch.setattr(workflow, "_exclusive_artifact", original)
    staged = dispatch(scenario, tmp_path)
    assert staged["status"] == "staged"
    Path(staged["prompt_path"]).unlink()
    assert dispatch(scenario, tmp_path) == staged
    assert Path(staged["prompt_path"]).is_file()
    actions = [event["action"] for event in scenario.service.store.audit(scenario.candidate_id)]
    assert actions.count("dispatch_reserved") == 1
    assert actions.count("dispatch_staged") == 1


def test_dispatch_refuses_tampered_existing_files(scenario, tmp_path):
    scenario.confirm()
    staged = dispatch(scenario, tmp_path)
    path = Path(staged["state_path"])
    path.write_text("operator notes")
    with pytest.raises(ConflictError, match="refusing overwrite"):
        dispatch(scenario, tmp_path)
    assert path.read_text() == "operator notes"


def test_dispatch_refuses_stale_replay_and_supports_next_stage(scenario, tmp_path):
    scenario.confirm()
    first = dispatch(scenario, tmp_path)
    scenario.approve_plan()
    with pytest.raises(ConflictError, match="stale"):
        dispatch(scenario, tmp_path)
    second = dispatch(scenario, tmp_path, request_id="stage-two")
    assert first["directory"] != second["directory"]
    assert second["role"] == "implementer"


def test_dispatch_prevents_duplicate_claims(scenario, tmp_path):
    scenario.confirm()
    dispatch(scenario, tmp_path)
    with pytest.raises(ConflictError, match="different arguments"):
        dispatch(scenario, tmp_path, output=tmp_path / "different")
    with pytest.raises(ConflictError, match="already reserved"):
        dispatch(scenario, tmp_path, actor="another", request_id="another-request")
    with pytest.raises(ConflictError, match="already reserved"):
        dispatch(scenario, tmp_path, output=tmp_path / "different", request_id="another-request")


def test_dispatch_checks_snapshot_after_handoff(scenario, tmp_path, monkeypatch):
    scenario.confirm()
    original = scenario.service.handoff

    def changed(candidate_id):
        packet = original(candidate_id)
        scenario.approve_plan()
        return packet

    monkeypatch.setattr(scenario.service, "handoff", changed)
    with pytest.raises(ConflictError, match="changed while preparing"):
        dispatch(scenario, tmp_path)
    assert not scenario.service.store.list("dispatch")


def test_dispatch_checks_snapshot_after_file_publication(scenario, tmp_path, monkeypatch):
    scenario.confirm()
    original = workflow._exclusive_artifact
    first = True

    def changed(path, content):
        nonlocal first
        original(path, content)
        if first:
            first = False
            scenario.approve_plan()

    monkeypatch.setattr(workflow, "_exclusive_artifact", changed)
    with pytest.raises(ConflictError, match="stale"):
        dispatch(scenario, tmp_path)
    assert scenario.service.store.list("dispatch")[0]["data"]["status"] == "reserved"
    assert "dispatch_staged" not in [
        e["action"] for e in scenario.service.store.audit(scenario.candidate_id)
    ]


def test_atomic_publication_leaves_no_partial_file(tmp_path, monkeypatch):
    path = tmp_path / "artifact.json"

    def failed_link(source, target):
        raise OSError("filesystem does not support hard links")

    monkeypatch.setattr(workflow.os, "link", failed_link)
    with pytest.raises(OSError, match="hard links"):
        workflow._exclusive_artifact(path, "complete contents")
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_workspace_routes_plan_and_review_to_independent_roles(scenario, tmp_path):
    scenario.confirm()
    root = scenario.repo / ".opencode/local/workspaces"
    staged = dispatch(scenario, tmp_path, workspace_root=root)
    task = json.loads(Path(staged["state_path"]).read_text())
    assert task["manager"] == "coordinator"
    assert [(step["id"], step["role"]) for step in task["stage_steps"]] == [
        ("research", "researcher"),
        ("plan", "architect"),
        ("plan-review", "reviewer"),
    ]
    assert task["stage_steps"][0]["submit_action"] is None
    assert task["stage_steps"][2]["submit_action"] == "approve_plan"
    assert task["stage_steps"][2]["independent_from"] == "plan"
    workspace = Path(staged["workspace_path"])
    assert workspace.parent == root
    for artifact in task["artifacts"]:
        path = Path(artifact["path"])
        assert path.parent == workspace / "artifacts" / artifact["step"]
        assert path.parent.is_dir()
        assert not path.exists()  # staging cannot create a role's claimed result
    assert "independent-plan-review.json" not in task["stage_steps"][1]["outputs"]
    assert all(not step["source_changes_allowed"] for step in task["stage_steps"])
    assert Path(staged["capsule_path"]).is_file()
    assert json.loads(Path(staged["execution_state_path"]).read_text())["attempts"] == []


def test_workspace_replay_preserves_capsule_progress_and_repairs_missing_seeds(scenario, tmp_path):
    scenario.confirm()
    root = scenario.repo / ".opencode/local/workspaces"
    first = dispatch(scenario, tmp_path, workspace_root=root)
    capsule = Path(first["capsule_path"])
    capsule.write_text("Role-updated evidence and open questions.")
    progress = Path(first["execution_state_path"])
    progress.write_text('{"next_step":"plan-review","attempts":["completed plan"]}')
    decisions = Path(first["workspace_path"]) / "decisions.md"
    decisions.unlink()
    assert dispatch(scenario, tmp_path, workspace_root=root) == first
    assert capsule.read_text() == "Role-updated evidence and open questions."
    assert json.loads(progress.read_text())["next_step"] == "plan-review"
    assert decisions.is_file()


def test_stage_only_export_can_be_materialized_without_replacing_source(scenario, tmp_path):
    scenario.confirm()
    staged = dispatch(scenario, tmp_path)
    assert staged["workspace_status"] == "not_configured"
    original = Path(staged["state_path"]).read_bytes()
    root = scenario.repo / ".opencode/local/workspaces"
    result = workflow.materialize_dispatch_workspace(scenario.service, staged["dispatch_id"], root)
    assert result["workspace_status"] == "materialized"
    assert Path(result["state_path"]).parent == Path(result["workspace_path"]) / "dispatch"
    assert Path(staged["state_path"]).read_bytes() == original
    assert (
        workflow.materialize_dispatch_workspace(scenario.service, staged["dispatch_id"], root)
        == result
    )
    with pytest.raises(ConflictError, match="already bound"):
        workflow.materialize_dispatch_workspace(
            scenario.service, staged["dispatch_id"], tmp_path / "other/.opencode/local/workspaces"
        )
    scenario.approve_plan()
    with pytest.raises(ConflictError, match="stale"):
        workflow.materialize_dispatch_workspace(scenario.service, staged["dispatch_id"], root)


def test_materialize_existing_configured_dispatch_keeps_identical_binding(scenario, tmp_path):
    scenario.confirm()
    root = scenario.repo / ".opencode/local/workspaces"
    staged = dispatch(scenario, tmp_path, workspace_root=root)
    result = workflow.materialize_dispatch_workspace(scenario.service, staged["dispatch_id"], root)
    assert result["state_path"] == staged["state_path"]
    assert result["capsule_path"] == staged["capsule_path"]
    with pytest.raises(ConflictError, match="already bound"):
        workflow.materialize_dispatch_workspace(
            scenario.service, staged["dispatch_id"], tmp_path / "other/.opencode/local/workspaces"
        )


def test_workspace_rejects_wrong_root_and_tampered_binding(scenario, tmp_path):
    scenario.confirm()
    with pytest.raises(GateError, match="workspace_root"):
        dispatch(scenario, tmp_path, workspace_root=tmp_path / "src")
    assert not scenario.service.store.list("dispatch")
    root = scenario.repo / ".opencode/local/workspaces"
    staged = dispatch(scenario, tmp_path, workspace_root=root)
    binding = Path(staged["workspace_path"]) / "evolution-binding.json"
    binding.write_text("a different candidate")
    with pytest.raises(ConflictError, match="refusing overwrite"):
        dispatch(scenario, tmp_path, workspace_root=root)


def test_damaged_workspace_projection_is_rejected_before_seeding_files(scenario, tmp_path):
    scenario.confirm()
    staged = dispatch(scenario, tmp_path)
    root = scenario.repo / ".opencode/local/workspaces"
    result = workflow.materialize_dispatch_workspace(scenario.service, staged["dispatch_id"], root)
    capsule = Path(result["capsule_path"])
    capsule.unlink()
    with scenario.service.store.transaction() as db:
        record = scenario.service.store.get(
            db, "dispatch_workspace", "workspace-" + staged["dispatch_id"]
        )
        data = record["data"]
        task = json.loads(data["artifacts"]["task.json"]["content"])
        task["title"] = "Damaged projection must not repair missing files"
        data["artifacts"]["task.json"]["content"] = json.dumps(task)
        scenario.service.store.put(db, "dispatch_workspace", record["id"], data, record["version"])
    with pytest.raises(ConflictError, match="integrity"):
        workflow.materialize_dispatch_workspace(scenario.service, staged["dispatch_id"], root)
    assert not capsule.exists()


def test_dispatch_contracts_drive_real_service_gates_and_preserve_distinct_repo_roots(
    scenario, tmp_path
):
    """Temporary Git/evidence fixtures exercise contracts, never a model or a device."""
    from hmopt.evolution.validation import ABReport

    s = scenario
    s.confirm()
    workbench = tmp_path / "workbench"
    workspace_root = workbench / ".opencode/local/workspaces"
    emitted = []

    def stage(name):
        staged = dispatch(s, tmp_path, workspace_root=workspace_root, request_id=name)
        task = json.loads(Path(staged["state_path"]).read_text(encoding="utf-8"))
        assert task["target_repo_path"] == str(s.repo)
        assert task["workbench_root"] == str(workbench)
        assert task["target_repo_path"] != task["workbench_root"]
        assert Path(staged["workspace_path"]).parent == workspace_root
        contract = task["submission_contract"]
        Draft202012Validator.check_schema(contract["json_schema"])
        emitted.append(staged)
        return contract, Draft202012Validator(contract["json_schema"])

    contract, schema = stage("plan")
    assert contract["tool"] == "evolution_submit"
    assert contract["argument"] == "payload"
    assert contract["action"] == "approve_plan"
    payload = s.plan_payload()
    schema.validate(payload)
    with pytest.raises(SchemaError):
        schema.validate({**payload, "status": "approved"})
    s.transition(contract["action"], "plan-reviewer", payload)

    contract, schema = stage("implementation")
    with pytest.raises(SchemaError):
        schema.validate({"revision": "HEAD"})
    (s.repo / s.path).write_text(
        "int target(int x) { return cached_lookup(x); }\n", encoding="utf-8"
    )
    payload = {"revision": s.commit("recorded immutable fixture implementation")}
    schema.validate(payload)
    s.transition(contract["action"], "implementer", payload)

    contract, schema = stage("code-review")
    payload = s.review(s.row["data"]["implementation_digest"], "code-reviewer")
    schema.validate(payload)
    with pytest.raises(SchemaError):
        schema.validate({"review": payload})
    s.transition(contract["action"], "code-reviewer", payload)

    contract, schema = stage("validation")
    assert contract["tool"] == "evolution_validate"
    assert contract["argument"] == "report"
    # Hardware flags here exercise the existing trusted collector contract; the
    # data are test fixtures and do not establish real device measurements.
    payload = s.report_data()
    schema.validate(payload)
    validated = s.validate(ABReport.model_validate(payload))
    assert validated["data"]["stage"] == "validated"
    assert len({item["state_path"] for item in emitted}) == 4
    journal = s.skill()["data"]
    assert journal["signal"] == "validation_result"
    assert journal["publication_status"] == "not_published"
    assert (
        s.service.store.read_evidence(journal["evidence"])["validation"]["report_evidence"]
        == (validated["data"]["validation"]["report_evidence"])
    )
    assert len(s.service.store.list("skill")) == 1  # no duplicate unverified capture is needed


def test_dispatch_correctness_submission_schema_is_the_approved_report_contract(scenario, tmp_path):
    from hmopt.evolution.correctness import CorrectnessReport

    scenario.confirm()
    payload = scenario.plan_payload()
    payload["plan"]["validation"] = {
        "kind": "correctness",
        "required_checks": ["reproducer"],
        "reproduction_checks": ["reproducer"],
        "execution_kind": "local",
        "device_id": "fixture-local",
        "workload_id": "fixture-check",
        "workload_config_sha256": "c" * 64,
        "environment_sha256": "d" * 64,
    }
    payload["review"] = scenario.review(digest(payload["plan"]), "plan-reviewer")
    scenario.transition("approve_plan", "plan-reviewer", payload)
    scenario.implement()
    scenario.approve_code()
    staged = dispatch(scenario, tmp_path)
    task = json.loads(Path(staged["state_path"]).read_text(encoding="utf-8"))
    schema = task["submission_contract"]["json_schema"]
    assert schema == CorrectnessReport.model_json_schema()
    assert "metrics" not in schema["properties"]


def test_recorded_implementation_cannot_be_silently_reworked_by_recipe(scenario):
    scenario.confirm()
    scenario.approve_plan()
    scenario.implement()
    original = scenario.row
    with pytest.raises(GateError, match="approved plan"):
        scenario.transition(
            "record_implementation",
            "implementer",
            {"revision": original["data"]["implementation"]["revision"]},
        )
    assert scenario.service.store.read("candidate", scenario.candidate_id) == original
