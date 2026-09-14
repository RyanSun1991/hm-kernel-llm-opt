"""Operational recovery against real Git trees and durable workflow checkpoints."""

import smtplib
from pathlib import Path

import pytest
from test_evolution_production import approval_config as shared_approval_config
from test_evolution_service import Scenario
from test_evolution_workspace import GIT, complete_pending
from test_evolution_workspace import workspace as shared_workspace

from hmopt.evolution.approval import _deliver, deliver_notifications, request_approval
from hmopt.evolution.production import ApprovalConfig
from hmopt.evolution.runs import advance_run, control_run, start_run
from hmopt.evolution.scan import scan_next, start_scan
from hmopt.evolution.store import ConflictError, digest

workspace = shared_workspace
approval_config = shared_approval_config


def start(scenario):
    return start_scan(
        scenario.service,
        scenario.repo,
        revision="HEAD",
        owners={"**": "owner"},
        request_id="recovery-scan",
        actor="operator",
    )


def test_failed_scan_retries_exact_frozen_checkpoint_and_rejects_stale_control(
    tmp_path, monkeypatch
):
    scenario = Scenario(tmp_path, GIT)
    service = scenario.service
    row = start(scenario)
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise OSError("transient fixture failure")

        patch.setattr(service, "scan", fail)
        with pytest.raises(OSError):
            scan_next(service, row["id"], expected_version=row["version"])
    failed = service.store.read("scan", row["id"])
    assert failed["data"]["stack"] == row["data"]["stack"]
    from hmopt.evolution.scan import control_scan

    with pytest.raises(ConflictError):
        control_scan(
            service, row["id"], action="retry", actor="operator", expected_version=row["version"]
        )
    retried = control_scan(
        service, row["id"], action="retry", actor="operator", expected_version=failed["version"]
    )
    assert retried["data"]["snapshot_sha256"] == row["data"]["snapshot_sha256"]
    assert "error" not in retried["data"]
    final = scan_next(service, row["id"], expected_version=retried["version"])
    assert final["data"]["status"] == "complete"
    assert final["data"]["candidate_count"] == 1


def test_scan_cannot_select_unselected_repository(workspace):
    service, config = workspace
    service.source_workspace = config
    with pytest.raises(ValueError, match="selected"):
        start_scan(
            service,
            Path(config["root"]) / "unselected",
            revision="HEAD",
            owners={"**": "owner"},
            request_id="outside",
            actor="operator",
        )
    assert not service.store.list("scan")


def retry_run(service, row, project="kernel"):
    current = service.store.read("workspace_run", row["id"])
    return control_run(
        service,
        row["id"],
        action="retry",
        actor="operator",
        expected_version=current["version"],
        project_id=project,
    )


def test_run_retry_forgets_previous_failure_stage(workspace, monkeypatch):
    service, config = workspace
    row = start_run(service, config, actor="operator", request_id="run")
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise OSError("transient history failure")

        patch.setattr("hmopt.evolution.runs.mine_git_history", fail)
        advance_run(service, row["id"])
    retry_run(service, row)
    advance_run(service, row["id"])
    advance_run(service, row["id"])
    retried = retry_run(service, row)
    state = retried["data"]["projects"]["kernel"]
    assert state["stage"] == "analysis"
    assert "resume_stage" not in state and "error" not in state


def attached_scan(service, config):
    run = start_run(service, config, actor="operator", request_id="run")
    advance_run(service, run["id"])
    complete_pending(service, config)
    advance_run(service, run["id"])
    scan = start_scan(
        service,
        Path(config["root"]) / "kernel",
        revision="HEAD",
        owners={"**": "kernel-owner"},
        request_id="attached",
        actor="operator",
    )
    # Simulate an in-progress independently scheduled page of the attached scan.
    with service.store.transaction() as db:
        current = service.store.get(db, "workspace_run", run["id"])
        current["data"]["projects"]["kernel"]["scan_id"] = scan["id"]
        service.store.put(db, "workspace_run", run["id"], current["data"], current["version"])
        scan["data"]["status"] = "running"
        scan = service.store.put(db, "scan", scan["id"], scan["data"], scan["version"])
    return run, scan


def test_run_retries_attached_scan_failed_by_another_worker(workspace):
    service, config = workspace
    run, scan = attached_scan(service, config)
    with service.store.transaction() as db:
        scan["data"].update(status="attention", error="independent scan failure")
        service.store.put(db, "scan", scan["id"], scan["data"], scan["version"])
    advance_run(service, run["id"])
    retried = retry_run(service, run)
    assert retried["data"]["projects"]["kernel"]["stage"] == "scan"
    assert service.store.read("scan", scan["id"])["data"]["status"] == "running"


def test_cancel_run_cancels_attached_scan(workspace):
    service, config = workspace
    run, scan = attached_scan(service, config)
    current = service.store.read("workspace_run", run["id"])
    control_run(
        service, run["id"], action="cancel", actor="operator", expected_version=current["version"]
    )
    canceled = service.store.read("scan", scan["id"])
    assert canceled["data"]["status"] == "cancelled"
    assert scan_next(service, scan["id"], expected_version=canceled["version"]) == canceled
    assert not service.store.list("workspace_claim")


def test_first_scan_page_failure_preserves_run_attachment(workspace, tmp_path, monkeypatch):
    service, config = workspace
    Scenario(tmp_path, GIT, service=service)  # Supply an active pattern to reach the first page.
    run = start_run(service, config, actor="operator", request_id="run")
    advance_run(service, run["id"])
    complete_pending(service, config)
    advance_run(service, run["id"])

    def fail(*args, **kwargs):
        raise OSError("interrupted immediately after creating scan")

    monkeypatch.setattr("hmopt.evolution.runs.scan_next", fail)
    result = advance_run(service, run["id"])["run"]
    ids = [state["scan_id"] for state in result["data"]["projects"].values()]
    assert all(ids) and set(ids) == {row["id"] for row in service.store.list("scan")}
    control_run(
        service, run["id"], action="cancel", actor="operator", expected_version=result["version"]
    )
    assert all(row["data"]["status"] == "cancelled" for row in service.store.list("scan"))


def test_smtp_starttls_negotiates_before_login(approval_config, monkeypatch):
    config = ApprovalConfig.model_validate(
        {
            **approval_config.model_dump(),
            "webhook_url": None,
            "smtp_host": "mail.example.invalid",
            "smtp_sender": "bot@example.invalid",
            "smtp_port": 587,
            "smtp_security": "starttls",
            "smtp_username": "bot",
            "smtp_password_env": "TEST_SMTP_PASSWORD",
        }
    )
    monkeypatch.setenv("TEST_SMTP_PASSWORD", "fixture-password")
    calls = []

    class SMTP:
        def __init__(self, host, port, **kwargs):
            assert port == 587

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def ehlo(self):
            calls.append("ehlo")

        def starttls(self, **kwargs):
            assert kwargs["context"].check_hostname
            calls.append("starttls")

        def login(self, *args):
            calls.append("login")

        def send_message(self, message):
            calls.append("send")

    monkeypatch.setattr(smtplib, "SMTP", SMTP)
    _deliver(
        config,
        "request-1",
        {
            "target": "expert@example.invalid",
            "path": "code.c",
            "owner": "owner",
            "candidate_id": "candidate-1",
            "revision": "fixture",
            "url": "https://example.invalid/review",
            "context_sha256": "a" * 64,
        },
    )
    assert calls == ["ehlo", "starttls", "ehlo", "login", "send"]


@pytest.mark.parametrize("change", ["candidate", "pattern"])
def test_notification_skips_obsolete_candidate_or_pattern(tmp_path, approval_config, change):
    scenario = Scenario(tmp_path, GIT)
    request_approval(
        scenario.service,
        scenario.candidate_id,
        approval_config,
        actor="coordinator",
        request_id="request",
    )
    if change == "candidate":
        scenario.confirm()
    else:
        with scenario.service.store.transaction() as db:
            row = scenario.service.store.get(db, "pattern", scenario.pattern_key)
            row["data"]["status"] = "retired"
            scenario.service.store.put(db, "pattern", row["id"], row["data"], row["version"])
    sent = []
    result = deliver_notifications(
        scenario.service,
        approval_config,
        actor="operator",
        transport=lambda *args: sent.append(args),
    )
    assert not sent
    assert result["notifications"][0]["data"]["status"] == "obsolete"


def test_notification_records_safe_failure_without_server_secrets(tmp_path, approval_config):
    scenario = Scenario(tmp_path, GIT)
    request_approval(
        scenario.service,
        scenario.candidate_id,
        approval_config,
        actor="coordinator",
        request_id="request",
    )

    def fail(*args):
        raise smtplib.SMTPAuthenticationError(535, b"secret fixture server text")

    result = deliver_notifications(
        scenario.service, approval_config, actor="operator", transport=fail
    )
    data = result["notifications"][0]["data"]
    assert data["status"] == "uncertain"
    assert data["error_type"] == "SMTPAuthenticationError" and data["smtp_code"] == 535
    assert "secret fixture" not in str(data)


def test_cancel_fences_inflight_candidate_archival(tmp_path, monkeypatch):
    from hmopt.evolution import service as service_module
    from hmopt.evolution.scan import control_scan

    scenario = Scenario(tmp_path, GIT)
    (scenario.repo / "second.c").write_text(
        "int x = redundant_lookup(x);\n", encoding="utf-8", newline="\n"
    )
    scenario.commit("another candidate")
    row = start(scenario)
    before = scenario.service.store.list("candidate")
    original = service_module.scan_candidates

    def cancel_after_matching(*args, **kwargs):
        found = original(*args, **kwargs)
        current = scenario.service.store.read("scan", row["id"])
        control_scan(
            scenario.service,
            row["id"],
            action="cancel",
            actor="operator",
            expected_version=current["version"],
        )
        return found

    monkeypatch.setattr(service_module, "scan_candidates", cancel_after_matching)
    with pytest.raises(ConflictError, match="superseded"):
        scan_next(scenario.service, row["id"], expected_version=row["version"])
    assert scenario.service.store.list("candidate") == before
    assert not scenario.service.store.list("scan_page")
    assert scenario.service.store.read("scan", row["id"])["data"]["status"] == "cancelled"


def test_changed_pattern_can_receive_fresh_expert_request(tmp_path, approval_config):
    scenario = Scenario(tmp_path, GIT)
    old = request_approval(
        scenario.service,
        scenario.candidate_id,
        approval_config,
        actor="coordinator",
        request_id="old-request",
    )
    with scenario.service.store.transaction() as db:
        row = scenario.service.store.get(db, "pattern", scenario.pattern_key)
        row["data"]["pattern"]["risks"].append("New independently recorded compatibility concern.")
        scenario.service.store.put(db, "pattern", row["id"], row["data"], row["version"])
    new = request_approval(
        scenario.service,
        scenario.candidate_id,
        approval_config,
        actor="coordinator",
        request_id="new-request",
    )
    assert new["id"] != old["id"]
    sent = []
    deliver_notifications(
        scenario.service,
        approval_config,
        actor="operator",
        transport=lambda config, identity, message: sent.append(identity),
    )
    assert sent == [new["id"]]


def test_default_ssl_preserves_existing_approval_directory_digest(tmp_path, approval_config):
    scenario = Scenario(tmp_path, GIT)
    legacy = approval_config.model_dump(mode="json")
    legacy.pop("smtp_security")
    request = request_approval(
        scenario.service,
        scenario.candidate_id,
        approval_config,
        actor="coordinator",
        request_id="legacy-directory",
    )
    assert request["data"]["directory_sha256"] == digest(legacy)


def test_scan_recovery_cli_and_mcp_share_versions(tmp_path):
    import asyncio

    from test_evolution_interfaces import invoke, local_session, success, tool_value

    from hmopt.evolution.cli import app

    scenario = Scenario(tmp_path, GIT)
    scan = start(scenario)
    with scenario.service.store.transaction() as db:
        scan["data"].update(status="attention", error="fixture interrupted page")
        scan = scenario.service.store.put(db, "scan", scan["id"], scan["data"], scan["version"])
    retried = success(
        invoke(
            (app, []),
            scenario.service.store.root,
            "scan-control",
            scan["id"],
            "retry",
            "--version",
            str(scan["version"]),
            git_bin=GIT,
        )
    )
    assert retried["data"]["status"] == "running"

    async def cancel():
        async with local_session(scenario) as session:
            row = tool_value(
                await session.call_tool(
                    "evolution_scan",
                    {
                        "action": "cancel",
                        "scan_id": scan["id"],
                        "expected_version": retried["version"],
                        "actor": "operator",
                    },
                )
            )
            assert row["data"]["status"] == "cancelled"

    asyncio.run(cancel())
