"""Real local HTTP requests exercise the OpenCode wire adapter and unified approval mounting."""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient
from test_evolution_production import (
    FixtureOpenCode,
    decision,
    post,
    unlock,
    worker_setup,
)
from test_evolution_production import approval_config as shared_approval_config
from test_evolution_production import change as shared_change
from test_evolution_production import git_bin as shared_git_bin
from test_evolution_production import repo as shared_repo
from test_evolution_production import scenario as shared_scenario

from hmopt.api.evolution_mcp_service import build_evolution_fastmcp_server, load_evolution_config
from hmopt.evolution.approval import request_approval
from hmopt.evolution.production import WorkerConfig
from hmopt.evolution.store import ConflictError, digest
from hmopt.evolution.worker import OpenCodeClient, _claim, retire_job, retry_job, worker_tick

approval_config = shared_approval_config
scenario = shared_scenario
change = shared_change
repo = shared_repo
git_bin = shared_git_bin


def test_real_http_opencode_adapter_freezes_digest_model_permissions_and_message_id(tmp_path):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, body))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"id": "ses-wire-fixture"}).encode())

        def do_GET(self):
            received.append((self.path, None))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"[]")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = WorkerConfig(
            url=f"http://127.0.0.1:{server.server_port}",
            directory=str(tmp_path),
            provider_id="fixture",
            model_id="model",
        )
        client = OpenCodeClient(config)
        session_id = client.create("wire fixture")
        packet = {"source_id": "real-wire-fixture", "before": "source evidence"}
        client.send(session_id, "msg-fixed-fixture", packet, {"skills": []})
        assert client.messages(session_id) == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert received[0][1]["permission"] == [{"permission": "*", "pattern": "*", "action": "deny"}]
    payload = received[1][1]
    prompt = json.loads(payload["parts"][0]["text"])
    assert prompt["packet_sha256"] == digest(packet)
    assert payload["messageID"] == "msg-fixed-fixture"
    assert payload["model"] == {"providerID": "fixture", "modelID": "model"}
    assert payload["format"]["schema"]["additionalProperties"] is False
    assert all("?directory=" in path for path, body in received)


def test_unified_http_mounts_gateway_and_mcp_cannot_forge_owner(
    scenario, approval_config, monkeypatch
):
    from hmopt.api.mcp_server import create_app

    monkeypatch.setenv(approval_config.signing_key_env, "fixture-key-" * 4)
    server = build_evolution_fastmcp_server(
        scenario.service.store.root,
        git_bin=scenario.service.git_bin,
        workspace_root=scenario.service.workspace_root,
        production={"approval": approval_config.model_dump(mode="json")},
    )
    request = request_approval(
        scenario.service,
        scenario.candidate_id,
        approval_config,
        actor="coordinator",
        request_id="test-request",
    )
    app = create_app(server, api_key="ordinary-mcp-key")
    with TestClient(app) as client:
        # Generic MCP authentication never replaces authenticated expert identity.
        response = client.post(
            "/evolution/approval/decide",
            json=decision(request),
            headers={"Authorization": "Bearer ordinary-mcp-key"},
        )
        assert response.status_code == 401
        response = post(client, "/evolution/approval/decide", decision(request))
        assert response.status_code == 200, response.text
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert len(names) == 38
    assert "evolution_request_approval" in names and "evolution_confirm" not in names


def test_same_config_loads_for_mcp_and_cli_without_secrets(
    scenario, approval_config, tmp_path, monkeypatch
):
    path = tmp_path / "production.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "root": str(scenario.service.store.root),
                "workspace_root": str(scenario.service.workspace_root),
                "production": {"approval": approval_config.model_dump(mode="json")},
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    config = load_evolution_config(path)
    assert config["production"]["approval"]["signing_key_env"] == "EVO_TEST_GATEWAY_KEY"
    from typer.testing import CliRunner

    from hmopt.evolution.cli import app

    received = {}

    class FakeServer:
        def run(self, **kwargs):
            assert kwargs == {"transport": "stdio"}

    def factory(**kwargs):
        received.update(kwargs)
        return FakeServer()

    monkeypatch.setattr("hmopt.evolution.mcp.build_server", factory)
    result = CliRunner().invoke(app, ["--config", str(path), "mcp-stdio"])
    assert result.exit_code == 0, result.output
    assert received["production"] == config["production"]
    schema = CliRunner().invoke(app, ["schema", "quality-evaluation"])
    assert schema.exit_code == 0 and "minimum_precision" in json.loads(schema.stdout)["properties"]
    config["production"]["surprise"] = True
    with pytest.raises(ValueError):
        build_evolution_fastmcp_server(**config)


def test_global_concurrency_includes_cancelled_remote_and_rejects_config_drift(change, tmp_path):
    service, config, campaign, analysis = worker_setup(change, tmp_path)
    client = FixtureOpenCode(analysis, config)
    row = worker_tick(service, config, worker_id="one", client=client)
    with service.store.transaction() as db:
        clone = {
            **row["data"],
            "status": "pending",
            "session_id": None,
            "remote_outstanding": False,
            "lease_until": 0,
        }
        service.store.put(db, "mining_job", "second-pending-job", clone)
    assert _claim(service, config, "two", time.time()) is None
    alternate = config.model_copy(update={"concurrency": 2})
    with pytest.raises(ConflictError, match="another config"):
        _claim(service, alternate, "two", time.time())
    from hmopt.evolution.worker import cancel_campaign

    cancel_campaign(service, campaign["id"], actor="operator")
    assert _claim(service, config, "two", time.time()) is None
    unlock(service, row["id"])
    with pytest.raises(ConflictError, match="worker stopped"):
        retire_job(service, row["id"], config, actor="operator", client=client)
    retire_job(service, row["id"], config, actor="operator", worker_stopped=True, client=client)
    next_job = _claim(service, config, "two", time.time())
    assert next_job["id"] == "second-pending-job"


def test_invalid_output_is_archived_and_explicit_retry_uses_new_message(change, tmp_path):
    service, config, _campaign, analysis = worker_setup(change, tmp_path)
    client = FixtureOpenCode({**analysis, "packet_sha256": "0" * 64}, config)
    row = worker_tick(service, config, worker_id="one", client=client)
    unlock(service, row["id"])
    failed = worker_tick(service, config, worker_id="one", client=client)
    # A stale/digest conflict must not be treated as an accepted model answer.
    assert failed["data"]["status"] == "attention"
    sha = failed["data"]["output_sha256"]
    assert service.store.read_evidence(sha)
    unlock(service, row["id"])
    failed = service.store.read("mining_job", row["id"])
    retried = retry_job(service, row["id"], actor="operator", expected_version=failed["version"])
    assert retried["data"]["message_id"] != failed["data"]["message_id"]
    assert retried["data"]["previous_attempts"][0]["output_sha256"] == sha


def test_worker_does_not_accept_intermediate_completed_message_while_session_busy(change, tmp_path):
    service, config, _campaign, analysis = worker_setup(change, tmp_path)
    client = FixtureOpenCode(analysis, config)
    first = worker_tick(service, config, worker_id="one", client=client)
    unlock(service, first["id"])
    client.idle = lambda session: False
    waiting = worker_tick(service, config, worker_id="one", client=client)
    assert waiting["data"]["status"] == "polling"
    assert waiting["data"]["remote_outstanding"]
    assert service.store.list("pattern") == []
    unlock(service, waiting["id"])
    client.idle = lambda session: True
    completed = worker_tick(service, config, worker_id="one", client=client)
    assert completed["data"]["status"] == "complete"


def test_smtp_adapter_builds_readable_expert_brief_with_stable_message_id(
    scenario, approval_config, monkeypatch
):
    from hmopt.evolution.approval import deliver_notifications
    from hmopt.evolution.production import ApprovalConfig

    config = ApprovalConfig.model_validate(
        {
            **approval_config.model_dump(mode="json"),
            "webhook_url": None,
            "smtp_host": "smtp.example.invalid",
            "smtp_sender": "hmopt@example.invalid",
        }
    )
    request = request_approval(
        scenario.service, scenario.candidate_id, config, actor="coordinator", request_id="mail-test"
    )
    sent = []

    class Mailer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def send_message(self, message):
            sent.append(message)

    monkeypatch.setattr(
        "hmopt.evolution.approval.smtplib.SMTP_SSL", lambda *args, **kwargs: Mailer()
    )
    result = deliver_notifications(scenario.service, config, actor="operator")
    assert result["notifications"][0]["data"]["status"] == "sent"
    assert str(sent[0]["To"]) == "expert@example.invalid"
    assert str(sent[0]["Message-ID"]) == f"<{request['id']}@hmopt.local>"
    assert "问题假设" in sent[0].get_content()
    assert "适用前提" in sent[0].get_content()
    assert request["data"]["context_sha256"] in sent[0].get_content()
