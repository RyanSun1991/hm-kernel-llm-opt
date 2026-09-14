"""Production protocol tests use explicit model and delivery fixtures, not live model-quality claims."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_evolution_change_analysis import change as shared_change
from test_evolution_change_analysis import repo as shared_repo
from test_evolution_change_analysis import report
from test_evolution_skill_workflow import configure_methods
from test_evolution_skill_workflow import git_bin as shared_git_bin
from test_evolution_skill_workflow import scenario as shared_scenario

from hmopt.api.evolution_approval import approval_router
from hmopt.evolution.approval import (
    deliver_notifications,
    request_approval,
    retry_notification,
    signature,
)
from hmopt.evolution.production import ApprovalConfig, WorkerConfig, check_production
from hmopt.evolution.quality_eval import EvaluationSet, evaluate, suggest_groups
from hmopt.evolution.store import ConflictError, canonical_json
from hmopt.evolution.worker import (
    _claim,
    _save,
    campaign_status,
    cancel_campaign,
    create_campaign,
    retire_job,
    worker_tick,
)

repo = shared_repo
change = shared_change
git_bin = shared_git_bin
scenario = shared_scenario


@pytest.fixture
def approval_config():
    return ApprovalConfig(
        contacts={"owner": {"principal": "tenant:user-17", "target": "expert@example.invalid"}},
        signing_key_env="EVO_TEST_GATEWAY_KEY",
        gateway_url="https://approvals.example.invalid/review",
        webhook_url="https://chat.example.invalid/notifications",
    )


def gateway(scenario, config, monkeypatch):
    monkeypatch.setenv(config.signing_key_env, "fixture-key-" * 4)
    app = FastAPI()
    app.include_router(approval_router(lambda: scenario.service, config))
    return TestClient(app)


def post(client, path, data, *, principal_key="fixture-key-" * 4, timestamp=None):
    body = canonical_json(data).encode()
    timestamp = timestamp or str(int(time.time()))
    return client.post(
        path,
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Evolution-Timestamp": timestamp,
            "X-Evolution-Signature": signature(principal_key, timestamp, path, body),
        },
    )


def decision(request):
    return {
        "approval_request_id": request["id"],
        "principal": "tenant:user-17",
        "context_sha256": request["data"]["context_sha256"],
        "decision": "confirm",
        "note": "Expert reviewed the immutable evidence and accepts investigation.",
        "request_id": "expert-decision-1",
    }


def test_authenticated_decision_archives_receipt_and_continuation_atomically(
    scenario, approval_config, monkeypatch
):
    service = scenario.service
    request = request_approval(
        service, scenario.candidate_id, approval_config, actor="coordinator", request_id="request-1"
    )
    same = request_approval(
        service, scenario.candidate_id, approval_config, actor="coordinator", request_id="request-2"
    )
    assert same == request
    assert len(service.store.list("notification")) == 1
    client = gateway(scenario, approval_config, monkeypatch)
    response = post(
        client,
        "/evolution/approval/context",
        {
            "approval_request_id": request["id"],
            "principal": "tenant:user-17",
        },
    )
    assert response.status_code == 200
    response = post(client, "/evolution/approval/decide", decision(request))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["candidate"]["data"]["stage"] == "confirmed"
    receipt = service.store.read("approval", result["approval_id"])["data"]
    assert receipt["identity_assurance"] == "authenticated_approval_gateway"
    assert receipt["authenticated_identity"]["principal"] == "tenant:user-17"
    assert result["continuation"]["data"]["automatic_execution"] is False
    assert post(client, "/evolution/approval/decide", decision(request)).json() == result
    changed = {**decision(request), "decision": "reject"}
    assert post(client, "/evolution/approval/decide", changed).status_code == 409


@pytest.mark.parametrize(
    "problem,code",
    [
        ("wrong_principal", 403),
        ("wrong_signature", 401),
        ("expired_signature", 401),
        ("stale_candidate", 409),
        ("wrong_context", 409),
        ("expired_request", 409),
        ("dirty_baseline", 400),
        ("changed_pattern", 409),
    ],
)
def test_gateway_rejects_invalid_decisions_without_partial_state(
    scenario, approval_config, monkeypatch, problem, code
):
    service = scenario.service
    request = request_approval(
        service, scenario.candidate_id, approval_config, actor="coordinator", request_id="request-1"
    )
    payload, kwargs = decision(request), {}
    if problem == "wrong_principal":
        payload["principal"] = "tenant:someone-else"
    if problem == "wrong_signature":
        kwargs["principal_key"] = "wrong-key"
    if problem == "expired_signature":
        kwargs["timestamp"] = "1"
    if problem == "wrong_context":
        payload["context_sha256"] = "0" * 64
    if problem in {"stale_candidate", "expired_request", "changed_pattern"}:
        kind, key = (
            ("candidate", scenario.candidate_id)
            if problem == "stale_candidate"
            else ("pattern", scenario.pattern_key)
            if problem == "changed_pattern"
            else ("approval_request", request["id"])
        )
        with service.store.transaction() as db:
            row = service.store.get(db, kind, key)
            if problem == "expired_request":
                row["data"]["expires_at"] = 1
            if problem == "changed_pattern":
                row["data"]["pattern"]["remedy"] = "Changed remedy for a different context."
            service.store.put(db, kind, key, row["data"], row["version"])
    if problem == "dirty_baseline":
        (scenario.repo / scenario.path).write_text("dirty", encoding="utf-8")
    client = gateway(scenario, approval_config, monkeypatch)
    response = post(client, "/evolution/approval/decide", payload, **kwargs)
    assert response.status_code == code, response.text
    assert service.store.read("candidate", scenario.candidate_id)["data"]["stage"] == "discovered"
    assert service.store.list("approval") == service.store.list("continuation") == []


def test_notifications_uncertain_delivery_requires_explicit_retry(scenario, approval_config):
    service = scenario.service
    request = request_approval(
        service, scenario.candidate_id, approval_config, actor="coordinator", request_id="request-1"
    )
    sent = []

    def transport(config, key, message):
        sent.append(key)
        assert message["target"] == "expert@example.invalid"
        raise OSError("uncertain transport fixture")

    result = deliver_notifications(service, approval_config, actor="operator", transport=transport)
    row = result["notifications"][0]
    assert row["data"]["status"] == "uncertain"
    assert (
        deliver_notifications(service, approval_config, actor="operator", transport=transport)[
            "notifications"
        ]
        == []
    )
    with pytest.raises(ConflictError):
        retry_notification(service, request["id"], actor="operator", expected_version=1)
    retry_notification(service, request["id"], actor="operator", expected_version=row["version"])
    result = deliver_notifications(
        service,
        approval_config,
        actor="operator",
        transport=lambda config, key, message: sent.append(key),
    )
    assert result["notifications"][0]["data"]["status"] == "sent"
    assert sent == [request["id"], request["id"]]


def test_concurrent_notification_workers_only_one_delivery(scenario, approval_config):
    request_approval(
        scenario.service,
        scenario.candidate_id,
        approval_config,
        actor="coordinator",
        request_id="request-1",
    )
    sent = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: deliver_notifications(
                    scenario.service,
                    approval_config,
                    actor="operator",
                    transport=lambda config, key, message: sent.append(key),
                ),
                range(4),
            )
        )
    assert len(sent) == 1 and sum(len(r["notifications"]) for r in results) == 1


class FixtureOpenCode:
    def __init__(self, analysis, config, *, uncertain=False):
        self.analysis, self.config, self.uncertain = analysis, config, uncertain
        self.creates = self.sends = 0
        self.message = None
        self.completed = True
        self.aborts = 0

    def create(self, title):
        self.creates += 1
        return "ses-fixture"

    def send(self, session, message, packet, method):
        self.sends += 1
        self.message = message
        if self.uncertain:
            raise OSError("Fixture accepted request but disconnected before acknowledgement")

    def messages(self, session):
        messages = [{"info": {"role": "user", "id": self.message}}]
        if self.completed:
            messages.append(
                {
                    "info": {
                        "role": "assistant",
                        "parentID": self.message,
                        "time": {"completed": 1},
                        "modelID": self.config.model_id,
                        "providerID": self.config.provider_id,
                        "tokens": {"input": 100, "output": 50},
                        "cost": 0,
                    },
                    "parts": [{"type": "text", "text": json.dumps(self.analysis)}],
                }
            )
        return messages

    def abort(self, session):
        self.aborts += 1

    def idle(self, session):
        return True


def worker_setup(change, tmp_path):
    service, repo, prepared, *_ = change
    configure_methods(service, tmp_path)
    config = WorkerConfig(
        url="http://127.0.0.1:12345",
        directory=str(service.workspace_root.parents[2]),
        provider_id="fixture",
        model_id="test-model",
        concurrency=1,
    )
    campaign = create_campaign(
        service, repo, [prepared["job"]["id"]], config, actor="coordinator", request_id="campaign-1"
    )
    return service, config, campaign, report(prepared)


def unlock(service, key):
    with service.store.transaction() as db:
        row = service.store.get(db, "mining_job", key)
        row["data"]["lease_until"] = 0
        return service.store.put(db, "mining_job", key, row["data"], row["version"])


@pytest.mark.parametrize("uncertain", [False, True])
def test_worker_resumes_original_session_and_validates_real_git_evidence(
    change, tmp_path, uncertain
):
    service, config, campaign, analysis = worker_setup(change, tmp_path)
    client = FixtureOpenCode(analysis, config, uncertain=uncertain)
    row = worker_tick(service, config, worker_id="a", client=client)
    assert row["data"]["status"] == ("sending" if uncertain else "polling")
    unlock(service, row["id"])
    result = worker_tick(service, config, worker_id="b", client=client)
    assert result["data"]["status"] == "complete"
    assert client.creates == client.sends == 1
    assert campaign_status(service, campaign["id"])["finished"]
    assert (
        service.store.read("history_analysis", analysis["source_id"])["data"]["status"]
        == "patterns"
    )
    assert all(p["data"]["status"] == "draft" for p in service.store.list("pattern"))


def test_worker_invalid_model_evidence_stops_without_patterns(change, tmp_path):
    service, config, _campaign, analysis = worker_setup(change, tmp_path)
    analysis["findings"][0]["citations"][0]["quote"] = "invented source line"
    client = FixtureOpenCode(analysis, config)
    first = worker_tick(service, config, worker_id="a", client=client)
    unlock(service, first["id"])
    result = worker_tick(service, config, worker_id="a", client=client)
    assert result["data"]["status"] == "attention"
    assert service.store.list("pattern") == []
    assert not result["data"]["remote_outstanding"]


def test_expired_worker_fenced_and_uncertain_remote_reserves_capacity(change, tmp_path):
    service, config, campaign, analysis = worker_setup(change, tmp_path)
    old = _claim(service, config, "old", time.time())
    assert _claim(service, config, "other", time.time()) is None
    unlock(service, old["id"])
    newer = _claim(service, config, "newer", time.time())
    with pytest.raises(ConflictError, match="superseded"):
        _save(service, old, status="complete")
    unlock(service, newer["id"])
    client = FixtureOpenCode(analysis, config)
    first = worker_tick(service, config, worker_id="a", client=client)
    cancel_campaign(service, campaign["id"], actor="operator")
    assert service.store.read("mining_job", first["id"])["data"]["remote_outstanding"]
    unlock(service, first["id"])
    retired = retire_job(
        service, first["id"], config, actor="operator", worker_stopped=True, client=client
    )
    assert not retired["data"]["remote_outstanding"] and client.aborts == 1
    assert service.store.list("pattern") == []


def test_grouping_is_explicitly_a_shortlist_not_semantic_proof(change, tmp_path):
    service, config, _campaign, analysis = worker_setup(change, tmp_path)
    client = FixtureOpenCode(analysis, config)
    row = worker_tick(service, config, worker_id="a", client=client)
    unlock(service, row["id"])
    worker_tick(service, config, worker_id="a", client=client)
    groups = suggest_groups(service, change[1])
    assert groups["sources_examined"] == 1 and groups["groups"] == []
    assert "not_semantic" in groups["method"]


def test_quality_holdout_counts_missing_and_abstentions_in_coverage(change):
    service, _repo, prepared, *_ = change
    analysis = report(prepared)
    with service.store.transaction() as db:
        input_sha = prepared["job"]["data"]["packet_sha256"]
        report_sha = service.store.evidence(db, analysis)
        negative_input = service.store.evidence(db, {"fixture": "separate-negative-case"})
        method_sha = service.store.evidence(db, {"fixture": "test-method"})
    label = {
        "case_id": "positive",
        "family": "parser-positive",
        "split": "holdout",
        "expected": True,
        "input_sha256": input_sha,
        "expert": "test-expert",
        "rationale": "Fixture expert label of whitespace parser change.",
    }
    data = {
        "name": "fixture-not-production-evaluation",
        "task": "history",
        "labels": [
            label,
            {
                **label,
                "case_id": "negative",
                "family": "negative-family",
                "expected": False,
                "input_sha256": negative_input,
            },
        ],
        "predictions": [
            {"case_id": "positive", "input_sha256": input_sha, "report_sha256": report_sha}
        ],
        "model_id": "fixture-model",
        "method_sha256": method_sha,
        "minimum_holdout": 2,
    }
    result = evaluate(service, EvaluationSet.model_validate(data), actor="evaluator")["data"]
    assert result["scores"]["holdout"]["precision"] == 1
    assert result["scores"]["holdout"]["coverage"] == 0.5
    assert result["quality_gate"] == "fail"
    data["labels"][1]["split"] = "development"
    data["labels"][1]["family"] = "parser-positive"
    with pytest.raises(ValueError, match="leaks"):
        EvaluationSet.model_validate(data)


def test_production_config_does_not_echo_secret_or_contact_service(monkeypatch, approval_config):
    monkeypatch.setenv("EVO_TEST_GATEWAY_KEY", "x" * 40)
    status = check_production({"approval": approval_config.model_dump(mode="json")})
    assert status["ready_for_connection_check"] and not status["external_services_checked"]
    assert "x" * 40 not in json.dumps(status)
    with pytest.raises(ValueError):
        WorkerConfig(
            url="http://remote.example.invalid", directory="/bad", provider_id="x", model_id="x"
        )
