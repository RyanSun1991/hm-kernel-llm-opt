"""Model replies are fixtures; actual read-only Git context and recovery are exercised."""

import pytest
from test_evolution_change_analysis import change as shared_change
from test_evolution_change_analysis import repo as shared_repo
from test_evolution_production import FixtureOpenCode, unlock, worker_setup

from hmopt.evolution.investigation import ContextRequest, resolve_requests
from hmopt.evolution.worker import worker_tick

repo = shared_repo
change = shared_change


def test_model_requests_context_then_completes_in_same_session(change, tmp_path):
    service, config, _campaign, analysis = worker_setup(change, tmp_path)
    request = {
        "context_requests": [
            {
                "operation": "read",
                "side": "before",
                "path": "count.py",
                "reason": "Inspect the full input normalization contract.",
            }
        ]
    }
    client = FixtureOpenCode(request, config)
    row = worker_tick(service, config, worker_id="a", client=client)
    unlock(service, row["id"])
    row = worker_tick(service, config, worker_id="b", client=client)
    assert row["data"]["status"] == "preparing"
    assert row["data"]["context_rounds"] == 1
    context = service.store.read_evidence(row["data"]["context_sha256"])[0]["result"]
    assert context["revision"] == change[2]["packet"]["parent_revision"]
    assert context["lines"][1] == "    return int(raw) if raw else 0"
    analysis["context_citations"] = [
        {"context_sha256": context["sha256"], "line_start": 1, "quote": context["lines"][0]}
    ]
    client.analysis = {"analysis": analysis}
    row = worker_tick(service, config, worker_id="c", client=client)
    unlock(service, row["id"])
    row = worker_tick(service, config, worker_id="d", client=client)
    assert row["data"]["status"] == "complete"
    assert client.creates == 1 and client.sends == 2
    assert row["data"]["reported_tokens"] == 300
    assert len(row["data"]["rounds"]) == 1


@pytest.mark.parametrize(
    "extra",
    [{"path": "../private.txt"}, {"revision": "HEAD"}, {"repo": "other"}, {"operation": "shell"}],
)
def test_investigation_cannot_expand_source_or_operation_scope(extra):
    with pytest.raises(ValueError):
        ContextRequest.model_validate(
            {
                "operation": "read",
                "side": "after",
                "path": "count.py",
                "reason": "Need exact immutable evidence.",
                **extra,
            }
        )


def test_search_is_literal_and_no_matches_is_valid_evidence(change):
    service, repo, prepared, *_ = change
    requests = [
        ContextRequest(
            operation="search",
            side="after",
            query=q,
            reason="Locate a definition before making a claim.",
        )
        for q in ("def count", "absent_symbol")
    ]
    results = resolve_requests(service, repo, prepared["packet"], requests)
    assert results[0]["result"]["matches"]
    assert results[1]["result"]["matches"] == []


def test_context_budget_stops_repeated_requests_without_resending(change, tmp_path):
    service, config, _campaign, analysis = worker_setup(change, tmp_path)
    config.max_context_rounds = 1
    # Job config is immutable: create the job using the same intended budget.
    from hmopt.evolution.worker import create_campaign

    with service.store.transaction() as db:
        db.execute("DELETE FROM records WHERE kind IN ('mining_job','mining_campaign')")
    create_campaign(
        service,
        change[1],
        [analysis["source_id"]],
        config,
        actor="coordinator",
        request_id="limited",
    )
    client = FixtureOpenCode(
        {
            "context_requests": [
                {
                    "operation": "read",
                    "side": "after",
                    "path": "count.py",
                    "reason": "Need exact immutable evidence.",
                }
            ]
        },
        config,
    )
    for _ in range(4):
        row = worker_tick(service, config, worker_id="a", client=client)
        unlock(service, row["id"])
    assert row["data"]["reason"] == "investigation_budget_exhausted"
    assert client.sends == 2
