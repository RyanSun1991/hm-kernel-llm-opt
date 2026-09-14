"""Cross-change research protocol, using explicit analysis fixtures and real Git diffs."""

from copy import deepcopy

import pytest
from test_evolution_change_analysis import change as shared_change
from test_evolution_change_analysis import report, submit
from test_evolution_mining import commit, history
from test_evolution_mining import repo as shared_repo
from test_evolution_skill_workflow import configure_methods

from hmopt.evolution.change_analysis import prepare_analysis
from hmopt.evolution.research import prepare_research, submit_research
from hmopt.evolution.store import ConflictError

repo = shared_repo
change = shared_change


@pytest.fixture
def synthesis(change, tmp_path):
    service, repo, first, old, new, _ = change
    configure_methods(service, tmp_path)
    submit(service, first, report(first))
    (repo / "another.py").write_text(
        old.replace("count", "another"), encoding="utf-8", newline="\n"
    )
    base = commit(repo, "add another parser")
    (repo / "another.py").write_text(
        new.replace("count", "another"), encoding="utf-8", newline="\n"
    )
    commit(repo, "update")
    changes = history(repo, after_revision=base)
    service.ingest_history(changes)
    second = prepare_analysis(service, repo, changes[0].source_id)
    value = report(second)
    value["proposals"][0]["exemplar_path"] = "another.py"
    submit(service, second, value, request_id="second-analysis")
    source_ids = [first["job"]["id"], second["job"]["id"]]
    prepared = prepare_research(service, repo, "synthesize", source_ids, actor="synthesizer")
    payload = {
        "summary": "Fixture: two independently changed parsers share a whitespace guard issue.",
        "groups": [
            {
                "proposal": report(first)["proposals"][0],
                "exemplars": [
                    {"source_id": source_ids[0], "path": "count.py"},
                    {"source_id": source_ids[1], "path": "another.py"},
                ],
                "differences_and_counterarguments": "Fixture: function names differ, while strict-parser APIs would exclude this remedy.",
            }
        ],
    }
    return service, repo, prepared, payload


def publish(service, prepared, payload, *, actor="synthesizer", request_id="synthesis"):
    row = prepared["research"]
    return submit_research(
        service,
        row["id"],
        payload,
        actor=actor,
        expected_version=row["version"],
        request_id=request_id,
    )


def test_multi_source_draft_requires_bound_independent_review(synthesis):
    service, repo, prepared, payload = synthesis
    result = publish(service, prepared, payload)
    assert result == publish(service, prepared, payload)
    key = result["data"]["output_ids"][0]
    pattern = service.store.read("pattern", key)["data"]
    assert len(pattern["pattern"]["source_ids"]) == 2
    assert pattern["requires_assessment"] and pattern["review_required"]
    with pytest.raises(ValueError, match="independent review"):
        service.activate_pattern(key, actor="curator", note="Reviewed the proposed generalization.")
    review = prepare_research(service, repo, "review", [key], actor="reviewer")
    decision = {
        "decision": "approve",
        "rationale": "Fixture independent review of mechanism, counterexamples and scope.",
    }
    with pytest.raises(ValueError, match="differ"):
        publish(service, review, decision, actor="synthesizer", request_id="self-review")
    publish(service, review, decision, actor="reviewer", request_id="independent-review")
    assert (
        service.activate_pattern(
            key, actor="curator", note="Reviewed independent evidence and applicability."
        )["data"]["status"]
        == "active"
    )


@pytest.mark.parametrize(
    "fault", ["duplicate-source", "unknown-source", "after-match", "finding", "primary-path"]
)
def test_unfounded_generalization_emits_no_patterns(synthesis, fault):
    service, _, prepared, payload = synthesis
    before = len(service.store.list("pattern"))
    group = payload["groups"][0]
    if fault == "duplicate-source":
        group["exemplars"][1] = deepcopy(group["exemplars"][0])
    elif fault == "unknown-source":
        group["exemplars"][1]["source_id"] = "not-selected"
    elif fault == "after-match":
        group["proposal"]["matcher"]["all_of"] = ["int(raw)"]
    elif fault == "finding":
        group["proposal"]["finding_indexes"] = [999]
    else:
        group["proposal"]["exemplar_path"] = "different.py"
    with pytest.raises(ValueError):
        publish(service, prepared, payload)
    assert len(service.store.list("pattern")) == before


def test_empty_synthesis_records_honest_no_generalization(synthesis):
    service, _, prepared, _ = synthesis
    result = publish(
        service,
        prepared,
        {"summary": "Insufficient shared mechanism for a supported generalization.", "groups": []},
    )
    assert result["data"]["output_ids"] == []
    with pytest.raises(ConflictError):
        publish(
            service,
            prepared,
            {"summary": "Changed answer cannot overwrite completed research.", "groups": []},
            request_id="different",
        )
