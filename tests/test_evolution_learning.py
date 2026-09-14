"""Bounded learning lifecycle tests using local stores and fixture measurements."""

import json
from copy import deepcopy

import pytest
from test_evolution_service import Scenario, history_fixture
from test_evolution_service import git_bin as shared_git_bin

from hmopt.evolution.mining import Hotspot, Pattern
from hmopt.evolution.service import EvolutionService, GateError
from hmopt.evolution.store import ConflictError, digest
from hmopt.evolution.validation import ABReport

git_bin = shared_git_bin


@pytest.fixture
def history_service(tmp_path):
    service = EvolutionService(tmp_path / "learning-state")
    source = history_fixture("learning-source", "1" * 40)
    service.ingest_history([source])
    return service


def capture(service, *, signal="structural_fact", **overrides):
    arguments = {
        "signal": signal,
        "recipe": "Lookup ownership remains protected by the caller lock.",
        "actor": "expert",
        "source_kind": "history",
        "source_id": "learning-source",
    }
    arguments.update(overrides)
    return service.capture(**arguments)


@pytest.mark.parametrize(
    "signal",
    [
        "validation_result",
        "failure_cause",
        "effective_recipe",
        "expert_decision",
        "structural_fact",
        "knowledge_correction",
    ],
)
def test_six_capture_signals_are_evidence_bound_and_never_self_promote(history_service, signal):
    service = history_service
    options = {}
    if signal == "knowledge_correction":
        options["corrects"] = capture(service)["id"]
    record = capture(service, signal=signal, **options)
    source = service.store.read("history", "learning-source")
    data = record["data"]
    assert data["signal"] == signal
    assert data["source_id"] == source["id"]
    assert data["source_version"] == source["version"]
    assert data["source_digest"] == digest(source["data"])
    assert data["tier"] == "journal"
    assert data["outcome"] == "unverified"
    assert data["eligible_for_promotion"] is False
    with service.store.transaction() as db:
        evidence = db.execute(
            "SELECT content FROM evidence WHERE sha256=?", (data["evidence"],)
        ).fetchone()[0]
    assert json.loads(evidence) == source
    with pytest.raises(GateError, match="real passing"):
        service.promote(
            record["id"],
            tier="staging",
            actor="curator",
            expected_version=1,
            request_id=f"promote-{signal}",
            note="Attempt to promote an unverified observation.",
        )


def test_repeated_capture_is_content_addressed_and_restartable(history_service):
    first = capture(history_service)
    repeated = capture(history_service)
    assert repeated == first
    assert len(history_service.store.list("skill")) == 1
    restarted = EvolutionService(history_service.store.root)
    assert capture(restarted) == first


def test_correction_records_link_without_mutating_original_knowledge(history_service):
    service = history_service
    original = capture(service)
    correction = capture(
        service,
        signal="knowledge_correction",
        corrects=original["id"],
        recipe="Correction: the helper requires a separate lifetime reference as well.",
    )
    assert correction["data"]["corrects"] == original["id"]
    assert service.store.read("skill", original["id"]) == original
    events = service.store.audit(original["id"])
    assert events[-1]["action"] == "correction_proposed"
    assert events[-1]["details"]["correction_skill_id"] == correction["id"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"signal": "unsupported_signal"},
        {"source_kind": "untrusted-arbitrary-path"},
        {"source_id": "missing-history"},
        {"recipe": "too short"},
        {"recipe": "x" * 20001},
        {"actor": "   "},
        {"signal": "knowledge_correction"},
        {"signal": "knowledge_correction", "corrects": "missing-knowledge"},
    ],
)
def test_invalid_capture_rejects_without_partial_knowledge(history_service, overrides):
    service = history_service
    before = service.store.list("skill")
    with pytest.raises(ValueError):
        capture(service, **overrides)
    assert service.store.list("skill") == before


def test_candidate_capture_cannot_turn_an_observation_into_a_validation(tmp_path, git_bin):
    scenario = Scenario(tmp_path, git_bin)
    service = scenario.service
    original = service.store.read("candidate", scenario.candidate_id)
    record = service.capture(
        signal="validation_result",
        recipe="An expert reports that the candidate looks promising.",
        actor="expert",
        source_kind="candidate",
        source_id=scenario.candidate_id,
    )
    assert record["data"]["candidate_id"] == scenario.candidate_id
    assert record["data"]["target"] == scenario.path
    assert record["data"]["pattern_key"] == scenario.pattern_key
    assert record["data"]["outcome"] == "unverified"
    assert record["data"]["eligible_for_promotion"] is False
    assert service.store.read("candidate", scenario.candidate_id) == original


def test_owner_can_retry_inconclusive_evidence_without_losing_first_attempt(tmp_path, git_bin):
    s = Scenario(tmp_path, git_bin).ready()
    first_report = ABReport.model_validate(s.report_data(values=(99.5,) * 3))
    first = s.validate(first_report)
    first_validation = deepcopy(first["data"]["validation"])
    original_skill = s.skill()
    with pytest.raises(GateError, match="No executable handoff"):
        s.service.handoff(s.candidate_id)
    retried = s.transition(
        "retry_validation",
        "owner",
        {"note": "Repeat paired measurements to resolve the inconclusive effect."},
    )
    assert retried["data"]["stage"] == "code_approved"
    assert "validation" not in retried["data"]
    assert retried["data"]["validation_history"] == [first_validation]
    assert s.service.handoff(s.candidate_id)["role"] == "validator"
    assert s.service.store.read("skill", original_skill["id"]) == original_skill
    with s.service.store.transaction() as db:
        assert (
            db.execute(
                "SELECT count(*) FROM outcomes WHERE candidate_id=?", (s.candidate_id,)
            ).fetchone()[0]
            == 0
        )
    passed = s.validate()
    assert passed["data"]["stage"] == "validated"
    assert passed["data"]["validation_history"] == [first_validation]
    skills = [
        row
        for row in s.service.store.list("skill")
        if row["data"]["candidate_id"] == s.candidate_id
    ]
    assert len(skills) == 2
    assert {row["data"]["outcome"] for row in skills} == {"pass", "inconclusive"}


@pytest.mark.parametrize("actor", ["validator", "implementer", "code-reviewer", "unassigned-owner"])
def test_inconclusive_retry_requires_assigned_owner(tmp_path, git_bin, actor):
    s = Scenario(tmp_path, git_bin).ready()
    s.validate(ABReport.model_validate(s.report_data(values=(99.5,) * 3)))
    before = s.service.store.read("candidate", s.candidate_id)
    with pytest.raises(GateError, match="Only the owner"):
        s.transition(
            "retry_validation", actor, {"note": "Attempt to reopen an inconclusive measurement."}
        )
    assert s.service.store.read("candidate", s.candidate_id) == before


@pytest.mark.parametrize("values", [(110.0,) * 3, (90.0,) * 3])
def test_retry_cannot_reopen_failed_or_passing_validation(tmp_path, git_bin, values):
    s = Scenario(tmp_path, git_bin).ready()
    s.validate(ABReport.model_validate(s.report_data(values=values)))
    with pytest.raises(GateError, match="Only the owner"):
        s.transition(
            "retry_validation", "owner", {"note": "This finalized outcome must not be reopened."}
        )


def test_retry_is_versioned_idempotent_and_preserves_immutable_policy(tmp_path, git_bin):
    s = Scenario(tmp_path, git_bin).ready()
    s.validate(ABReport.model_validate(s.report_data(values=(99.5,) * 3)))
    source = deepcopy(s.row)
    params = {
        "actor": "owner",
        "expected_version": source["version"],
        "request_id": "retry-once",
        "payload": {"note": "Repeat the approved experiment after an inconclusive result."},
    }
    first = s.service.transition(s.candidate_id, "retry_validation", **params)
    assert s.service.transition(s.candidate_id, "retry_validation", **params) == first
    assert len(first["data"]["validation_history"]) == 1
    assert first["data"]["plan_digest"] == source["data"]["plan_digest"]
    assert first["data"]["implementation_digest"] == source["data"]["implementation_digest"]
    assert first["data"]["plan"] == source["data"]["plan"]
    with pytest.raises(ConflictError):
        s.service.transition(
            s.candidate_id, "retry_validation", **{**params, "request_id": "stale-retry"}
        )


def test_retired_latest_pattern_does_not_fall_back_to_older_active_version(tmp_path, git_bin):
    s = Scenario(tmp_path, git_bin)
    service = s.service
    original = service.store.read("pattern", s.pattern_key)
    version_two = Pattern.model_validate({**original["data"]["pattern"], "version": 2})
    service.import_pattern(version_two)
    key_two = f"{version_two.pattern_id}@2"
    service.activate_pattern(
        key_two, actor="curator", note="Updated and reviewed the second pattern version."
    )
    current = service.scan(
        s.repo,
        owners={"**/*.c": "owner"},
        hotspots=[Hotspot(path=s.path, weight=0.9, revision=s.base)],
    )
    assert {row["data"]["candidate"]["pattern_version"] for row in current["candidates"]} == {2}
    retired = service.retire_pattern(
        key_two, actor="curator", note="Evidence invalidates this pattern family."
    )
    assert retired["data"]["status"] == "retired"
    assert service.store.read("pattern", s.pattern_key)["data"]["status"] == "active"
    assert service.scan(s.repo, owners={"**/*.c": "owner"})["candidates"] == []
    with pytest.raises(GateError, match="Only a draft"):
        service.activate_pattern(
            key_two, actor="curator", note="Retired version must not be reactivated."
        )
    assert (
        service.retire_pattern(
            key_two, actor="curator", note="Same retirement does not duplicate an event."
        )
        == retired
    )
    assert [event["action"] for event in service.store.audit(key_two)] == [
        "activate_pattern",
        "retire_pattern",
    ]


def test_newer_explicitly_curated_version_can_replace_a_retired_pattern(tmp_path, git_bin):
    s = Scenario(tmp_path, git_bin)
    service = s.service
    service.retire_pattern(
        s.pattern_key, actor="curator", note="Retire the original unsafe applicability rule."
    )
    original = service.store.read("pattern", s.pattern_key)["data"]["pattern"]
    successor = Pattern.model_validate(
        {
            **original,
            "version": 2,
            "status": "draft",
            "remedy": "A revised remedy addresses the earlier lifetime failure.",
        }
    )
    service.import_pattern(successor)
    assert service.scan(s.repo, owners={"**/*.c": "owner"})["candidates"] == []
    service.activate_pattern(
        f"{successor.pattern_id}@2",
        actor="curator",
        note="Reviewed revised remedy and its new constraints.",
    )
    found = service.scan(s.repo, owners={"**/*.c": "owner"})["candidates"]
    assert len(found) == 1
    assert found[0]["data"]["candidate"]["pattern_version"] == 2


def test_recall_finds_matching_knowledge_beyond_the_oldest_thousand_records(history_service):
    service = history_service
    # Populate storage in one transaction to keep this a bounded search test,
    # rather than performing a thousand separate capture transactions.
    with service.store.transaction() as db:
        for index in range(1001):
            skill_id = f"older-{index:04d}"
            service.store.put(
                db,
                "skill",
                skill_id,
                {
                    "skill_id": skill_id,
                    "tier": "journal",
                    "target": "unrelated.c",
                    "recipe": "Unrelated historical observation.",
                    "pattern_key": "other@1",
                    "eligible_for_promotion": False,
                },
            )
    newest = capture(service, recipe="Zephyrlookup requires a stable lifetime reference.")
    recalled = service.recall("Zephyrlookup", limit=3)
    assert [row["id"] for row in recalled] == [newest["id"]]
    assert service.recall("zznonexistentlookupzz", limit=3) == []


def test_recall_preserves_result_limit_and_prefers_hub_on_equal_relevance(history_service):
    service = history_service
    with service.store.transaction() as db:
        for index in range(6):
            skill_id = f"relevant-{index}"
            service.store.put(
                db,
                "skill",
                skill_id,
                {
                    "skill_id": skill_id,
                    "tier": "hub" if index == 5 else "journal",
                    "target": "kernel.c",
                    "recipe": "Lookup lifetime constraints.",
                    "pattern_key": "lookup@1",
                    "eligible_for_promotion": index == 5,
                },
            )
    assert len(service.recall("lookup", limit=4)) == 4
    assert service.recall("lookup", limit=3)[0]["id"] == "relevant-5"
