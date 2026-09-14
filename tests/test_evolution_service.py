"""Local Git/SQLite integration tests; measurements are explicitly test fixtures.

No test builds a kernel or reaches hardware. ``hardware=True`` below exercises
the trusted-operator attestation contract, not a claim of device execution.
"""

import hashlib
import json
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from hmopt.evolution.mining import ChangeRecord, Hotspot, Matcher, Pattern
from hmopt.evolution.service import EvolutionService, GateError, Plan, ValidationPolicy
from hmopt.evolution.store import ConflictError, EvolutionStore, digest
from hmopt.evolution.validation import ABReport


@pytest.fixture
def git_bin():
    binary = shutil.which("git")
    bundled = (
        Path.home()
        / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
        / "native/git/cmd/git.exe"
    )
    if binary:
        return binary
    if bundled.is_file():
        return str(bundled)
    pytest.skip("These integration tests require a local Git executable")


class Scenario:
    def __init__(self, tmp_path, git_bin, *, service=None, name="one"):
        self.git_bin = git_bin
        self.repo = tmp_path / f"repo-{name}"
        self.repo.mkdir()
        self.path = f"module_{name}.c"
        self.service = service or EvolutionService(tmp_path / "state", git_bin=git_bin)
        self.git("init", "-q")
        self.git("config", "user.name", "Integration Test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "core.autocrlf", "false")
        (self.repo / self.path).write_text(
            "int target(int x) { return redundant_lookup(x); }\n", encoding="utf-8"
        )
        (self.repo / "unrelated.txt").write_text("original\n", encoding="utf-8")
        self.commit("baseline")
        self.base = self.git("rev-parse", "HEAD").strip()
        self.pattern_key = "remove-redundant-lookup@1"
        if not self.service.store.list("pattern"):
            source = ChangeRecord(
                source_id="reviewed-history",
                repo_id="historical-repo",
                revision=self.base,
                subject="Reviewed optimization",
                message="Historical evidence for manual curation.",
                paths=[self.path],
                patch="-redundant_lookup(x);\n+cached_lookup(x);\n",
            )
            self.service.ingest_history([source])
            self.service.import_pattern(
                Pattern(
                    pattern_id="remove-redundant-lookup",
                    title="Remove repeated lookup",
                    kind="optimization",
                    problem="The measured path repeats a stable lookup.",
                    diagnosis="The lookup is invariant across this operation.",
                    remedy="Reuse the value while preserving ownership and locking.",
                    matcher=Matcher(file_globs=["**/*.c"], all_of=["redundant_lookup(x)"]),
                    primary_metric="instructions",
                    unit="count",
                    direction="minimize",
                    preconditions=["Owner must verify lookup stability and lifetime."],
                    risks=["A changed lifetime can invalidate cached values."],
                    source_ids=["reviewed-history"],
                )
            )
            self.service.activate_pattern(
                self.pattern_key,
                actor="pattern-curator",
                note="Reviewed historical evidence and bounded applicability.",
            )
        scan = self.service.scan(
            self.repo,
            owners={"**/*.c": "owner"},
            hotspots=[Hotspot(path=self.path, weight=0.9, revision=self.base)],
        )
        assert len(scan["candidates"]) == 1
        self.row = scan["candidates"][0]
        self.candidate_id = self.row["id"]
        self.sequence = 0

    def git(self, *args):
        env = dict(os.environ)
        for key in (
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_COMMON_DIR",
        ):
            env.pop(key, None)
        env.update(
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_TERMINAL_PROMPT="0",
            GIT_NO_REPLACE_OBJECTS="1",
        )
        result = subprocess.run(
            [
                self.git_bin,
                "-c",
                "core.hooksPath=" + str(self.repo / "empty-hooks"),
                "-C",
                str(self.repo),
                *args,
            ],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            env=env,
            timeout=30,
        )
        return result.stdout.decode("utf-8")

    def commit(self, message):
        self.git("add", "--all")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD").strip()

    def request_id(self, action):
        self.sequence += 1
        return f"{self.candidate_id}-{action}-{self.sequence}"

    def transition(self, action, actor, payload=None, **overrides):
        params = {
            "actor": actor,
            "expected_version": self.row["version"],
            "request_id": self.request_id(action),
            "payload": payload,
        }
        params.update(overrides)
        self.row = self.service.transition(self.candidate_id, action, **params)
        return self.row

    def confirm(self):
        return self.transition("confirm", "owner", {"note": "Hot path and ownership confirmed."})

    def plan_payload(self):
        plan = Plan(
            candidate_id=self.candidate_id,
            base_revision=self.base,
            author="architect",
            hypothesis="Remove repeated lookup instructions without changing returned values.",
            bottleneck="cpu",
            metric_rationale="Instruction count measures repeated lookup cost.",
            allowed_paths=[self.path],
            validation=ValidationPolicy(
                metrics=[
                    {
                        "name": "instructions",
                        "unit": "count",
                        "direction": "minimize",
                        "primary": True,
                    }
                ],
                minimum_pairs=3,
                device_id="fixture-device",
                workload_id="fixture-workload",
                workload_config_sha256="c" * 64,
                environment_sha256="d" * 64,
            ),
        ).model_dump(mode="json")
        return {"plan": plan, "review": self.review(digest(plan), "plan-reviewer")}

    def approve_plan(self):
        return self.transition("approve_plan", "plan-reviewer", self.plan_payload())

    def review(self, subject, author):
        return {
            "candidate_id": self.candidate_id,
            "subject_digest": subject,
            "author": author,
            "decision": "approve",
            "rationale": "Independent review confirms evidence and preserved semantics.",
        }

    def implement(self, *, extra_file=False):
        (self.repo / self.path).write_text(
            "int target(int x) { return cached_lookup(x); }\n", encoding="utf-8"
        )
        if extra_file:
            (self.repo / "unrelated.txt").write_text("unauthorized\n", encoding="utf-8")
        revision = self.commit("feature optimization")
        return self.transition("record_implementation", "implementer", {"revision": revision})

    def approve_code(self):
        return self.transition(
            "approve_code",
            "code-reviewer",
            self.review(self.row["data"]["implementation_digest"], "code-reviewer"),
        )

    def ready(self):
        self.confirm()
        self.approve_plan()
        self.implement()
        self.approve_code()
        return self

    def report_data(self, *, values=(90.0, 90.0, 90.0), hardware=True):
        policy = self.row["data"]["plan"]["validation"]
        feature = self.row["data"]["implementation"]["revision"]
        shared = {
            field: policy[field]
            for field in (
                "device_id",
                "workload_id",
                "workload_config_sha256",
                "environment_sha256",
            )
        }
        return {
            "candidate_id": self.candidate_id,
            "implementation_revision": feature,
            "functional_passed": True,
            "metrics": policy["metrics"],
            "minimum_pairs": policy["minimum_pairs"],
            "baseline": {
                **shared,
                "repo_revision": self.base,
                "image_sha256": hashlib.sha256(self.base.encode()).hexdigest(),
                "hardware": hardware,
                "measurements": [
                    {"pair_id": str(i), "metrics": {"instructions": 100.0}}
                    for i in range(len(values))
                ],
            },
            "candidate": {
                **shared,
                "repo_revision": feature,
                "image_sha256": hashlib.sha256(feature.encode()).hexdigest(),
                "hardware": hardware,
                "measurements": [
                    {"pair_id": str(i), "metrics": {"instructions": value}}
                    for i, value in enumerate(values)
                ],
            },
        }

    def validate(self, report=None, **overrides):
        report = report or ABReport.model_validate(self.report_data())
        params = {
            "actor": "validator",
            "expected_version": self.row["version"],
            "request_id": self.request_id("validate"),
        }
        params.update(overrides)
        self.row = self.service.validate(self.candidate_id, report, **params)
        return self.row

    def skill(self):
        skills = [
            r
            for r in self.service.store.list("skill")
            if r["data"]["candidate_id"] == self.candidate_id
        ]
        assert len(skills) == 1
        return skills[0]

    def promote(self, *, tier="staging", actor="knowledge-curator", skill=None, **overrides):
        skill = skill or self.skill()
        params = {
            "tier": tier,
            "actor": actor,
            "expected_version": skill["version"],
            "request_id": self.request_id("promote"),
            "note": "Independent curator reviewed applicability and validation evidence.",
        }
        params.update(overrides)
        return self.service.promote(skill["id"], **params)


@pytest.fixture
def scenario(tmp_path, git_bin):
    return Scenario(tmp_path, git_bin)


def test_attested_fixture_runs_full_pipeline_and_persists_evidence(scenario):
    s = scenario
    assert s.row["data"]["candidate"]["lane"] == "pipeline"
    with pytest.raises(GateError):
        s.service.handoff(s.candidate_id)
    s.confirm()
    assert s.service.handoff(s.candidate_id)["role"] == "architect"
    s.approve_plan()
    handoff = s.service.handoff(s.candidate_id)
    assert handoff["role"] == "implementer"
    assert handoff["source_changes_allowed"] is True
    assert handoff["allowed_paths"] == [s.path]
    s.implement()
    assert s.service.handoff(s.candidate_id)["role"] == "reviewer"
    s.approve_code()
    assert s.service.handoff(s.candidate_id)["role"] == "validator"
    result = s.validate()
    assert result["data"]["stage"] == "validated"
    assert result["data"]["validation"]["eligible_for_promotion"] is True
    skill = s.skill()
    assert skill["data"]["tier"] == "journal"
    assert skill["data"]["outcome"] == "pass"
    assert s.promote()["data"]["tier"] == "staging"
    with pytest.raises(GateError, match="two independently"):
        s.promote(tier="hub")
    with pytest.raises(GateError, match="No executable handoff"):
        s.service.handoff(s.candidate_id)
    actions = [event["action"] for event in s.service.store.audit(s.candidate_id)]
    assert actions == [
        "discovered",
        "confirm",
        "approve_plan",
        "record_implementation",
        "approve_code",
        "validate",
    ]
    sha = result["data"]["validation"]["report_evidence"]
    with s.service.store.transaction() as db:
        content = db.execute("SELECT content FROM evidence WHERE sha256=?", (sha,)).fetchone()[0]
    assert digest(json.loads(content)) == sha


def test_hub_requires_two_curated_contexts_of_the_same_pattern(tmp_path, git_bin):
    first = Scenario(tmp_path, git_bin, name="first").ready()
    first.validate()
    first.promote()
    second = Scenario(tmp_path, git_bin, name="second", service=first.service).ready()
    second.validate()
    with pytest.raises(GateError, match="two independently"):
        first.promote(tier="hub")
    second.promote()
    hub = first.promote(tier="hub")
    assert hub["data"]["tier"] == "hub"
    assert set(hub["data"]["replication_skill_ids"]) == {first.skill()["id"], second.skill()["id"]}


@pytest.mark.parametrize("hardware,allow_synthetic", [(False, True), (True, True), (False, False)])
def test_simulation_never_validates_or_promotes(scenario, hardware, allow_synthetic):
    s = scenario.ready()
    result = s.validate(
        ABReport.model_validate(s.report_data(hardware=hardware)), allow_synthetic=allow_synthetic
    )
    assert result["data"]["stage"] == "code_approved"
    assert result["data"]["validation"]["eligible_for_promotion"] is False
    assert s.skill()["data"]["tier"] == "journal"
    with pytest.raises(GateError, match="real passing"):
        s.promote()
    with pytest.raises(GateError, match="No executable handoff"):
        s.service.handoff(s.candidate_id)


@pytest.mark.parametrize("values,expected", [((110.0,) * 3, "fail"), ((99.5,) * 3, "inconclusive")])
def test_failed_and_inconclusive_attempts_are_journaled_and_cannot_retry(
    scenario, values, expected
):
    s = scenario.ready()
    result = s.validate(ABReport.model_validate(s.report_data(values=values)))
    assert result["data"]["validation"]["result"]["verdict"] == expected
    assert s.skill()["data"]["outcome"] == expected
    with pytest.raises(GateError, match="No executable handoff"):
        s.service.handoff(s.candidate_id)
    with pytest.raises(GateError, match="previously finalized"):
        s.validate()
    with pytest.raises(GateError):
        s.promote()


def test_owner_rejection_is_durable_negative_knowledge_and_suppresses_rediscovery(scenario):
    s = scenario
    s.transition(
        "reject", "owner", {"note": "Existing ownership semantics make this optimization unsafe."}
    )
    assert s.skill()["data"]["signal"] == "expert_decision"
    assert s.skill()["data"]["eligible_for_promotion"] is False
    found = s.service.scan(s.repo, owners={"**/*.c": "owner"})
    assert found["candidates"] == []
    assert found["suppressed_prior_outcomes"] == 1
    with pytest.raises(GateError):
        s.service.handoff(s.candidate_id)


@pytest.mark.parametrize(
    "gate", ["confirm", "plan", "implementation", "code", "validation", "promotion"]
)
def test_stage_owners_and_reviewers_are_independent(scenario, gate):
    s = scenario
    if gate == "confirm":
        with pytest.raises(GateError, match="assigned owner"):
            s.transition("confirm", "someone-else", {"note": "Attempted owner confirmation."})
        return
    s.confirm()
    if gate == "plan":
        payload = s.plan_payload()
        payload["review"]["author"] = "architect"
        with pytest.raises(GateError, match="independent reviewer"):
            s.transition("approve_plan", "architect", payload)
        return
    s.approve_plan()
    if gate == "implementation":
        with pytest.raises(GateError, match="plan reviewer cannot implement"):
            s.transition("record_implementation", "plan-reviewer", {"revision": s.base})
        return
    s.implement()
    if gate == "code":
        review = s.review(s.row["data"]["implementation_digest"], "implementer")
        with pytest.raises(GateError, match="independent reviewer"):
            s.transition("approve_code", "implementer", review)
        return
    s.approve_code()
    if gate == "validation":
        with pytest.raises(GateError, match="independent validator"):
            s.validate(actor="implementer")
        return
    s.validate()
    for actor in ("implementer", "validator"):
        with pytest.raises(GateError, match="independent curator"):
            s.promote(actor=actor)


@pytest.mark.parametrize("mutation", ["digest", "candidate", "author", "decision", "base", "scope"])
def test_plan_review_is_bound_to_exact_plan_subject_and_actor(scenario, mutation):
    s = scenario
    s.confirm()
    payload = s.plan_payload()
    if mutation == "digest":
        payload["plan"]["hypothesis"] = "A changed plan requires a fresh independent review."
    elif mutation == "candidate":
        payload["review"]["candidate_id"] = "different-candidate"
    elif mutation == "author":
        payload["review"]["author"] = "different-reviewer"
    elif mutation == "decision":
        payload["review"]["decision"] = "reject"
    elif mutation == "base":
        payload["plan"]["base_revision"] = "f" * 40
    else:
        payload["plan"]["allowed_paths"] = ["unrelated.txt"]
    before = s.service.store.audit(s.candidate_id)
    with pytest.raises(GateError):
        s.transition("approve_plan", "plan-reviewer", payload)
    assert s.service.store.audit(s.candidate_id) == before
    assert s.service.store.read("candidate", s.candidate_id)["data"]["stage"] == "confirmed"


def test_implementation_cannot_expand_approved_file_scope(scenario):
    s = scenario
    s.confirm()
    s.approve_plan()
    with pytest.raises(GateError, match="outside the approved scope"):
        s.implement(extra_file=True)
    assert s.service.store.read("candidate", s.candidate_id)["data"]["stage"] == "plan_approved"


def test_implementation_must_descend_from_baseline(scenario):
    s = scenario
    s.confirm()
    s.approve_plan()
    tree = s.git("rev-parse", "HEAD^{tree}").strip()
    orphan = s.git("commit-tree", tree, "-m", "unrelated root history").strip()
    with pytest.raises(GateError):
        s.transition("record_implementation", "implementer", {"revision": orphan})


def test_implementation_cannot_reuse_unchanged_baseline(scenario):
    s = scenario
    s.confirm()
    s.approve_plan()
    with pytest.raises(GateError, match="change the approved baseline"):
        s.transition("record_implementation", "implementer", {"revision": s.base})


def test_code_review_binds_implementation_digest(scenario):
    s = scenario
    s.confirm()
    s.approve_plan()
    s.implement()
    with pytest.raises(GateError, match="stale"):
        s.transition("approve_code", "code-reviewer", s.review("f" * 64, "code-reviewer"))


@pytest.mark.parametrize(
    "mutation",
    [
        "candidate",
        "baseline",
        "implementation",
        "metric",
        "pairs",
        "device_id",
        "workload_id",
        "workload_config_sha256",
        "environment_sha256",
    ],
)
def test_validation_cannot_substitute_revision_context_or_policy(scenario, mutation):
    s = scenario.ready()
    report = deepcopy(s.report_data())
    if mutation == "candidate":
        report["candidate_id"] = "unrelated-candidate"
    elif mutation == "baseline":
        report["baseline"]["repo_revision"] = "f" * 40
    elif mutation == "implementation":
        report["implementation_revision"] = "f" * 40
        report["candidate"]["repo_revision"] = "f" * 40
    elif mutation == "metric":
        report["metrics"][0]["min_improvement_pct"] = 0.0
    elif mutation == "pairs":
        report["minimum_pairs"] = 4
    else:
        for arm in ("baseline", "candidate"):
            report[arm][mutation] = "f" * 64 if mutation.endswith("sha256") else "other-context"
    before = s.service.store.audit(s.candidate_id)
    with pytest.raises(GateError):
        s.validate(ABReport.model_validate(report))
    assert s.service.store.audit(s.candidate_id) == before
    assert s.service.store.list("skill") == []


@pytest.mark.parametrize(
    "when,dirty", [("confirm", False), ("confirm", True), ("plan", False), ("plan", True)]
)
def test_stale_head_or_dirty_worktree_blocks_approval(scenario, when, dirty):
    s = scenario
    if when == "plan":
        s.confirm()
    (s.repo / "unrelated.txt").write_text("changed outside workflow\n", encoding="utf-8")
    if not dirty:
        s.commit("unrelated change")
    with pytest.raises(GateError, match="HEAD changed|tracked changes"):
        s.confirm() if when == "confirm" else s.approve_plan()


def test_repeat_requests_replay_exactly_without_duplicate_events_or_evidence(scenario):
    s = scenario
    params = {
        "actor": "owner",
        "expected_version": s.row["version"],
        "request_id": "confirm-once",
        "payload": {"note": "Hot path and ownership confirmed."},
    }
    first = s.service.transition(s.candidate_id, "confirm", **params)
    with s.service.store.transaction() as db:
        counts = {
            table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("records", "evidence", "events", "requests")
        }
    assert s.service.transition(s.candidate_id, "confirm", **params) == first
    with s.service.store.transaction() as db:
        after = {
            table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in counts
        }
    assert counts == after
    with pytest.raises(ConflictError, match="different arguments"):
        s.service.transition(
            s.candidate_id,
            "confirm",
            **{**params, "payload": {"note": "This reused request now has different content."}},
        )


def test_validation_and_promotion_replays_survive_restart(scenario):
    s = scenario.ready()
    report = ABReport.model_validate(s.report_data())
    params = {
        "actor": "validator",
        "expected_version": s.row["version"],
        "request_id": "validate-once",
    }
    first = s.service.validate(s.candidate_id, report, **params)
    s.service = EvolutionService(s.service.store.root, git_bin=s.git_bin)
    assert s.service.validate(s.candidate_id, report, **params) == first
    assert len(s.service.store.list("skill")) == 1
    skill = s.skill()
    promotion = {
        "tier": "staging",
        "actor": "knowledge-curator",
        "expected_version": skill["version"],
        "request_id": "promote-once",
        "note": "Independent curator checked evidence quality.",
    }
    first_promotion = s.service.promote(skill["id"], **promotion)
    assert s.service.promote(skill["id"], **promotion) == first_promotion
    assert len(s.service.store.audit(skill["id"])) == 1


def test_stale_optimistic_version_never_applies_a_second_decision(scenario):
    s = scenario
    stale = s.row["version"]
    s.confirm()
    with pytest.raises(ConflictError, match="version changed"):
        s.transition(
            "reject",
            "owner",
            {"note": "A concurrent stale decision should not apply."},
            expected_version=stale,
        )
    assert s.service.store.read("candidate", s.candidate_id)["data"]["stage"] == "confirmed"


def test_restart_preserves_stage_and_handoff_contract(scenario):
    s = scenario
    s.confirm()
    s.approve_plan()
    before = s.service.handoff(s.candidate_id)
    restarted = EvolutionService(s.service.store.root, git_bin=s.git_bin)
    assert restarted.handoff(s.candidate_id) == before
    assert restarted.store.read("candidate", s.candidate_id) == s.row


def test_untrusted_payload_cannot_choose_stage_owner_or_permission(scenario):
    s = scenario
    s.transition(
        "confirm",
        "owner",
        {
            "note": "Confirmed applicability after actual review.",
            "stage": "validated",
            "owner": "attacker",
            "eligible_for_promotion": True,
            "source_changes_allowed": True,
        },
    )
    assert s.row["data"]["stage"] == "confirmed"
    assert s.row["data"]["candidate"]["owner"] == "owner"
    assert "eligible_for_promotion" not in s.row["data"]
    assert s.service.handoff(s.candidate_id)["source_changes_allowed"] is False


@pytest.mark.parametrize(
    "path", ["../outside.c", "/absolute.c", "C:/outside.c", "*.c", ".git/config"]
)
def test_approved_paths_must_be_exact_repository_files(scenario, path):
    payload = scenario.plan_payload()["plan"]
    payload["allowed_paths"] = [path]
    with pytest.raises(ValidationError):
        Plan.model_validate(payload)


def test_unsupported_actions_and_oversized_collection_parameters_are_rejected(scenario):
    s = scenario
    with pytest.raises(GateError, match="Unknown transition"):
        s.transition("set_validated", "owner", {"stage": "validated"})
    with pytest.raises(ValueError):
        s.service.scan(s.repo, owners={}, top_k=1001)
    with pytest.raises(ValueError):
        s.service.recall("lookup", limit=5)
    with pytest.raises(ValueError):
        s.service.store.list("candidate", limit=1001)
    with pytest.raises(ValueError):
        s.service.store.audit(s.candidate_id, limit=1001)


def test_store_rolls_back_failed_transactions_and_keeps_immutable_history(tmp_path):
    store = EvolutionStore(tmp_path / "store")
    with pytest.raises(RuntimeError), store.transaction() as db:
        store.put(db, "history", "one", {"proof": "uncommitted"})
        store.event(db, "one", "attempt", "actor", {})
        raise RuntimeError("abort")
    assert store.list("history") == []
    assert store.audit("one") == []
    with store.transaction() as db:
        store.put(db, "history", "one", {"proof": "original"})
    with pytest.raises(ConflictError, match="Immutable record"), store.transaction() as db:
        store.put(db, "history", "one", {"proof": "changed"})
    assert store.read("history", "one")["data"] == {"proof": "original"}


def test_git_replacement_objects_cannot_hide_out_of_scope_implementation(scenario):
    s = scenario
    s.confirm()
    s.approve_plan()
    (s.repo / s.path).write_text(
        "int target(int x) { return cached_lookup(x); }\n", encoding="utf-8"
    )
    (s.repo / "unrelated.txt").write_text("unauthorized change\n", encoding="utf-8")
    actual = s.commit("actual implementation includes unauthorized scope")
    s.git("checkout", "--detach", s.base)
    (s.repo / s.path).write_text(
        "int target(int x) { return cached_lookup(x); }\n", encoding="utf-8"
    )
    replacement = s.commit("replacement has only approved scope")
    s.git("replace", actual, replacement)
    with pytest.raises(GateError, match="outside the approved scope"):
        s.transition("record_implementation", "implementer", {"revision": actual})
    assert s.service.store.read("candidate", s.candidate_id)["data"]["stage"] == "plan_approved"


def test_service_git_reads_ignore_inherited_repository_location_overrides(scenario, monkeypatch):
    s = scenario
    monkeypatch.setenv("GIT_DIR", str(s.repo / "nonexistent-hostile-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(s.repo / "different-tree"))
    assert s.service._revision(s.repo, "HEAD") == s.base


def history_fixture(source_id, revision):
    return ChangeRecord(
        source_id=source_id,
        repo_id="history-fixture",
        revision=revision,
        subject="perf remove redundant lookup",
        message="Reviewed historical optimization.",
        paths=["kernel.c"],
        patch="-redundant_lookup(x);\n+cached_lookup(x);\n",
    )


def test_repeated_historical_literal_keeps_both_sources_without_overwriting_history(tmp_path):
    service = EvolutionService(tmp_path / "history")
    first = history_fixture("source-one", "1" * 40)
    second = history_fixture("source-two", "2" * 40)
    service.ingest_history([first])
    service.ingest_history([second])
    assert len(service.store.list("history")) == 2
    assert service.store.list("pattern") == []
    jobs = service.store.list("history_analysis")
    assert {row["id"] for row in jobs} == {"source-one", "source-two"}
    assert all(row["data"]["status"] == "pending" for row in jobs)
    assert service.store.read("history", "source-one")["data"] == first.model_dump(mode="json")


def test_reimporting_history_preserves_curated_pattern_and_avoids_duplicates(tmp_path):
    service = EvolutionService(tmp_path / "history")
    change = history_fixture("source-one", "1" * 40)
    service.ingest_history([change])
    pattern = Pattern(
        pattern_id="manual-curation",
        title="Curated repeated lookup",
        kind="optimization",
        problem="A repeated stable lookup may be cached.",
        diagnosis="Verify lookup stability.",
        remedy="Cache only within the stable lifetime.",
        matcher=Matcher(file_globs=["**/*.c"], all_of=["redundant_lookup(x)"]),
        primary_metric="instructions",
        preconditions=["The lookup remains stable."],
        risks=["Lifetime may change."],
        source_ids=[change.source_id],
    )
    pattern_key = service.import_pattern(pattern)["id"]
    activated = service.activate_pattern(
        pattern_key, actor="curator", note="Reviewed applicability and source evidence."
    )
    service.ingest_history([change])
    assert service.store.read("pattern", pattern_key) == activated
    assert len(service.store.list("pattern")) == 1
    assert len(service.store.list("history")) == 1
    assert len(service.store.list("history_analysis")) == 1


def test_negative_outcome_is_not_forgotten_after_an_unrelated_commit(scenario):
    s = scenario
    s.transition(
        "reject", "owner", {"note": "Reject repeated lookup idea because lifetime is unstable."}
    )
    s.git("commit", "--allow-empty", "-q", "-m", "unrelated metadata-only commit")
    result = s.service.scan(s.repo, owners={"**/*.c": "owner"})
    assert result["candidates"] == []
    assert result["suppressed_prior_outcomes"] == 1


def test_rescan_refreshes_unconfirmed_owner_and_invalidates_stale_owner_decision(scenario):
    s = scenario
    old_version = s.row["version"]
    result = s.service.scan(
        s.repo,
        owners={"**/*.c": "new-owner"},
        hotspots=[Hotspot(path=s.path, weight=0.9, revision=s.base)],
    )
    refreshed = result["candidates"][0]
    assert refreshed["id"] == s.candidate_id
    assert refreshed["data"]["candidate"]["owner"] == "new-owner"
    assert refreshed["version"] > old_version
    with pytest.raises(ConflictError):
        s.transition(
            "confirm",
            "owner",
            {"note": "Stale old-owner confirmation must not apply."},
            expected_version=old_version,
        )


def test_rescan_refreshes_unconfirmed_hotspot_ranking_and_lane(scenario):
    s = scenario
    original = s.row["data"]["candidate"]
    assert original["lane"] == "pipeline"
    result = s.service.scan(s.repo, owners={"**/*.c": "owner"}, hotspots=[])
    refreshed = result["candidates"][0]["data"]["candidate"]
    assert refreshed["lane"] == "workbench"
    assert refreshed["score"] < original["score"]


def test_history_batches_cannot_leave_patterns_beyond_schema_provenance_budget(tmp_path):
    service = EvolutionService(tmp_path / "history")
    first = [history_fixture(f"source-{index}", f"{index + 1:040x}") for index in range(1000)]
    service.ingest_history(first)
    before = service.store.list("pattern")
    overflow = [
        history_fixture(f"source-{index}", f"{index + 1:040x}") for index in range(1000, 1025)
    ]
    try:
        service.ingest_history(overflow)
    except ValueError:
        assert service.store.list("pattern") == before
        with service.store.transaction() as db:
            assert (
                db.execute("SELECT count(*) FROM records WHERE kind='history'").fetchone()[0]
                == 1000
            )
    for row in service.store.list("pattern"):
        Pattern.model_validate(row["data"]["pattern"])


def test_rename_cannot_hide_deletion_of_an_unapproved_file(scenario):
    s = scenario
    s.confirm()
    payload = s.plan_payload()
    payload["plan"]["allowed_paths"].append("approved.txt")
    payload["review"] = s.review(digest(payload["plan"]), "plan-reviewer")
    s.transition("approve_plan", "plan-reviewer", payload)
    (s.repo / s.path).write_text(
        "int target(int x) { return cached_lookup(x); }\n", encoding="utf-8"
    )
    s.git("mv", "unrelated.txt", "approved.txt")
    implementation = s.commit("rename out-of-scope source into approved destination")
    with pytest.raises(GateError, match="outside the approved scope"):
        s.transition("record_implementation", "implementer", {"revision": implementation})
