"""Skill-driven synthesis, independent pattern review and candidate applicability evidence."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .change_analysis import PatternProposal, _matches_text
from .methods import skill_snapshot, verify_citation, verify_snapshot
from .mining import Pattern, _matches
from .store import ConflictError, digest


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class Citation(Strict):
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    line_start: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=4000)

    # Code indentation is evidence, not prose whitespace.
    model_config = ConfigDict(extra="forbid", strict=True)


class Check(Strict):
    criterion_id: str = Field(min_length=1, max_length=128)
    result: Literal["met", "violated", "unknown"]
    explanation: str = Field(min_length=10, max_length=4000)
    citations: list[Citation] = Field(default_factory=list, max_length=16)


class Assessment(Strict):
    result: Literal["applicable", "not_applicable", "needs_context"]
    summary: str = Field(min_length=10, max_length=8000)
    checks: list[Check] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def verdict(self):
        values = {check.result for check in self.checks}
        expected = (
            "not_applicable"
            if "violated" in values
            else "needs_context"
            if "unknown" in values
            else "applicable"
        )
        if self.result != expected:
            raise ValueError("Assessment result must follow its individual condition verdicts")
        return self


class Exemplar(Strict):
    source_id: str = Field(min_length=1, max_length=200)
    path: str = Field(min_length=1, max_length=2000)


class SynthesisGroup(Strict):
    proposal: PatternProposal
    exemplars: list[Exemplar] = Field(min_length=2, max_length=8)
    differences_and_counterarguments: str = Field(min_length=10, max_length=8000)


class Synthesis(Strict):
    summary: str = Field(min_length=10, max_length=8000)
    groups: list[SynthesisGroup] = Field(max_length=8)


class PatternReview(Strict):
    decision: Literal["approve", "revise", "reject"]
    rationale: str = Field(min_length=10, max_length=8000)


MODELS = {"synthesize": Synthesis, "assess": Assessment, "review": PatternReview}
SKILLS = {
    "synthesize": "pattern-synthesis",
    "assess": "candidate-assessment",
    "review": "pattern-synthesis",
}


def pattern_digest(data):
    return digest(
        {
            key: value
            for key, value in data.items()
            if key not in {"status", "review", "curator", "activation_note"}
        }
    )


def criteria(pattern):
    result = [
        {
            "id": "mechanism",
            "assertion": "This exact target exhibits: " + pattern["pattern"]["problem"],
        }
    ]
    result += [
        {"id": f"precondition:{i}", "assertion": text}
        for i, text in enumerate(pattern["pattern"]["preconditions"])
    ]
    result += [
        {"id": f"exclusion:{i}", "assertion": "The target is NOT an instance of: " + text}
        for i, text in enumerate(pattern.get("negative_examples", []))
    ]
    return result


def prepare_research(service, repo, kind, subject_ids, *, actor):
    actor = service._actor(actor)
    if (
        kind not in MODELS
        or not 1 <= len(subject_ids) <= 8
        or len(set(subject_ids)) != len(subject_ids)
    ):
        raise ValueError("Select a supported method and one to eight distinct subject IDs")
    from pathlib import Path

    repo_id = service.repo_identity(repo)
    inputs = {}
    if kind == "synthesize":
        if len(subject_ids) < 2:
            raise ValueError("Synthesis requires at least two analyzed historical sources")
        for source_id in subject_ids:
            row = service.store.read("history_analysis", source_id)
            if row["data"]["repo_id"] != repo_id or row["data"]["status"] not in {
                "patterns",
                "no_pattern",
            }:
                raise ValueError("Synthesis requires complete analyses from this repository")
            inputs[source_id] = {
                "analysis_sha256": row["data"]["analysis_sha256"],
                "packet_sha256": row["data"]["packet_sha256"],
            }
    elif kind == "assess":
        if len(subject_ids) != 1:
            raise ValueError("Assess one candidate at a time")
        row = service.store.read("candidate", subject_ids[0])
        candidate = row["data"]["candidate"]
        if (
            Path(candidate["repo_path"]).resolve() != Path(repo).resolve()
            or row["data"]["stage"] != "discovered"
        ):
            raise ValueError("Assess a discovered candidate from this profile")
        pattern = service.store.read(
            "pattern", f"{candidate['pattern_id']}@{candidate['pattern_version']}"
        )
        if pattern["data"]["status"] != "active":
            raise ValueError("Assess candidates against an active pattern")
        inputs = {"candidate": row, "pattern": pattern, "criteria": criteria(pattern["data"])}
    else:
        if len(subject_ids) != 1:
            raise ValueError("Review one draft pattern at a time")
        row = service.store.read("pattern", subject_ids[0])
        if row["data"]["status"] != "draft":
            raise ValueError("Review a draft pattern")
        for source_id in row["data"]["pattern"]["source_ids"]:
            if service.store.read("history", source_id)["data"]["repo_id"] != repo_id:
                raise ValueError("Pattern sources must belong to the profile repository")
        inputs = {"pattern": row, "subject_digest": pattern_digest(row["data"])}
    method = skill_snapshot(service, [SKILLS[kind]])
    data = {
        "kind": kind,
        "repo_id": repo_id,
        "subject_ids": subject_ids,
        "inputs": inputs,
        "method_sha256": method["sha256"],
        "prepared_by": actor,
        "status": "pending",
    }
    if kind == "assess":
        data["candidate_id"] = subject_ids[0]
    identity = "research_" + digest(data)
    with service.store.transaction() as db:
        existing = db.execute(
            "SELECT 1 FROM records WHERE kind='research' AND id=?", (identity,)
        ).fetchone()
        row = (
            service.store.get(db, "research", identity)
            if existing
            else service.store.put(db, "research", identity, data)
        )
    return {
        "research": row,
        "method": method,
        "submission_schema": MODELS[kind].model_json_schema(),
    }


def submit_research(service, research_id, payload, *, actor, expected_version, request_id):
    actor = service._actor(actor)
    if type(expected_version) is not int or expected_version < 1:
        raise ValueError("expected_version must be a positive integer")
    row = service.store.read("research", research_id)
    report = MODELS[row["data"]["kind"]].model_validate(payload).model_dump(mode="json")
    request = {
        "action": "submit_research",
        "research_id": research_id,
        "report": report,
        "actor": actor,
        "expected_version": expected_version,
    }
    verify_snapshot(service, row["data"]["method_sha256"])
    # Verify source evidence before taking the write transaction.
    if row["data"]["kind"] == "assess":
        candidate = row["data"]["inputs"]["candidate"]["data"]["candidate"]
        for check in report["checks"]:
            if check["result"] != "unknown" and not check["citations"]:
                raise ValueError("Decisive applicability checks require immutable source citations")
            for citation in check["citations"]:
                verify_citation(
                    service,
                    citation,
                    repo_id=row["data"]["repo_id"],
                    revision=candidate["repo_revision"],
                )
            if check["criterion_id"] == "mechanism" and check["result"] == "met":
                contexts = [
                    (c, service.store.read_evidence(c["context_sha256"]))
                    for c in check["citations"]
                ]
                if not any(
                    ctx["path"] == candidate["path"]
                    and c["line_start"]
                    <= candidate["line_start"]
                    < c["line_start"] + len(c["quote"].splitlines())
                    for c, ctx in contexts
                ):
                    raise ValueError("Mechanism evidence must cover the actual candidate location")
    packets = {}
    if row["data"]["kind"] == "synthesize":
        packets = {
            source: service.store.read_evidence(value["packet_sha256"])
            for source, value in row["data"]["inputs"].items()
        }
        for group in report["groups"]:
            proposal = PatternProposal.model_validate(group["proposal"])
            examples = group["exemplars"]
            if proposal.exemplar_path != examples[0]["path"]:
                raise ValueError("Primary exemplar must be the first selected example")
            analysis = service.store.read_evidence(
                row["data"]["inputs"].get(examples[0]["source_id"], {}).get("analysis_sha256", "")
            )
            for index in proposal.finding_indexes:
                if (
                    index >= len(analysis["findings"])
                    or proposal.exemplar_path not in analysis["findings"][index]["paths"]
                ):
                    raise ValueError("Generalized proposal must refer to primary exemplar findings")
            if len({e["source_id"] for e in examples}) < 2:
                raise ValueError("Use at least two distinct historical sources")
            fingerprints = set()
            for example in examples:
                packet = packets.get(example["source_id"])
                file = next(
                    (f for f in (packet or {}).get("files", []) if f["path"] == example["path"]),
                    None,
                )
                if not packet or not packet["coverage_complete"] or not file:
                    raise ValueError("Exemplar is not available in the selected complete evidence")
                before, after = file["before"].get("content", ""), file["after"].get("content", "")
                if (
                    not any(_matches(example["path"], g) for g in proposal.matcher.file_globs)
                    or not _matches_text(proposal.matcher, before)
                    or _matches_text(proposal.matcher, after)
                ):
                    raise ValueError("Generalized matcher must pass every before/after exemplar")
                fingerprints.add(digest([before, after]))
            if len(fingerprints) < 2:
                raise ValueError(
                    "Identical changes are duplicate evidence, not independent examples"
                )
    with service.store.transaction() as db:
        replay = service.store.replay(db, request_id, request)
        if replay is not None:
            return replay
        current = service.store.get(db, "research", research_id)
        if current["version"] != expected_version or current["data"]["status"] != "pending":
            raise ConflictError("Research run is stale or already complete")
        data = deepcopy(current["data"])
        report_sha = service.store.evidence(db, report)
        outputs = []
        if data["kind"] == "assess":
            original = data["inputs"]["candidate"]
            candidate = service.store.get(db, "candidate", original["id"])
            if candidate != original:
                raise ConflictError("Candidate changed during applicability investigation")
            original_pattern = data["inputs"]["pattern"]
            if service.store.get(db, "pattern", original_pattern["id"]) != original_pattern:
                raise ConflictError("Pattern changed during applicability investigation")
            required = {c["id"] for c in data["inputs"]["criteria"]}
            supplied = [check["criterion_id"] for check in report["checks"]]
            if set(supplied) != required or len(supplied) != len(required):
                raise ValueError("Assess every criterion exactly once")
            value = deepcopy(candidate["data"])
            value["assessment"] = {
                "research_id": research_id,
                "result": report["result"],
                "subject_digest": digest(value["candidate"]),
                "pattern_digest": pattern_digest(original_pattern["data"]),
                "report_sha256": report_sha,
                "method_sha256": data["method_sha256"],
                "actor": actor,
            }
            service.store.put(db, "candidate", candidate["id"], value, candidate["version"])
            service.store.event(
                db, candidate["id"], "assess_applicability", actor, value["assessment"]
            )
        elif data["kind"] == "review":
            original = data["inputs"]["pattern"]
            pattern = service.store.get(db, "pattern", original["id"])
            if pattern != original:
                raise ConflictError("Pattern changed during independent review")
            if actor == pattern["data"].get("created_by"):
                raise ValueError("Pattern reviewer must differ from its synthesizer")
            value = deepcopy(pattern["data"])
            value["review"] = {
                "decision": report["decision"],
                "actor": actor,
                "subject_digest": data["inputs"]["subject_digest"],
                "report_sha256": report_sha,
                "method_sha256": data["method_sha256"],
            }
            service.store.put(db, "pattern", pattern["id"], value, pattern["version"])
        else:
            for group in report["groups"]:
                proposal = PatternProposal.model_validate(group["proposal"])
                sources = sorted({e["source_id"] for e in group["exemplars"]})
                identity = "pattern_" + digest([research_id, group])
                fields = proposal.model_dump(
                    exclude={
                        "mechanism",
                        "exemplar_path",
                        "finding_indexes",
                        "negative_examples",
                        "validation_plan",
                        "metric_rationale",
                    }
                )
                fields["risks"] = fields["risks"] + [
                    "SEMANTIC_REVIEW: each target requires cited applicability assessment."
                ]
                pattern = Pattern(pattern_id=identity, source_ids=sources, **fields)
                key = f"{identity}@1"
                service.store.put(
                    db,
                    "pattern",
                    key,
                    {
                        "pattern": pattern.model_dump(mode="json"),
                        "status": "draft",
                        "created_by": actor,
                        "research_id": research_id,
                        "report_sha256": report_sha,
                        "method_sha256": data["method_sha256"],
                        "requires_assessment": True,
                        "review_required": True,
                        "mechanism": proposal.mechanism,
                        "negative_examples": proposal.negative_examples,
                        "validation_plan": proposal.validation_plan,
                        "metric_rationale": proposal.metric_rationale,
                        "source_analyses": {source: data["inputs"][source] for source in sources},
                    },
                )
                outputs.append(key)
        data.update(
            status="completed", report_sha256=report_sha, submitted_by=actor, output_ids=outputs
        )
        result = service.store.put(db, "research", research_id, data, expected_version)
        service.store.event(
            db,
            research_id,
            "submit_research",
            actor,
            {"report_sha256": report_sha, "output_ids": outputs},
        )
        service.store.remember(db, request_id, request, result)
    return result
