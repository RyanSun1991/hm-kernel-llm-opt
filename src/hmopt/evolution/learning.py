"""Evidence-based screening quality, explicit overlays, and sanitized review export.

Owner acceptance is a workflow selection rate, not classifier precision. Validation
is neither a merge nor publication. Export creates an adapter-neutral review
manifest, not an installable native Hub package. Hardware flags remain trusted
operator attestations; this module does not execute or authenticate device tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .correctness import CorrectnessReport, evaluate_correctness
from .store import ConflictError, canonical_json, digest
from .validation import ABReport, evaluate_ab

if TYPE_CHECKING:
    from .service import EvolutionService

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_VERDICTS = ("pass", "fail", "inconclusive")
_MAX_EXPORT = 1000
_CHECKLIST = """# Adapter-neutral evidence review checklist

This bundle is a local review manifest, not a native Hub package. Opaque hashes
alone do not install reusable skills. This export does not merge or publish changes.

- Resolve the referenced evidence in the originating trusted local store.
- Review the optimization context, independent reviews, and measurement limits.
- Prepare useful, explicitly approved shareable instructions and applicability constraints.
- Remove proprietary source, private paths, identities, credentials, and device details.
- Map the reviewed content to the actual destination Hub schema once it is available.
- Obtain the destination's normal publication approval before any network submission.

No source text or unreviewed recipes are included in this export.
"""


def _actor(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError("actor must be a nonempty string of at most 200 characters")
    return value.strip()


def _read_evidence(db: Any, sha: str) -> Any:
    if not isinstance(sha, str) or not _HASH.fullmatch(sha):
        raise ValueError("A full SHA-256 evidence reference is required")
    row = db.execute("SELECT content FROM evidence WHERE sha256=?", (sha,)).fetchone()
    if row is None:
        raise ValueError("Referenced evidence is missing")
    content = json.loads(row["content"])
    if digest(content) != sha:
        raise ConflictError("Referenced evidence failed its digest integrity check")
    return content


def _source_references(db: Any, attempt: dict) -> list[str]:
    """Verify only provenance frozen at validation, never a later conversion index."""
    sources = attempt.get("source_evidence", [])
    if (
        not isinstance(sources, list)
        or len(sources) > 32
        or any(not isinstance(sha, str) or not _HASH.fullmatch(sha) for sha in sources)
        or len(set(sources)) != len(sources)
    ):
        raise ValueError(
            "Validation source evidence must be a bounded list of unique SHA-256 references"
        )
    for sha in sources:
        _read_evidence(db, sha)
    return sources


def _attempt(db: Any, data: dict, attempt: Any) -> tuple[str, ABReport | CorrectnessReport | None]:
    """Classify a persisted attempt without trusting its cached verdict alone."""
    if not isinstance(attempt, dict):
        return "invalid", None
    if type(attempt.get("simulation", False)) is not bool:
        return "invalid", None
    try:
        sources = _source_references(db, attempt)
        if attempt.get("simulation") is True:
            return "simulation", None
        candidate, implementation, policy = (
            data["candidate"],
            data["implementation"],
            data["plan"]["validation"],
        )
        correctness = policy.get("kind") == "correctness"
        if not correctness:
            from .reports import verify_measurement_profile

            verify_measurement_profile(db, sources, policy)
        if attempt.get("kind", "performance") != ("correctness" if correctness else "performance"):
            return "invalid", None
        report_type = CorrectnessReport if correctness else ABReport
        report = report_type.model_validate(_read_evidence(db, attempt["report_evidence"]))
        expected_execution = report.execution_kind if correctness else "hardware"
        if attempt.get("execution_kind", expected_execution) != expected_execution:
            return "invalid", None
        if correctness:
            if report.execution_kind == "simulation":
                return "simulation", None
            if report.policy.model_dump(mode="json") != policy:
                return "invalid", None
        elif not report.baseline.hardware or not report.candidate.hardware:
            return "simulation", None
        if (
            report.candidate_id != candidate["candidate_id"]
            or report.baseline.repo_revision != candidate["repo_revision"]
            or report.candidate.repo_revision != implementation["revision"]
            or report.implementation_revision != implementation["revision"]
        ):
            return "invalid", None
        if not correctness and (
            report.minimum_pairs != policy["minimum_pairs"]
            or [rule.model_dump(mode="json") for rule in report.metrics] != policy["metrics"]
        ):
            return "invalid", None
        for arm in (report.baseline, report.candidate):
            if any(
                getattr(arm, field) != policy[field]
                for field in (
                    "device_id",
                    "workload_id",
                    "workload_config_sha256",
                    "environment_sha256",
                )
            ):
                return "invalid", None
        result = evaluate_correctness(report) if correctness else evaluate_ab(report)
        if not correctness and not result.hardware_verified:
            return "invalid", None
        if result.verdict != attempt["result"]["verdict"]:
            return "invalid", None
        if attempt["result"].get("hardware_verified") is not result.hardware_verified:
            return "invalid", None
        return result.verdict, report
    except (KeyError, TypeError, ValueError, OverflowError):
        return "invalid", None


def _empty_summary() -> dict:
    return {
        "candidates": 0,
        "owner_confirmed": 0,
        "owner_rejected": 0,
        "owner_undecided": 0,
        "owner_decisions": 0,
        "acceptance_rate": None,
        "true_precision": None,
        "validated_candidates": 0,
        "latest_reports": {key: 0 for key in (*_VERDICTS, "simulation", "invalid", "none")},
        "attempts": {key: 0 for key in (*_VERDICTS, "simulation", "invalid")},
        "attempts_by_execution": {
            key: {verdict: 0 for verdict in _VERDICTS}
            for key in ("performance_hardware", "correctness_local", "correctness_hardware")
        },
        "latest_reports_by_execution": {
            key: {verdict: 0 for verdict in _VERDICTS}
            for key in ("performance_hardware", "correctness_local", "correctness_hardware")
        },
        "distinct_report_evidence": 0,
    }


def _quality(db: Any, pattern_key: str | None = None) -> dict:
    keys = [
        row["id"] for row in db.execute("SELECT id FROM records WHERE kind='pattern' ORDER BY id")
    ]
    if pattern_key is not None:
        if pattern_key not in keys:
            raise ValueError("Unknown pattern version")
        keys = [pattern_key]
    summaries = {key: _empty_summary() for key in keys}
    snapshots = {key: hashlib.sha256() for key in keys}
    report_hashes: dict[str, set[str]] = {key: set() for key in keys}
    for row in db.execute(
        "SELECT id,version,payload FROM records WHERE kind='candidate' ORDER BY id"
    ):
        data = json.loads(row["payload"])
        candidate = data.get("candidate", {})
        key = f"{candidate.get('pattern_id')}@{candidate.get('pattern_version')}"
        if key not in summaries:
            continue
        item = summaries[key]
        item["candidates"] += 1
        snapshots[key].update(
            canonical_json(
                {"id": row["id"], "version": row["version"], "data_digest": digest(data)}
            ).encode("utf-8")
            + b"\n"
        )
        owner = candidate.get("owner")
        rejection, confirmation = data.get("rejection"), data.get("owner_decision")
        if owner and isinstance(rejection, dict) and rejection.get("actor") == owner:
            item["owner_rejected"] += 1
        elif owner and isinstance(confirmation, dict) and confirmation.get("actor") == owner:
            item["owner_confirmed"] += 1
        else:
            item["owner_undecided"] += 1
        history = data.get("validation_history", [])
        attempts = list(history) if isinstance(history, list) else [None]
        if data.get("validation") is not None:
            attempts.append(data["validation"])
        latest = "none"
        execution = None
        for attempt in attempts:
            latest, report = _attempt(db, data, attempt)
            execution = None
            item["attempts"][latest] += 1
            if latest in _VERDICTS:
                report_hashes[key].add(attempt["report_evidence"])
                execution = (
                    "correctness_" + report.execution_kind
                    if isinstance(report, CorrectnessReport)
                    else "performance_hardware"
                )
                item["attempts_by_execution"][execution][latest] += 1
        item["latest_reports"][latest] += 1
        if execution:
            item["latest_reports_by_execution"][execution][latest] += 1
        if latest == "pass":
            item["validated_candidates"] += 1
    for key, item in summaries.items():
        item["owner_decisions"] = item["owner_confirmed"] + item["owner_rejected"]
        if item["owner_decisions"]:
            item["acceptance_rate"] = item["owner_confirmed"] / item["owner_decisions"]
        item["distinct_report_evidence"] = len(report_hashes[key])
        item["candidate_snapshot_sha256"] = snapshots[key].hexdigest()
    report = {
        "schema_version": 1,
        "patterns": summaries,
        "interpretation": {
            "acceptance_rate": "owner-confirmed / owner-decided distinct candidates; final rejection wins",
            "true_precision": "unknown; no independent ground-truth labels are available",
            "validation": "latest eligible report is separate from all recorded attempts",
            "exclusions": "simulation and invalid reports are shown separately and excluded from verdict counts",
            "delivery": "validated is not merged or published",
        },
    }
    report["quality_sha256"] = digest(report)
    return report


def quality_report(service: EvolutionService, pattern_key: str | None = None) -> dict:
    """Summarize candidate facts, not repeated events or unverified captured notes."""
    with service.store.transaction() as db:
        return _quality(db, pattern_key)


def set_pattern_overlay(
    service: EvolutionService,
    pattern_key: str,
    factor: float,
    state: str,
    actor: str,
    note: str,
    request_id: str,
    expected_version: int | None = None,
) -> dict:
    """Set an absolute ranking factor; never recursively multiply or retire patterns."""
    actor = _actor(actor)
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise ValueError("factor must be a finite number between 0 and 1")  # noqa: TRY004
    if not 0 <= factor <= 1 or not math.isfinite(factor):
        raise ValueError("factor must be a finite number between 0 and 1")
    if not isinstance(state, str) or state not in {"active", "probation"}:
        raise ValueError("overlay state must be active or probation")
    if not isinstance(note, str) or not 10 <= len(note.strip()) <= 20000:
        raise ValueError("Explain the overlay in 10..20000 characters")
    if expected_version is not None and (type(expected_version) is not int or expected_version < 1):
        raise ValueError("expected_version must be a positive integer")
    if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 200:
        raise ValueError("request_id must be a nonempty string of at most 200 characters")
    request = {
        "operation": "set_pattern_overlay",
        "pattern_key": pattern_key,
        "factor": float(factor),
        "state": state,
        "actor": actor,
        "note": note.strip(),
        "expected_version": expected_version,
    }
    with service.store.transaction() as db:
        replay = service.store.replay(db, request_id, request)
        if replay is not None:
            return replay
        pattern = service.store.get(db, "pattern", pattern_key)
        exists = db.execute(
            "SELECT 1 FROM records WHERE kind='overlay' AND id=?", (pattern_key,)
        ).fetchone()
        if exists and expected_version is None:
            raise ConflictError("An existing overlay requires its expected_version")
        quality = _quality(db, pattern_key)
        data = {
            "pattern_key": pattern_key,
            "factor": float(factor),
            "state": state,
            "actor": actor,
            "note": note.strip(),
            "quality": quality,
            "quality_sha256": quality["quality_sha256"],
            "pattern_record_version": pattern["version"],
            "pattern_sha256": digest(pattern["data"]),
        }
        result = service.store.put(db, "overlay", pattern_key, data, expected_version)
        service.store.event(
            db,
            pattern_key,
            "set_pattern_overlay",
            actor,
            {"factor": float(factor), "state": state, "quality_sha256": data["quality_sha256"]},
        )
        service.store.remember(db, request_id, request, result)
        return result


def _proposal(service: EvolutionService, db: Any, record: dict) -> dict:
    skill = record["data"]
    if not (
        skill.get("tier") in {"staging", "hub"}
        and skill.get("eligible_for_promotion") is True
        and skill.get("signal") == "validation_result"
        and skill.get("outcome") == "pass"
        and isinstance(skill.get("curator"), str)
        and skill["curator"].strip()
    ):
        raise ValueError("Only curated real passing validation skills are export eligible")
    snapshot = _read_evidence(db, skill.get("evidence"))
    candidate_row = service.store.get(db, "candidate", skill.get("candidate_id"))
    data = candidate_row["data"]
    attempt = data.get("validation")
    verdict, report = _attempt(db, data, attempt)
    if verdict != "pass" or report is None or not attempt.get("eligible_for_promotion"):
        raise ValueError("Export requires a valid, real passing candidate report")
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("validation", {}).get("report_evidence") != attempt["report_evidence"]
    ):
        raise ValueError("Skill evidence does not bind the current validated report")
    sources = _source_references(db, snapshot["validation"])
    if sources != attempt.get("source_evidence", []):
        raise ValueError("Skill evidence does not bind the frozen report source provenance")
    if snapshot.get("candidate", {}).get("candidate_id") != skill["candidate_id"]:
        raise ValueError("Skill evidence is bound to another candidate")
    if (
        snapshot.get("plan_digest") != data.get("plan_digest")
        or snapshot.get("implementation_digest") != data.get("implementation_digest")
        or snapshot.get("validation", {}).get("simulation") is not False
        or snapshot.get("validation", {}).get("eligible_for_promotion") is not True
    ):
        raise ValueError("Curated snapshot does not bind the approved real implementation")
    expected_pattern = f"{data['candidate']['pattern_id']}@{data['candidate']['pattern_version']}"
    if skill.get("pattern_key") != expected_pattern:
        raise ValueError("Skill pattern reference does not match its candidate")
    implementation = data["implementation"]
    if (
        data.get("plan_digest") != digest(data["plan"])
        or data.get("implementation_digest") != digest(implementation)
        or data.get("plan_review", {}).get("subject_digest") != data["plan_digest"]
        or data.get("code_review", {}).get("subject_digest") != data["implementation_digest"]
    ):
        raise ValueError("Plan or implementation review digest does not match")
    patch = _read_evidence(db, implementation.get("patch_evidence"))
    if not isinstance(patch, dict) or not isinstance(patch.get("patch"), str):
        raise ValueError("Implementation patch evidence is malformed")  # noqa: TRY004
    if hashlib.sha256(patch["patch"].encode("utf-8")).hexdigest() != implementation.get(
        "patch_sha256"
    ):
        raise ValueError("Implementation patch does not match its recorded SHA-256")
    if skill["curator"] in {implementation["actor"], attempt["actor"]}:
        raise ValueError("Export requires an independent curator")
    return {
        "schema_version": 1,
        "proposal_id": "proposal-"
        + digest({"skill": record["id"], "version": record["version"]})[:32],
        "skill_reference_sha256": digest({"skill_id": record["id"], "version": record["version"]}),
        "pattern_reference_sha256": digest(skill.get("pattern_key")),
        "candidate_reference_sha256": digest(skill["candidate_id"]),
        "tier": skill["tier"],
        "signal": "validation_result",
        "outcome": "pass",
        "validation_kind": "correctness"
        if isinstance(report, CorrectnessReport)
        else "performance",
        "execution_kind": report.execution_kind
        if isinstance(report, CorrectnessReport)
        else "hardware",
        "metric_count": len(report.metrics) if isinstance(report, ABReport) else 0,
        "check_count": len(report.policy.required_checks)
        if isinstance(report, CorrectnessReport)
        else 0,
        "matched_pairs": len(report.baseline.measurements) if isinstance(report, ABReport) else 0,
        "evidence_sha256": sorted(
            {
                skill["evidence"],
                attempt["report_evidence"],
                implementation["patch_evidence"],
                *sources,
            }
        ),
        "validation_status": "validated",
        "merge_status": "not_asserted",
        "publication_status": "not_published",
        "summary": "Curated evidence passed validation on a declared executor; shareable instructions require review.",
    }


def export_bundle(
    service: EvolutionService,
    output: Path,
    actor: str,
    skill_ids: list[str] | None = None,
) -> dict:
    """Write an exclusive, sanitized local review bundle; never a native Hub package."""
    actor = _actor(actor)
    if skill_ids is not None and (
        not isinstance(skill_ids, list)
        or not 1 <= len(skill_ids) <= _MAX_EXPORT
        or any(
            not isinstance(value, str) or not value.strip() or len(value) > 256
            for value in skill_ids
        )
        or len(set(skill_ids)) != len(skill_ids)
    ):
        raise ValueError("Select 1..1000 unique skill IDs")
    output = Path(output).absolute()
    created: list[Path] = []
    made_directory = False
    try:
        with service.store.transaction() as db:
            if skill_ids is None:
                rows = db.execute(
                    "SELECT id FROM records WHERE kind='skill' "
                    "AND json_extract(payload,'$.eligible_for_promotion')=1 "
                    "AND json_extract(payload,'$.tier') IN ('staging','hub') ORDER BY id LIMIT ?",
                    (_MAX_EXPORT + 1,),
                ).fetchall()
                skill_ids = [row["id"] for row in rows]
            if not skill_ids or len(skill_ids) > _MAX_EXPORT:
                raise ValueError("Export requires 1..1000 eligible curated skills")
            proposals = [
                _proposal(service, db, service.store.get(db, "skill", key))
                for key in sorted(skill_ids)
            ]
            proposal_text = "".join(canonical_json(proposal) + "\n" for proposal in proposals)
            file_hashes = {
                "proposals.jsonl": hashlib.sha256(proposal_text.encode("utf-8")).hexdigest(),
                "review_checklist.md": hashlib.sha256(_CHECKLIST.encode("utf-8")).hexdigest(),
            }
            manifest = {
                "schema_version": 1,
                "format": "hmopt-adapter-neutral-review-manifest",
                "native_hub_package": False,
                "status": "exported",
                "review_required": True,
                "proposal_count": len(proposals),
                "files": file_hashes,
                "published": False,
                "merged": False,
                "exporter_reference_sha256": digest(actor),
            }
            manifest_text = canonical_json(manifest) + "\n"
            bundle_id = "bundle-" + digest(manifest)[:32]
            output.mkdir(parents=False, exist_ok=False)
            made_directory = True
            for name, text in (
                ("proposals.jsonl", proposal_text),
                ("review_checklist.md", _CHECKLIST),
                ("manifest.json", manifest_text),
            ):
                path = output / name
                with path.open("x", encoding="utf-8", newline="\n") as stream:
                    created.append(path)
                    stream.write(text)
            result = service.store.put(
                db,
                "bundle",
                bundle_id,
                {
                    "bundle_id": bundle_id,
                    "status": "exported",
                    "manifest": manifest,
                    "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
                    "proposal_count": len(proposals),
                    "native_hub_package": False,
                },
            )
            service.store.event(
                db,
                bundle_id,
                "export_bundle",
                actor,
                {
                    "proposal_count": len(proposals),
                    "manifest_sha256": result["data"]["manifest_sha256"],
                },
            )
        return result
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        if made_directory:
            output.rmdir()
        raise


__all__ = ["export_bundle", "quality_report", "set_pattern_overlay"]
