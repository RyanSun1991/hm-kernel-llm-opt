"""Strict conversion of existing IC compare reports using an explicit manifest.

The converter never reads a 'previous' result or trusts aggregate PASS text. A
manifest supplies collector-declared provenance; this is not device attestation.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .store import digest
from .validation import ABReport, ExperimentArm, Identifier, Measurement, Sha256


class ICManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1] = 1
    candidate_id: Identifier
    plan_digest: Sha256
    implementation_digest: Sha256
    compare_sha256: Sha256
    metric_name: Identifier
    level: Literal["total", "process", "thread", "lib", "function"]
    target: dict[str, str] = Field(max_length=4)
    baseline_dir: Identifier
    candidate_dir: Identifier
    baseline: ExperimentArm
    candidate: ExperimentArm
    functional_passed: bool


def convert_ic_report(service, candidate_id: str, compare: dict, manifest: ICManifest) -> dict:
    from .service import GateError

    manifest = ICManifest.model_validate(manifest.model_dump(mode="python"))
    if not isinstance(compare, dict) or compare.get("success") is not True:
        raise GateError("IC compare must contain an explicitly successful result")
    if digest(compare) != manifest.compare_sha256:
        raise GateError("IC input digest differs from the collection manifest")
    data = service.store.read("candidate", candidate_id)["data"]
    if data["stage"] != "code_approved" or data.get("validation"):
        raise GateError("IC conversion requires an unsealed code-approved candidate")
    if (
        manifest.candidate_id != candidate_id
        or manifest.plan_digest != data["plan_digest"]
        or manifest.implementation_digest != data["implementation_digest"]
    ):
        raise GateError("Manifest is not bound to the approved plan and implementation")
    policy = data["plan"]["validation"]
    if policy.get("kind") == "correctness":
        raise GateError("IC measurements cannot replace correctness acceptance")
    if len(policy["metrics"]) != 1 or policy["metrics"][0]["name"] != manifest.metric_name:
        raise GateError(
            "IC conversion requires exactly the frozen IC metric; additional guardrails need another collector"
        )
    rule = policy["metrics"][0]
    if rule["unit"] not in {"count", "instructions"} or rule["direction"] != "minimize":
        raise GateError("IC requires an instruction-count metric minimized by the approved plan")
    relevant_names = {
        "total": [],
        "process": ["process"],
        "thread": ["process", "thread"],
        "lib": ["process", "thread", "lib"],
        "function": ["process", "thread", "lib", "function"],
    }[manifest.level]
    required_names = set() if manifest.level == "total" else {manifest.level}
    if not required_names <= set(manifest.target) <= set(relevant_names) or any(
        not value.strip() for value in manifest.target.values()
    ):
        raise GateError("Comparison target requires the selected level and only relevant filters")
    for field in ("level", "target", "baseline_dir", "candidate_dir"):
        if compare.get(field) != getattr(manifest, field):
            raise GateError(f"Compare {field} differs from the explicit manifest")
    if manifest.baseline_dir == manifest.candidate_dir:
        raise GateError("Baseline and candidate directories must differ")
    if manifest.baseline.measurements or manifest.candidate.measurements:
        raise GateError("Manifest arms must not contain prefilled measurements")
    if (
        manifest.baseline.repo_revision != data["candidate"]["repo_revision"]
        or manifest.candidate.repo_revision != data["implementation"]["revision"]
    ):
        raise GateError("Manifest revisions do not match reviewed source")
    for arm in (manifest.baseline, manifest.candidate):
        for field in ("device_id", "workload_id", "workload_config_sha256", "environment_sha256"):
            if getattr(arm, field) != policy[field]:
                raise GateError(f"Manifest {field} differs from the frozen plan")
    relay = compare.get("relay_result")
    if relay is not None and (
        not isinstance(relay, dict)
        or type(relay.get("returncode")) is not int
        or relay["returncode"] != 0
    ):
        raise GateError("Underlying compare process did not exit successfully")
    rows = compare.get("reports")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
        raise GateError("IC report must include bounded per-pair measurements")
    seen, before, after = set(), [], []
    for row in rows:
        if not isinstance(row, dict) or "missing" in row or "error" in row:
            raise GateError("Incomplete or failed IC pairs cannot be converted")
        if row.get("baseline_found") is not True or row.get("candidate_found") is not True:
            raise GateError("Requested target is missing from an IC pair")
        if (
            not isinstance(row.get("case"), str)
            or not row["case"].strip()
            or any(type(row.get(key)) is not int or row[key] < 0 for key in ("round", "step"))
        ):
            raise GateError("IC pair requires case and nonnegative integer round/step")
        pair_id = digest([row["case"], row["round"], row["step"]])
        if pair_id in seen:
            raise GateError("Duplicate IC pair identity")
        seen.add(pair_id)
        for key, destination in (("baseline", before), ("candidate", after)):
            value = row.get(key)
            if type(value) is not int or not 0 <= value <= 2**53:
                raise GateError(
                    "IC counts must be nonnegative exact integers within float precision"
                )
            destination.append(
                Measurement(pair_id=pair_id, metrics={manifest.metric_name: float(value)})
            )
    baseline = manifest.baseline.model_copy(update={"measurements": before})
    candidate = manifest.candidate.model_copy(update={"measurements": after})
    report = ABReport(
        candidate_id=candidate_id,
        implementation_revision=data["implementation"]["revision"],
        baseline=baseline,
        candidate=candidate,
        functional_passed=manifest.functional_passed,
        metrics=policy["metrics"],
        minimum_pairs=policy["minimum_pairs"],
    )
    return register_report_conversion(
        service,
        candidate_id,
        report,
        candidate_snapshot=data,
        source={
            "kind": "ic_compare",
            "compare": compare,
            "manifest": manifest.model_dump(mode="json"),
        },
    )


def register_report_conversion(
    service, candidate_id: str, report: ABReport, *, candidate_snapshot: dict, source: dict
) -> dict:
    """Atomically retain collector evidence and link it to the canonical report.

    Collectors must validate source formats and approved experiment bindings
    before calling. The state check here prevents a concurrent approval or
    finalized validation from accepting a conversion of an obsolete snapshot.
    """
    from .service import GateError

    data = candidate_snapshot
    report = ABReport.model_validate(report.model_dump(mode="python"))
    if report.candidate_id != candidate_id:
        raise GateError("Converted report belongs to a different candidate")
    with service.store.transaction() as db:
        current = service.store.get(db, "candidate", candidate_id)["data"]
        if (
            current["stage"] != "code_approved"
            or current.get("validation")
            or current["plan_digest"] != data["plan_digest"]
            or current["implementation_digest"] != data["implementation_digest"]
        ):
            raise GateError("Candidate changed during report conversion")
        evidence = service.store.evidence(db, source)
        report_digest = digest(report.model_dump(mode="json"))
        exists = db.execute(
            "SELECT 1 FROM records WHERE kind='report_conversion' AND id=?", (report_digest,)
        ).fetchone()
        previous = service.store.get(db, "report_conversion", report_digest) if exists else None
        sources = list(previous["data"]["source_evidence"]) if previous else []
        if evidence not in sources:
            if len(sources) >= 32:
                raise GateError("Equivalent report already has 32 source manifests")
            sources.append(evidence)
        conversion = {
            "report_digest": report_digest,
            "candidate_id": candidate_id,
            "plan_digest": data["plan_digest"],
            "implementation_digest": data["implementation_digest"],
            "source_evidence": sources,
        }
        if not previous or conversion != previous["data"]:
            service.store.put(
                db,
                "report_conversion",
                report_digest,
                conversion,
                previous["version"] if previous else None,
            )
    return {
        "report": report.model_dump(mode="json"),
        "source_evidence": evidence,
        "note": "Converted collector declarations; submit through validate for the final gate. Hardware origin is not authenticated.",
    }


def report_sources(store, db, report_data: dict, candidate_data: dict) -> list[str]:
    """Bind persisted collector conversion provenance to a validation snapshot.

    Ordinary reports need no conversion record. Converted reports are indexed
    by their complete canonical digest so a CLI report file retains the link.
    """
    from .service import GateError

    report_digest = digest(report_data)
    exists = db.execute(
        "SELECT 1 FROM records WHERE kind='report_conversion' AND id=?", (report_digest,)
    ).fetchone()
    if not exists:
        return []
    conversion = store.get(db, "report_conversion", report_digest)["data"]
    for field, expected in (
        ("report_digest", report_digest),
        ("candidate_id", report_data["candidate_id"]),
        ("plan_digest", candidate_data["plan_digest"]),
        ("implementation_digest", candidate_data["implementation_digest"]),
    ):
        if conversion.get(field) != expected:
            raise GateError("Report conversion provenance does not match reviewed evidence")
    sources = conversion.get("source_evidence")
    if not isinstance(sources, list) or not 1 <= len(sources) <= 32:
        raise GateError("Report conversion has no bounded source evidence")
    for sha in sources:
        row = db.execute("SELECT content FROM evidence WHERE sha256=?", (sha,)).fetchone()
        if row is None or digest(json.loads(row["content"])) != sha:
            raise GateError("Report source evidence is missing or corrupted")
    return sources


def verify_measurement_profile(db, source_ids: list[str], policy: dict) -> None:
    """Require intact native collector evidence for a plan with a frozen profile.

    Verification uses the caller's existing transaction.
    A generic ABReport must not bypass an approved native measurement profile.
    """
    from .service import GateError

    expected = policy.get("measurement_profile_sha256")
    if expected is None:
        return
    if not source_ids:
        raise GateError("Frozen measurement profile requires native collector source evidence")
    from .lmbench import LmbenchProfile

    for sha in source_ids:
        row = db.execute("SELECT content FROM evidence WHERE sha256=?", (sha,)).fetchone()
        if row is None:
            raise GateError("Measurement profile source evidence is missing or corrupted")
        try:
            source = json.loads(row["content"])
            if digest(source) != sha:
                raise GateError("Measurement profile source evidence is missing or corrupted")
            if (
                source.get("kind") != "lmbench_paired_suites"
                or source.get("measurement_profile_sha256") != expected
            ):
                raise GateError("Collector source differs from the frozen measurement profile")
            manifest = source["manifest"]
            profile = LmbenchProfile.model_validate(
                {"kind": manifest["kind"], "metrics": manifest["metrics"]}
            )
            if digest(profile.model_dump(mode="json")) != expected:
                raise GateError(
                    "Collector metric mapping differs from the frozen measurement profile"
                )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise GateError(f"Invalid measurement profile source evidence: {exc}") from exc
