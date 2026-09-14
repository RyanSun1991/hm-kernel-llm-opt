"""Frozen correctness acceptance without inventing a performance improvement.

Evidence hashes and execution metadata are trusted local collector declarations,
not authenticated execution proofs. The service binds these to the reviewed code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .validation import Identifier, Sha256, ValidationResult


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    @field_validator(
        "artifact_sha256",
        "evidence_sha256",
        "workload_config_sha256",
        "environment_sha256",
        check_fields=False,
    )
    @classmethod
    def normalize_hash(cls, value: str) -> str:
        return value.lower()


class CorrectnessPolicy(StrictContract):
    kind: Literal["correctness"]
    required_checks: list[Identifier] = Field(min_length=1, max_length=100)
    reproduction_checks: list[Identifier] = Field(min_length=1, max_length=100)
    execution_kind: Literal["local", "hardware"]
    device_id: Identifier
    workload_id: Identifier
    workload_config_sha256: Sha256
    environment_sha256: Sha256

    @model_validator(mode="after")
    def check_names(self) -> CorrectnessPolicy:
        for values in (self.required_checks, self.reproduction_checks):
            if len(set(values)) != len(values):
                raise ValueError("Check names must be unique")
        if not set(self.reproduction_checks).issubset(self.required_checks):
            raise ValueError("Reproduction checks must be required checks")
        return self


class CheckResult(StrictContract):
    outcome: Literal["pass", "fail", "skipped"]
    evidence_sha256: Sha256


class CorrectnessArm(StrictContract):
    repo_revision: Identifier
    project_revisions: dict[Identifier, Identifier] = Field(default_factory=dict)
    artifact_sha256: Sha256
    device_id: Identifier
    workload_id: Identifier
    workload_config_sha256: Sha256
    environment_sha256: Sha256
    checks: dict[Identifier, CheckResult] = Field(max_length=100)


class CorrectnessReport(StrictContract):
    kind: Literal["correctness"]
    candidate_id: Identifier
    implementation_revision: Identifier
    policy: CorrectnessPolicy
    baseline: CorrectnessArm
    candidate: CorrectnessArm
    execution_kind: Literal["local", "hardware", "simulation"]
    functional_passed: bool


def evaluate_correctness(
    report: CorrectnessReport, *, allow_synthetic: bool = False
) -> ValidationResult:
    if type(allow_synthetic) is not bool:
        raise ValueError("allow_synthetic must be a boolean")
    report = CorrectnessReport.model_validate(report.model_dump(mode="python"))
    missing, failed = [], []
    if not report.functional_passed:
        failed.append("Functional regression checks failed")
    if report.execution_kind == "simulation":
        if not allow_synthetic:
            missing.append("Simulation is not accepted as executed correctness evidence")
    elif report.execution_kind != report.policy.execution_kind:
        missing.append("Execution kind differs from the frozen correctness policy")
    for arm in (report.baseline, report.candidate):
        for field in ("device_id", "workload_id", "workload_config_sha256", "environment_sha256"):
            if getattr(arm, field) != getattr(report.policy, field):
                missing.append(f"Experiment {field} differs from the frozen policy")
    if report.baseline.repo_revision == report.candidate.repo_revision:
        missing.append("Baseline and candidate revisions must differ")
    if report.candidate.repo_revision != report.implementation_revision:
        missing.append("Implementation revision does not match candidate evidence")
    if report.baseline.artifact_sha256 == report.candidate.artifact_sha256:
        missing.append("Baseline and candidate build artifacts must differ")
    for name in report.policy.reproduction_checks:
        old = report.baseline.checks.get(name)
        if old is None or old.outcome != "fail":
            missing.append(f"Baseline does not reproduce required defect: {name}")
    for name in report.policy.required_checks:
        result = report.candidate.checks.get(name)
        if result is None or result.outcome == "skipped":
            missing.append(f"Required candidate check was not executed: {name}")
        elif result.outcome == "fail":
            failed.append(f"Required candidate check failed: {name}")
    verdict = "fail" if failed else "inconclusive" if missing else "pass"
    return ValidationResult(
        verdict=verdict,
        reasons=failed + missing
        or ["Baseline reproduces the defect and all required candidate checks pass"],
        metrics={},
        pairs=0,
        hardware_verified=report.execution_kind == "hardware" and not missing,
    )
