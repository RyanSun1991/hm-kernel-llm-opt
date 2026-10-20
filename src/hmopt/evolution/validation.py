"""Deterministic, fail-closed evaluation of paired A/B measurements.

The primary statistic is the mean of per-pair percentage improvements, using
``abs(baseline)`` as denominator and reversing the sign for maximization. A
two-sided 95% Student-t interval must lie above the required improvement.
Critical values are rounded upward; for more than 31 pairs, the df=30 value is
retained conservatively. This is a screening rule, not production certainty:
the interval assumes independent, representative pairs and approximately normal
pair effects. It cannot detect biased workloads, correlated repeats, thermal
drift, or fabricated provenance. Equal weighting of pairs is intentional.

Hardware verification here means internally consistent *reported* hardware
metadata. Authenticating measurements and binding revisions, images, and policy
to a persisted approved experiment remain responsibilities of the caller.
"""

from __future__ import annotations

import math
import statistics
from decimal import Decimal, localcontext
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Identifier = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
Sha256 = Annotated[
    str, StringConstraints(strict=True, min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")
]
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NonnegativeNumber = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class MetricRule(_StrictModel):
    name: Identifier
    unit: Identifier
    direction: Literal["minimize", "maximize"]
    min_improvement_pct: NonnegativeNumber = 1.0
    max_regression_pct: NonnegativeNumber = 0.0
    primary: bool = False


class Measurement(_StrictModel):
    pair_id: Identifier
    metrics: dict[Identifier, FiniteNumber]


class ExperimentArm(_StrictModel):
    repo_revision: Identifier
    project_revisions: dict[Identifier, Identifier] = Field(default_factory=dict)
    image_sha256: Sha256
    device_id: Identifier
    workload_id: Identifier
    workload_config_sha256: Sha256
    environment_sha256: Sha256
    measurements: list[Measurement]
    hardware: bool = True

    @field_validator("image_sha256", "workload_config_sha256", "environment_sha256")
    @classmethod
    def normalize_hash(cls, value: str) -> str:
        return value.lower()


class ABReport(_StrictModel):
    candidate_id: Identifier
    implementation_revision: Identifier
    baseline: ExperimentArm
    candidate: ExperimentArm
    functional_passed: bool
    metrics: list[MetricRule]
    minimum_pairs: Annotated[int, Field(ge=3)] = 3

    @model_validator(mode="after")
    def validate_policy(self) -> ABReport:
        if sum(rule.primary for rule in self.metrics) != 1:
            raise ValueError("exactly one primary metric rule is required")
        names = [rule.name for rule in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("metric rule names must be unique")
        return self


class ValidationResult(_StrictModel):
    verdict: Literal["pass", "fail", "inconclusive"]
    reasons: list[str]
    metrics: dict[str, dict[str, Any]]
    pairs: Annotated[int, Field(ge=0)]
    hardware_verified: bool


# Two-sided 95% Student-t critical values for degrees of freedom 1..30,
# rounded upward to six decimal places. Reusing t_30 beyond df=30 is conservative.
_T95 = (
    12.706205,
    4.302653,
    3.182447,
    2.776446,
    2.570582,
    2.446912,
    2.364625,
    2.306005,
    2.262158,
    2.228139,
    2.200986,
    2.178813,
    2.160369,
    2.144787,
    2.131450,
    2.119906,
    2.109816,
    2.100923,
    2.093025,
    2.085964,
    2.079614,
    2.073874,
    2.068658,
    2.063899,
    2.059539,
    2.055530,
    2.051831,
    2.048408,
    2.045230,
    2.042273,
)


def _duplicate_ids(measurements: list[Measurement]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in measurements:
        if item.pair_id in seen:
            duplicates.add(item.pair_id)
        seen.add(item.pair_id)
    return sorted(duplicates)


def evaluate_ab(report: ABReport, *, allow_synthetic: bool = False) -> ValidationResult:
    """Evaluate an experiment without I/O or hardware execution.

    Invalid schema/policy values raise Pydantic ValidationError. Incomparable or
    incomplete evidence is inconclusive. Functional failures and regressions
    beyond a metric's allowed mean percentage regression fail. Otherwise the
    primary metric needs a positive effect and its entire 95% interval at or
    above its minimum improvement. A zero baseline with a nonzero candidate
    cannot support a finite percentage win; a directionally worse value fails.
    """
    if not isinstance(allow_synthetic, bool):
        raise TypeError("allow_synthetic must be an explicit boolean")
    # Revalidate even an instance assembled with model_construct/model_copy, or
    # one whose mutable nested measurements were changed after construction.
    report = ABReport.model_validate(report.model_dump(mode="python"))
    baseline, candidate = report.baseline, report.candidate
    reasons: list[str] = []
    failures: list[str] = []
    if not report.functional_passed:
        failures.append("Functional validation failed.")

    hardware = baseline.hardware and candidate.hardware
    if not hardware and not allow_synthetic:
        reasons.append("Both experiment arms must report hardware measurements.")
    for field in ("device_id", "workload_id", "workload_config_sha256", "environment_sha256"):
        if getattr(baseline, field) != getattr(candidate, field):
            reasons.append(f"Experiment arms have different {field} values.")
    if baseline.repo_revision == candidate.repo_revision:
        reasons.append("Baseline and candidate repository revisions must be distinct.")
    if baseline.image_sha256 == candidate.image_sha256:
        reasons.append("Baseline and candidate image SHA-256 hashes must be distinct.")
    if candidate.repo_revision != report.implementation_revision:
        reasons.append("Candidate revision does not match implementation_revision.")

    for label, arm in (("Baseline", baseline), ("Candidate", candidate)):
        duplicates = _duplicate_ids(arm.measurements)
        if duplicates:
            reasons.append(f"{label} contains duplicate pair IDs: {', '.join(duplicates)}.")
    before = {item.pair_id: item.metrics for item in baseline.measurements}
    after = {item.pair_id: item.metrics for item in candidate.measurements}
    matched = sorted(before.keys() & after.keys())
    missing_baseline = sorted(after.keys() - before.keys())
    missing_candidate = sorted(before.keys() - after.keys())
    if missing_baseline:
        reasons.append(f"Missing baseline pairs: {', '.join(missing_baseline)}.")
    if missing_candidate:
        reasons.append(f"Missing candidate pairs: {', '.join(missing_candidate)}.")
    if len(matched) < report.minimum_pairs:
        reasons.append(
            f"Only {len(matched)} matched pairs; at least {report.minimum_pairs} are required."
        )
    for rule in report.metrics:
        for pair in matched:
            for label, values in (("baseline", before[pair]), ("candidate", after[pair])):
                if rule.name not in values:
                    reasons.append(f"Missing metric {rule.name!r} in {label} pair {pair!r}.")
    if reasons:
        return ValidationResult(
            verdict="fail" if failures else "inconclusive",
            reasons=failures + reasons,
            metrics={},
            pairs=len(matched),
            hardware_verified=False,
        )

    summaries: dict[str, dict[str, Any]] = {}
    for rule in report.metrics:
        sign = 1.0 if rule.direction == "minimize" else -1.0
        effects: list[float] = []
        undefined: list[str] = []
        for pair in matched:
            old, new = before[pair][rule.name], after[pair][rule.name]
            if old == 0:
                if new == 0:
                    effects.append(0.0)
                    continue
                undefined.append(pair)
                if (rule.direction == "minimize" and new > 0) or (
                    rule.direction == "maximize" and new < 0
                ):
                    failures.append(f"Metric {rule.name!r} regressed from zero in pair {pair!r}.")
                continue
            # Decimal arithmetic prevents binary cancellation from making an
            # exact policy boundary (e.g. 100 -> 102, allowance 2%) look worse.
            # It also avoids intermediate overflow for opposite-sign inputs.
            with localcontext() as context:
                context.prec = 50
                old_decimal, new_decimal = Decimal(str(old)), Decimal(str(new))
                effect = float(
                    Decimal(str(sign)) * (old_decimal - new_decimal) / abs(old_decimal) * 100
                )
            if not math.isfinite(effect):
                undefined.append(pair)
            else:
                effects.append(effect)
        if undefined:
            reasons.append(
                f"Metric {rule.name!r} has undefined or overflowing percentage changes "
                f"in pairs: {', '.join(undefined)}."
            )
            summaries[rule.name] = {"unquantifiable_pairs": undefined}
            continue

        try:
            mean = statistics.mean(effects)
            stddev = statistics.stdev(effects)
            critical = _T95[min(len(effects) - 1, 30) - 1]
            margin = critical * stddev / math.sqrt(len(effects))
            lower, upper = mean - margin, mean + margin
            baseline_mean = statistics.mean(before[pair][rule.name] for pair in matched)
            candidate_mean = statistics.mean(after[pair][rule.name] for pair in matched)
            if not all(
                math.isfinite(value)
                for value in (mean, stddev, margin, lower, upper, baseline_mean, candidate_mean)
            ):
                raise OverflowError
        except (OverflowError, ValueError):
            reasons.append(f"Metric {rule.name!r} exceeds the finite statistical range.")
            summaries[rule.name] = {"unquantifiable_pairs": matched}
            continue

        summaries[rule.name] = {
            "unit": rule.unit,
            "direction": rule.direction,
            "primary": rule.primary,
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "mean_improvement_pct": mean,
            "confidence_interval_pct": {"lower": lower, "upper": upper},
            "confidence_level": 0.95,
            "method": "paired_percent_student_t_conservative",
            "minimum_improvement_pct": rule.min_improvement_pct,
            "maximum_regression_pct": rule.max_regression_pct,
            "improvement_pct_by_pair": dict(zip(matched, effects)),
        }
        if mean < -rule.max_regression_pct:
            failures.append(
                f"Metric {rule.name!r} regressed {-mean:.6g}% beyond the "
                f"{rule.max_regression_pct:.6g}% allowance."
            )
        if rule.primary and (mean <= 0 or lower < rule.min_improvement_pct):
            reasons.append(
                f"Primary metric {rule.name!r} lacks a demonstrated positive improvement "
                f"of {rule.min_improvement_pct:.6g}%: mean={mean:.6g}%, "
                f"95% interval=[{lower:.6g}%, {upper:.6g}%]."
            )

    verdict = "fail" if failures else "inconclusive" if reasons else "pass"
    return ValidationResult(
        verdict=verdict,
        reasons=failures + reasons
        or ["Primary improvement cleared its 95% confidence bound; all guardrails passed."],
        metrics=summaries,
        pairs=len(matched),
        hardware_verified=hardware,
    )
