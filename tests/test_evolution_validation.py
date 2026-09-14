"""Contract and fail-closed regression tests for generic paired validation."""

import json

import pytest
from pydantic import ValidationError

from hmopt.evolution.validation import ABReport, Measurement, MetricRule, evaluate_ab


def report_data(*, values=(90.0, 90.0, 90.0), direction="minimize"):
    common = {
        "device_id": "device-1",
        "workload_id": "scroll",
        "workload_config_sha256": "c" * 64,
        "environment_sha256": "d" * 64,
    }
    return {
        "candidate_id": "candidate-1",
        "implementation_revision": "feature-revision",
        "baseline": {
            **common,
            "repo_revision": "baseline-revision",
            "image_sha256": "a" * 64,
            "measurements": [
                {"pair_id": str(i), "metrics": {"work": 100.0}} for i in range(len(values))
            ],
        },
        "candidate": {
            **common,
            "repo_revision": "feature-revision",
            "image_sha256": "b" * 64,
            "measurements": [
                {"pair_id": str(i), "metrics": {"work": value}} for i, value in enumerate(values)
            ],
        },
        "functional_passed": True,
        "metrics": [
            {"name": "work", "unit": "instructions", "direction": direction, "primary": True}
        ],
    }


def evaluate(data, **kwargs):
    return evaluate_ab(ABReport.model_validate(data), **kwargs)


@pytest.mark.parametrize(
    "direction,values",
    [
        ("minimize", (90.0, 91.0, 89.0)),
        ("maximize", (110.0, 111.0, 109.0)),
    ],
)
def test_clear_paired_improvement_passes_both_directions(direction, values):
    result = evaluate(report_data(values=values, direction=direction))
    assert result.verdict == "pass"
    assert result.pairs == 3
    assert result.hardware_verified is True
    assert result.metrics["work"]["confidence_interval_pct"]["lower"] > 1.0
    assert json.loads(result.model_dump_json())["verdict"] == "pass"


def test_pair_identity_controls_matching_not_order():
    data = report_data(values=(90.0, 91.0, 89.0))
    expected = evaluate(data)
    data["candidate"]["measurements"].reverse()
    assert evaluate(data) == expected


@pytest.mark.parametrize("values", [(99.5, 99.5, 99.5), (80.0, 99.0, 101.0), (100.0,) * 3])
def test_noise_or_uncertain_effect_cannot_pass(values):
    result = evaluate(report_data(values=values))
    assert result.verdict == "inconclusive"
    assert "lacks a demonstrated" in " ".join(result.reasons)


@pytest.mark.parametrize(
    "direction,values", [("minimize", (100.5,) * 3), ("maximize", (99.5,) * 3)]
)
def test_even_small_regression_fails_when_guardrail_is_zero(direction, values):
    assert evaluate(report_data(values=values, direction=direction)).verdict == "fail"


@pytest.mark.parametrize(
    "direction,baseline,candidate", [("minimize", 100.0, 102.0), ("maximize", 100.0, 98.0)]
)
def test_secondary_guardrail_is_direction_aware(direction, baseline, candidate):
    data = report_data()
    data["metrics"].append(
        {"name": "guard", "unit": "units", "direction": direction, "max_regression_pct": 1.0}
    )
    for arm, value in (("baseline", baseline), ("candidate", candidate)):
        for item in data[arm]["measurements"]:
            item["metrics"]["guard"] = value
    result = evaluate(data)
    assert result.verdict == "fail"
    assert "guard" in " ".join(result.reasons)
    data["metrics"][1]["max_regression_pct"] = 3.0
    assert evaluate(data).verdict == "pass"


def test_functional_failure_overrides_performance_gain():
    data = report_data()
    data["functional_passed"] = False
    assert evaluate(data).verdict == "fail"


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("device_id", "another-device"),
        ("workload_id", "another-workload"),
        ("workload_config_sha256", "e" * 64),
        ("environment_sha256", "f" * 64),
        ("image_sha256", "a" * 64),
        ("repo_revision", "baseline-revision"),
        ("repo_revision", "unrelated-feature-revision"),
    ],
)
def test_mismatched_provenance_cannot_pass(field, replacement):
    data = report_data()
    data["candidate"][field] = replacement
    result = evaluate(data)
    assert result.verdict == "inconclusive"
    assert result.hardware_verified is False
    assert result.metrics == {}


def test_synthetic_measurements_require_explicit_opt_in_and_remain_marked_synthetic():
    data = report_data()
    data["candidate"]["hardware"] = False
    assert evaluate(data).verdict == "inconclusive"
    result = evaluate(data, allow_synthetic=True)
    assert result.verdict == "pass"
    assert result.hardware_verified is False


def test_opt_in_does_not_relax_provenance_validation():
    data = report_data()
    data["candidate"]["hardware"] = False
    data["candidate"]["device_id"] = "different"
    assert evaluate(data, allow_synthetic=True).verdict == "inconclusive"


def test_synthetic_opt_in_rejects_truthy_strings():
    with pytest.raises(TypeError, match="explicit boolean"):
        evaluate(report_data(), allow_synthetic="false")


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "metric", "few", "empty"])
def test_incomplete_or_ambiguous_measurements_fail_closed(mutation):
    data = report_data()
    values = data["candidate"]["measurements"]
    if mutation == "duplicate":
        values.append(values[0].copy())
    elif mutation == "missing":
        values[0]["pair_id"] = "unmatched"
    elif mutation == "metric":
        values[0]["metrics"] = {}
    elif mutation == "few":
        data["baseline"]["measurements"].pop()
        values.pop()
    else:
        data["baseline"]["measurements"] = []
        data["candidate"]["measurements"] = []
    assert evaluate(data).verdict == "inconclusive"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "90", True])
def test_nonfinite_or_coerced_measurement_values_are_rejected(value):
    with pytest.raises(ValidationError):
        Measurement(pair_id="one", metrics={"work": value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_improvement_pct", float("nan")),
        ("max_regression_pct", float("inf")),
        ("min_improvement_pct", -1.0),
        ("max_regression_pct", -1.0),
        ("min_improvement_pct", "1.0"),
        ("primary", "true"),
    ],
)
def test_invalid_policy_numbers_and_flags_are_rejected(field, value):
    with pytest.raises(ValidationError):
        MetricRule(name="work", unit="count", direction="minimize", **{field: value})


@pytest.mark.parametrize("mutation", ["no-primary", "two-primary", "duplicate-name", "too-few"])
def test_policy_requires_one_unique_primary_and_at_least_three_pairs(mutation):
    data = report_data()
    if mutation == "no-primary":
        data["metrics"][0]["primary"] = False
    elif mutation == "two-primary":
        data["metrics"].append({**data["metrics"][0], "name": "another"})
    elif mutation == "duplicate-name":
        data["metrics"].append({**data["metrics"][0], "primary": False})
    else:
        data["minimum_pairs"] = 2
    with pytest.raises(ValidationError):
        ABReport.model_validate(data)


@pytest.mark.parametrize("mutation", ["extra", "empty", "bad-hash", "bool-pairs", "nested-extra"])
def test_schema_rejects_unknown_fields_and_invalid_identifiers(mutation):
    data = report_data()
    if mutation == "extra":
        data["allow_unreviewed"] = True
    elif mutation == "empty":
        data["candidate_id"] = "   "
    elif mutation == "bad-hash":
        data["candidate"]["image_sha256"] = "short"
    elif mutation == "bool-pairs":
        data["minimum_pairs"] = True
    else:
        data["candidate"]["measurements"][0]["extra"] = "ignored?"
    with pytest.raises(ValidationError):
        ABReport.model_validate(data)


def test_hashes_are_compared_case_insensitively():
    data = report_data()
    data["candidate"]["image_sha256"] = "A" * 64
    assert evaluate(data).verdict == "inconclusive"


@pytest.mark.parametrize(
    "direction,value,verdict",
    [
        ("minimize", 0.0, "inconclusive"),
        ("maximize", 0.0, "inconclusive"),
        ("minimize", 1.0, "fail"),
        ("maximize", -1.0, "fail"),
        ("minimize", -1.0, "inconclusive"),
        ("maximize", 1.0, "inconclusive"),
    ],
)
def test_zero_baselines_do_not_fabricate_percentage_wins(direction, value, verdict):
    data = report_data(values=(value,) * 3, direction=direction)
    for item in data["baseline"]["measurements"]:
        item["metrics"]["work"] = 0.0
    assert evaluate(data).verdict == verdict


def test_finite_inputs_that_overflow_statistics_are_inconclusive_and_json_safe():
    data = report_data(values=(1e308,) * 3, direction="maximize")
    for item in data["baseline"]["measurements"]:
        item["metrics"]["work"] = 1e-308
    result = evaluate(data)
    assert result.verdict == "inconclusive"
    json.dumps(result.model_dump(), allow_nan=False)


def test_mutated_nested_model_is_revalidated_before_evaluation():
    report = ABReport.model_validate(report_data())
    report.candidate.measurements[0].metrics["work"] = float("nan")
    with pytest.raises(ValidationError):
        evaluate_ab(report)


def test_caller_requested_pair_minimum_is_enforced():
    data = report_data()
    data["minimum_pairs"] = 5
    assert evaluate(data).verdict == "inconclusive"


def test_negative_metric_values_have_directionally_correct_absolute_denominators():
    data = report_data(values=(-110.0,) * 3)
    for item in data["baseline"]["measurements"]:
        item["metrics"]["work"] = -100.0
    result = evaluate(data)
    assert result.verdict == "pass"
    assert result.metrics["work"]["mean_improvement_pct"] == pytest.approx(10.0)


def test_large_pair_count_uses_finite_deterministic_interval():
    data = report_data(values=(90.0, 91.0, 89.0) * 20)
    first = evaluate(data)
    assert first.verdict == "pass"
    assert first == evaluate(data)


def test_exact_primary_threshold_passes_without_binary_cancellation():
    data = report_data(values=(99.0,) * 3)
    result = evaluate(data)
    assert result.verdict == "pass"
    assert result.metrics["work"]["confidence_interval_pct"]["lower"] == 1.0


def test_exact_guardrail_allowance_does_not_become_a_floating_point_regression():
    data = report_data()
    data["metrics"].append(
        {"name": "latency", "unit": "ms", "direction": "minimize", "max_regression_pct": 2.0}
    )
    for arm, value in (("baseline", 100.0), ("candidate", 102.0)):
        for item in data[arm]["measurements"]:
            item["metrics"]["latency"] = value
    assert evaluate(data).verdict == "pass"


def test_large_finite_values_with_a_valid_ratio_remain_evaluable():
    data = report_data(values=(9e307,) * 3)
    for item in data["baseline"]["measurements"]:
        item["metrics"]["work"] = 1e308
    result = evaluate(data)
    assert result.verdict == "pass"
    json.dumps(result.model_dump(), allow_nan=False)
