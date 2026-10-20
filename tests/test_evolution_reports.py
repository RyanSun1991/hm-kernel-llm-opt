from copy import deepcopy

import pytest
from test_evolution_service import Scenario
from test_evolution_service import git_bin as shared_git_bin

from hmopt.evolution.reports import ICManifest, convert_ic_report
from hmopt.evolution.service import GateError
from hmopt.evolution.store import digest
from hmopt.evolution.validation import ABReport

git_bin = shared_git_bin


@pytest.fixture
def conversion(tmp_path, git_bin):
    s = Scenario(tmp_path, git_bin).ready()
    data = s.row["data"]
    report = s.report_data()
    report["baseline"]["measurements"] = []
    report["candidate"]["measurements"] = []
    compare = {
        "success": True,
        "level": "total",
        "target": {},
        "baseline_dir": "C:/fixture/stock",
        "candidate_dir": "C:/fixture/feature",
        "aggregate": {"delta_pct": -9999},
        "reports": [
            {
                "case": "case",
                "round": i,
                "step": 0,
                "baseline": 100,
                "candidate": 90,
                "baseline_found": True,
                "candidate_found": True,
            }
            for i in range(3)
        ],
    }
    manifest = {
        "candidate_id": s.candidate_id,
        "plan_digest": data["plan_digest"],
        "implementation_digest": data["implementation_digest"],
        "compare_sha256": digest(compare),
        "metric_name": "instructions",
        "level": "total",
        "target": {},
        "baseline_dir": compare["baseline_dir"],
        "candidate_dir": compare["candidate_dir"],
        "baseline": report["baseline"],
        "candidate": report["candidate"],
        "functional_passed": True,
    }
    return s, compare, manifest


def test_convert_raw_pairs_then_validate_not_aggregate(conversion):
    s, compare, manifest = conversion
    converted = convert_ic_report(
        s.service, s.candidate_id, compare, ICManifest.model_validate(manifest)
    )
    report = ABReport.model_validate(converted["report"])
    assert len(report.baseline.measurements) == 3
    assert s.service.store.read_evidence(converted["source_evidence"])["compare"] == compare
    row = s.service.validate(
        s.candidate_id,
        report,
        actor="validator",
        expected_version=s.row["version"],
        request_id="converted",
    )
    assert (
        row["data"]["validation"]["result"]["metrics"]["instructions"]["mean_improvement_pct"] == 10
    )
    assert row["data"]["validation"]["source_evidence"] == [converted["source_evidence"]]
    assert s.service.store.audit(s.candidate_id)[-1]["details"]["source_evidence"] == [
        converted["source_evidence"]
    ]


def test_missing_ic_source_evidence_blocks_validation(conversion):
    s, compare, manifest = conversion
    converted = convert_ic_report(
        s.service, s.candidate_id, compare, ICManifest.model_validate(manifest)
    )
    with s.service.store.transaction() as db:
        db.execute("DELETE FROM evidence WHERE sha256=?", (converted["source_evidence"],))
    with pytest.raises(GateError, match="missing or corrupted"):
        s.service.validate(
            s.candidate_id,
            ABReport.model_validate(converted["report"]),
            actor="validator",
            expected_version=s.row["version"],
            request_id="missing-source",
        )


def test_equivalent_report_retains_each_raw_manifest(conversion):
    s, compare, manifest = conversion
    first = convert_ic_report(
        s.service, s.candidate_id, compare, ICManifest.model_validate(manifest)
    )
    compare["aggregate"] = {"delta_pct": 999}
    manifest["compare_sha256"] = digest(compare)
    second = convert_ic_report(
        s.service, s.candidate_id, compare, ICManifest.model_validate(manifest)
    )
    assert first["report"] == second["report"]
    row = s.service.validate(
        s.candidate_id,
        ABReport.model_validate(second["report"]),
        actor="validator",
        expected_version=s.row["version"],
        request_id="both-sources",
    )
    assert row["data"]["validation"]["source_evidence"] == [
        first["source_evidence"],
        second["source_evidence"],
    ]


@pytest.mark.parametrize(
    "patch",
    [
        {"baseline_found": False},
        {"candidate_found": False},
        {"baseline_found": 1},
        {"missing": "candidate"},
        {"error": "parse failed"},
        {"baseline": -1},
        {"candidate": True},
        {"candidate": 1.2},
        {"baseline": 2**54},
        {"round": -1},
        {"step": True},
        {"case": ""},
    ],
)
def test_bad_pairs_never_convert(conversion, patch):
    s, compare, manifest = conversion
    compare["reports"][0].update(patch)
    manifest["compare_sha256"] = digest(compare)
    with pytest.raises(GateError):
        convert_ic_report(s.service, s.candidate_id, compare, ICManifest.model_validate(manifest))


@pytest.mark.parametrize("field", ["plan_digest", "implementation_digest", "compare_sha256"])
def test_manifest_is_bound_to_immutable_evidence(conversion, field):
    s, compare, manifest = conversion
    manifest[field] = "f" * 64
    with pytest.raises(GateError):
        convert_ic_report(s.service, s.candidate_id, compare, ICManifest.model_validate(manifest))


def test_duplicate_pairs_rejected(conversion):
    s, compare, manifest = conversion
    compare["reports"].append(deepcopy(compare["reports"][0]))
    manifest["compare_sha256"] = digest(compare)
    with pytest.raises(GateError, match="Duplicate"):
        convert_ic_report(s.service, s.candidate_id, compare, ICManifest.model_validate(manifest))


@pytest.mark.parametrize(
    "change",
    [
        {"success": False},
        {"relay_result": {"returncode": 1}},
        {"baseline_dir": "C:/fixture/latest"},
        {"target": {"function": "wrong"}},
    ],
)
def test_exit_status_and_explicit_comparison_scope(conversion, change):
    s, compare, manifest = conversion
    compare.update(change)
    manifest["compare_sha256"] = digest(compare)
    with pytest.raises(GateError):
        convert_ic_report(s.service, s.candidate_id, compare, ICManifest.model_validate(manifest))


@pytest.mark.parametrize(
    "level,target",
    [
        ("thread", {"thread": "RenderThread"}),
        ("lib", {"lib": "kernel"}),
        ("function", {"function": "target"}),
        ("function", {"process": "app", "function": "target"}),
    ],
)
def test_optional_parent_filters_follow_native_compare_contract(conversion, level, target):
    scenario, compare, manifest = conversion
    compare.update(level=level, target=target)
    manifest.update(level=level, target=target, compare_sha256=digest(compare))
    result = convert_ic_report(
        scenario.service, scenario.candidate_id, compare, ICManifest.model_validate(manifest)
    )
    assert len(result["report"]["baseline"]["measurements"]) == 3


@pytest.mark.parametrize(
    "level,target",
    [
        ("thread", {"process": "app"}),
        ("thread", {"thread": "worker", "function": "target"}),
        ("function", {"function": " "}),
        ("total", {"process": "app"}),
    ],
)
def test_optional_filters_cannot_change_granularity(conversion, level, target):
    scenario, compare, manifest = conversion
    compare.update(level=level, target=target)
    manifest.update(level=level, target=target, compare_sha256=digest(compare))
    with pytest.raises(GateError, match="selected level"):
        convert_ic_report(
            scenario.service, scenario.candidate_id, compare, ICManifest.model_validate(manifest)
        )
