"""Local fixture workbooks exercise the bridge; no hardware is accessed."""

import hashlib
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_evolution_service import Scenario
from test_evolution_service import git_bin as shared_git_bin

from hmopt.evolution.lmbench import LmbenchManifest, LmbenchProfile, convert_lmbench_report
from hmopt.evolution.service import GateError
from hmopt.evolution.store import digest
from hmopt.evolution.validation import ABReport, evaluate_ab

openpyxl = pytest.importorskip("openpyxl")
git_bin = shared_git_bin

_METRICS = [
    {
        "name": "syscall_latency",
        "system": "big",
        "tool": "lmbench-lat",
        "metric": "lat_sys",
        "command": "lat_syscall null",
        "units": "microseconds",
    },
    {
        "name": "read_bandwidth",
        "system": "big",
        "tool": "lmbench-mem",
        "metric": "bw_rd",
        "command": "bw_mem rd",
        "units": "MB/s",
    },
]
_HEADER = ["system", "tool", "metric", "command", "average", "value0", "value1", "units"]
_CONTEXT = (
    "repo_revision",
    "image_sha256",
    "device_id",
    "workload_id",
    "workload_config_sha256",
    "environment_sha256",
)


class LmbenchScenario(Scenario):
    def plan_payload(self):
        payload = super().plan_payload()
        plan = payload["plan"]
        plan["validation"]["metrics"] = [
            {
                "name": "syscall_latency",
                "unit": "microseconds",
                "direction": "minimize",
                "primary": True,
                "min_improvement_pct": 1.0,
                "max_regression_pct": 0.0,
            },
            {
                "name": "read_bandwidth",
                "unit": "MB/s",
                "direction": "maximize",
                "primary": False,
                "min_improvement_pct": 0.0,
                "max_regression_pct": 1.0,
            },
        ]
        plan["validation"]["measurement_profile_sha256"] = digest(
            LmbenchProfile(metrics=_METRICS).model_dump(mode="json")
        )
        payload["review"] = self.review(digest(plan), "plan-reviewer")
        return payload


def _workbook(path, latency, bandwidth, token):
    workbook = openpyxl.Workbook()
    workbook.properties.title = token
    sheet = workbook.active
    sheet.title = "result"
    sheet.append(_HEADER)
    for metric, value in zip(_METRICS, (latency, bandwidth)):
        sheet.append(
            [
                metric["system"],
                metric["tool"],
                metric["metric"],
                metric["command"],
                "=AVERAGE(F2:G2)",
                value * 0.99,
                value * 1.01,
                metric["units"],
            ]
        )
    workbook.save(path)
    workbook.close()


def _rehash(run):
    run["xlsx_sha256"] = hashlib.sha256(Path(run["xlsx_path"]).read_bytes()).hexdigest()


def _edit(run, callback):
    path = run["xlsx_path"]
    workbook = openpyxl.load_workbook(path)
    callback(workbook["result"])
    workbook.save(path)
    workbook.close()
    _rehash(run)


@pytest.fixture
def conversion(tmp_path, git_bin):
    scenario = LmbenchScenario(tmp_path, git_bin).ready()
    report = scenario.report_data()
    for arm in ("baseline", "candidate"):
        report[arm]["measurements"] = []
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    pairs = []
    for i in range(3):
        pair = {"pair_id": f"suite-{i}"}
        for arm in ("baseline", "candidate"):
            token = f"run-{arm}-{i}"
            path = artifacts / f"{token}.xlsx"
            _workbook(
                path,
                (100 + i) * (0.9 if arm == "candidate" else 1),
                (1000 + i) * (1.1 if arm == "candidate" else 1),
                token,
            )
            pair[arm] = {
                **{field: report[arm][field] for field in _CONTEXT},
                "run_token": token,
                "xlsx_path": str(path),
                "xlsx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        pairs.append(pair)
    data = scenario.row["data"]
    manifest = {
        "candidate_id": scenario.candidate_id,
        "plan_digest": data["plan_digest"],
        "implementation_digest": data["implementation_digest"],
        "baseline": report["baseline"],
        "candidate": report["candidate"],
        "metrics": deepcopy(_METRICS),
        "pairs": pairs,
        "functional_passed": True,
    }
    return scenario, manifest, artifacts


def _convert(fixture, **kwargs):
    scenario, manifest, _ = fixture
    return convert_lmbench_report(
        scenario.service, scenario.candidate_id, LmbenchManifest.model_validate(manifest), **kwargs
    )


def test_raw_suite_means_preserve_sources_through_validation(conversion):
    scenario, manifest, _ = conversion
    converted = _convert(conversion)
    report = ABReport.model_validate(converted["report"])
    assert len(report.baseline.measurements) == 3  # Six raw values are only three suite runs.
    assert report.baseline.measurements[0].metrics["syscall_latency"] == 100
    source = scenario.service.store.read_evidence(converted["source_evidence"])
    assert source["kind"] == "lmbench_paired_suites"
    assert source["artifacts"][0]["sha256"] == manifest["pairs"][0]["baseline"]["xlsx_sha256"]
    assert source["artifacts"][0]["metrics"]["syscall_latency"]["raw_samples"] == [99, 101]
    row = scenario.service.validate(
        scenario.candidate_id,
        report,
        actor="validator",
        expected_version=scenario.row["version"],
        request_id="lmbench-validate",
    )
    assert row["data"]["stage"] == "validated"
    assert row["data"]["validation"]["result"]["verdict"] == "pass"
    assert row["data"]["validation"]["source_evidence"] == [converted["source_evidence"]]
    assert scenario.service.store.audit(scenario.candidate_id)[-1]["details"][
        "source_evidence"
    ] == [converted["source_evidence"]]


def test_guardrail_regression_fails(conversion):
    for pair in conversion[1]["pairs"]:
        _edit(
            pair["candidate"],
            lambda sheet: (
                setattr(sheet["F3"], "value", 700),
                setattr(sheet["G3"], "value", 700),
            ),
        )
    result = evaluate_ab(ABReport.model_validate(_convert(conversion)["report"]))
    assert result.verdict == "fail"
    assert result.metrics["read_bandwidth"]["mean_improvement_pct"] < -29


def test_insufficient_improvement_is_inconclusive(conversion):
    for i, pair in enumerate(conversion[1]["pairs"]):
        value = 100 + i
        _edit(
            pair["candidate"],
            lambda sheet, value=value: (
                setattr(sheet["F2"], "value", value),
                setattr(sheet["G2"], "value", value),
            ),
        )
    assert (
        evaluate_ab(ABReport.model_validate(_convert(conversion)["report"])).verdict
        == "inconclusive"
    )


def test_functional_failure_and_simulation_are_not_accepted(conversion):
    conversion[1]["functional_passed"] = False
    conversion[1]["candidate"]["hardware"] = False
    result = evaluate_ab(ABReport.model_validate(_convert(conversion)["report"]))
    assert result.verdict == "fail" and not result.hardware_verified


@pytest.mark.parametrize("field", ["candidate_id", "plan_digest", "implementation_digest"])
def test_wrong_approval_binding_rejected(conversion, field):
    conversion[1][field] = "f" * 64
    with pytest.raises(GateError, match="approved plan"):
        _convert(conversion)


@pytest.mark.parametrize("field", list(_CONTEXT))
def test_wrong_per_run_provenance_rejected(conversion, field):
    conversion[1]["pairs"][0]["candidate"][field] = "f" * 64
    with pytest.raises(GateError, match="provenance"):
        _convert(conversion)


@pytest.mark.parametrize("field", list(_CONTEXT[2:]))
def test_wrong_arm_environment_rejected(conversion, field):
    conversion[1]["candidate"][field] = "f" * 64
    with pytest.raises(GateError, match="frozen plan"):
        _convert(conversion)


def test_same_image_and_wrong_revision_rejected(conversion):
    manifest = conversion[1]
    manifest["candidate"]["image_sha256"] = manifest["baseline"]["image_sha256"]
    with pytest.raises(GateError, match="distinct image"):
        _convert(conversion)
    manifest["candidate"]["repo_revision"] = manifest["baseline"]["repo_revision"]
    with pytest.raises(GateError, match="revisions"):
        _convert(conversion)


@pytest.mark.parametrize(
    "cell,value",
    [
        ("F2", None),
        ("F2", "NaN"),
        ("F2", True),
        ("F2", -1),
        ("F2", "=1+1"),
        ("H2", "milliseconds"),
        ("D2", "another command"),
    ],
)
def test_bad_raw_samples_or_metric_identity_rejected(conversion, cell, value):
    _edit(conversion[1]["pairs"][0]["baseline"], lambda sheet: setattr(sheet[cell], "value", value))
    with pytest.raises(GateError, match="sample|identity"):
        _convert(conversion)


def test_average_only_workbook_is_rejected(conversion):
    _edit(conversion[1]["pairs"][0]["baseline"], lambda sheet: sheet.delete_cols(6, 2))
    with pytest.raises(GateError, match="raw value0"):
        _convert(conversion)


def test_missing_and_duplicate_sample_headers_rejected(conversion):
    run = conversion[1]["pairs"][0]["baseline"]
    _edit(run, lambda sheet: setattr(sheet["G1"], "value", "value2"))
    with pytest.raises(GateError, match="contiguous"):
        _convert(conversion)
    _edit(run, lambda sheet: setattr(sheet["G1"], "value", "value0"))
    with pytest.raises(GateError, match="Duplicate"):
        _convert(conversion)


def test_duplicate_metric_rows_rejected(conversion):
    _edit(
        conversion[1]["pairs"][0]["baseline"],
        lambda sheet: sheet.append([cell.value for cell in sheet[2]]),
    )
    with pytest.raises(GateError, match="Duplicate lmbench metric row"):
        _convert(conversion)


@pytest.mark.parametrize("case", ["pair", "run_token", "path", "content"])
def test_reused_suite_evidence_rejected(conversion, case):
    manifest = conversion[1]
    before, after = manifest["pairs"][0], manifest["pairs"][1]
    if case == "pair":
        after["pair_id"] = before["pair_id"]
    elif case == "run_token":
        after["baseline"]["run_token"] = before["baseline"]["run_token"]
    elif case == "path":
        after["baseline"]["xlsx_path"] = before["baseline"]["xlsx_path"]
    else:
        Path(after["baseline"]["xlsx_path"]).write_bytes(
            Path(before["baseline"]["xlsx_path"]).read_bytes()
        )
        _rehash(after["baseline"])
    with pytest.raises(GateError, match="Duplicate"):
        _convert(conversion)


def test_hash_tampering_rejected(conversion):
    conversion[1]["pairs"][0]["baseline"]["xlsx_sha256"] = "f" * 64
    with pytest.raises(GateError, match="SHA-256"):
        _convert(conversion)


def test_digest_and_insufficient_pair_manifest_are_rejected(conversion):
    manifest = conversion[1]
    manifest["pairs"] = manifest["pairs"][:2]
    with pytest.raises(ValidationError):
        _convert(conversion)
    with pytest.raises(ValidationError):
        LmbenchManifest.model_validate({"digest": {"ok": True, "vs_previous": {"improved": 100}}})


def test_metric_mapping_cannot_omit_guardrails(conversion):
    conversion[1]["metrics"] = conversion[1]["metrics"][:1]
    with pytest.raises(GateError, match="profile"):
        _convert(conversion)


def test_native_metric_profile_cannot_be_selected_after_approval(conversion):
    conversion[1]["metrics"][0]["command"] = "lat_syscall read"
    with pytest.raises(GateError, match="profile"):
        _convert(conversion)


def test_plan_without_frozen_lmbench_profile_is_rejected(conversion, tmp_path, git_bin):
    directory = tmp_path / "generic"
    directory.mkdir()
    scenario = Scenario(directory, git_bin).ready()
    manifest = conversion[1]
    manifest.update(
        candidate_id=scenario.candidate_id,
        plan_digest=scenario.row["data"]["plan_digest"],
        implementation_digest=scenario.row["data"]["implementation_digest"],
    )
    with pytest.raises(GateError, match="profile"):
        convert_lmbench_report(
            scenario.service, scenario.candidate_id, LmbenchManifest.model_validate(manifest)
        )


def test_fixed_artifacts_root_supports_relative_paths(conversion):
    for pair in conversion[1]["pairs"]:
        for arm in ("baseline", "candidate"):
            pair[arm]["xlsx_path"] = Path(pair[arm]["xlsx_path"]).name
    assert (
        len(
            _convert(conversion, artifacts_root=conversion[2])["report"]["baseline"]["measurements"]
        )
        == 3
    )


def test_paths_outside_root_and_parent_traversal_rejected(conversion):
    with pytest.raises(GateError, match="outside"):
        _convert(conversion, artifacts_root=conversion[0].repo)
    conversion[1]["pairs"][0]["baseline"]["xlsx_path"] = "../outside.xlsx"
    with pytest.raises(GateError, match="traversal"):
        _convert(conversion, artifacts_root=conversion[2])


def test_symlink_artifact_rejected_under_fixed_root(conversion):
    run = conversion[1]["pairs"][0]["baseline"]
    link = conversion[2] / "link.xlsx"
    try:
        link.symlink_to(run["xlsx_path"])
    except OSError:
        pytest.skip("Creating symlinks requires an OS privilege")
    run["xlsx_path"] = str(link)
    with pytest.raises(GateError, match="symlink"):
        _convert(conversion, artifacts_root=conversion[2])


def test_missing_source_snapshot_blocks_validation(conversion):
    scenario = conversion[0]
    converted = _convert(conversion)
    with scenario.service.store.transaction() as db:
        db.execute("DELETE FROM evidence WHERE sha256=?", (converted["source_evidence"],))
    with pytest.raises(GateError, match="missing or corrupted"):
        scenario.service.validate(
            scenario.candidate_id,
            ABReport.model_validate(converted["report"]),
            actor="validator",
            expected_version=scenario.row["version"],
            request_id="missing-lmbench",
        )


def test_equivalent_conversion_is_idempotent(conversion):
    first = _convert(conversion)
    assert _convert(conversion) == first
    assert len(conversion[0].service.store.list("report_conversion")) == 1


def test_generic_report_cannot_bypass_frozen_collector_profile(conversion):
    scenario = conversion[0]
    report = _convert(conversion)["report"]
    # A new, otherwise passing generic report has no registered native evidence.
    for arm in ("baseline", "candidate"):
        for i, measurement in enumerate(report[arm]["measurements"]):
            measurement["pair_id"] = f"unregistered-{i}"
    with pytest.raises(GateError, match="native collector source"):
        scenario.validate(ABReport.model_validate(report))


@pytest.mark.parametrize("damage", ["source-kind", "profile-hash", "metric-mapping"])
def test_quality_and_native_export_reject_validly_hashed_wrong_profile(
    conversion, tmp_path, damage
):
    from hmopt.evolution.learning import quality_report
    from hmopt.evolution.native_memory import export_native_memory

    scenario = conversion[0]
    converted = _convert(conversion)
    scenario.validate(ABReport.model_validate(converted["report"]))
    skill = scenario.promote(actor="curator")
    assert (
        quality_report(scenario.service)["patterns"][scenario.pattern_key]["validated_candidates"]
        == 1
    )
    source = scenario.service.store.read_evidence(converted["source_evidence"])
    if damage == "source-kind":
        source["kind"] = "ic_compare"
    elif damage == "profile-hash":
        source["measurement_profile_sha256"] = "f" * 64
    else:
        source["manifest"]["metrics"][0]["command"] = "lat_syscall read"
    with scenario.service.store.transaction() as db:
        wrong_sha = scenario.service.store.evidence(db, source)
        row = scenario.service.store.get(db, "candidate", scenario.candidate_id)
        row["data"]["validation"]["source_evidence"] = [wrong_sha]
        scenario.service.store.put(
            db, "candidate", scenario.candidate_id, row["data"], row["version"]
        )
    result = quality_report(scenario.service)["patterns"][scenario.pattern_key]
    assert result["validated_candidates"] == 0
    assert result["latest_reports"]["invalid"] == 1
    memory_root, hub_root = tmp_path / "memory", tmp_path / "hub"
    memory_root.mkdir()
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "hm-skill-hub/schemas", hub_root / "schemas"
    )
    with pytest.raises(ValueError, match="valid, real passing"):
        export_native_memory(
            scenario.service,
            skill["id"],
            memory_root=memory_root,
            hub_root=hub_root,
            contributor="owner",
            project="kernel",
            actor="curator",
            request_id="invalid-native",
            title="Preserve syscall semantics",
            body="Apply only after independent paired suite verification.",
            applies_when=["The same workload exercises the modified path."],
            invalidated_by=["The device or workload changes."],
        )
    assert not list(memory_root.rglob("*.md"))
    assert not (hub_root / "staging").exists()


def test_raw_snapshot_survives_original_artifact_removal(conversion):
    scenario, manifest, _ = conversion
    converted = _convert(conversion)
    Path(manifest["pairs"][0]["baseline"]["xlsx_path"]).unlink()
    row = scenario.service.validate(
        scenario.candidate_id,
        ABReport.model_validate(converted["report"]),
        actor="validator",
        expected_version=scenario.row["version"],
        request_id="saved-raw-snapshot",
    )
    assert row["data"]["validation"]["result"]["verdict"] == "pass"


def test_prefilled_measurements_rejected(conversion):
    conversion[1]["baseline"]["measurements"] = [
        {"pair_id": "manufactured", "metrics": {"syscall_latency": 100.0}}
    ]
    with pytest.raises(GateError, match="prefilled"):
        _convert(conversion)


def test_unapproved_candidate_rejected(conversion):
    conversion[0].transition("reject", "owner", {"note": "Experiment is no longer authorized."})
    with pytest.raises(GateError, match="code-approved"):
        _convert(conversion)


def test_correctness_strategy_cannot_use_lmbench(conversion, tmp_path, git_bin):
    from test_evolution_correctness import ready

    directory = tmp_path / "correctness"
    directory.mkdir()
    scenario = ready(directory, git_bin)
    manifest = conversion[1]
    manifest.update(
        candidate_id=scenario.candidate_id,
        plan_digest=scenario.row["data"]["plan_digest"],
        implementation_digest=scenario.row["data"]["implementation_digest"],
    )
    with pytest.raises(GateError, match="correctness"):
        convert_lmbench_report(
            scenario.service, scenario.candidate_id, LmbenchManifest.model_validate(manifest)
        )


def test_digest_disguised_as_xlsx_is_rejected(conversion):
    run = conversion[1]["pairs"][0]["baseline"]
    Path(run["xlsx_path"]).write_text('{"digest":{"ok":true}}', encoding="utf-8")
    _rehash(run)
    with pytest.raises(GateError, match="Invalid lmbench workbook"):
        _convert(conversion)


def test_candidate_change_during_parsing_is_rejected(conversion, monkeypatch):
    from hmopt.evolution import lmbench

    scenario = conversion[0]
    parse = lmbench._read_workbook
    calls = []

    def changed(*args):
        parsed = parse(*args)
        if not calls:
            calls.append(True)
            scenario.transition("reject", "owner", {"note": "Owner rejects stale experiment."})
        return parsed

    monkeypatch.setattr(lmbench, "_read_workbook", changed)
    with pytest.raises(GateError, match="Candidate changed"):
        _convert(conversion)
    assert not scenario.service.store.list("report_conversion")
