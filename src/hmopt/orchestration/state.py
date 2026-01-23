"""Shared RunState used by LangGraph."""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict


class RunState(TypedDict, total=False):
    run_id: str
    iteration: int
    max_iterations: int
    decision: str
    next_action: Optional[str]
    stop_reason: Optional[str]
    force_stop: bool
    baseline_metrics: Dict[str, float]
    candidate_metrics: Dict[str, float]
    best_metrics: Dict[str, float]
    perf_improved: Optional[bool]
    verification_success: Optional[bool]
    snapshot_artifact_id: Optional[str]
    evidence_artifact_id: Optional[str]
    evidence_report_artifact_id: Optional[str]
    evidence_summary: Optional[str]
    query_summary: Optional[str]
    query_artifact_id: Optional[str]
    review_decision: Optional[str]
    review_artifact_id: Optional[str]
    patch_artifact_id: Optional[str]
    patch_apply_status: Optional[str]
    patch_apply_log_artifact_id: Optional[str]
    build_log_artifact_id: Optional[str]
    test_log_artifact_id: Optional[str]
    best_run_id: Optional[str]
    report_artifact_id: Optional[str]
    code_index_versions: List[dict]
    runtime_index_versions: List[dict]
    hotspots: List[dict]
    logs: List[str]
    trace_insights: List[dict]


def initial_state(run_id: str, max_iterations: int) -> RunState:
    return {
        "run_id": run_id,
        "iteration": 0,
        "max_iterations": max_iterations,
        "decision": "continue",
        "next_action": None,
        "stop_reason": None,
        "force_stop": False,
        "baseline_metrics": {},
        "candidate_metrics": {},
        "best_metrics": {},
        "perf_improved": None,
        "verification_success": None,
        "snapshot_artifact_id": None,
        "evidence_artifact_id": None,
        "evidence_report_artifact_id": None,
        "evidence_summary": None,
        "query_summary": None,
        "query_artifact_id": None,
        "review_decision": None,
        "review_artifact_id": None,
        "patch_artifact_id": None,
        "patch_apply_status": None,
        "patch_apply_log_artifact_id": None,
        "build_log_artifact_id": None,
        "test_log_artifact_id": None,
        "best_run_id": run_id,
        "report_artifact_id": None,
        "code_index_versions": [],
        "runtime_index_versions": [],
        "hotspots": [],
        "logs": [],
        "trace_insights": [],
    }
