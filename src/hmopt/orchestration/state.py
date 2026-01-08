"""Shared RunState used by LangGraph."""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict


class RunState(TypedDict, total=False):
    run_id: str
    iteration: int
    max_iterations: int
    decision: str
    stop_reason: Optional[str]
    baseline_metrics: Dict[str, float]
    candidate_metrics: Dict[str, float]
    best_metrics: Dict[str, float]
    evidence_artifact_id: Optional[str]
    evidence_report_artifact_id: Optional[str]
    patch_artifact_id: Optional[str]
    best_run_id: Optional[str]
    hotspots: List[dict]
    logs: List[str]
    trace_insights: List[dict]


def initial_state(run_id: str, max_iterations: int) -> RunState:
    return {
        "run_id": run_id,
        "iteration": 0,
        "max_iterations": max_iterations,
        "decision": "continue",
        "stop_reason": None,
        "baseline_metrics": {},
        "candidate_metrics": {},
        "best_metrics": {},
        "evidence_artifact_id": None,
        "evidence_report_artifact_id": None,
        "patch_artifact_id": None,
        "best_run_id": run_id,
        "hotspots": [],
        "logs": [],
        "trace_insights": [],
    }
