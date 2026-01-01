"""LangGraph-based optimization loop."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from langgraph.graph import END, START, StateGraph

from hmopt.agents import (
    CoderAgent,
    ConductorAgent,
    ProfilerAgent,
    SafetyGuard,
    TraceAnalystAgent,
    VerifierAgent,
)
from hmopt.analysis.correlation.aligner import align_hotspots_to_psg
from hmopt.analysis.correlation.ranker import rank_correlated
from hmopt.analysis.runtime.hotspot import HotspotCandidate, persist_hotspots, rank_hotspots
from hmopt.analysis.runtime.metrics import Metric, compute_delta, record_metrics
from hmopt.analysis.runtime.traces import parse_framegraph, parse_hiperf, parse_hitrace
from hmopt.analysis.static import build_psg, index_repo
from hmopt.core.config import AppConfig
from hmopt.core.llm import LLMClient
from hmopt.core.run_context import RunContext, build_context, register_run
from hmopt.orchestration.state import RunState, initial_state
from hmopt.storage.db import models
from hmopt.tools import (
    DummyBuildAdapter,
    DummyProfilerAdapter,
    DummyTestAdapter,
    ShellBuildAdapter,
    ShellProfilerAdapter,
    ShellTestAdapter,
    get_repo_state,
    snapshot_files,
)


@dataclass
class PipelineServices:
    config: AppConfig
    ctx: RunContext
    safety: SafetyGuard
    llm: LLMClient
    conductor: ConductorAgent
    coder: CoderAgent
    trace_analyst: TraceAnalystAgent
    verifier: VerifierAgent
    profiler: ProfilerAgent
    psg: object | None = None


logger = logging.getLogger(__name__)


def _metrics_to_map(metrics: Iterable[Metric]) -> Dict[str, float]:
    return {m.metric_name: m.value for m in metrics}


def _hotspot_to_dict(h: HotspotCandidate) -> dict:
    return {
        "symbol": h.symbol,
        "file_path": h.file_path,
        "line_start": h.line_start,
        "line_end": h.line_end,
        "score": h.score,
        "evidence_artifacts": h.evidence_artifacts or [],
    }


def _hotspot_from_dict(data: dict) -> HotspotCandidate:
    return HotspotCandidate(
        symbol=data.get("symbol"),
        file_path=data.get("file_path"),
        line_start=data.get("line_start"),
        line_end=data.get("line_end"),
        score=data.get("score", 0.0),
        evidence_artifacts=data.get("evidence_artifacts", []),
    )


def make_services(config: AppConfig) -> PipelineServices:
    ctx = build_context(config)
    safety = SafetyGuard(config.llm.allow_external_proxy_models)
    llm = LLMClient(
        api_base=config.llm.base_url,
        api_key=config.llm.api_key,
        default_model=config.llm.model,
        allow_external_proxy_models=config.llm.allow_external_proxy_models,
    )

    if config.adapters.dummy:
        build_adapter = DummyBuildAdapter()
        test_adapter = DummyTestAdapter()
        profiler_adapter = DummyProfilerAdapter()
    else:
        build_cmd = config.adapters.build_command or "make -j"
        test_cmd = config.adapters.test_command or "ctest || true"
        profile_cmd = config.adapters.profile_command or config.adapters.workload_command or "true"
        build_adapter = ShellBuildAdapter(build_cmd)
        test_adapter = ShellTestAdapter(test_cmd)
        profiler_adapter = ShellProfilerAdapter(profile_cmd)

    verifier = VerifierAgent(build_adapter, test_adapter)
    profiler = ProfilerAgent(profiler_adapter)
    conductor = ConductorAgent(llm, safety)
    coder = CoderAgent(llm, safety)
    trace_analyst = TraceAnalystAgent(llm, safety)

    return PipelineServices(
        config=config,
        ctx=ctx,
        safety=safety,
        llm=llm,
        conductor=conductor,
        coder=coder,
        trace_analyst=trace_analyst,
        verifier=verifier,
        profiler=profiler,
    )


def _profile_and_analyze(services: PipelineServices, state: RunState, label: str) -> RunState:
    """Run profiler adapter, persist artifacts, compute metrics/hotspots."""
    run_id = state["run_id"]
    logger.info("Profiling phase start: run=%s label=%s", run_id, label)
    output_dir = services.ctx.run_dir / label
    profile_result = services.profiler.profile(
        services.config.project.workload or "default", output_dir
    )
    artifacts = {}
    for kind, path in profile_result.artifacts.items():
        art = services.ctx.artifact_store.store_file(
            path, kind=kind, run_id=run_id, session=services.ctx.session
        )
        artifacts[kind] = art

    metrics: list[Metric] = []
    hotspots: list[HotspotCandidate] = []
    if "framegraph" in profile_result.artifacts:
        fg = parse_framegraph(profile_result.artifacts["framegraph"])
        metrics.extend(fg.to_metrics())
    if "hitrace" in profile_result.artifacts:
        ht = parse_hitrace(profile_result.artifacts["hitrace"])
        metrics.extend(ht.to_metrics())
    if "hiperf" in profile_result.artifacts:
        hp = parse_hiperf(profile_result.artifacts["hiperf"])
        hotspots = rank_hotspots(hp.hotspot_costs, hp.edge_costs, top_n=10)
        if services.psg:
            hotspots = align_hotspots_to_psg(hotspots, services.psg)
        hotspots = rank_correlated(hotspots, {m.metric_name: m.value for m in metrics})
        persist_hotspots(services.ctx.session, run_id, hotspots)

    record_metrics(services.ctx.session, run_id, metrics)
    services.ctx.session.commit()

    metrics_map = _metrics_to_map(metrics)
    state["candidate_metrics"] = metrics_map
    state["hotspots"] = [_hotspot_to_dict(h) for h in hotspots]
    if label == "baseline":
        state["baseline_metrics"] = metrics_map
        state["best_metrics"] = metrics_map
    logger.info(
        "Profiling phase done: label=%s metrics=%s hotspots=%d",
        label,
        {k: round(v, 3) for k, v in metrics_map.items()},
        len(hotspots),
    )
    return state


def _build_evidence(services: PipelineServices, state: RunState) -> RunState:
    logger.info("Building evidence pack for iteration=%s", state["iteration"])
    metrics = state.get("candidate_metrics") or state.get("baseline_metrics") or {}
    hotspots = [_hotspot_from_dict(h) for h in state.get("hotspots", [])]
    summary = services.trace_analyst.analyze(metrics, hotspots)
    pack = {
        "iteration": state["iteration"],
        "metrics": metrics,
        "hotspots": [h.__dict__ for h in hotspots],
        "summary": summary,
    }
    artifact = services.ctx.artifact_store.store_json(
        pack, kind="evidence_pack", run_id=state["run_id"], session=services.ctx.session
    )
    state["evidence_artifact_id"] = artifact.artifact_id
    logger.info("Evidence pack stored: artifact=%s", artifact.artifact_id)
    return state


def _conductor_decide(services: PipelineServices, state: RunState) -> RunState:
    metrics = state.get("candidate_metrics", {})
    best_summary = f"best fps {state.get('best_metrics', {}).get('fps_avg', 0):.2f}"
    evidence_summary = json.dumps(metrics)[:800]
    decision = services.conductor.decide(
        evidence_summary=evidence_summary,
        best_summary=best_summary,
        iteration=state["iteration"],
        max_iterations=state["max_iterations"],
    )
    state["decision"] = decision["decision"]
    state.setdefault("logs", []).append(decision["rationale"])
    state["next_action"] = decision["next_action"]
    logger.info(
        "Conductor decision: decision=%s next=%s iteration=%s",
        state["decision"],
        state["next_action"],
        state["iteration"],
    )
    return state


def _coder_generate_patch(services: PipelineServices, state: RunState) -> RunState:
    if state.get("decision") == "stop":
        return state
    logger.info("Coder generating patch for iteration=%s", state["iteration"])
    instructions = state.get("next_action", "Improve hotspot.")
    patch_text = services.coder.generate_patch(
        Path(services.config.project.repo_path), instructions, iteration=state["iteration"]
    )
    art = services.ctx.artifact_store.store_text(
        patch_text,
        kind="patch_diff",
        run_id=state["run_id"],
        extension=".patch",
        session=services.ctx.session,
    )
    state["patch_artifact_id"] = art.artifact_id
    patch_row = models.Patch(
        run_id=state["run_id"],
        iteration=state["iteration"],
        diff_artifact_id=art.artifact_id,
        apply_status="pending",
        files_changed_json=[],
    )
    services.ctx.session.add(patch_row)
    services.ctx.session.commit()
    logger.info("Patch generated: artifact=%s", art.artifact_id)
    return state


def _apply_patch(services: PipelineServices, state: RunState) -> RunState:
    if not state.get("patch_artifact_id"):
        return state
    logger.info("Applying patch artifact=%s", state["patch_artifact_id"])
    patch_artifact = (
        services.ctx.session.query(models.Artifact)
        .filter(models.Artifact.artifact_id == state["patch_artifact_id"])
        .one()
    )
    patch_text = Path(patch_artifact.path).read_text(encoding="utf-8")
    status = "skipped"
    log = "dummy adapters set; patch not applied"
    if not services.config.adapters.dummy:
        try:
            proc = subprocess.run(
                ["patch", "-p1"],
                input=patch_text,
                text=True,
                cwd=services.config.project.repo_path,
                capture_output=True,
                check=False,
            )
            status = "applied" if proc.returncode == 0 else "failed"
            log = proc.stdout + "\n" + proc.stderr
        except Exception as exc:
            status = "failed"
            log = f"apply failed: {exc}"

    patch_row = (
        services.ctx.session.query(models.Patch)
        .filter(models.Patch.diff_artifact_id == state["patch_artifact_id"])
        .one()
    )
    patch_row.apply_status = status
    log_art = services.ctx.artifact_store.store_text(
        log, kind="patch_apply_log", run_id=state["run_id"], session=services.ctx.session
    )
    services.ctx.session.commit()
    state["patch_log_artifact_id"] = log_art.artifact_id
    logger.info("Patch apply result: status=%s log_artifact=%s", status, log_art.artifact_id)
    return state


def _verify(services: PipelineServices, state: RunState) -> RunState:
    logger.info("Verification start: build+test")
    result = services.verifier.verify(Path(services.config.project.repo_path))
    build_log = services.ctx.artifact_store.store_text(
        result.build.log, kind="build_log", run_id=state["run_id"], session=services.ctx.session
    )
    test_log = services.ctx.artifact_store.store_text(
        result.tests.log, kind="test_log", run_id=state["run_id"], session=services.ctx.session
    )
    services.ctx.session.commit()
    state["verification_success"] = result.success
    state["build_log_artifact_id"] = build_log.artifact_id
    state["test_log_artifact_id"] = test_log.artifact_id
    if not result.success:
        state["force_stop"] = True
        state["stop_reason"] = "verification_failed"
    logger.info("Verification done: success=%s", result.success)
    return state


def _evaluate(services: PipelineServices, state: RunState) -> RunState:
    baseline = state.get("baseline_metrics", {})
    candidate = state.get("candidate_metrics", {})
    delta = compute_delta(baseline, candidate)
    perf_improved = candidate.get("fps_avg", 0.0) > state.get("best_metrics", {}).get("fps_avg", 0.0)
    if perf_improved:
        state["best_metrics"] = candidate
    eval_row = models.Evaluation(
        run_id=state["run_id"],
        baseline_run_id=state["best_run_id"],
        delta_metrics_json=delta,
        correctness_passed=bool(state.get("verification_success", True)),
        perf_improved=perf_improved,
        notes=state.get("stop_reason"),
    )
    services.ctx.session.add(eval_row)
    services.ctx.session.commit()
    state["iteration"] += 1
    state["perf_improved"] = perf_improved
    logger.info(
        "Evaluation: iteration=%s perf_improved=%s delta_keys=%s",
        state["iteration"],
        perf_improved,
        list(delta.keys()),
    )
    return state


def _report(services: PipelineServices, state: RunState) -> RunState:
    logger.info("Generating report for run=%s", state["run_id"])
    metrics = state.get("candidate_metrics", {})
    hotspots = state.get("hotspots", [])
    lines = [
        f"# HMOPT run {state['run_id']}",
        f"Status: {state.get('stop_reason') or 'completed'}",
        f"Iterations: {state.get('iteration', 0)}",
        "## Metrics",
    ]
    for k, v in metrics.items():
        lines.append(f"- {k}: {v}")
    lines.append("## Hotspots")
    for hs in hotspots:
        lines.append(f"- {hs.get('symbol')} score={hs.get('score')}")
    report_text = "\n".join(lines)
    art = services.ctx.artifact_store.store_text(
        report_text, kind="report", run_id=state["run_id"], extension=".md", session=services.ctx.session
    )
    export_dataset(services.ctx.session, [state["run_id"]], services.ctx.run_dir / "dataset.json")
    run_row = services.ctx.session.query(models.Run).filter(models.Run.run_id == state["run_id"]).one()
    run_row.status = "succeeded" if not state.get("stop_reason") else "stopped"
    services.ctx.session.commit()
    state["report_artifact_id"] = art.artifact_id
    logger.info("Report generated: artifact=%s", art.artifact_id)
    return state


def _stop_or_continue(state: RunState) -> str:
    if state.get("force_stop"):
        return "stop"
    if state.get("iteration", 0) >= state.get("max_iterations", 1):
        state["stop_reason"] = "iteration_budget"
        return "stop"
    return "continue"


def _conductor_branch(state: RunState) -> str:
    return "continue" if state.get("decision", "continue") == "continue" else "stop"


def build_graph(services: PipelineServices, max_iterations: int) -> StateGraph:
    graph = StateGraph(RunState)

    def init_node(state: RunState) -> RunState:
        register_run(services.ctx, workload_id=services.config.project.workload)
        logger.info("Run initialized: run_id=%s pipeline=%s", state["run_id"], services.config.pipeline)
        repo_state = get_repo_state(services.config.project.repo_path)
        snapshot = snapshot_files(services.config.project.repo_path)
        snapshot_art = services.ctx.artifact_store.store_json(
            snapshot, kind="repo_snapshot", run_id=state["run_id"], session=services.ctx.session
        )
        run_row = services.ctx.session.query(models.Run).filter(models.Run.run_id == state["run_id"]).one()
        run_row.repo_rev = repo_state.get("commit")
        run_row.repo_uri = repo_state.get("remote")
        run_row.repo_dirty = bool(repo_state.get("dirty"))
        run_row.status = "running"
        services.ctx.session.commit()
        state["snapshot_artifact_id"] = snapshot_art.artifact_id
        state["max_iterations"] = max_iterations
        return state

    def static_analysis_node(state: RunState) -> RunState:
        logger.info("Static analysis start (PSG)")
        symbols = index_repo(services.config.project.repo_path)
        services.psg = build_psg(symbols)
        art = services.ctx.artifact_store.store_text(
            services.psg.to_json(), kind="psg", run_id=state["run_id"], extension=".json", session=services.ctx.session
        )
        graph_row = models.Graph(
            run_id=state["run_id"],
            kind="psg",
            format="json",
            payload_artifact_id=art.artifact_id,
            metadata_json={"nodes": len(services.psg.nodes), "edges": len(services.psg.edges)},
        )
        services.ctx.session.add(graph_row)
        services.ctx.session.commit()
        logger.info("Static analysis done: nodes=%d edges=%d", len(services.psg.nodes), len(services.psg.edges))
        return state

    graph.add_node("init_run", init_node)
    graph.add_node("static_analysis", static_analysis_node)
    graph.add_node("baseline_profile", lambda s: _profile_and_analyze(services, s, "baseline"))
    graph.add_node("build_evidence", lambda s: _build_evidence(services, s))
    graph.add_node("conductor_decide", lambda s: _conductor_decide(services, s))
    graph.add_node("coder_generate_patch", lambda s: _coder_generate_patch(services, s))
    graph.add_node("apply_patch", lambda s: _apply_patch(services, s))
    graph.add_node("verify_build_test", lambda s: _verify(services, s))
    graph.add_node("profile_candidate", lambda s: _profile_and_analyze(services, s, f"iter_{s.get('iteration', 0)}"))
    graph.add_node("evaluate", lambda s: _evaluate(services, s))
    graph.add_node("stop_or_continue", lambda s: s)
    graph.add_node("generate_report", lambda s: _report(services, s))

    graph.add_edge(START, "init_run")
    graph.add_edge("init_run", "static_analysis")
    graph.add_edge("static_analysis", "baseline_profile")
    graph.add_edge("baseline_profile", "build_evidence")
    graph.add_edge("build_evidence", "conductor_decide")
    graph.add_conditional_edges("conductor_decide", _conductor_branch, {"continue": "coder_generate_patch", "stop": "generate_report"})
    graph.add_edge("coder_generate_patch", "apply_patch")
    graph.add_edge("apply_patch", "verify_build_test")
    graph.add_edge("verify_build_test", "profile_candidate")
    graph.add_edge("profile_candidate", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        lambda s: "stop" if s.get("force_stop") else "continue",
        {"continue": "stop_or_continue", "stop": "generate_report"},
    )
    graph.add_conditional_edges("stop_or_continue", _stop_or_continue, {"continue": "build_evidence", "stop": "generate_report"})
    graph.add_edge("generate_report", END)

    return graph


def run_pipeline(config: AppConfig) -> str:
    """Run optimization pipeline and return run_id."""
    services = make_services(config)
    state = initial_state(services.ctx.run_id, config.iterations)
    graph = build_graph(services, config.iterations).compile()
    graph.invoke(state)
    services.ctx.session.close()
    return services.ctx.run_id


def run_artifact_analysis(
    config: AppConfig,
    artifacts: list[dict],
    *,
    run_conductor: bool = True,
    run_coder: bool = True,
    run_verify: bool = False,
    run_profile: bool = False,
) -> str:
    """Run a shortened pipeline that ingests existing artifacts and optionally drives LLM suggestions."""
    services = make_services(config)
    state: RunState = initial_state(services.ctx.run_id, max_iterations=1)

    # init + static
    register_run(services.ctx, workload_id=config.project.workload)
    repo_state = get_repo_state(config.project.repo_path)
    snapshot = snapshot_files(config.project.repo_path)
    snapshot_art = services.ctx.artifact_store.store_json(
        snapshot, kind="repo_snapshot", run_id=state["run_id"], session=services.ctx.session
    )
    run_row = (
        services.ctx.session.query(models.Run).filter(models.Run.run_id == state["run_id"]).one()
    )
    run_row.repo_rev = repo_state.get("commit")
    run_row.repo_uri = repo_state.get("remote")
    run_row.repo_dirty = bool(repo_state.get("dirty"))
    run_row.status = "running"
    services.ctx.session.commit()
    state["snapshot_artifact_id"] = snapshot_art.artifact_id

    logger.info("Artifact analysis: ingest %d artifacts", len(artifacts))
    symbols = index_repo(config.project.repo_path)
    services.psg = build_psg(symbols)
    psg_art = services.ctx.artifact_store.store_text(
        services.psg.to_json(), kind="psg", run_id=state["run_id"], extension=".json", session=services.ctx.session
    )
    graph_row = models.Graph(
        run_id=state["run_id"],
        kind="psg",
        format="json",
        payload_artifact_id=psg_art.artifact_id,
        metadata_json={"nodes": len(services.psg.nodes), "edges": len(services.psg.edges)},
    )
    services.ctx.session.add(graph_row)

    metrics: list[Metric] = []
    hotspots: list[HotspotCandidate] = []

    for item in artifacts:
        kind = item.get("kind")
        path = Path(item.get("path"))
        art = services.ctx.artifact_store.store_file(
            path, kind=kind or "unknown", run_id=state["run_id"], session=services.ctx.session
        )
        if kind == "framegraph":
            fg = parse_framegraph(path)
            metrics.extend(fg.to_metrics())
        elif kind == "hitrace":
            ht = parse_hitrace(path)
            metrics.extend(ht.to_metrics())
        elif kind == "hiperf":
            hp = parse_hiperf(path)
            hs = rank_hotspots(hp.hotspot_costs, hp.edge_costs, top_n=15)
            if services.psg:
                hs = align_hotspots_to_psg(hs, services.psg)
            hotspots.extend(rank_correlated(hs, {}))
        else:
            logger.info("Stored artifact with no parser: %s", kind)

    record_metrics(services.ctx.session, state["run_id"], metrics)
    persist_hotspots(services.ctx.session, state["run_id"], hotspots)
    services.ctx.session.commit()

    state["baseline_metrics"] = _metrics_to_map(metrics)
    state["hotspots"] = [_hotspot_to_dict(h) for h in hotspots]

    state = _build_evidence(services, state)

    if run_conductor:
        state = _conductor_decide(services, state)
    else:
        state["decision"] = "continue" if run_coder else "stop"
        state["next_action"] = "analyze hotspots"

    if run_coder and state.get("decision") == "continue":
        state = _coder_generate_patch(services, state)
        state = _apply_patch(services, state)
        if run_verify:
            state = _verify(services, state)
        if run_profile and not state.get("force_stop"):
            state = _profile_and_analyze(services, state, f"iter_{state.get('iteration', 0)}")
            state = _evaluate(services, state)

    state = _report(services, state)
    run_row.status = "succeeded"
    services.ctx.session.commit()
    services.ctx.session.close()
    logger.info("Artifact analysis finished: run_id=%s", state["run_id"])
    return state["run_id"]
