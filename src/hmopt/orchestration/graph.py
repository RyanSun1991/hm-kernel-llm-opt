"""LangGraph-based optimization loop."""

from __future__ import annotations

import logging
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from hmopt.analysis.correlation.aligner import align_hotspots_to_psg
from hmopt.analysis.correlation.ranker import rank_correlated
from hmopt.analysis.runtime.hotspot import HotspotCandidate, persist_hotspots, rank_hotspots
from hmopt.analysis.runtime.metrics import Metric, record_metrics
from hmopt.analysis.runtime.traces import (
    parse_flamegraph,
    parse_hiperf,
    parse_hitrace,
    parse_sysfs_trace,
)
from hmopt.analysis.static import build_psg, index_repo
from hmopt.core.config import AppConfig
from hmopt.orchestration import nodes as pipeline_nodes
from hmopt.orchestration.state import RunState, initial_state
from hmopt.storage.db import models
from hmopt.tools import get_repo_state

logger = logging.getLogger(__name__)


def build_graph(
    services: pipeline_nodes.PipelineServices, max_iterations: int
) -> StateGraph:
    graph = StateGraph(RunState)

    graph.add_node("init_run", lambda s: pipeline_nodes.init_run_node(services, s))
    graph.add_node("static_analysis", lambda s: pipeline_nodes.static_analysis_node(services, s))
    graph.add_node("index_code", lambda s: pipeline_nodes.index_code_node(services, s))
    graph.add_node(
        "baseline_profile",
        lambda s: pipeline_nodes._profile_and_analyze(services, s, "baseline"),
    )
    graph.add_node("build_evidence", lambda s: pipeline_nodes._build_evidence(services, s))
    graph.add_node("index_runtime", lambda s: pipeline_nodes.index_runtime_node(services, s))
    graph.add_node("query_insights", lambda s: pipeline_nodes.query_insights_node(services, s))
    graph.add_node("conductor_decide", lambda s: pipeline_nodes._conductor_decide(services, s))
    graph.add_node("coder_generate_patch", lambda s: pipeline_nodes._coder_generate_patch(services, s))
    graph.add_node("review_patch", lambda s: pipeline_nodes.review_patch_node(services, s))
    graph.add_node("apply_patch", lambda s: pipeline_nodes._apply_patch(services, s))
    graph.add_node("verify_build_test", lambda s: pipeline_nodes._verify(services, s))
    graph.add_node(
        "profile_candidate",
        lambda s: pipeline_nodes._profile_and_analyze(
            services, s, f"iter_{s.get('iteration', 0)}"
        ),
    )
    graph.add_node("evaluate", lambda s: pipeline_nodes._evaluate(services, s))
    graph.add_node("stop_or_continue", lambda s: s)
    graph.add_node("generate_report", lambda s: pipeline_nodes._report(services, s))

    graph.add_edge(START, "init_run")
    graph.add_edge("init_run", "static_analysis")
    graph.add_edge("static_analysis", "index_code")
    graph.add_edge("index_code", "baseline_profile")
    graph.add_edge("baseline_profile", "build_evidence")
    graph.add_edge("build_evidence", "index_runtime")
    graph.add_edge("index_runtime", "query_insights")
    graph.add_edge("query_insights", "conductor_decide")
    graph.add_conditional_edges(
        "conductor_decide",
        pipeline_nodes._conductor_branch,
        {"continue": "coder_generate_patch", "stop": "generate_report"},
    )
    graph.add_edge("coder_generate_patch", "review_patch")
    graph.add_edge("review_patch", "apply_patch")
    graph.add_edge("apply_patch", "verify_build_test")
    graph.add_edge("verify_build_test", "profile_candidate")
    graph.add_edge("profile_candidate", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        lambda s: "stop" if s.get("force_stop") else "continue",
        {"continue": "stop_or_continue", "stop": "generate_report"},
    )
    graph.add_conditional_edges(
        "stop_or_continue",
        pipeline_nodes._stop_or_continue,
        {"continue": "build_evidence", "stop": "generate_report"},
    )
    graph.add_edge("generate_report", END)

    return graph


def run_pipeline(config: AppConfig) -> str:
    """Run optimization pipeline and return run_id."""
    services = pipeline_nodes.make_services(config)
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
    services = pipeline_nodes.make_services(config)
    state: RunState = initial_state(services.ctx.run_id, max_iterations=1)

    state = pipeline_nodes.init_run_node(services, state)

    repo_state = get_repo_state(config.project.repo_path)
    repo_rev = repo_state.get("commit")
    repo_dirty = bool(repo_state.get("dirty"))

    reused_psg = False
    if repo_rev:
        existing_graph = (
            services.ctx.session.query(models.Graph)
            .join(models.Run, models.Graph.run_id == models.Run.run_id)
            .filter(
                models.Graph.kind == "psg",
                models.Run.repo_rev == repo_rev,
                models.Run.repo_dirty == repo_dirty,
                models.Run.run_id != state["run_id"],
            )
            .order_by(models.Run.created_at.desc())
            .first()
        )
        if existing_graph:
            loaded_psg = pipeline_nodes._load_psg_from_artifact(
                services, existing_graph.payload_artifact_id
            )
            if loaded_psg:
                services.psg = loaded_psg
                graph_row = models.Graph(
                    run_id=state["run_id"],
                    kind="psg",
                    format="json",
                    payload_artifact_id=existing_graph.payload_artifact_id,
                    metadata_json=existing_graph.metadata_json,
                )
                services.ctx.session.add(graph_row)
                reused_psg = True
                logger.info("Reused PSG from run %s", existing_graph.run_id)

    if not reused_psg:
        symbols = index_repo(config.project.repo_path)
        services.psg = build_psg(symbols)
        psg_art = services.ctx.artifact_store.store_text(
            services.psg.to_json(),
            kind="psg",
            run_id=state["run_id"],
            extension=".json",
            session=services.ctx.session,
        )
        graph_row = models.Graph(
            run_id=state["run_id"],
            kind="psg",
            format="json",
            payload_artifact_id=psg_art.artifact_id,
            metadata_json={"nodes": len(services.psg.nodes), "edges": len(services.psg.edges)},
        )
        services.ctx.session.add(graph_row)
        services.ctx.session.commit()

    state = pipeline_nodes.index_code_node(services, state)

    metrics: list[Metric] = []
    hotspots: list[HotspotCandidate] = []

    for item in artifacts:
        kind = item.get("kind")
        if kind == "framegraph":
            kind = "flamegraph"
        path = Path(item.get("path"))
        if path.is_dir() and kind != "flamegraph":
            logger.warning("Skipping directory artifact (unsupported kind=%s): %s", kind, path)
            continue
        if kind == "flamegraph":
            if path.is_dir():
                html_files = sorted(path.rglob("*__sysmgr_hiperfReport.html"))
                if html_files:
                    for html_file in html_files:
                        services.ctx.artifact_store.store_file(
                            html_file,
                            kind=kind,
                            run_id=state["run_id"],
                            session=services.ctx.session,
                            metadata={"source_dir": str(path)},
                        )
                else:
                    logger.warning("Flamegraph directory had no report HTML: %s", path)
            else:
                services.ctx.artifact_store.store_file(
                    path,
                    kind=kind or "unknown",
                    run_id=state["run_id"],
                    session=services.ctx.session,
                )
            fg_results = parse_flamegraph(path)
            fg_hotspots_all: list[HotspotCandidate] = []
            for fg in fg_results:
                metrics.extend(fg.to_metrics())
                pipeline_nodes._store_flamegraph_maps(services, state["run_id"], fg, path)
                if fg.symbol_counts:
                    fg_hotspots = pipeline_nodes._hotspots_from_symbol_counts(
                        fg.symbol_counts,
                        top_n=services.config.indexing.hotspot_top_k,
                        min_ratio=services.config.indexing.hotspot_min_ratio,
                        min_abs=services.config.indexing.hotspot_min_abs,
                        total=fg.event_count_total,
                    )
                    if services.psg:
                        fg_hotspots = align_hotspots_to_psg(fg_hotspots, services.psg)
                    fg_hotspots_all.extend(fg_hotspots)
                state.setdefault("trace_insights", []).append(
                    pipeline_nodes._flamegraph_trace_insight(fg, top_n=20)
                )
            hotspots.extend(fg_hotspots_all)
            comparison = pipeline_nodes._flamegraph_comparison_insight(fg_results, top_n=10)
            if comparison:
                state.setdefault("trace_insights", []).append(comparison)
        elif kind == "hitrace":
            services.ctx.artifact_store.store_file(
                path,
                kind=kind or "unknown",
                run_id=state["run_id"],
                session=services.ctx.session,
            )
            ht = parse_hitrace(path)
            metrics.extend(ht.to_metrics())
        elif kind == "sysfs":
            services.ctx.artifact_store.store_file(
                path,
                kind=kind or "unknown",
                run_id=state["run_id"],
                session=services.ctx.session,
            )
            st = parse_sysfs_trace(path)
            metrics.extend(st.to_metrics())
        elif kind == "hiperf":
            services.ctx.artifact_store.store_file(
                path,
                kind=kind or "unknown",
                run_id=state["run_id"],
                session=services.ctx.session,
            )
            hp = parse_hiperf(path)
            hs = rank_hotspots(hp.hotspot_costs, hp.edge_costs, top_n=15)
            if services.psg:
                hs = align_hotspots_to_psg(hs, services.psg)
            hotspots.extend(hs)
        else:
            services.ctx.artifact_store.store_file(
                path,
                kind=kind or "unknown",
                run_id=state["run_id"],
                session=services.ctx.session,
            )
            logger.info("Stored artifact with no parser: %s", kind)

    metrics_map = pipeline_nodes._metrics_to_map(metrics)
    if hotspots:
        hotspots = rank_correlated(hotspots, metrics_map, limit=20)
    record_metrics(services.ctx.session, state["run_id"], metrics)
    persist_hotspots(services.ctx.session, state["run_id"], hotspots)
    services.ctx.session.commit()

    state["baseline_metrics"] = metrics_map
    state["candidate_metrics"] = metrics_map
    state["best_metrics"] = metrics_map
    state["hotspots"] = [pipeline_nodes._hotspot_to_dict(h) for h in hotspots]

    state = pipeline_nodes._build_evidence(services, state)
    state = pipeline_nodes.index_runtime_node(services, state)
    state = pipeline_nodes.query_insights_node(services, state)

    if run_conductor:
        state = pipeline_nodes._conductor_decide(services, state)
    else:
        state["decision"] = "continue" if run_coder else "stop"
        state["next_action"] = "analyze hotspots"

    if run_coder and state.get("decision") == "continue":
        state = pipeline_nodes._coder_generate_patch(services, state)
        state = pipeline_nodes.review_patch_node(services, state)
        state = pipeline_nodes._apply_patch(services, state)
        if run_verify:
            state = pipeline_nodes._verify(services, state)
        if run_profile and not state.get("force_stop"):
            state = pipeline_nodes._profile_and_analyze(
                services, state, f"iter_{state.get('iteration', 0)}"
            )
            state = pipeline_nodes._evaluate(services, state)

    state = pipeline_nodes._report(services, state)
    services.ctx.session.close()
    logger.info("Artifact analysis finished: run_id=%s", state["run_id"])
    return state["run_id"]
