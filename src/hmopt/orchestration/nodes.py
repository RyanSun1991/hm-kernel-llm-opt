"""Pipeline node implementations for orchestration graph."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from hmopt.agents import (
    CoderAgent,
    ConductorAgent,
    ProfilerAgent,
    ReviewAgent,
    SafetyGuard,
    TraceAnalystAgent,
    VerifierAgent,
)
from hmopt.analysis.correlation.aligner import align_hotspots_to_psg
from hmopt.analysis.correlation.ranker import rank_correlated
from hmopt.analysis.runtime.hotspot import HotspotCandidate, persist_hotspots, rank_hotspots
from hmopt.analysis.runtime.metrics import Metric, compute_delta, record_metrics
from hmopt.analysis.runtime.traces import (
    parse_flamegraph,
    parse_hiperf,
    parse_hitrace,
    parse_sysfs_trace,
)
from hmopt.analysis.static import build_psg, index_repo
from hmopt.analysis.static.psg import PsgGraph
from hmopt.core.config import AppConfig
from hmopt.core.llm import LLMClient
from hmopt.core.run_context import RunContext, build_context, register_run
from hmopt.indexing import (
    build_kernel_index,
    build_kernel_indexes,
    build_runtime_index,
    route_query,
)
from hmopt.orchestration.state import RunState
from hmopt.prompting import PromptRegistry, build_prompt_registry, resolve_prompt_name
from hmopt.storage.db import models
from hmopt.storage.vector.store import VectorRecord
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

logger = logging.getLogger(__name__)


@dataclass
class PipelineServices:
    config: AppConfig
    ctx: RunContext
    safety: SafetyGuard
    llm: LLMClient
    prompts: PromptRegistry
    conductor: ConductorAgent
    coder: CoderAgent
    reviewer: ReviewAgent
    trace_analyst: TraceAnalystAgent
    verifier: VerifierAgent
    profiler: ProfilerAgent
    psg: object | None = None


def make_services(config: AppConfig) -> PipelineServices:
    ctx = build_context(config)
    safety = SafetyGuard(config.llm.allow_external_proxy_models)
    llm = LLMClient(
        api_base=config.llm.base_url,
        api_key=config.llm.api_key,
        default_model=config.llm.model,
        allow_external_proxy_models=config.llm.allow_external_proxy_models,
    )
    prompts = build_prompt_registry(config)

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
    conductor_prompt = resolve_prompt_name(config, "conductor", "conductor")
    coder_prompt = resolve_prompt_name(config, "coder", "coder")
    trace_prompt = resolve_prompt_name(config, "trace_analyst", "trace_analyst")
    review_prompt = resolve_prompt_name(config, "review", "code_review")
    conductor = ConductorAgent(llm, safety, prompts=prompts, prompt_name=conductor_prompt)
    coder = CoderAgent(llm, safety, prompts=prompts, prompt_name=coder_prompt)
    reviewer = ReviewAgent(llm, safety, prompts=prompts, prompt_name=review_prompt)
    trace_analyst = TraceAnalystAgent(llm, safety, prompts=prompts, prompt_name=trace_prompt)

    return PipelineServices(
        config=config,
        ctx=ctx,
        safety=safety,
        llm=llm,
        prompts=prompts,
        conductor=conductor,
        coder=coder,
        reviewer=reviewer,
        trace_analyst=trace_analyst,
        verifier=verifier,
        profiler=profiler,
    )


def _hotspots_from_symbol_counts(
    symbol_counts: dict[str, float],
    *,
    top_n: int = 20,
    min_ratio: float = 0.001,
    min_abs: float = 0.0,
    total: float | None = None,
) -> list[HotspotCandidate]:
    threshold = min_abs
    if total:
        threshold = max(threshold, total * min_ratio)
    filtered = [(name, score) for name, score in symbol_counts.items() if score >= threshold]
    ordered = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]
    return [
        HotspotCandidate(
            symbol=name,
            file_path=None,
            line_start=None,
            line_end=None,
            score=score,
            evidence_artifacts=[],
        )
        for name, score in ordered
    ]


def _top_weighted_items(
    weights: dict[str, float], *, top_n: int, total: float | None = None
) -> list[dict[str, float]]:
    if not weights:
        return []
    ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:top_n]
    result: list[dict[str, float]] = []
    for name, weight in ordered:
        entry: dict[str, float] = {"name": name, "weight": weight}
        if total and total > 0:
            entry["ratio"] = weight / total
        result.append(entry)
    return result


def _top_summaries(
    summaries: dict[str, dict], *, top_n: int, total: float | None = None
) -> list[dict[str, Any]]:
    if not summaries:
        return []
    ordered = sorted(
        summaries.values(), key=lambda entry: entry.get("event_count", 0.0), reverse=True
    )[:top_n]
    result: list[dict[str, Any]] = []
    for entry in ordered:
        enriched = dict(entry)
        enriched["weight"] = enriched.get("event_count", 0.0)
        if total and total > 0:
            ratio = enriched.get("event_count", 0.0) / total
            enriched["event_ratio"] = ratio
            enriched["ratio"] = ratio
        result.append(enriched)
    return result


def _flamegraph_trace_insight(
    result, *, top_n: int = 20, source_path: Path | None = None
) -> dict[str, Any]:
    event_total = result.event_count_total or sum(result.symbol_counts.values())
    source_value = source_path or getattr(result, "source_path", None)
    source_text = str(source_value) if source_value else None
    thread_symbol_insights: list[dict[str, Any]] = []
    for tid, weights in (result.symbol_counts_per_thread or {}).items():
        summary = (result.thread_summaries or {}).get(str(tid), {})
        thread_total = sum(weights.values())
        thread_symbol_insights.append(
            {
                "tid": tid,
                "name": summary.get("name", ""),
                "event_count": summary.get("event_count", 0.0),
                "top_symbols": _top_weighted_items(weights, top_n=top_n, total=thread_total),
            }
        )
    return {
        "event_total": event_total,
        "top_symbols": _top_weighted_items(result.symbol_counts, top_n=top_n, total=event_total),
        "top_symbols_raw": _top_weighted_items(
            result.symbol_counts_raw, top_n=top_n, total=event_total
        ),
        "top_processes": _top_summaries(
            result.process_summaries, top_n=top_n, total=event_total
        ),
        "top_threads": _top_summaries(
            result.thread_summaries, top_n=top_n, total=event_total
        ),
        "top_libs": _top_summaries(result.lib_summaries, top_n=top_n, total=event_total),
        "top_symbols_per_thread": thread_symbol_insights,
        "source_path": source_text,
        "source": "flamegraph",
    }


def _flamegraph_comparison_insight(results: list, *, top_n: int = 10) -> dict[str, Any] | None:
    if len(results) < 2:
        return None

    summaries: list[dict[str, Any]] = []
    symbol_weights_by_source: dict[str, dict[str, float]] = {}
    thread_comparisons: dict[str, dict[str, Any]] = {}
    for idx, result in enumerate(results):
        source_path = getattr(result, "source_path", None)
        source_label = str(source_path) if source_path else f"flamegraph_{idx + 1}"
        event_total = result.event_count_total or sum(result.symbol_counts.values())
        summaries.append(
            {
                "source": source_label,
                "event_total": event_total,
                "process_count": result.process_count,
                "thread_count": result.thread_count,
                "sample_count_total": result.sample_count_total,
                "top_symbols": _top_weighted_items(
                    result.symbol_counts, top_n=top_n, total=event_total
                ),
                "top_processes": _top_summaries(
                    result.process_summaries, top_n=top_n, total=event_total
                ),
                "top_threads": _top_summaries(
                    result.thread_summaries, top_n=top_n, total=event_total
                ),
                "top_libs": _top_summaries(
                    result.lib_summaries, top_n=top_n, total=event_total
                ),
            }
        )
        symbol_weights_by_source[source_label] = dict(result.symbol_counts or {})
        for tid, weights in (result.symbol_counts_per_thread or {}).items():
            summary = (result.thread_summaries or {}).get(str(tid), {})
            thread_label = summary.get("name") or str(tid)
            thread_key = f"{tid}:{thread_label}"
            thread_entry = thread_comparisons.setdefault(
                thread_key,
                {"tid": tid, "name": summary.get("name", ""), "per_source": {}},
            )
            thread_total = sum(weights.values())
            thread_entry["per_source"][source_label] = {
                "event_count": summary.get("event_count", 0.0),
                "top_symbols": _top_weighted_items(weights, top_n=top_n, total=thread_total),
                "symbol_weights": dict(weights),
            }

    event_totals = {summary["source"]: summary["event_total"] for summary in summaries}
    min_source = min(event_totals, key=event_totals.get)
    max_source = max(event_totals, key=event_totals.get)
    event_total_range = {
        "min": event_totals[min_source],
        "max": event_totals[max_source],
        "delta": event_totals[max_source] - event_totals[min_source],
        "min_source": min_source,
        "max_source": max_source,
    }

    symbol_sources: dict[str, dict[str, float]] = {}
    for source, weights in symbol_weights_by_source.items():
        for symbol, weight in weights.items():
            symbol_sources.setdefault(symbol, {})[source] = weight

    symbol_spreads: list[dict[str, Any]] = []
    for symbol, weights in symbol_sources.items():
        for source in symbol_weights_by_source:
            weights.setdefault(source, 0.0)
        max_source_symbol = max(weights, key=weights.get)
        min_source_symbol = min(weights, key=weights.get)
        max_weight = weights[max_source_symbol]
        min_weight = weights[min_source_symbol]
        ratio = (max_weight / min_weight) if min_weight > 0 else None
        symbol_spreads.append(
            {
                "symbol": symbol,
                "max_weight": max_weight,
                "min_weight": min_weight,
                "max_source": max_source_symbol,
                "min_source": min_source_symbol,
                "spread": max_weight - min_weight,
                "ratio": ratio,
                "weights": weights,
            }
        )

    symbol_spreads.sort(key=lambda entry: entry["spread"], reverse=True)
    symbol_spreads = symbol_spreads[:top_n]

    thread_spreads: list[dict[str, Any]] = []
    for thread_key, entry in thread_comparisons.items():
        per_source = entry.get("per_source", {})
        symbol_sources: dict[str, dict[str, float]] = {}
        for source, payload in per_source.items():
            weights = payload.get("symbol_weights", {})
            for symbol, weight in weights.items():
                symbol_sources.setdefault(symbol, {})[source] = weight
        symbol_spreads = []
        for symbol, weights in symbol_sources.items():
            for source in symbol_weights_by_source:
                weights.setdefault(source, 0.0)
            max_source_symbol = max(weights, key=weights.get)
            min_source_symbol = min(weights, key=weights.get)
            max_weight = weights[max_source_symbol]
            min_weight = weights[min_source_symbol]
            ratio = (max_weight / min_weight) if min_weight > 0 else None
            symbol_spreads.append(
                {
                    "symbol": symbol,
                    "max_weight": max_weight,
                    "min_weight": min_weight,
                    "max_source": max_source_symbol,
                    "min_source": min_source_symbol,
                    "spread": max_weight - min_weight,
                    "ratio": ratio,
                }
            )
        symbol_spreads.sort(key=lambda entry: entry["spread"], reverse=True)
        thread_spreads.append(
            {
                "thread": entry,
                "symbol_spreads": symbol_spreads[:top_n],
            }
        )

    return {
        "summary": summaries,
        "event_totals": event_total_range,
        "symbol_spreads": symbol_spreads,
        "thread_spreads": thread_spreads,
        "source": "flamegraph_comparison",
    }


def _load_psg_from_artifact(services: PipelineServices, artifact_id: str) -> PsgGraph | None:
    if not artifact_id:
        return None
    artifact = (
        services.ctx.session.query(models.Artifact)
        .filter(models.Artifact.artifact_id == artifact_id)
        .one_or_none()
    )
    if not artifact:
        return None
    return PsgGraph.from_json(Path(artifact.path).read_text(encoding="utf-8"))


def _format_evidence_report(
    run_id: str,
    iteration: int,
    metrics: dict[str, float],
    hotspots: list[HotspotCandidate],
    trace_insights: list[dict[str, Any]],
    summary: str,
) -> str:
    lines = [
        f"# Evidence Pack for {run_id}",
        f"Iteration: {iteration}",
        "## Metrics",
    ]
    for k, v in metrics.items():
        lines.append(f"- {k}: {v}")
    lines.append("## Hotspots")
    for h in hotspots:
        lines.append(f"- {h.symbol} score={h.score}")
    lines.append("## Trace insights")
    lines.append(json.dumps(trace_insights, indent=2))
    lines.append("## Summary")
    lines.append(summary)
    return "\n".join(lines)


def _store_flamegraph_maps(
    services: PipelineServices, run_id: str, result, source_path: Path | None
) -> None:
    if not result:
        return
    meta = {
        "process_count": result.process_count,
        "thread_count": result.thread_count,
        "event_count_total": result.event_count_total,
        "sample_count_total": result.sample_count_total,
        "source_path": str(source_path) if source_path else None,
    }
    artifact = services.ctx.artifact_store.store_json(
        {
            "process_name_map": result.process_name_map,
            "thread_name_map": result.thread_name_map,
            "symbols_file_list": result.symbols_file_list,
            "symbol_map": result.symbol_map,
        },
        kind="flamegraph_name_map",
        run_id=run_id,
        session=services.ctx.session,
        metadata=meta,
    )
    services.ctx.session.add(
        models.Artifact(
            artifact_id=artifact.artifact_id,
            run_id=run_id,
            kind="flamegraph_symbol_counts_raw",
            sha256=artifact.sha256,
            path=artifact.path,
            bytes=artifact.bytes,
            mime=artifact.mime,
            metadata_json={"symbol_counts_raw": result.symbol_counts_raw},
        )
    )
    services.ctx.session.add(
        models.Artifact(
            artifact_id=artifact.artifact_id,
            run_id=run_id,
            kind="flamegraph_symbol_counts",
            sha256=artifact.sha256,
            path=artifact.path,
            bytes=artifact.bytes,
            mime=artifact.mime,
            metadata_json={"symbol_counts": result.symbol_counts},
        )
    )
    services.ctx.session.add(
        models.Artifact(
            artifact_id=artifact.artifact_id,
            run_id=run_id,
            kind="flamegraph_symbol_counts_per_thread",
            sha256=artifact.sha256,
            path=artifact.path,
            bytes=artifact.bytes,
            mime=artifact.mime,
            metadata_json={"symbol_counts_per_thread": result.symbol_counts_per_thread},
        )
    )
    services.ctx.session.commit()
    _store_name_maps_in_vector_store(services, run_id, result)
    _store_name_map_summaries(services, run_id, result)
    _store_flamegraph_pcg(services, run_id, result, meta)


def _store_name_maps_in_vector_store(
    services: PipelineServices, run_id: str, result
) -> None:
    if not result:
        return
    vector_store = services.ctx.vector_store
    if not vector_store:
        return
    records: list[VectorRecord] = []
    for pid, name in (result.process_name_map or {}).items():
        records.append(
            VectorRecord(
                kind="process_name_map",
                ref_id=f"{run_id}:{pid}",
                embedding=[],
                run_id=run_id,
                metadata={"pid": pid, "name": name},
            )
        )
    for tid, name in (result.thread_name_map or {}).items():
        records.append(
            VectorRecord(
                kind="thread_name_map",
                ref_id=f"{run_id}:{tid}",
                embedding=[],
                run_id=run_id,
                metadata={"tid": tid, "name": name},
            )
        )
    for idx, name in enumerate(result.symbols_file_list or []):
        records.append(
            VectorRecord(
                kind="symbols_file_list",
                ref_id=f"{run_id}:{idx}",
                embedding=[],
                run_id=run_id,
                metadata={"file_id": idx, "name": name},
            )
        )
    if records:
        vector_store.add(records)


def _store_name_map_summaries(
    services: PipelineServices, run_id: str, result
) -> None:
    if not result:
        return
    vector_store = services.ctx.vector_store
    if not vector_store:
        return
    summaries = result.summaries_for_vector_store()
    records = []
    for summary in summaries:
        records.append(
            VectorRecord(
                kind="flamegraph_name_map_summary",
                ref_id=f"{run_id}:{summary.get('key')}",
                embedding=[],
                run_id=run_id,
                metadata=summary,
            )
        )
    if records:
        vector_store.add(records)


def _store_flamegraph_pcg(
    services: PipelineServices, run_id: str, result, metadata: dict | None = None
) -> None:
    if not result:
        return
    pcg = result.to_pcg()
    art = services.ctx.artifact_store.store_json(
        pcg.to_json(),
        kind="pcg",
        run_id=run_id,
        session=services.ctx.session,
        metadata=metadata,
    )
    services.ctx.session.add(
        models.Graph(
            run_id=run_id,
            kind="pcg",
            format="json",
            payload_artifact_id=art.artifact_id,
            metadata_json={"nodes": len(pcg.nodes), "edges": len(pcg.edges)},
        )
    )
    services.ctx.session.commit()


def _metrics_to_map(metrics: Iterable[Metric]) -> Dict[str, float]:
    return {m.metric_name: m.value for m in metrics}


def _hotspot_to_dict(h: HotspotCandidate) -> dict:
    return {
        "symbol": h.symbol,
        "file_path": h.file_path,
        "line_start": h.line_start,
        "line_end": h.line_end,
        "score": h.score,
        "evidence_artifacts": h.evidence_artifacts,
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


def _index_code(services: PipelineServices, state: RunState) -> RunState:
    if not services.config.indexing.enabled:
        return state
    repos = getattr(services.config.project, "repos", []) or []
    if repos:
        paths = build_kernel_indexes(services.config)
        state["code_index_versions"] = [
            {"repo": p.code_dir.parent.name, "version": p.code_version, "path": str(p.code_dir)}
            for p in paths
        ]
    else:
        path = build_kernel_index(services.config)
        state["code_index_versions"] = [
            {"repo": services.config.project.name, "version": path.code_version, "path": str(path.code_dir)}
        ]
    logger.info("Code indexing done: repos=%d", len(state.get("code_index_versions", [])))
    return state


def _index_runtime(services: PipelineServices, state: RunState) -> RunState:
    if not services.config.indexing.enabled:
        return state
    run_id = state["run_id"]
    repos = getattr(services.config.project, "repos", []) or []
    if repos:
        versions = []
        for repo in repos:
            path = build_runtime_index(services.config, run_id, repo_name=repo.name)
            versions.append({"repo": repo.name, "version": path.runtime_version, "path": str(path.runtime_dir)})
        state["runtime_index_versions"] = versions
    else:
        path = build_runtime_index(services.config, run_id)
        state["runtime_index_versions"] = [
            {"repo": services.config.project.name, "version": path.runtime_version, "path": str(path.runtime_dir)}
        ]
    logger.info("Runtime indexing done: repos=%d", len(state.get("runtime_index_versions", [])))
    return state


def _query_across_repos(
    services: PipelineServices, query: str, *, run_id: str, mode: str = "runtime_code"
) -> str:
    repos = getattr(services.config.project, "repos", []) or []
    if not repos:
        return route_query(services.config, query, mode=mode, run_id=run_id)
    sections = []
    for repo in repos:
        try:
            response = route_query(
                services.config,
                query,
                mode=mode,
                run_id=run_id,
                repo_name=repo.name,
            )
        except Exception as exc:
            logger.warning("Query failed for repo=%s err=%s", repo.name, exc)
            response = f"Query failed for repo {repo.name}: {exc}"
        sections.append(f"## Repo: {repo.name}\n{response}")
    return "\n\n".join(sections)


def _query_insights(services: PipelineServices, state: RunState) -> RunState:
    if not services.config.indexing.enabled:
        return state
    run_id = state["run_id"]
    metrics = state.get("candidate_metrics", {})
    hotspots = [h.get("symbol") for h in state.get("hotspots", []) if h.get("symbol")]
    summary = state.get("evidence_summary") or ""
    query = (
        "Use runtime hotspots/metrics to locate relevant kernel code and suggest optimization focus.\n"
        f"Hotspots: {hotspots[:10]}\n"
        f"Metrics: {metrics}\n"
        f"Summary: {summary}"
    )
    response = _query_across_repos(services, query, run_id=run_id)
    state["query_summary"] = response
    art = services.ctx.artifact_store.store_text(
        response,
        kind="query_summary",
        run_id=run_id,
        extension=".md",
        session=services.ctx.session,
    )
    state["query_artifact_id"] = art.artifact_id
    services.ctx.session.commit()
    logger.info("Query insights generated: artifact=%s", art.artifact_id)
    return state


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
    if "flamegraph" in profile_result.artifacts:
        fg_results = parse_flamegraph(profile_result.artifacts["flamegraph"])
        fg_hotspots_all: list[HotspotCandidate] = []
        for fg in fg_results:
            metrics.extend(fg.to_metrics())
            _store_flamegraph_maps(services, run_id, fg, profile_result.artifacts["flamegraph"])
            if fg.symbol_counts:
                fg_hotspots = _hotspots_from_symbol_counts(
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
                _flamegraph_trace_insight(fg, top_n=20)
            )
        hotspots.extend(fg_hotspots_all)
        comparison = _flamegraph_comparison_insight(fg_results, top_n=10)
        if comparison:
            state.setdefault("trace_insights", []).append(comparison)
    if "hitrace" in profile_result.artifacts:
        ht = parse_hitrace(profile_result.artifacts["hitrace"])
        metrics.extend(ht.to_metrics())
    if "sysfs" in profile_result.artifacts:
        st = parse_sysfs_trace(profile_result.artifacts["sysfs"])
        metrics.extend(st.to_metrics())
    if "hiperf" in profile_result.artifacts:
        hp = parse_hiperf(profile_result.artifacts["hiperf"])
        hp_hotspots = rank_hotspots(hp.hotspot_costs, hp.edge_costs, top_n=10)
        if services.psg:
            hp_hotspots = align_hotspots_to_psg(hp_hotspots, services.psg)
        hotspots.extend(hp_hotspots)

    if hotspots:
        hotspots = rank_correlated(hotspots, {m.metric_name: m.value for m in metrics}, limit=20)
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
    trace_insights = state.get("trace_insights", [])
    summary = services.trace_analyst.analyze(metrics, hotspots, trace_insights)
    pack = {
        "iteration": state["iteration"],
        "metrics": metrics,
        "hotspots": [h.__dict__ for h in hotspots],
        "summary": summary,
        "trace_insights": trace_insights,
    }
    artifact = services.ctx.artifact_store.store_json(
        pack, kind="evidence_pack", run_id=state["run_id"], session=services.ctx.session
    )
    report_text = _format_evidence_report(
        state["run_id"], state["iteration"], metrics, hotspots, trace_insights, summary
    )
    report_artifact = services.ctx.artifact_store.store_text(
        report_text,
        kind="evidence_report",
        run_id=state["run_id"],
        extension=".md",
        session=services.ctx.session,
    )
    state["evidence_artifact_id"] = artifact.artifact_id
    state["evidence_report_artifact_id"] = report_artifact.artifact_id
    state["evidence_summary"] = summary
    logger.info(
        "Evidence pack stored: artifact=%s report_artifact=%s",
        artifact.artifact_id,
        report_artifact.artifact_id,
    )
    return state


def _conductor_decide(services: PipelineServices, state: RunState) -> RunState:
    metrics = state.get("candidate_metrics", {})
    best_summary = f"best fps {state.get('best_metrics', {}).get('fps_avg', 0):.2f}"
    evidence_summary = json.dumps(metrics)[:800]
    query_summary = state.get("query_summary")
    if query_summary:
        evidence_summary = f"{evidence_summary}\n\nQuery insights:\n{query_summary[:2000]}"
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
    if state.get("query_summary"):
        instructions = f"{instructions}\n\nQuery insights:\n{state.get('query_summary')}"
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
            log = str(exc)
    state["patch_apply_status"] = status
    patch_row = (
        services.ctx.session.query(models.Patch)
        .filter(models.Patch.run_id == state["run_id"], models.Patch.iteration == state["iteration"])
        .one_or_none()
    )
    if patch_row:
        patch_row.apply_status = status
    art = services.ctx.artifact_store.store_text(
        log, kind="patch_apply_log", run_id=state["run_id"], session=services.ctx.session
    )
    services.ctx.session.commit()
    state["patch_apply_log_artifact_id"] = art.artifact_id
    logger.info("Patch apply status=%s artifact=%s", status, art.artifact_id)
    if status != "applied":
        state["force_stop"] = True
        state["stop_reason"] = "patch_apply_failed"
    return state


def _review_patch(services: PipelineServices, state: RunState) -> RunState:
    if not getattr(services.config, "prompts", None):
        return state
    if not services.config.prompts.review_enabled:
        return state
    if not state.get("patch_artifact_id"):
        return state
    logger.info("Reviewing patch artifact=%s", state["patch_artifact_id"])
    patch_artifact = (
        services.ctx.session.query(models.Artifact)
        .filter(models.Artifact.artifact_id == state["patch_artifact_id"])
        .one()
    )
    patch_text = Path(patch_artifact.path).read_text(encoding="utf-8")
    evidence_summary = state.get("evidence_summary") or json.dumps(
        state.get("candidate_metrics", {})
    )
    decision = services.reviewer.review_patch(
        patch_diff=patch_text,
        evidence_summary=evidence_summary,
        iteration=state["iteration"],
    )
    art = services.ctx.artifact_store.store_text(
        decision.get("rationale", ""),
        kind="patch_review",
        run_id=state["run_id"],
        extension=".md",
        session=services.ctx.session,
    )
    state["review_decision"] = decision.get("decision")
    state["review_artifact_id"] = art.artifact_id
    services.ctx.session.commit()
    if (
        services.config.prompts.review_block_on_reject
        and decision.get("decision") == "reject"
    ):
        state["force_stop"] = True
        state["stop_reason"] = "review_reject"
        logger.warning("Patch review rejected; stopping run.")
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
    from hmopt.evaluation.dataset import export_dataset

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


def init_run_node(services: PipelineServices, state: RunState) -> RunState:
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
    return state


def static_analysis_node(services: PipelineServices, state: RunState) -> RunState:
    logger.info("Static analysis start (PSG)")
    symbols = index_repo(services.config.project.repo_path)
    services.psg = build_psg(symbols)
    art = services.ctx.artifact_store.store_text(
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
        payload_artifact_id=art.artifact_id,
        metadata_json={"nodes": len(services.psg.nodes), "edges": len(services.psg.edges)},
    )
    services.ctx.session.add(graph_row)
    services.ctx.session.commit()
    logger.info(
        "Static analysis done: nodes=%d edges=%d", len(services.psg.nodes), len(services.psg.edges)
    )
    return state


def index_code_node(services: PipelineServices, state: RunState) -> RunState:
    return _index_code(services, state)


def index_runtime_node(services: PipelineServices, state: RunState) -> RunState:
    return _index_runtime(services, state)


def query_insights_node(services: PipelineServices, state: RunState) -> RunState:
    return _query_insights(services, state)


def review_patch_node(services: PipelineServices, state: RunState) -> RunState:
    return _review_patch(services, state)
