"""LangGraph-based optimization loop."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
from hmopt.analysis.runtime.traces import (
    parse_framegraph,
    parse_framegraph_html,
    parse_hiperf,
    parse_hitrace,
    parse_sysfs_trace,
)
from hmopt.analysis.static import build_psg, index_repo
from hmopt.core.config import AppConfig
from hmopt.core.llm import LLMClient
from hmopt.core.run_context import RunContext, build_context, register_run
from hmopt.orchestration.state import RunState, initial_state
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


def _top_weighted_items(weights: dict[str, float], *, top_n: int, total: float | None = None) -> list[dict[str, float]]:
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


def _top_summaries(summaries: dict[str, dict], *, top_n: int, total: float | None = None) -> list[dict[str, Any]]:
    if not summaries:
        return []
    ordered = sorted(summaries.values(), key=lambda entry: entry.get("event_count", 0.0), reverse=True)[:top_n]
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


def _top_threads_by_symbols(result, *, top_n: int = 10, symbols_per_thread: int = 5) -> list[dict[str, Any]]:
    thread_weights: list[tuple[str, float]] = []
    for tid, weights in result.symbol_counts_per_thread.items():
        thread_weights.append((str(tid), sum(weights.values())))
    ordered = sorted(thread_weights, key=lambda item: item[1], reverse=True)[:top_n]
    output: list[dict[str, Any]] = []
    for tid, total in ordered:
        name = ""
        thread_summary = result.thread_summaries.get(str(tid), {})
        if thread_summary:
            name = thread_summary.get("name", "")
        label = f"{tid} {name}".strip()
        weights = result.symbol_counts_per_thread.get(str(tid), {})
        output.append(
            {
                "thread": label or tid,
                "total": total,
                "top_symbols": _top_weighted_items(weights, top_n=symbols_per_thread, total=total),
            }
        )
    return output


def _framegraph_trace_insight(result, *, top_n: int = 20, source_path: Path | None = None) -> dict[str, Any]:
    event_total = result.event_count_total or sum(result.symbol_counts.values())
    return {
        "event_total": event_total,
        "top_symbols": _top_weighted_items(result.symbol_counts, top_n=top_n, total=event_total),
        "top_symbols_raw": _top_weighted_items(result.symbol_counts_raw, top_n=top_n, total=event_total),
        "top_processes": _top_summaries(result.process_summaries, top_n=top_n, total=event_total),
        "top_threads": _top_summaries(result.thread_summaries, top_n=top_n, total=event_total),
        "top_libs": _top_summaries(result.lib_summaries, top_n=top_n, total=event_total),
        "top_threads_by_symbols": _top_threads_by_symbols(result, top_n=top_n),
        "symbol_counts_per_thread": result.symbol_counts_per_thread,
        "source_path": str(source_path) if source_path else None,
        "source": "framegraph",
    }


def _collect_framegraph_paths(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.rglob("*__sysmgr_hiperfReport.html"))
    return [path]


def _format_evidence_report(
    run_id: str,
    iteration: int,
    metrics: dict[str, float],
    hotspots: list[HotspotCandidate],
    trace_insights: list[dict[str, Any]],
    summary: str,
) -> str:
    lines: list[str] = [
        f"# Evidence snapshot for run {run_id}",
        f"Iteration: {iteration}",
        "## Trace Analyst Summary",
        summary or "No summary available.",
        "## Metrics",
    ]
    if metrics:
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- (none)")

    lines.append("## Hotspots")
    if hotspots:
        for hs in hotspots[:20]:
            lines.append(f"- {hs.symbol} score={hs.score:.2f}")
    else:
        lines.append("- (none)")

    for insight in trace_insights:
        source_label = insight.get("source_path") or insight.get("source", "trace")
        lines.append(f"## Trace hotspot detail ({source_label})")
        lines.append(f"Total events: {insight.get('event_total', 0)}")

        def _emit_list(title: str, items: list[dict[str, Any]], key: str = "name") -> None:
            lines.append(f"- {title}:")
            if not items:
                lines.append("  - (none)")
                return
            for item in items:
                name = (
                    item.get(key)
                    or item.get("name")
                    or item.get("symbol")
                    or item.get("pid")
                    or item.get("tid")
                    or item.get("file_path")
                    or str(item.get("file_id") or "unknown")
                )
                weight = item.get("weight", 0.0)
                ratio = item.get("ratio")
                ratio_text = f" ({ratio:.2%})" if isinstance(ratio, float) else ""
                lines.append(f"  - {name}: {weight:.2f}{ratio_text}")

        _emit_list("Top symbols (normalized)", insight.get("top_symbols", []))
        _emit_list("Top symbols (raw)", insight.get("top_symbols_raw", []))
        _emit_list("Top processes", insight.get("top_processes", []), key="name")
        _emit_list("Top threads", insight.get("top_threads", []), key="name")
        _emit_list("Top libs", insight.get("top_libs", []), key="file_path")
        lines.append("- Top symbols per thread:")
        thread_items = insight.get("top_threads_by_symbols", [])
        if not thread_items:
            lines.append("  - (none)")
        else:
            for thread_item in thread_items:
                thread_label = thread_item.get("thread", "unknown")
                lines.append(f"  - {thread_label} (total {thread_item.get('total', 0):.2f}):")
                for symbol in thread_item.get("top_symbols", []):
                    ratio = symbol.get("ratio")
                    ratio_text = f" ({ratio:.2%})" if isinstance(ratio, float) else ""
                    lines.append(
                        f"    - {symbol.get('name', 'unknown')}: {symbol.get('weight', 0.0):.2f}{ratio_text}"
                    )

    return "\n".join(lines)


def _store_framegraph_maps(services: PipelineServices, run_id: str, result, source_path: Path | None) -> None:
    metadata = {"source": str(source_path)} if source_path else {}
    if getattr(result, "name_maps", None):
        payload = result.name_maps.to_dict()
        map_art = services.ctx.artifact_store.store_json(
            payload, kind="framegraph_name_map", run_id=run_id, session=services.ctx.session, metadata=metadata
        )
        logger.info("Stored framegraph name maps: artifact=%s", map_art.artifact_id)
    if getattr(result, "symbol_counts_raw", None):
        counts_raw = result.symbol_counts_raw
        if counts_raw:
            counts_art = services.ctx.artifact_store.store_json(
                counts_raw,
                kind="framegraph_symbol_counts_raw",
                run_id=run_id,
                session=services.ctx.session,
                metadata=metadata,
            )
            logger.info("Stored framegraph raw symbol counts: artifact=%s", counts_art.artifact_id)
    if getattr(result, "symbol_counts", None):
        counts = result.symbol_counts
        if counts:
            counts_art = services.ctx.artifact_store.store_json(
                counts,
                kind="framegraph_symbol_counts",
                run_id=run_id,
                session=services.ctx.session,
                metadata=metadata,
            )
            logger.info("Stored framegraph normalized symbol counts: artifact=%s", counts_art.artifact_id)
    _store_framegraph_pcg(services, run_id, result, metadata)
    _store_name_maps_in_vector_store(services, run_id, result)
    _store_name_map_summaries(services, run_id, result, metadata)
    services.ctx.session.commit()


def _store_name_maps_in_vector_store(services: PipelineServices, run_id: str, result) -> None:
    if not getattr(result, "name_maps", None):
        return
    name_maps = result.name_maps
    entries: list[tuple[str, dict]] = []

    for pid, name in name_maps.process_name_map.items():
        entries.append((f"process {pid} name {name}", {"kind": "process", "id": str(pid)}))
    for tid, name in name_maps.thread_name_map.items():
        entries.append((f"thread {tid} name {name}", {"kind": "thread", "id": str(tid)}))
    for idx, path in enumerate(name_maps.symbols_file_list):
        entries.append((f"lib {idx} path {path}", {"kind": "symbols_file", "id": str(idx)}))
    for sym_id, entry in name_maps.symbol_map.items():
        if isinstance(entry, dict):
            sym_name = entry.get("symbol", "")
            file_id = entry.get("file", "")
            entries.append(
                (
                    f"symbol {sym_id} file {file_id} name {sym_name}",
                    {"kind": "symbol", "id": str(sym_id), "file": str(file_id)},
                )
            )

    for pid, summary in (result.process_summaries or {}).items():
        name = summary.get("name", "")
        top_symbols = summary.get("top_symbols", [])
        symbol_text = ", ".join(f"{s['symbol']}:{s['weight']:.1f}" for s in top_symbols)
        entries.append(
            (
                f"process {pid} name {name} events {summary.get('event_count', 0)} "
                f"samples {summary.get('sample_count', 0)} threads {summary.get('thread_count', 0)} "
                f"top_symbols {symbol_text}",
                {"kind": "process_summary", "id": str(pid)},
            )
        )
    for tid, summary in (result.thread_summaries or {}).items():
        name = summary.get("name", "")
        top_symbols = summary.get("top_symbols", [])
        symbol_text = ", ".join(f"{s['symbol']}:{s['weight']:.1f}" for s in top_symbols)
        entries.append(
            (
                f"thread {tid} name {name} pid {summary.get('pid')} "
                f"events {summary.get('event_count', 0)} samples {summary.get('sample_count', 0)} "
                f"top_symbols {symbol_text}",
                {"kind": "thread_summary", "id": str(tid), "pid": str(summary.get('pid'))},
            )
        )
    for lib_id, summary in (result.lib_summaries or {}).items():
        top_symbols = summary.get("top_symbols", [])
        symbol_text = ", ".join(f"{s['symbol']}:{s['weight']:.1f}" for s in top_symbols)
        entries.append(
            (
                f"lib {lib_id} path {summary.get('file_path', '')} "
                f"events {summary.get('event_count', 0)} top_symbols {symbol_text}",
                {"kind": "lib_summary", "id": str(lib_id)},
            )
        )

    if not entries:
        return

    texts = [t for t, _ in entries]
    metas = [m for _, m in entries]
    batch_size = 128
    for idx in range(0, len(texts), batch_size):
        batch_texts = texts[idx : idx + batch_size]
        batch_metas = metas[idx : idx + batch_size]
        embeddings = services.ctx.embedding_client.embed_texts(batch_texts)
        records = [
            VectorRecord(
                kind="framegraph_name_map",
                ref_id=f"{meta['kind']}:{meta['id']}",
                embedding=emb,
                run_id=run_id,
                metadata=meta,
            )
            for emb, meta in zip(embeddings, batch_metas)
        ]
        services.ctx.vector_store.add(records)
    logger.info("Stored framegraph name maps in vector store: count=%d", len(entries))


def _store_name_map_summaries(
    services: PipelineServices, run_id: str, result, metadata: dict | None = None
) -> None:
    payload = {
        "process_summaries": result.process_summaries,
        "thread_summaries": result.thread_summaries,
        "lib_summaries": result.lib_summaries,
    }
    if result.process_summaries or result.thread_summaries or result.lib_summaries:
        art = services.ctx.artifact_store.store_json(
            payload,
            kind="framegraph_name_map_summary",
            run_id=run_id,
            session=services.ctx.session,
            metadata=metadata or {},
        )
        logger.info("Stored framegraph name map summary: artifact=%s", art.artifact_id)


def _store_framegraph_pcg(services: PipelineServices, run_id: str, result, metadata: dict | None = None) -> None:
    if not getattr(result, "pcg_edges", None):
        return
    if not result.pcg_edges:
        return
    payload = {"nodes": result.pcg_nodes, "edges": result.pcg_edges}
    art = services.ctx.artifact_store.store_json(
        payload,
        kind="pcg_framegraph",
        run_id=run_id,
        session=services.ctx.session,
        metadata=metadata or {},
    )
    graph_row = models.Graph(
        run_id=run_id,
        kind="pcg",
        format="json",
        payload_artifact_id=art.artifact_id,
        metadata_json={
            "nodes": len(result.pcg_nodes),
            "edges": len(result.pcg_edges),
            "source": "framegraph",
        },
    )
    services.ctx.session.add(graph_row)
    logger.info("Stored framegraph PCG: artifact=%s", art.artifact_id)


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
        _store_framegraph_maps(services, run_id, fg, profile_result.artifacts["framegraph"])
        if fg.symbol_counts:
            fg_hotspots = _hotspots_from_symbol_counts(
                fg.symbol_counts,
                top_n=20,
                min_ratio=0.001,
                min_abs=10.0,
                total=fg.event_count_total,
            )
            if services.psg:
                fg_hotspots = align_hotspots_to_psg(fg_hotspots, services.psg)
            hotspots.extend(fg_hotspots)
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
        if kind == "framegraph":
            framegraph_paths = _collect_framegraph_paths(path)
            if not framegraph_paths:
                logger.warning("No framegraph HTML files found in %s", path)
            for fg_path in framegraph_paths:
                services.ctx.artifact_store.store_file(
                    fg_path, kind=kind or "unknown", run_id=state["run_id"], session=services.ctx.session
                )
                fg = parse_framegraph_html(fg_path) if fg_path.suffix.lower() == ".html" else parse_framegraph(fg_path)
                metrics.extend(fg.to_metrics())
                _store_framegraph_maps(services, state["run_id"], fg, fg_path)
                if fg.symbol_counts:
                    fg_hotspots = _hotspots_from_symbol_counts(
                        fg.symbol_counts,
                        top_n=20,
                        min_ratio=0.001,
                        min_abs=10.0,
                        total=fg.event_count_total,
                    )
                    if services.psg:
                        fg_hotspots = align_hotspots_to_psg(fg_hotspots, services.psg)
                    hotspots.extend(fg_hotspots)
                state.setdefault("trace_insights", []).append(
                    _framegraph_trace_insight(fg, top_n=20, source_path=fg_path)
                )
        else:
            services.ctx.artifact_store.store_file(
                path, kind=kind or "unknown", run_id=state["run_id"], session=services.ctx.session
            )
            if kind == "hitrace":
                ht = parse_hitrace(path)
                metrics.extend(ht.to_metrics())
            elif kind == "sysfs":
                st = parse_sysfs_trace(path)
                metrics.extend(st.to_metrics())
            elif kind == "hiperf":
                hp = parse_hiperf(path)
                hs = rank_hotspots(hp.hotspot_costs, hp.edge_costs, top_n=15)
                if services.psg:
                    hs = align_hotspots_to_psg(hs, services.psg)
                hotspots.extend(hs)
            else:
                logger.info("Stored artifact with no parser: %s", kind)

    if hotspots:
        metrics_map = {m.metric_name: m.value for m in metrics}
        hotspots = rank_correlated(hotspots, metrics_map, limit=20)
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
