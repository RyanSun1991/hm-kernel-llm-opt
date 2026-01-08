"""Trace Analyst agent."""

from __future__ import annotations

from typing import Iterable, Mapping

import logging

from hmopt.analysis.runtime.hotspot import HotspotCandidate
from hmopt.core.llm import ChatMessage, LLMClient

from .safety import SafetyGuard

logger = logging.getLogger(__name__)


class TraceAnalystAgent:
    def __init__(self, llm: LLMClient, safety: SafetyGuard):
        self.llm = llm
        self.safety = safety

    def analyze(
        self,
        metrics: Mapping[str, float],
        hotspots: Iterable[HotspotCandidate],
        trace_insights: list[dict] | None = None,
    ) -> str:
        hotspot_lines = [f"- {h.symbol}: score={h.score:.2f}" for h in hotspots]
        metric_lines = [f"{k}={v}" for k, v in metrics.items()]
        insight_lines: list[str] = []
        for insight in trace_insights or []:
            source = insight.get("source_path", "framegraph")
            insight_lines.append(f"Framegraph source: {source}")
            insight_lines.append(f"Total events: {insight.get('event_total', 0)}")
            top_symbols = insight.get("top_symbols", [])[:5]
            if top_symbols:
                sym_text = ", ".join(f"{sym['name']}={sym['weight']:.2f}" for sym in top_symbols)
                insight_lines.append(f"- top symbols: {sym_text}")
            for thread_entry in insight.get("top_threads_by_symbols", []):
                thread_label = thread_entry.get("thread", "unknown")
                top_syms = thread_entry.get("top_symbols", [])
                if top_syms:
                    sym_text = ", ".join(f"{sym['name']}={sym['weight']:.2f}" for sym in top_syms)
                    insight_lines.append(f"- thread {thread_label}: {sym_text}")
        prompt = (
            "You are the Trace Analyst. Summarize performance symptoms and hotspot classes.\n"
            f"Metrics: {', '.join(metric_lines)}\n"
            f"Hotspots:\n" + "\n".join(hotspot_lines)
        )
        if insight_lines:
            prompt += "\nFramegraph per-thread symbol hotspots:\n" + "\n".join(insight_lines)
        messages = [
            ChatMessage(role="system", content="Trace analyst focusing on HM kernel perf."),
            ChatMessage(role="user", content=self.safety.redact(prompt)),
        ]
        logger.info(
            "TraceAnalyst analyzing metrics=%d hotspots=%d framegraph_sources=%d",
            len(metrics),
            len(hotspot_lines),
            len(trace_insights or []),
        )
        return self.llm.chat(messages)
