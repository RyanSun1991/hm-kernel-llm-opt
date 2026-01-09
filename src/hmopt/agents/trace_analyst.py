"""Trace Analyst agent."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

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
        trace_insights: Sequence[Mapping[str, object]] | None = None,
    ) -> str:
        hotspot_lines = [f"- {h.symbol}: score={h.score:.2f}" for h in hotspots]
        metric_lines = [f"{k}={v}" for k, v in metrics.items()]
        insight_lines = []
        if trace_insights:
            for idx, insight in enumerate(trace_insights, start=1):
                source = insight.get("source_path") or insight.get("source") or "flamegraph"
                total = insight.get("event_total", 0)
                top_symbols = insight.get("top_symbols", [])
                top_threads = insight.get("top_symbols_per_thread", [])
                insight_lines.append(f"Flamegraph {idx} ({source}) total_events={total}")
                for symbol in top_symbols[:5]:
                    name = symbol.get("name") or symbol.get("symbol")
                    weight = symbol.get("weight", 0.0)
                    ratio = symbol.get("ratio")
                    ratio_text = f" ({ratio:.2%})" if isinstance(ratio, float) else ""
                    insight_lines.append(f"  - top_symbol {name}: {weight:.2f}{ratio_text}")
                for thread in top_threads[:5]:
                    thread_name = thread.get("name") or thread.get("tid")
                    insight_lines.append(f"  - thread {thread_name}:")
                    for symbol in (thread.get("top_symbols") or [])[:5]:
                        name = symbol.get("name") or symbol.get("symbol")
                        weight = symbol.get("weight", 0.0)
                        ratio = symbol.get("ratio")
                        ratio_text = f" ({ratio:.2%})" if isinstance(ratio, float) else ""
                        insight_lines.append(f"    - {name}: {weight:.2f}{ratio_text}")
        prompt = (
            "You are the Trace Analyst. Summarize performance symptoms and hotspot classes.\n"
            f"Metrics: {', '.join(metric_lines)}\n"
            f"Hotspots:\n" + "\n".join(hotspot_lines)
        )
        if insight_lines:
            prompt += (
                "\nFlamegraph insights (compare patterns across files, highlight per-thread differences):\n"
                + "\n".join(insight_lines)
            )
        messages = [
            ChatMessage(role="system", content="Trace analyst focusing on HM kernel perf."),
            ChatMessage(role="user", content=self.safety.redact(prompt)),
        ]
        logger.info(
            "TraceAnalyst analyzing metrics=%d hotspots=%d flamegraphs=%d",
            len(metrics),
            len(hotspot_lines),
            len(trace_insights or []),
        )
        return self.llm.chat(messages)
