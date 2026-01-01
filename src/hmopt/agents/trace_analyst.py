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

    def analyze(self, metrics: Mapping[str, float], hotspots: Iterable[HotspotCandidate]) -> str:
        hotspot_lines = [f"- {h.symbol}: score={h.score:.2f}" for h in hotspots]
        metric_lines = [f"{k}={v}" for k, v in metrics.items()]
        prompt = (
            "You are the Trace Analyst. Summarize performance symptoms and hotspot classes.\n"
            f"Metrics: {', '.join(metric_lines)}\n"
            f"Hotspots:\n" + "\n".join(hotspot_lines)
        )
        messages = [
            ChatMessage(role="system", content="Trace analyst focusing on HM kernel perf."),
            ChatMessage(role="user", content=self.safety.redact(prompt)),
        ]
        logger.info("TraceAnalyst analyzing metrics=%d hotspots=%d", len(metrics), len(hotspot_lines))
        return self.llm.chat(messages)
