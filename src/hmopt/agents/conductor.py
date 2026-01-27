"""Conductor agent."""

from __future__ import annotations

from typing import Mapping

import logging

from hmopt.core.llm import ChatMessage, LLMClient

from .safety import SafetyGuard

logger = logging.getLogger(__name__)


class ConductorDecision(dict):
    decision: str
    rationale: str
    next_action: str


class ConductorAgent:
    def __init__(self, llm: LLMClient, safety: SafetyGuard):
        self.llm = llm
        self.safety = safety

    def decide(
        self,
        *,
        evidence_summary: str,
        best_summary: str,
        iteration: int,
        max_iterations: int,
    ) -> ConductorDecision:
        if iteration >= max_iterations:
            logger.info("Conductor stop due to iteration budget: %s/%s", iteration, max_iterations)
            return ConductorDecision(
                decision="stop",
                rationale="Reached iteration budget",
                next_action="report",
            )

        prompt = (
            "You are the Conductor agent coordinating an optimization loop.\n"
            f"Iteration: {iteration}/{max_iterations}\n"
            f"Best run summary: {best_summary}\n"
            f"Current evidence:\n{evidence_summary}\n"
            "Decide whether to continue optimizing or stop. "
            "If continuing, propose a concise next action for the coder."
        )
        messages = [
            ChatMessage(role="system", content="Conductor focusing on perf + correctness."),
            ChatMessage(role="user", content=self.safety.redact(prompt)),
        ]
        reply = self.llm.chat(messages)
        decision = "continue" if "continue" in reply.lower() else "stop"
        next_action = "refine code paths in hotspot" if decision == "continue" else "finalize report"
        logger.info("Conductor decision=%s next_action=%s", decision, next_action)
        return ConductorDecision(
            decision=decision,
            rationale=reply,
            next_action=next_action,
        )
