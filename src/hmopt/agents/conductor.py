"""Conductor agent."""

from __future__ import annotations

import logging
import re

from hmopt.core.llm import ChatMessage, LLMClient

from .prompting import render_prompt_template
from .safety import SafetyGuard

logger = logging.getLogger(__name__)

# Negation guard: "do not continue", "don't continue", "discontinue", "stop"
# must not be read as "continue" by a naive substring check.
_DECISION_RE = re.compile(r"(?im)^\s*decision\s*:\s*(continue|stop)\b")
_STOP_RE = re.compile(r"(?i)\b(stop|halt|finalize|do not continue|don'?t continue|discontinue)\b")
_CONTINUE_RE = re.compile(r"(?i)\b(continue|keep going|iterate again|another iteration)\b")
_NEXT_RE = re.compile(r"(?im)^\s*(?:next(?: action)?|action)\s*:\s*(.+)$")


def _parse_decision(reply: str) -> str:
    """Continue only on an explicit, un-negated continue signal; else stop.

    Fail safe toward stopping: an ambiguous or unparseable reply ends the loop
    rather than burning iterations on a fabricated 'continue'.
    """
    m = _DECISION_RE.search(reply)
    if m:
        return m.group(1).lower()
    if _STOP_RE.search(reply):
        return "stop"
    if _CONTINUE_RE.search(reply):
        return "continue"
    return "stop"


def _parse_next_action(reply: str, decision: str) -> str:
    m = _NEXT_RE.search(reply)
    if m and m.group(1).strip():
        return m.group(1).strip()[:300]
    return "report" if decision == "stop" else "refine the identified hotspot"


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

        prompt = render_prompt_template(
            "conductor.md",
            (
                "# Conductor Prompt\n\n"
                "You are the Conductor agent coordinating an optimization loop.\n\n"
                "Iteration: {iteration}/{max_iterations}\n"
                "Best run summary: {best_summary}\n\n"
                "Current evidence:\n"
                "{evidence_summary}\n\n"
                "Decide whether to continue optimizing or stop.\n"
                "Answer on two lines:\n"
                "Decision: CONTINUE or STOP\n"
                "Next: <one concise next action for the coder, if continuing>\n"
            ),
            iteration=iteration,
            max_iterations=max_iterations,
            best_summary=best_summary,
            evidence_summary=evidence_summary,
        )
        messages = [
            ChatMessage(role="system", content="Conductor focusing on perf, correctness, and disciplined iteration."),
            ChatMessage(role="user", content=self.safety.redact(prompt)),
        ]
        reply = self.llm.chat(messages)
        decision = _parse_decision(reply)
        next_action = _parse_next_action(reply, decision)
        logger.info("Conductor decision=%s next_action=%s", decision, next_action)
        return ConductorDecision(
            decision=decision,
            rationale=reply,
            next_action=next_action,
        )
