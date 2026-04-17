"""Reviewer agent."""

from __future__ import annotations

import logging
import re

from hmopt.core.llm import ChatMessage, LLMClient

from .prompting import render_prompt_template
from .safety import SafetyGuard

logger = logging.getLogger(__name__)


class ReviewerDecision(dict):
    decision: str
    rationale: str
    risk_summary: str


class ReviewerAgent:
    def __init__(self, llm: LLMClient, safety: SafetyGuard):
        self.llm = llm
        self.safety = safety

    @staticmethod
    def _default_prompt() -> str:
        return (
            "# Reviewer Prompt\n\n"
            "You are the Reviewer agent for HMOPT.\n\n"
            "Iteration: {iteration}\n"
            "Build success: {build_success}\n"
            "Test success: {test_success}\n\n"
            "Evidence summary:\n"
            "{evidence_summary}\n\n"
            "Patch summary:\n"
            "{patch_summary}\n\n"
            "Build log excerpt:\n"
            "{build_log_excerpt}\n\n"
            "Test log excerpt:\n"
            "{test_log_excerpt}\n\n"
            "Decide whether the candidate patch should proceed to profiling.\n"
            "Respond with:\n"
            "Decision: APPROVE or REJECT\n"
            "Risks: ...\n"
            "Rationale: ...\n"
        )

    @staticmethod
    def _parse_decision(reply: str, *, build_success: bool, test_success: bool) -> str:
        match = re.search(r"(?im)^decision\s*:\s*(approve|reject)\b", reply)
        if match:
            return match.group(1).lower()
        return "approve" if build_success and test_success else "reject"

    @staticmethod
    def _extract_risk_summary(reply: str) -> str:
        match = re.search(r"(?im)^risks?\s*:\s*(.+)$", reply)
        if match:
            return match.group(1).strip()
        lines = [line.strip() for line in reply.splitlines() if line.strip()]
        return lines[0][:300] if lines else "No risk summary."

    def review(
        self,
        *,
        evidence_summary: str,
        patch_summary: str,
        build_success: bool,
        test_success: bool,
        build_log_excerpt: str = "",
        test_log_excerpt: str = "",
        iteration: int,
    ) -> ReviewerDecision:
        prompt = render_prompt_template(
            "reviewer.md",
            self._default_prompt(),
            iteration=iteration,
            build_success=build_success,
            test_success=test_success,
            evidence_summary=evidence_summary,
            patch_summary=patch_summary,
            build_log_excerpt=build_log_excerpt,
            test_log_excerpt=test_log_excerpt,
        )
        messages = [
            ChatMessage(role="system", content="Reviewer focused on correctness, safety, and validation discipline."),
            ChatMessage(role="user", content=self.safety.redact(prompt)),
        ]
        reply = self.llm.chat(messages)
        decision = self._parse_decision(reply, build_success=build_success, test_success=test_success)
        risk_summary = self._extract_risk_summary(reply)
        logger.info("Reviewer decision=%s iteration=%s", decision, iteration)
        return ReviewerDecision(
            decision=decision,
            rationale=reply,
            risk_summary=risk_summary,
        )
