"""Patch review agent."""

from __future__ import annotations

import logging

from hmopt.core.llm import ChatMessage, LLMClient
from hmopt.prompting import PromptRegistry

from .safety import SafetyGuard

logger = logging.getLogger(__name__)


class ReviewDecision(dict):
    decision: str
    rationale: str


class ReviewAgent:
    def __init__(
        self,
        llm: LLMClient,
        safety: SafetyGuard,
        *,
        prompts: PromptRegistry | None = None,
        prompt_name: str = "code_review",
    ):
        self.llm = llm
        self.safety = safety
        self.prompts = prompts
        self.prompt_name = prompt_name
        self.default_system_prompt = "You are a senior kernel reviewer focused on correctness and performance risk."
        self.default_template = (
            "Review the proposed patch for correctness and performance impact.\n"
            "Evidence summary:\n{evidence_summary}\n\n"
            "Patch diff:\n{patch_diff}\n\n"
            "Return a short decision (approve/reject) and rationale."
        )

    def _render_prompt(self, *, evidence_summary: str, patch_diff: str) -> tuple[str, str]:
        system_prompt = self.default_system_prompt
        template = self.default_template
        if self.prompts:
            try:
                spec = self.prompts.load(self.prompt_name)
                system_prompt = spec.system or system_prompt
                template = spec.template or template
            except Exception as exc:
                logger.warning("Prompt load failed for %s: %s", self.prompt_name, exc)
        try:
            prompt = template.format(evidence_summary=evidence_summary, patch_diff=patch_diff)
        except KeyError as exc:
            logger.warning("Prompt render failed for %s: %s", self.prompt_name, exc)
            prompt = self.default_template.format(evidence_summary=evidence_summary, patch_diff=patch_diff)
            system_prompt = self.default_system_prompt
        return system_prompt, prompt

    def review_patch(
        self,
        *,
        patch_diff: str,
        evidence_summary: str,
        iteration: int,
    ) -> ReviewDecision:
        system_prompt, prompt = self._render_prompt(
            evidence_summary=evidence_summary,
            patch_diff=patch_diff,
        )
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=self.safety.redact(prompt)),
        ]
        reply = self.llm.chat(messages)
        decision = "reject" if "reject" in reply.lower() else "approve"
        logger.info("Review decision=%s iteration=%s", decision, iteration)
        return ReviewDecision(decision=decision, rationale=reply)
