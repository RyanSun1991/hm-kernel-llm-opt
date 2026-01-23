"""Coder agent."""

from __future__ import annotations

from pathlib import Path

import logging

from hmopt.core.llm import ChatMessage, LLMClient
from hmopt.prompting import PromptRegistry

from .safety import SafetyGuard

logger = logging.getLogger(__name__)


class CoderAgent:
    def __init__(
        self,
        llm: LLMClient,
        safety: SafetyGuard,
        *,
        prompts: PromptRegistry | None = None,
        prompt_name: str = "coder",
    ):
        self.llm = llm
        self.safety = safety
        self.prompts = prompts
        self.prompt_name = prompt_name
        self.default_system_prompt = "Return only a unified diff."
        self.default_template = (
            "You are the Coder agent. Produce a minimal unified diff. "
            "Prefer editing or creating documentation/config files unless a specific "
            "code path is given.\n"
            "Iteration: {iteration}\n"
            "Instruction: {instructions}"
        )

    def _render_prompt(self, *, instructions: str, iteration: int) -> tuple[str, str]:
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
            prompt = template.format(instructions=instructions, iteration=iteration)
        except KeyError as exc:
            logger.warning("Prompt render failed for %s: %s", self.prompt_name, exc)
            prompt = self.default_template.format(instructions=instructions, iteration=iteration)
            system_prompt = self.default_system_prompt
        return system_prompt, prompt

    def _offline_patch(self, instructions: str, iteration: int) -> str:
        content = instructions.strip() or "No-op patch"
        return f"""--- /dev/null
+++ b/HMOPT_PATCH_NOTES.md
@@
# HMOPT iteration {iteration}
{content}
"""

    def generate_patch(self, repo_path: Path, instructions: str, iteration: int) -> str:
        system_prompt, prompt = self._render_prompt(
            instructions=instructions,
            iteration=iteration,
        )
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=self.safety.redact(prompt)),
        ]
        patch = self.llm.chat(messages)
        if "---" not in patch:
            patch = self._offline_patch(patch, iteration)
            logger.warning("Coder offline patch fallback used for iteration=%s", iteration)
        else:
            logger.info("Coder produced patch for iteration=%s", iteration)
        return patch.strip() + "\n"
