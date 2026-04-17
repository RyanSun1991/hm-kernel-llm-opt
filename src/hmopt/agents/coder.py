"""Coder agent."""

from __future__ import annotations

from pathlib import Path

import logging

from hmopt.core.llm import ChatMessage, LLMClient

from .prompting import render_prompt_template
from .safety import SafetyGuard

logger = logging.getLogger(__name__)


class CoderAgent:
    def __init__(self, llm: LLMClient, safety: SafetyGuard):
        self.llm = llm
        self.safety = safety

    def _offline_patch(self, instructions: str, iteration: int) -> str:
        content = instructions.strip() or "No-op patch"
        return f"""--- /dev/null
+++ b/HMOPT_PATCH_NOTES.md
@@
# HMOPT iteration {iteration}
{content}
"""

    def generate_patch(self, repo_path: Path, instructions: str, iteration: int, context: str = "") -> str:
        prompt = render_prompt_template(
            "coder.md",
            (
                "# Coder Prompt\n\n"
                "You are the Coder agent.\n\n"
                "Iteration: {iteration}\n"
                "Repository root: {repo_path}\n"
                "Instruction: {instructions}\n\n"
                "Optional context:\n"
                "{context}\n\n"
                "Produce a minimal unified diff.\n"
                "Preserve correctness and keep scope tight.\n"
                "Prefer editing or creating documentation/config files unless a specific code path is given.\n"
                "Return only the diff.\n"
            ),
            iteration=iteration,
            repo_path=repo_path,
            instructions=instructions,
            context=context,
        )
        messages = [
            ChatMessage(role="system", content="Return only a unified diff."),
            ChatMessage(role="user", content=self.safety.redact(prompt)),
        ]
        patch = self.llm.chat(messages)
        if "---" not in patch:
            patch = self._offline_patch(patch, iteration)
            logger.warning("Coder offline patch fallback used for iteration=%s", iteration)
        else:
            logger.info("Coder produced patch for iteration=%s", iteration)
        return patch.strip() + "\n"
