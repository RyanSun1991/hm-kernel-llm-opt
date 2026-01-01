"""Coder agent."""

from __future__ import annotations

from pathlib import Path

from hmopt.core.llm import ChatMessage, LLMClient

from .safety import SafetyGuard


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

    def generate_patch(self, repo_path: Path, instructions: str, iteration: int) -> str:
        prompt = (
            "You are the Coder agent. Produce a minimal unified diff. "
            "Prefer editing or creating documentation/config files unless a specific "
            "code path is given.\n"
            f"Iteration: {iteration}\n"
            f"Instruction: {instructions}"
        )
        messages = [
            ChatMessage(role="system", content="Return only a unified diff."),
            ChatMessage(role="user", content=self.safety.redact(prompt)),
        ]
        patch = self.llm.chat(messages)
        if "---" not in patch:
            patch = self._offline_patch(patch, iteration)
        return patch.strip() + "\n"
