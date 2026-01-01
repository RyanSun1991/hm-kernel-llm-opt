"""Safety/policy enforcement."""

from __future__ import annotations

import re
from typing import Iterable


class SafetyGuard:
    def __init__(self, allow_external_proxy_models: bool = False):
        self.allow_external_proxy_models = allow_external_proxy_models
        self._blocked_patterns = [
            re.compile(r"(?i)api[_-]?key"),
            re.compile(r"(?i)password"),
        ]

    def redact(self, text: str) -> str:
        scrubbed = text
        for pat in self._blocked_patterns:
            scrubbed = pat.sub("[REDACTED]", scrubbed)
        return scrubbed

    def model_allowed(self, model_id: str) -> bool:
        if self.allow_external_proxy_models:
            return True
        # Heuristic: block obvious external models
        return not model_id.lower().startswith("gpt-4")

    def sanitize_messages(self, messages: Iterable[dict]) -> list[dict]:
        return [{"role": m["role"], "content": self.redact(str(m["content"]))} for m in messages]
