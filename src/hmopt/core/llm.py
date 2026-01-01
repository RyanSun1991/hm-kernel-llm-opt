"""OpenAI-compatible chat wrapper used by agents."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Optional

from openai import OpenAI


@dataclass
class ChatMessage:
    role: str
    content: str


class LLMClient:
    def __init__(
        self,
        api_base: str,
        api_key: str | None,
        default_model: str,
        allow_external_proxy_models: bool = False,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.default_model = default_model
        self.allow_external_proxy_models = allow_external_proxy_models
        self._client = OpenAI(base_url=api_base, api_key=api_key) if api_key else None
        self._offline = api_key is None

    def _offline_answer(self, messages: List[ChatMessage]) -> str:
        # Deterministic summary of the last user/system message for offline mode
        content = messages[-1].content if messages else "no prompt"
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        return f"[offline-llm:{digest}] {content[:300]}"

    def chat(self, messages: Iterable[ChatMessage], model: Optional[str] = None) -> str:
        msgs = list(messages)
        if self._offline or not self._client:
            return self._offline_answer(msgs)

        payload = [{"role": m.role, "content": m.content} for m in msgs]
        chosen_model = model or self.default_model
        try:
            resp = self._client.chat.completions.create(
                model=chosen_model, messages=payload, temperature=0.2
            )
            return resp.choices[0].message.content  # type: ignore[return-value]
        except Exception:
            self._offline = True
            return self._offline_answer(msgs)
