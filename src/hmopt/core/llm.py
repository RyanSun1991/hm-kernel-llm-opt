"""OpenAI-compatible chat wrapper used by agents."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


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
        allow_offline: bool = False,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.default_model = default_model
        self.allow_external_proxy_models = allow_external_proxy_models
        # Offline mode must be explicitly allowed. Otherwise a run with no API
        # key (or a failing gateway) would silently emit deterministic fake
        # answers and still report "succeeded" — the fail-open footgun. With
        # allow_offline=False we raise loudly instead. Dummy/demo runs opt in.
        self.allow_offline = allow_offline
        self._client = OpenAI(base_url=api_base, api_key=api_key) if api_key else None
        self._offline = api_key is None
        if self._offline and not allow_offline:
            logger.warning(
                "LLMClient has no API key; calls will fail unless allow_offline=True "
                "(set HMOPT_LLM_API_KEY or run with dummy adapters)"
            )

    def _offline_answer(self, messages: List[ChatMessage]) -> str:
        # Deterministic summary of the last user/system message for offline mode
        content = messages[-1].content if messages else "no prompt"
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        logger.warning("LLM offline fallback used for model=%s", self.default_model)
        return f"[offline-llm:{digest}] {content[:300]}"

    def chat(self, messages: Iterable[ChatMessage], model: Optional[str] = None) -> str:
        msgs = list(messages)
        if self._offline or not self._client:
            if not self.allow_offline:
                raise RuntimeError(
                    "LLM is offline (no API key or the gateway failed) and "
                    "allow_offline is False. Set HMOPT_LLM_API_KEY / a reachable "
                    "base_url, or run with dummy adapters. Refusing to fabricate "
                    "a response that would make the run look successful."
                )
            return self._offline_answer(msgs)

        payload = [{"role": m.role, "content": m.content} for m in msgs]
        chosen_model = model or self.default_model
        try:
            resp = self._client.chat.completions.create(
                model=chosen_model, messages=payload, temperature=0.2
            )
            return resp.choices[0].message.content  # type: ignore[return-value]
        except Exception:
            logger.exception("LLM call failed for model=%s", chosen_model)
            if not self.allow_offline:
                raise
            self._offline = True
            logger.warning("Switching to offline mode after LLM failure (allow_offline=True)")
            return self._offline_answer(msgs)
