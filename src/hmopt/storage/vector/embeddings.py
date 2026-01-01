"""Embedding model integration."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, List

from openai import OpenAI


class EmbeddingClient:
    """Wrapper around OpenAI-compatible embeddings with an offline fallback."""

    def __init__(self, api_base: str, api_key: str | None, model: str):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self._client = OpenAI(base_url=api_base, api_key=api_key) if api_key else None
        self._offline_mode = api_key is None

    def _hash_fallback(self, text: str) -> list[float]:
        # Deterministic pseudo-embedding
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [int(b) / 255.0 for b in digest[:64]]

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        items = list(texts)
        if self._offline_mode or not self._client:
            return [self._hash_fallback(t) for t in items]
        try:
            resp = self._client.embeddings.create(model=self.model, input=items)
            return [record.embedding for record in resp.data]
        except Exception:
            # Avoid failing the pipeline if the gateway is unreachable
            self._offline_mode = True
            return [self._hash_fallback(t) for t in items]

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]
