"""Embedders for skill-hub retrieval.

The default `HashingEmbedder` is offline, deterministic, and *semantic-ish*: it
uses the hashing trick (each token hashed into one of D buckets, tf-weighted,
L2-normalized) so two texts that share tokens get a non-zero cosine. That makes
the read path fully testable without a live embedding gateway, while real
production runs can inject an OpenAI-compatible embedder via `ClientEmbedder`.

Design §12.3: index vectors are a *derived cache* of the markdown source; the
embedder is swappable without touching the records.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-identifier chars, and expand snake_case / paths.

    `mm/vmscan.c::shrink_node` -> ['mm','vmscan','c','shrink_node','shrink','node']
    so both the full symbol and its parts are matchable (design §12.1: pure
    vectors are weak on symbol names; lexical parts help).
    """
    lowered = text.lower()
    toks: list[str] = []
    for raw in _TOKEN_RE.findall(lowered):
        toks.append(raw)
        if "_" in raw:
            toks.extend(part for part in raw.split("_") if part)
    return toks


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:  # pragma: no cover - protocol
        ...


class HashingEmbedder:
    """Deterministic offline embedder via the hashing trick."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _bucket(self, token: str) -> int:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big") % self.dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            vec[self._bucket(tok)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


class ClientEmbedder:
    """Adapter wrapping hmopt.storage.vector.embeddings.EmbeddingClient."""

    def __init__(self, client) -> None:  # type: ignore[no-untyped-def]
        self._client = client

    def embed(self, text: str) -> list[float]:
        return list(self._client.embed_text(text))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
