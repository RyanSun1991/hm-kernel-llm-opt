"""Pure-stdlib similarity primitives for the central Curator tools.

Kept dependency-free (no faiss / no embedding gateway) so the hub stays
split-out-ready and its CI runs offline-deterministically. The token-hashing
embedder is *fuzzy lexical*, not semantic (shared-token cosine); a real embedder
can replace `embed` in production without touching the merge logic — markdown is
the source of truth and the index is a derived cache (design §12.3).
"""
from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    for raw in _TOKEN_RE.findall((text or "").lower()):
        toks.append(raw)
        if "_" in raw:
            toks.extend(p for p in raw.split("_") if p)
    return toks


def _bucket(token: str, dim: int) -> int:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % dim


def embed(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for tok in tokenize(text):
        vec[_bucket(tok, dim)] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    return dot / (na * nb) if na and nb else 0.0


def jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def text_similarity(a: str, b: str) -> float:
    """Blend embedding cosine with token Jaccard (robust offline signal)."""
    return 0.5 * cosine(embed(a), embed(b)) + 0.5 * jaccard(a, b)


def alias_hit(a_names: list[str], b_names: list[str]) -> bool:
    """Case-insensitive overlap of mechanism names / aliases."""
    na = {x.strip().lower() for x in a_names if x}
    nb = {x.strip().lower() for x in b_names if x}
    return bool(na & nb)
