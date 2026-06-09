"""Hybrid retrieval over knowledge records (design §12.1).

Implements the read-path stack literally:

    1) scalar pre-filter (status / maturity / scope)   -- cheap, schema fields
    2) BM25 + vector cosine, fused with RRF + entity bonus
    3) weight by the record's promotion `score` (§9 feeds ranking)
    4) return top-k

BM25 carries symbol-name queries (`shrink_node`) where pure (bag-of-words)
vectors dilute; the vector arm adds a fuzzy token-overlap signal — and, with a
real embedder injected, paraphrase matching. The `mode` knob exposes bm25-only /
vector-only / hybrid for the symbol-name ablation (plan P1-10 ④).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from .embeddings import Embedder, HashingEmbedder, cosine, tokenize
from .records import Record, maturity_at_least

Mode = Literal["hybrid", "bm25", "vector"]


@dataclass
class RetrievalHit:
    record: Record
    score: float
    bm25: float = 0.0
    vector: float = 0.0
    rrf: float = 0.0
    entity_bonus: float = 0.0
    score_weight: float = 1.0


@dataclass
class ScopeFilter:
    subsystem: str | None = None
    target_slug: str | None = None
    # If True, global-level records always pass scope (broadly applicable).
    include_global: bool = True


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class _BM25:
    """Okapi BM25 over a fixed corpus of token lists."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(toks) for toks in corpus]
        self.dl = [sum(c.values()) for c in self.docs]
        self.N = len(self.docs)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        df: Counter[str] = Counter()
        for c in self.docs:
            df.update(c.keys())
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def score(self, q_tokens: Iterable[str], idx: int) -> float:
        c = self.docs[idx]
        dl = self.dl[idx] or 1
        s = 0.0
        for t in q_tokens:
            f = c.get(t, 0)
            if not f:
                continue
            idf = self.idf.get(t, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s += idf * (f * (self.k1 + 1)) / denom
        return s


def _rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


def extract_entities(query: str) -> set[str]:
    """Symbol-ish tokens from a query (contain '_' or look like identifiers)."""
    ents: set[str] = set()
    for tok in tokenize(query):
        if "_" in tok or len(tok) >= 6:
            ents.add(tok)
    return ents


class HybridRetriever:
    def __init__(self, records: Sequence[Record], embedder: Embedder | None = None):
        self.records = list(records)
        self.embedder = embedder or HashingEmbedder()
        self._tokens = [tokenize(r.search_text()) for r in self.records]
        self._bm25 = _BM25(self._tokens)
        self._vecs = [self.embedder.embed(r.search_text()) for r in self.records]

    # --- scalar pre-filter --------------------------------------------------

    def _in_scope(self, r: Record, scope: ScopeFilter | None) -> bool:
        if scope is None:
            return True
        if scope.include_global and r.level == "global":
            return True
        if scope.target_slug and r.target_slug == scope.target_slug:
            return True
        if scope.subsystem and r.subsystem == scope.subsystem:
            return True
        # No positive scope signal provided at all -> don't over-filter.
        return scope.subsystem is None and scope.target_slug is None

    def _candidates(
        self,
        scope: ScopeFilter | None,
        maturity_floor: str | None,
        status: Sequence[str],
        mechanism: str | None,
    ) -> list[int]:
        out = []
        for i, r in enumerate(self.records):
            if status and r.status not in status:
                continue
            if maturity_floor and not maturity_at_least(r.maturity, maturity_floor):
                continue
            if mechanism and r.mechanism and r.mechanism != mechanism:
                continue
            if not self._in_scope(r, scope):
                continue
            out.append(i)
        return out

    # --- main retrieve ------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        scope: ScopeFilter | None = None,
        maturity_floor: str | None = "L2",
        status: Sequence[str] = ("active",),
        mechanism: str | None = None,
        mode: Mode = "hybrid",
        entity_bonus_weight: float = 0.1,
    ) -> list[RetrievalHit]:
        cand = self._candidates(scope, maturity_floor, status, mechanism)
        if not cand:
            return []

        q_tokens = tokenize(query)
        q_vec = self.embedder.embed(query)
        entities = extract_entities(query)

        bm = {i: self._bm25.score(q_tokens, i) for i in cand}
        vec = {i: cosine(q_vec, self._vecs[i]) for i in cand}

        pool = 4 * k
        bm_rank = [i for i in sorted(cand, key=lambda j: bm[j], reverse=True) if bm[i] > 0][:pool]
        vec_rank = sorted(cand, key=lambda j: vec[j], reverse=True)[:pool]

        if mode == "bm25":
            fused = {i: bm[i] for i in cand}
        elif mode == "vector":
            fused = {i: vec[i] for i in cand}
        else:
            fused = _rrf([bm_rank, vec_rank])

        hits: list[RetrievalHit] = []
        for i in cand:
            base = fused.get(i, 0.0)
            if base <= 0.0 and mode == "hybrid":
                continue
            rtoks = set(self._tokens[i])
            ent = sum(1 for e in entities if e in rtoks)
            # Entity matching is a hybrid feature; keep bm25/vector arms pure so
            # the ablation (P1-10 ④) measures the raw single-arm signal.
            ebonus = (entity_bonus_weight * ent) if mode == "hybrid" else 0.0
            sw = _sigmoid(self.records[i].score)
            final = (base + ebonus) * sw
            hits.append(
                RetrievalHit(
                    record=self.records[i],
                    score=final,
                    bm25=bm[i],
                    vector=vec[i],
                    rrf=fused.get(i, 0.0),
                    entity_bonus=ebonus,
                    score_weight=sw,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
