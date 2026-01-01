"""Lightweight local vector store backed by the SQL DB."""

from __future__ import annotations

import math
import uuid
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

from sqlalchemy.orm import Session

from ..db import models

logger = logging.getLogger(__name__)


@dataclass
class VectorRecord:
    kind: str
    ref_id: str
    embedding: List[float]
    run_id: str | None = None
    metadata: dict | None = None


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class LocalVectorStore:
    """Stores embeddings in the DB and performs simple cosine search."""

    def __init__(self, session: Session):
        self.session = session

    def add(self, records: Iterable[VectorRecord]) -> list[str]:
        ids: list[str] = []
        records = list(records)
        for rec in records:
            vid = str(uuid.uuid4())
            ids.append(vid)
            row = models.VectorEmbedding(
                id=vid,
                kind=rec.kind,
                ref_id=rec.ref_id,
                run_id=rec.run_id,
                embedding_json=rec.embedding,
                metadata_json=rec.metadata or {},
            )
            self.session.add(row)
        self.session.flush()
        logger.info("Stored embeddings: count=%d", len(records))
        return ids

    def similarity_search(
        self, query_embedding: List[float], *, top_k: int = 5, kind: Optional[str] = None
    ) -> list[tuple[models.VectorEmbedding, float]]:
        rows = (
            self.session.query(models.VectorEmbedding)
            .filter(models.VectorEmbedding.kind == kind) if kind else self.session.query(models.VectorEmbedding)
        )
        scored = [
            (row, _cosine(query_embedding, list(row.embedding_json or []))) for row in rows.all()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
