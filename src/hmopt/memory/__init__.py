"""Local online memory consolidation (design §10.1, plan P1-8).

The local curator classifies an incoming sedimented item against its nearest
neighbors into the §10.1.0 relation taxonomy and applies it with the CRDT
discipline: **never physically delete** temporal / conditional / selector /
evidence relations — only `superseded` tombstones, only for a contradiction
backed by stronger evidence. This keeps `local/memory/` from rotting between
central merges, at ~1 LLM-call latency.
"""

from __future__ import annotations

__all__ = ["local_curator", "curator_benchmark"]
