"""hmopt skill-hub consumer side (Phase 1 read/write path).

This package is the *business-repo* side of the Team Skill Hub (design
docs/Team_Skill_Hub_Design_CN.md). It consumes the hub's markdown records
(treated as the single source of truth, design §6.1) and provides:

- `records`   — load hub/local knowledge records from markdown frontmatter.
- `embeddings`— offline-deterministic token-hashing embedder (+ pluggable real one).
- `retrieval` — hybrid BM25 + vector RRF retrieval with scalar pre-filter (§12.1).
- `resolver`  — runtime composition: skills via selectors + knowledge via retrieval,
                trimmed to a per-stage context budget (§12.2).
- `sediment`  — Tier 0→1 distillation: run artifacts → schema-valid candidates (§8).
- `curator`   — local online relation classifier (seven-way taxonomy, §10.1).

Everything has an offline path so it is testable without a live LLM / gateway.
"""

from __future__ import annotations

__all__ = [
    "records",
    "embeddings",
    "retrieval",
    "resolver",
]
