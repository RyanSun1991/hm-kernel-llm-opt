"""Retrieval eval hard gate (design §12, plan P1-10).

Loads a fixture corpus + a labelled query set, runs the hybrid retriever, and
reports:
  - strict (must-hit) and lenient (must+optional) recall@k, overall + per type;
  - a BM25 / vector / hybrid ablation on symbol-name queries (P1-10 ④),
    demonstrating hybrid >= each single arm;
  - a regression check against a recorded baseline (early gate), plus an
    absolute must-hit recall@5 >= 0.8 line.

markdown/yaml is the source of truth; the index is rebuilt each run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .records import Record
from .retrieval import HybridRetriever, Mode, ScopeFilter

DEFAULT_EVAL_DIR = "hm-skill-hub/eval/retrieval"


def _record_from_dict(d: dict[str, Any]) -> Record:
    d = dict(d)
    rid = str(d.pop("id"))
    kind = str(d.pop("kind", "memory_item"))
    body = str(d.pop("body", ""))
    return Record(id=rid, kind=kind, path="corpus", fields=d, body=body, origin="hub")


def load_corpus(path: str | Path) -> list[Record]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [_record_from_dict(r) for r in raw.get("records", [])]


def load_queries(path: str | Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return list(raw.get("queries", []))


def _scope_of(q: dict[str, Any]) -> ScopeFilter | None:
    sc = q.get("scope")
    if not sc:
        return None
    return ScopeFilter(subsystem=sc.get("subsystem"), target_slug=sc.get("target_slug"))


def _recall_for_query(hits_ids: list[str], expect: list[dict], lenient: bool,
                      match: str = "all") -> tuple[int, int]:
    """Return (found, total) for must-hit (or must+optional when lenient).

    `match="any"` (e.g. an ambiguous bare-symbol query satisfied by any of
    several equally-valid records) scores 1/1 if at least one wanted id is in
    the top-k, else 0/1.
    """
    wanted = [e["id"] for e in expect if lenient or e.get("hit", "must") == "must"]
    if not wanted:
        return (0, 0)
    if match == "any":
        return (1, 1) if any(w in hits_ids for w in wanted) else (0, 1)
    found = sum(1 for w in wanted if w in hits_ids)
    return (found, len(wanted))


@dataclass
class EvalReport:
    k: int
    must_recall: float
    lenient_recall: float
    per_type: dict[str, float] = field(default_factory=dict)
    type_counts: dict[str, int] = field(default_factory=dict)
    ablation: dict[str, float] = field(default_factory=dict)  # mode -> must recall on symbol queries
    n_queries: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "must_recall": round(self.must_recall, 4),
            "lenient_recall": round(self.lenient_recall, 4),
            "per_type": {k: round(v, 4) for k, v in self.per_type.items()},
            "type_counts": self.type_counts,
            "ablation": {k: round(v, 4) for k, v in self.ablation.items()},
            "n_queries": self.n_queries,
        }


def run_eval(eval_dir: str | Path = DEFAULT_EVAL_DIR, *, k: int = 5,
             ablation_k: int = 1, embedder=None) -> EvalReport:
    eval_dir = Path(eval_dir)
    corpus = load_corpus(eval_dir / "corpus.yaml")
    queries = load_queries(eval_dir / "queries.yaml")
    retr = HybridRetriever(corpus, embedder=embedder)

    def hits_for(q: dict[str, Any], mode: Mode = "hybrid", topk: int | None = None) -> list[str]:
        return [
            h.record.id
            for h in retr.retrieve(
                q["query"], k=topk or k, scope=_scope_of(q), mechanism=q.get("mechanism"),
                maturity_floor="L2", mode=mode,
            )
        ]

    must_f = must_t = len_f = len_t = 0
    per_type_f: dict[str, int] = {}
    per_type_t: dict[str, int] = {}
    type_counts: dict[str, int] = {}

    for q in queries:
        qtype = q.get("type", "free-form")
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
        ids = hits_for(q)
        match = q.get("match", "all")
        mf, mt = _recall_for_query(ids, q["expect"], lenient=False, match=match)
        lf, lt = _recall_for_query(ids, q["expect"], lenient=True, match=match)
        must_f += mf
        must_t += mt
        len_f += lf
        len_t += lt
        per_type_f[qtype] = per_type_f.get(qtype, 0) + mf
        per_type_t[qtype] = per_type_t.get(qtype, 0) + mt

    # symbol-name ablation (P1-10 ④): run at a tight ablation_k so a single arm's
    # weakness is visible (pure vectors dilute on bare symbol names; BM25 rescues).
    ablation: dict[str, float] = {}
    sym_qs = [q for q in queries if q.get("symbol")]
    for mode in ("bm25", "vector", "hybrid"):
        f = t = 0
        for q in sym_qs:
            ids = hits_for(q, mode=mode, topk=ablation_k)  # type: ignore[arg-type]
            sf, st = _recall_for_query(ids, q["expect"], lenient=False, match=q.get("match", "all"))
            f += sf
            t += st
        ablation[mode] = (f / t) if t else 0.0

    return EvalReport(
        k=k,
        must_recall=(must_f / must_t) if must_t else 0.0,
        lenient_recall=(len_f / len_t) if len_t else 0.0,
        per_type={t: per_type_f[t] / per_type_t[t] for t in type_counts if per_type_t.get(t)},
        type_counts=type_counts,
        ablation=ablation,
        n_queries=len(queries),
    )


def load_baseline(eval_dir: str | Path = DEFAULT_EVAL_DIR) -> dict[str, Any] | None:
    p = Path(eval_dir) / "baseline.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def write_baseline(report: EvalReport, eval_dir: str | Path = DEFAULT_EVAL_DIR) -> Path:
    p = Path(eval_dir) / "baseline.json"
    p.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
    return p
