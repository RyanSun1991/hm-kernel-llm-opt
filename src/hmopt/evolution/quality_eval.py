"""Reproducible labelled evaluation and bounded mechanism retrieval suggestions."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Literal

from pydantic import Field, model_validator

from .change_analysis import HistoryAnalysis
from .production import Strict
from .research import Assessment
from .store import digest


class Label(Strict):
    case_id: str = Field(min_length=1, max_length=200)
    family: str = Field(min_length=1, max_length=200)
    split: Literal["development", "holdout"]
    expected: bool
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expert: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=10, max_length=4000)


class Prediction(Strict):
    case_id: str = Field(min_length=1, max_length=200)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvaluationSet(Strict):
    name: str = Field(min_length=1, max_length=200)
    task: Literal["history", "applicability"]
    labels: list[Label] = Field(min_length=1, max_length=10000)
    predictions: list[Prediction] = Field(max_length=10000)
    model_id: str = Field(min_length=1, max_length=200)
    method_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    minimum_holdout: int = Field(default=30, ge=1, le=10000)
    minimum_precision: float = Field(default=0.8, ge=0, le=1)
    minimum_recall: float = Field(default=0.7, ge=0, le=1)
    minimum_coverage: float = Field(default=0.9, ge=0, le=1)

    @model_validator(mode="after")
    def split_integrity(self):
        if len({label.input_sha256 for label in self.labels}) != len(self.labels):
            raise ValueError("Duplicate labelled input snapshots would inflate evaluation counts")
        for rows in (self.labels, self.predictions):
            if len({row.case_id for row in rows}) != len(rows):
                raise ValueError("Duplicate evaluation case IDs")
        families, inputs = {}, {}
        for row in self.labels:
            for mapping, key in ((families, row.family), (inputs, row.input_sha256)):
                if key in mapping and mapping[key] != row.split:
                    raise ValueError("Code family or input leaks between development and holdout")
                mapping[key] = row.split
        if not {p.case_id for p in self.predictions} <= {label.case_id for label in self.labels}:
            raise ValueError("Predictions contain unlabelled cases")
        return self


def evaluate(service, dataset: EvaluationSet, *, actor):
    actor = service._actor(actor)
    service.store.read_evidence(dataset.method_sha256)
    predictions = {p.case_id: p for p in dataset.predictions}
    scores = {}
    outcomes = []
    for label in dataset.labels:
        service.store.read_evidence(label.input_sha256)
        prediction = predictions.get(label.case_id)
        result = None
        if prediction:
            if prediction.input_sha256 != label.input_sha256:
                raise ValueError("Prediction input does not match the labelled input snapshot")
            raw = service.store.read_evidence(prediction.report_sha256)
            if dataset.task == "history":
                report = HistoryAnalysis.model_validate(raw)
                if report.packet_sha256 != label.input_sha256:
                    raise ValueError("History prediction references a different packet")
                result = {"patterns": True, "no_pattern": False, "needs_context": None}[
                    report.outcome
                ]
            else:
                # The envelope binds a semantic report to its prepared research input.
                if raw.get("input_sha256") != label.input_sha256:
                    raise ValueError("Applicability prediction references a different input")
                report = Assessment.model_validate(raw["assessment"])
                result = {"applicable": True, "not_applicable": False, "needs_context": None}[
                    report.result
                ]
        split = scores.setdefault(
            label.split,
            {
                "total": 0,
                "positive": 0,
                "negative": 0,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "abstained": 0,
                "missing": 0,
            },
        )
        split["total"] += 1
        split["positive" if label.expected else "negative"] += 1
        if result is None:
            split["abstained"] += 1
            split["missing"] += int(prediction is None)
            if label.expected:
                split["fn"] += 1  # Missing positives count against end-to-end recall.
        else:
            split[
                "tp"
                if result and label.expected
                else "fp"
                if result
                else "fn"
                if label.expected
                else "tn"
            ] += 1
        outcomes.append(
            {
                "case_id": label.case_id,
                "split": label.split,
                "expected": label.expected,
                "prediction": result,
            }
        )
    for split in scores.values():
        split["precision"] = (
            split["tp"] / (split["tp"] + split["fp"]) if split["tp"] + split["fp"] else None
        )
        split["recall"] = split["tp"] / split["positive"] if split["positive"] else None
        split["coverage"] = (split["total"] - split["abstained"]) / split["total"]
    held = scores.get("holdout", {})
    reasons = []
    if held.get("total", 0) < dataset.minimum_holdout:
        reasons.append("insufficient_holdout_cases")
    if not held.get("positive") or not held.get("negative"):
        reasons.append("holdout_requires_positive_and_negative_examples")
    for metric in ("precision", "recall", "coverage"):
        if held.get(metric) is None or held[metric] < getattr(dataset, "minimum_" + metric):
            reasons.append("holdout_" + metric + "_below_threshold_or_undefined")
    data = dataset.model_dump(mode="json")
    with service.store.transaction() as db:
        sha = service.store.evidence(db, data)
        return service.store.put(
            db,
            "quality_evaluation",
            "evaluation_" + sha,
            {
                "dataset_sha256": sha,
                "scores": scores,
                "outcomes": outcomes,
                "quality_gate": "pass" if not reasons else "fail",
                "reasons": reasons,
                "model_id": dataset.model_id,
                "method_sha256": dataset.method_sha256,
                "label_identity": "trusted_operator_supplied_expert_labels",
                "model_identity": "operator_declared_not_provider_attestation",
                "scope": "labelled dataset only; not proof of code safety or device benefit",
            },
        )


def _terms(text):
    terms = set(re.findall(r"[a-z_][a-z_0-9]{2,}", text.lower()))
    for phrase in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.update(phrase[i : i + 2] for i in range(len(phrase) - 1))
    return terms


def suggest_groups(service, repo, *, limit=200, offset=0, threshold=0.35):
    """Paged lexical mechanism retrieval. A Skill must judge every proposed synthesis group."""

    if type(limit) is not int or not 2 <= limit <= 500 or type(offset) is not int or offset < 0:
        raise ValueError("Grouping window requires limit=2..500 and nonnegative offset")
    if isinstance(threshold, bool) or not 0 < threshold <= 1:
        raise ValueError("Grouping threshold must be greater than zero and at most one")
    repo_id = service.repo_identity(repo)
    with service.store.transaction() as db:
        rows = db.execute(
            "SELECT id FROM records WHERE kind='history_analysis' "
            "AND json_extract(payload,'$.repo_id')=? "
            "AND json_extract(payload,'$.status')='patterns' ORDER BY rowid LIMIT ? OFFSET ?",
            (repo_id, limit + 1, offset),
        ).fetchall()
        selected = [service.store.get(db, "history_analysis", row[0]) for row in rows[:limit]]
    docs, inverted = {}, defaultdict(set)
    for row in selected:
        report = service.store.read_evidence(row["data"]["analysis_sha256"])
        terms = _terms(" ".join(p["mechanism"] + " " + p["problem"] for p in report["proposals"]))
        docs[row["id"]] = terms
        for term in terms:
            inverted[term].add(row["id"])
    pairs = set()
    for source, terms in docs.items():
        for term in terms:
            for other in inverted[term]:
                if source != other:
                    pairs.add(tuple(sorted((source, other))))
    ranked = []
    for left, right in sorted(pairs):
        union = docs[left] | docs[right]
        score = len(docs[left] & docs[right]) / len(union) if union else 0
        if score >= threshold:
            ranked.append({"source_ids": [left, right], "retrieval_score": round(score, 6)})
    ranked.sort(key=lambda group: (-group["retrieval_score"], group["source_ids"]))
    result = {
        "groups": ranked[:100],
        "groups_truncated": len(ranked) > 100,
        "sources_examined": len(selected),
        "next_offset": offset + limit if len(rows) > limit else None,
        "method": "lexical_jaccard_shortlist_not_semantic_equivalence",
        "scope": "pairs inside this page; cross-page relationships are not covered",
        "next_step": "Review selected source IDs with /evolve-research <profile> synthesize",
        "input_sha256": digest(selected),
    }
    with service.store.transaction() as db:
        sha = service.store.evidence(db, result)
    return {**result, "evidence_sha256": sha}
