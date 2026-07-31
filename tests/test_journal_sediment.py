"""Journal tier of the sediment flow (Team Memory design §4.6).

Deterministic journal→candidate mapping (no LLM), outcome anti-optimism gate,
schema-valid-by-construction against the real in-repo hub, non-overwriting
timestamped bundles, and the sediment marker feeding memory_status `pending`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from hmopt.sediment import journal as jm
from hmopt.sediment.pipeline import bundle_staging, sediment_journal
from hmopt.sediment.validate import validate_candidate

REPO = Path(__file__).resolve().parent.parent
HUB = REPO / "hm-skill-hub"


def _seed_journal(root: Path, *, project: str = "projA") -> list[jm.JournalEntry]:
    specs = [
        ("fact", "validated", "mm-vmscan-c-shrink-node"),
        ("heuristic", "accepted", ""),
        ("anti_pattern", "validated", ""),
        ("validation_pitfall", "validated", ""),
        ("bad_plan", "failed", "lmbench-relay"),
        ("idea", "validated", "mm-vmscan-c-shrink-node"),
        ("fact", "attempted", ""),  # gated
        ("idea", "unknown", ""),  # gated
    ]
    entries = []
    for i, (t, outcome, slug) in enumerate(specs):
        e, errs = jm.write_entry(
            root,
            contributor="ryan",
            project=project,
            type=t,
            title=f"{t} finding {i} about shrink_node reclaim",
            body=f"Body {i}: because X, do Y on the {t} path.",
            tags=["mm", "reclaim"],
            target_slug=slug,
            outcome=outcome,
            evidence=[f"mm/vmscan.c:{100 + i}"],
            applies_when=["kernel 4.x custom build"],
        )
        assert errs == [], errs
        entries.append(e)
    return entries


# --- mapping -----------------------------------------------------------------


def test_journal_to_candidates_maps_all_six_types(tmp_path):
    _seed_journal(tmp_path)
    entries, _ = jm.iter_entries(tmp_path, "ryan")
    cands, gated, errors = jm.journal_to_candidates(entries, contributor="ryan")
    assert errors == []
    assert len(cands) == 6
    assert len(gated) == 2  # outcome gate: attempted + unknown withheld
    assert {e.outcome for e in gated} == {"attempted", "unknown"}
    assert Counter(c["schema"] for c in cands) == Counter(
        {"memory_item": 1, "global_lesson": 3, "bad_plan": 1, "idea": 1}
    )
    # every candidate is lint-clean against the real hub schemas
    for cand in cands:
        assert validate_candidate(cand, HUB) == [], (cand, validate_candidate(cand, HUB))


def test_fact_mapping_details(tmp_path):
    e, errs = jm.write_entry(
        tmp_path,
        contributor="ryan",
        project="projA",
        type="fact",
        title="generated headers go stale across branch switches",
        body="Regenerate before blaming the compiler.",
        target_slug="build-headers",
        outcome="validated",
        evidence=["scripts/gen_headers.sh:12"],
        applies_when=["custom build 4.x"],
        invalidated_by=["build system v2 removes codegen"],
    )
    assert errs == []
    cands, gated, _ = jm.journal_to_candidates([e], contributor="ryan")
    assert gated == []
    (cand,) = cands
    rec = cand["record"]
    assert cand["schema"] == "memory_item" and rec["id"] == "F901"
    assert rec["scope"] == {"level": "function", "target_slug": "build-headers"}
    assert rec["contributor"] == "ryan"  # bare name: (target_slug, contributor) distinctness
    assert rec["source"] == [
        {"kind": "doc", "ref": f"journal:ryan/projA/{e.id}"},
        {"kind": "doc", "ref": "scripts/gen_headers.sh:12"},
    ]
    assert "Evidence:" in rec["body"] and "scripts/gen_headers.sh:12" in rec["body"]
    assert rec["applies_when"] == "custom build 4.x"
    assert rec["invalidation"] == "build system v2 removes codegen"
    assert validate_candidate(cand, HUB) == []


def test_lesson_prefixes_match_kind_pinning(tmp_path):
    for t in ("heuristic", "anti_pattern", "validation_pitfall"):
        _entry, errs = jm.write_entry(
            tmp_path,
            contributor="ryan",
            project="p",
            type=t,
            title=f"{t} lesson",
            body="do: the thing.",
            outcome="validated",
        )
        assert errs == []
    entries, _ = jm.iter_entries(tmp_path, "ryan")
    cands, _, _ = jm.journal_to_candidates(entries, contributor="ryan")
    by_kind = {c["record"]["kind"]: c["record"]["id"] for c in cands}
    # hub global_lesson schema pins H->heuristic, A->anti_pattern, V->validation_pitfall
    assert by_kind["heuristic"].startswith("H")
    assert by_kind["anti_pattern"].startswith("A")
    assert by_kind["validation_pitfall"].startswith("V")
    for c in cands:
        assert c["record"]["added_by"] == "ryan"
        assert c["record"]["confidence"] == "tentative"  # unreviewed layer never over-claims
        assert validate_candidate(c, HUB) == []


def test_idea_outcome_status_mapping(tmp_path):
    for outcome in ("validated", "accepted", "failed", "reverted"):
        _entry, errs = jm.write_entry(
            tmp_path,
            contributor="ryan",
            project="p",
            type="idea",
            title=f"idea {outcome}",
            body="try mechanism Z.",
            outcome=outcome,
            tags=["mech-z"],
        )
        assert errs == []
    entries, _ = jm.iter_entries(tmp_path, "ryan")
    cands, gated, _ = jm.journal_to_candidates(entries, contributor="ryan")
    assert gated == []
    statuses = sorted(c["record"]["status"] for c in cands)
    assert statuses == ["approved", "approved", "rejected", "reverted"]
    for c in cands:
        rec = c["record"]
        if rec["status"] == "approved":
            assert "approved_on" in rec
        if rec["status"] == "rejected":
            assert "rejected_on" in rec
        assert validate_candidate(c, HUB) == []


def test_provisional_ids_unique_within_batch(tmp_path):
    for i in range(3):
        _entry, errs = jm.write_entry(
            tmp_path,
            contributor="ryan",
            project="p",
            type="fact",
            title=f"fact {i}",
            body="b.",
            outcome="validated",
        )
        assert errs == []
    entries, _ = jm.iter_entries(tmp_path, "ryan")
    cands, _, _ = jm.journal_to_candidates(entries, contributor="ryan")
    ids = [c["record"]["id"] for c in cands]
    assert len(ids) == len(set(ids)) == 3
    assert all(i.startswith("F9") for i in ids)


# --- sediment_journal entrypoint ----------------------------------------------


def test_sediment_journal_end_to_end(tmp_path):
    root = tmp_path / "mem"
    _seed_journal(root)
    out = tmp_path / "staging"
    res = sediment_journal(root, contributor="ryan", out_dir=out, hub_root=HUB)
    assert res.run_id == "journal-ryan"
    assert res.n_valid == 6
    assert res.invalid == [], res.invalid
    assert res.scanned == {"journal": 8}
    assert any("outcome gate: 2" in e for e in res.parse_errors)
    out_path = Path(res.out_path)
    assert out_path.name == "journal-ryan.jsonl"
    for line in out_path.read_text(encoding="utf-8").splitlines():
        assert validate_candidate(json.loads(line), HUB) == []
    # Only successfully emitted entries are covered. attempted/unknown remain
    # pending so the outcome gate stays visible in memory_status.
    st = jm.journal_status(root, "ryan")
    assert st["pending_sediment"] == 2


def test_sediment_journal_without_hub_emits_unvalidated(tmp_path):
    root = tmp_path / "mem"
    _seed_journal(root)
    res = sediment_journal(
        root, contributor="ryan", out_dir=tmp_path / "s", hub_root=tmp_path / "nohub"
    )
    assert res.n_valid == 6  # never blocks: emitted unvalidated with a loud note
    assert any("schemas not found" in e for e in res.parse_errors)


def test_sediment_journal_project_filter_and_run_id(tmp_path):
    root = tmp_path / "mem"
    _seed_journal(root, project="projA")
    _entry, errs = jm.write_entry(
        root,
        contributor="ryan",
        project="projB",
        type="fact",
        title="projB only",
        body="b.",
        outcome="validated",
    )
    assert errs == []
    res = sediment_journal(
        root, contributor="ryan", project="projB", out_dir=tmp_path / "s", hub_root=HUB
    )
    assert res.run_id == "journal-ryan-projB"
    assert res.n_valid == 1 and res.scanned == {"journal": 1}


# --- bundling (design §4.6 item 3: timestamped, never self-swallowing) --------


def test_bundle_skips_prior_bundles_and_renumbers(tmp_path):
    root = tmp_path / "mem"
    _seed_journal(root)
    out = tmp_path / "staging"
    res = sediment_journal(root, contributor="ryan", out_dir=out, hub_root=HUB)
    _b1, n1 = bundle_staging(out, out / "_bundle_20260730T000000Z.jsonl")
    b2, n2 = bundle_staging(out, out / "_bundle_20260730T000001Z.jsonl")
    assert n1 == n2 == res.n_valid  # second bundle must not re-ingest the first
    ids = [json.loads(x)["record"]["id"] for x in b2.read_text().splitlines()]
    assert len(ids) == len(set(ids))


def test_targetless_fact_and_idea_fall_back_to_project_scope(tmp_path):
    for entry_type in ("fact", "idea"):
        _entry, errors = jm.write_entry(
            tmp_path,
            contributor="ryan",
            project="MyProject",
            type=entry_type,
            title=f"{entry_type} finding",
            body="Reusable project-level conclusion.",
            outcome="validated",
        )
        assert errors == []
    entries, _errors = jm.iter_entries(tmp_path, "ryan")
    candidates, gated, errors = jm.journal_to_candidates(entries, contributor="ryan")
    assert gated == [] and errors == []
    fact = next(
        candidate["record"] for candidate in candidates if candidate["schema"] == "memory_item"
    )
    idea = next(candidate["record"] for candidate in candidates if candidate["schema"] == "idea")
    assert fact["scope"] == {"level": "architectural", "target_slug": "myproject"}
    assert idea["target_slug"] == "myproject"
