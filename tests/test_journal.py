"""Team Memory journal store (design §4.2/§4.3/§4.4/§4.5/§4.7).

Covers the P1 DoD surface: storage roundtrip, redact-on-write rejection,
lexical recall with layer-relevant scoring, get/forget scoping per
contributor, feedback validation, status (pending vs sediment marker),
and non-blocking degradation on malformed files.
"""

from __future__ import annotations

import stat
from datetime import datetime, timezone
from pathlib import Path

from hmopt.sediment import journal as jm

REPO = Path(__file__).resolve().parent.parent
HUB = REPO / "hm-skill-hub"


def _log(
    root,
    *,
    contributor="ryan",
    project="projA",
    type="fact",
    title="shrink_node fact",
    body="Stable fact about shrink_node.",
    **kw,
):
    entry, errs = jm.write_entry(
        root, contributor=contributor, project=project, type=type, title=title, body=body, **kw
    )
    assert errs == [], errs
    assert entry is not None
    return entry


# --- ids ---------------------------------------------------------------------


def test_ulid_shape_and_time_ordering():
    a = jm.new_ulid(now_ms=1_000_000)
    b = jm.new_ulid(now_ms=2_000_000)
    assert len(a) == len(b) == 26
    assert a[:10] < b[:10]  # time prefix is lexicographically sortable
    assert jm.is_journal_id(f"J-{a}")
    assert not jm.is_journal_id("F031")
    assert not jm.is_journal_id("J-short")
    ids = [jm.new_ulid() for _ in range(50)]
    assert ids == sorted(ids) and len(set(ids)) == 50  # strictly monotonic in-process


def test_safe_name_blocks_traversal():
    assert jm.safe_name("../../etc", fallback="x") == "etc"
    assert jm.safe_name("a/b\\c", fallback="x") == "a-b-c"
    assert jm.safe_name("..", fallback="x") == "x"
    assert jm.safe_name("", fallback="general") == "general"


# --- storage roundtrip -------------------------------------------------------


def test_write_and_read_roundtrip(tmp_path):
    e = _log(
        tmp_path,
        type="anti_pattern",
        title="DETACHED_PROCESS breaks hdc reconnect",
        body="Use CREATE_NEW_CONSOLE instead of DETACHED_PROCESS.",
        tags=["windows", "hdc"],
        target_slug="lmbench-relay",
        outcome="validated",
        evidence=["tools/windows_relay/lmbench_pipeline.py:126"],
        applies_when=["Windows spawn needing device-restart reconnect"],
        confidence="high",
    )
    assert e.id.startswith("J-") and e.ts
    # layout: <root>/<contributor>/<project>/journal/<YYYY-MM>/<id>.md
    rel = Path(e.path).relative_to(tmp_path)
    assert rel.parts[0] == "ryan" and rel.parts[1] == "projA" and rel.parts[2] == "journal"
    assert rel.parts[3] == e.ts[:7]

    entries, errors = jm.iter_entries(tmp_path, "ryan")
    assert errors == []
    assert len(entries) == 1
    got = entries[0]
    assert got.id == e.id
    assert got.tags == ["windows", "hdc"]
    assert got.outcome == "validated"
    assert got.evidence == ["tools/windows_relay/lmbench_pipeline.py:126"]
    assert got.body.startswith("Use CREATE_NEW_CONSOLE")


def test_validation_rejects_bad_input(tmp_path):
    entry, errs = jm.write_entry(
        tmp_path,
        contributor="ryan",
        project="p",
        type="rumor",
        title="",
        body="",
        outcome="hopeful",
    )
    assert entry is None
    joined = "\n".join(errs)
    assert "type must be one of" in joined
    assert "title is required" in joined
    assert "body is required" in joined
    assert "outcome must be one of" in joined
    assert not (Path(tmp_path) / "ryan").exists()  # nothing written on reject


# --- redact-on-write (design §4.7: reject, never silently store) -------------


def test_redact_rejects_secret_with_reason(tmp_path):
    entry, errs = jm.write_entry(
        tmp_path,
        contributor="ryan",
        project="p",
        type="fact",
        title="leak",
        body="the key is AKIAABCDEFGHIJKLMNOP",
        outcome="validated",
    )
    assert entry is None
    assert any("aws-akid" in e for e in errs)
    assert any("[REDACTED]" in e for e in errs)  # remediation hint present


def test_redact_uses_hub_rules_and_rejects_public_allow_tag(tmp_path):
    # 40-hex git SHA trips the hub's generic-hex-key rule…
    sha = "a" * 40
    entry, errs = jm.write_entry(
        tmp_path,
        contributor="ryan",
        project="p",
        type="fact",
        title="sha",
        body=f"commit {sha} landed",
        outcome="validated",
        hub_root=HUB,
    )
    assert entry is None and any("generic-hex-key" in e for e in errs)
    # Journal calls are stricter than trusted hub curation: a public caller
    # cannot use an allow tag to smuggle the same token through.
    entry, errs = jm.write_entry(
        tmp_path,
        contributor="ryan",
        project="p",
        type="fact",
        title="sha ok",
        body=f"commit {sha} landed <!-- allow-secret -->",
        outcome="validated",
        hub_root=HUB,
    )
    assert entry is None and any("generic-hex-key" in e for e in errs)
    assert sha not in "\n".join(errs)  # reject responses never echo the secret
    _, _, version = jm.load_redact_rules(HUB)
    assert version.startswith("hub:")
    _, _, version = jm.load_redact_rules(None)
    assert version.startswith("builtin")


# --- recall (design §4.4: token overlap + tag/target + decay) -----------------


def test_recall_matches_and_explains(tmp_path):
    hit = _log(
        tmp_path,
        title="hdc hangs under DETACHED_PROCESS on windows",
        body="Use CREATE_NEW_CONSOLE.",
        tags=["windows", "hdc"],
        outcome="validated",
    )
    _log(
        tmp_path, title="unrelated build cache note", body="ccache speeds rebuilds.", tags=["build"]
    )
    hits = jm.recall_entries("windows hdc detached", jm.iter_entries(tmp_path, "ryan")[0], k=5)
    assert [h[0].id for h in hits] == [hit.id]  # zero-overlap entry dropped
    score, matched = hits[0][1], hits[0][2]
    assert score > 0
    assert "hdc" in matched and "windows" in matched


def test_recall_tag_and_target_bonus_rank_higher(tmp_path):
    plain = _log(tmp_path, title="windows note", body="windows detail one.")
    tagged = _log(
        tmp_path,
        title="windows note",
        body="windows detail two.",
        tags=["windows"],
        target_slug="windows-relay",
    )
    hits = jm.recall_entries("windows", jm.iter_entries(tmp_path, "ryan")[0], k=2)
    assert hits[0][0].id == tagged.id and hits[1][0].id == plain.id
    assert hits[0][1] > hits[1][1]


def test_recall_time_decay_prefers_recent_on_equal_match():
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    old = jm.JournalEntry(
        id="J-" + jm.new_ulid(),
        type="fact",
        title="windows fact",
        body="same",
        ts="2026-01-01T00:00:00+00:00",
    )
    new = jm.JournalEntry(
        id="J-" + jm.new_ulid(),
        type="fact",
        title="windows fact",
        body="same",
        ts="2026-07-29T00:00:00+00:00",
    )
    hits = jm.recall_entries("windows fact", [old, new], k=2, now=now)
    assert hits[0][0].id == new.id
    assert hits[0][1] > hits[1][1]
    assert jm.time_decay("2020-01-01T00:00:00+00:00", now=now) == 0.3  # floor
    assert jm.time_decay("2020-01-01T00:00:00Z", now=now) == 0.3  # Python 3.10-safe Z
    assert jm.time_decay("not-a-ts", now=now) == 1.0  # unparseable -> no decay


def test_recall_supports_chinese_bigrams(tmp_path):
    hit = _log(
        tmp_path,
        title="切换分支后生成头文件可能过期",
        body="先重新生成头文件，再排查编译器。",
        outcome="validated",
    )
    hits = jm.recall_entries("头文件过期", jm.iter_entries(tmp_path, "ryan")[0])
    assert hits and hits[0][0].id == hit.id
    assert "头文" in hits[0][2] or "过期" in hits[0][2]


# --- get / forget scoping ----------------------------------------------------


def test_find_and_forget_are_contributor_scoped(tmp_path):
    e = _log(tmp_path, contributor="ryan")
    assert jm.find_entry(tmp_path, "ryan", e.id) is not None
    assert jm.find_entry(tmp_path, "mallory", e.id) is None  # not visible cross-contributor
    assert jm.forget_entry(tmp_path, "mallory", e.id) is False
    assert jm.forget_entry(tmp_path, "ryan", e.id) is True
    assert not Path(e.path).exists()  # journal delete IS physical (personal layer)
    assert jm.forget_entry(tmp_path, "ryan", e.id) is False


# --- feedback ----------------------------------------------------------------


def test_feedback_appends_and_validates(tmp_path):
    e = _log(tmp_path)
    path, errs = jm.append_feedback(
        tmp_path, contributor="ryan", entry_id=e.id, verdict="helpful", note="saved an hour"
    )
    assert errs == [] and path is not None and path.name == "feedback.jsonl"
    _path2, errs2 = jm.append_feedback(
        tmp_path, contributor="ryan", entry_id="F031", verdict="stale"
    )
    assert errs2 == []  # hub ids accepted too — curators consume these
    _, errs3 = jm.append_feedback(tmp_path, contributor="ryan", entry_id=e.id, verdict="great")
    assert errs3 and "verdict must be one of" in errs3[0]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"verdict": "helpful"' in lines[0] and '"note"' in lines[0]


def test_feedback_rejects_secrets_and_uses_private_permissions(tmp_path):
    path, errs = jm.append_feedback(
        tmp_path,
        contributor="ryan",
        entry_id="F031",
        verdict="harmful",
        note="leaked AKIAABCDEFGHIJKLMNOP",
    )
    assert path is None and any("aws-akid" in error for error in errs)
    assert "AKIAABCDEFGHIJKLMNOP" not in "\n".join(errs)

    path, errs = jm.append_feedback(
        tmp_path, contributor="ryan", entry_id="F031", verdict="helpful"
    )
    assert errs == [] and path is not None
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


# --- status + pending marker --------------------------------------------------


def test_status_counts_and_pending_marker(tmp_path):
    a = _log(tmp_path, project="projA", outcome="validated")
    b = _log(tmp_path, project="projB", title="another windows fact", outcome="validated")
    st = jm.journal_status(tmp_path, "ryan")
    assert st["entries"] == 2 and st["pending_sediment"] == 2
    assert st["per_project"] == {"projA": 1, "projB": 1}
    assert st["latest"] is not None and st["latest"]["id"].startswith("J-")

    # markers record the newest covered entry id (what sediment_journal writes)
    jm.write_sediment_marker(tmp_path, "ryan", "projA", last_id=a.id)
    jm.write_sediment_marker(tmp_path, "ryan", "projB", last_id=b.id)
    st = jm.journal_status(tmp_path, "ryan")
    assert st["pending_sediment"] == 0

    _log(tmp_path, project="projA", title="post-sediment windows note")
    st = jm.journal_status(tmp_path, "ryan")
    assert st["pending_sediment"] == 1


# --- degradation: never raise on bad files ------------------------------------


def test_iter_entries_survives_malformed_files(tmp_path):
    e = _log(tmp_path)
    jdir = Path(e.path).parent
    (jdir / "J-0000000000AAAAAAAAAAAAAAAA.md").write_text("no frontmatter", encoding="utf-8")
    entries, errors = jm.iter_entries(tmp_path, "ryan")
    assert [x.id for x in entries] == [e.id]
    assert len(errors) == 1 and "frontmatter" in errors[0]


def test_manual_entry_validation_is_non_blocking(tmp_path):
    valid = _log(tmp_path, outcome="validated")
    jdir = Path(valid.path).parent
    bad_id = "J-" + jm.new_ulid()
    (jdir / f"{bad_id}.md").write_text(
        f"---\nid: {bad_id}\ntype: idea\ntitle: bad\nproject: projA\n"
        "outcome: impossible\ncontributor: ryan\nts: 2026-07-30T00:00:00Z\n---\n\nbody\n",
        encoding="utf-8",
    )
    entries, errors = jm.iter_entries(tmp_path, "ryan")
    assert [entry.id for entry in entries] == [valid.id]
    assert len(errors) == 1 and "outcome must be one of" in errors[0]


def test_parse_entry_survives_yaml_date_coercion(tmp_path):
    # Unquoted ISO ts is coerced to datetime by yaml.safe_load; the parser must
    # stringify it back so entries stay json-serializable downstream.
    jdir = tmp_path / "ryan" / "p" / "journal" / "2026-07"
    jdir.mkdir(parents=True)
    eid = "J-" + jm.new_ulid()
    (jdir / f"{eid}.md").write_text(
        f"---\nid: {eid}\ntype: fact\ntitle: t\nproject: p\noutcome: validated\n"
        "contributor: ryan\nts: 2026-07-30T14:31:00Z\n---\n\nbody\n",
        encoding="utf-8",
    )
    entries, errors = jm.iter_entries(tmp_path, "ryan")
    assert errors == []
    assert isinstance(entries[0].ts, str) and entries[0].ts.startswith("2026-07-30")


def test_resolve_memory_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HMOPT_MEMBER_MEMORY_ROOT", str(tmp_path / "envroot"))
    assert jm.resolve_memory_root(None) == tmp_path / "envroot"
    assert jm.resolve_memory_root(tmp_path / "explicit") == tmp_path / "explicit"
    monkeypatch.delenv("HMOPT_MEMBER_MEMORY_ROOT")
    assert str(jm.resolve_memory_root(None)) == jm.DEFAULT_MEMORY_ROOT


def test_write_permissions_and_input_bounds(tmp_path):
    entry = _log(tmp_path, outcome="validated")
    path = Path(entry.path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "ryan").stat().st_mode) == 0o700

    rejected, errors = jm.write_entry(
        tmp_path,
        contributor="ryan",
        project="p",
        type="fact",
        title="too much",
        body="\n".join(f"line {i}" for i in range(11)),
        outcome="validated",
    )
    assert rejected is None and any("<= 10 lines" in error for error in errors)
    rejected, errors = jm.write_entry(
        tmp_path,
        contributor="ryan",
        project="p",
        type="fact",
        title="unsafe target",
        body="body",
        target_slug="../../escape",
        outcome="validated",
    )
    assert rejected is None and any("target_slug" in error for error in errors)
