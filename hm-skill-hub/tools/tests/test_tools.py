"""Tests for the hub toolchain: parse_memory, path_scope, lint.

Runnable two ways (the hub is split-out-ready, so it must not hard-depend on a
test runner being present):

    cd hm-skill-hub && python -m pytest tools/tests/      # via pytest
    cd hm-skill-hub && python tools/tests/test_tools.py   # standalone
"""
from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
HUB = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import lint  # type: ignore[import-not-found]
from parse_memory import ParseError, parse, parse_frontmatter  # type: ignore[import-not-found]
from path_scope import check_scope_consistency, derive_path_scope  # type: ignore[import-not-found]


# ---- helpers ---------------------------------------------------------------

def _write(tmp: Path, rel: str, text: str) -> Path:
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


F001 = HUB / "knowledge/targets/mm-vmscan-shrink_node/facts/F001-hoist-sc-priority.md"


# ---- parse_memory ----------------------------------------------------------

def test_parse_frontmatter_splits_fm_and_body():
    fm, body = parse_frontmatter("---\nid: X001\ntype: fact\n---\n\nhello body\n")
    assert fm == {"id": "X001", "type": "fact"}
    assert body == "hello body"


def test_parse_one_file_one_record():
    recs = parse(F001)
    assert len(recs) == 1
    r = recs[0]
    assert r.id == "F001"
    assert r.fields["type"] == "pattern"
    assert r.fields["scope"]["target_slug"] == "mm-vmscan-shrink_node"
    assert r.body.startswith("# F001")


def test_parse_missing_frontmatter_raises(tmp_path: Path):
    p = _write(tmp_path, "x.md", "# no frontmatter here\n")
    try:
        parse(p)
    except ParseError:
        return
    raise AssertionError("expected ParseError on missing frontmatter")


def test_parse_dates_stringified():
    fm, _ = parse_frontmatter("---\nid: H001\nadded_on: 2026-04-20\n---\n")
    assert fm["added_on"] == "2026-04-20"  # date -> ISO string, not a date object


# ---- path_scope.derive -----------------------------------------------------

def test_derive_global_lesson():
    ps = derive_path_scope("global/anti_patterns/A001-x.md")
    assert ps is not None and ps.schema == "global_lesson"
    assert ps.level == "global" and ps.category == "anti_patterns"


def test_derive_global_bad_plan():
    ps = derive_path_scope("global/bad_plans/B001-x.md")
    assert ps is not None and ps.schema == "bad_plan" and ps.level == "global"


def test_derive_subsystem_memory_item():
    ps = derive_path_scope("subsystems/mm-reclaim/G001-x.md")
    assert ps is not None and ps.schema == "memory_item"
    assert ps.level == "subsystem" and ps.subsystem == "mm-reclaim"


def test_derive_subsystem_bad_plan():
    ps = derive_path_scope("subsystems/workqueue-threadpool/bad_plans/B002-x.md")
    assert ps is not None and ps.schema == "bad_plan"
    assert ps.level == "subsystem" and ps.subsystem == "workqueue-threadpool"


def test_derive_target_fact_and_idea():
    ps = derive_path_scope("targets/foo/facts/F009-x.md")
    assert ps is not None and ps.schema == "memory_item" and ps.target_slug == "foo"
    ps2 = derive_path_scope("targets/foo/idea_ledger/L009-x.md")
    assert ps2 is not None and ps2.schema == "idea" and ps2.target_slug == "foo"


def test_derive_unrecognized_returns_none():
    assert derive_path_scope("global/whoops/Z001-x.md") is None
    assert derive_path_scope("random/file.md") is None


# ---- path_scope.check (the gate AC: mismatches must be caught precisely) ----

def test_scope_ok_for_real_F001():
    ps = derive_path_scope("targets/mm-vmscan-shrink_node/facts/F001-x.md")
    recs = parse(F001)
    assert check_scope_consistency(ps, recs[0].id, recs[0].fields) == []


def test_memory_item_target_level_global_is_mismatch():
    ps = derive_path_scope("targets/foo/facts/F009-x.md")
    fields = {"scope": {"level": "global"}}
    errs = check_scope_consistency(ps, "F009", fields)
    assert errs and "targets/foo" in errs[0]


def test_memory_item_wrong_target_slug_is_mismatch():
    ps = derive_path_scope("targets/foo/facts/F009-x.md")
    fields = {"scope": {"level": "function", "target_slug": "bar"}}
    errs = check_scope_consistency(ps, "F009", fields)
    assert any("slug" in e for e in errs)


def test_idea_missing_target_slug_is_mismatch():
    ps = derive_path_scope("targets/foo/idea_ledger/L009-x.md")
    errs = check_scope_consistency(ps, "L009", {"mechanism": "hoist-invariant"})
    assert errs and "target_slug" in errs[0]


def test_idea_wrong_target_slug_is_mismatch():
    ps = derive_path_scope("targets/foo/idea_ledger/L009-x.md")
    errs = check_scope_consistency(ps, "L009", {"target_slug": "bar"})
    assert any("bar" in e for e in errs)


def test_global_bad_plan_non_star_subsystems_is_mismatch():
    ps = derive_path_scope("global/bad_plans/B009-x.md")
    errs = check_scope_consistency(ps, "B009", {"applies_to": {"subsystems": ["mm-reclaim"]}})
    assert errs and "['*']" in errs[0]


def test_subsystem_memory_item_wrong_subsystem_is_mismatch():
    ps = derive_path_scope("subsystems/mm-reclaim/G009-x.md")
    fields = {"scope": {"level": "subsystem", "subsystem": "sched"}}
    errs = check_scope_consistency(ps, "G009", fields)
    assert any("subsystem" in e for e in errs)


def test_global_lesson_kind_mismatch_with_path():
    ps = derive_path_scope("global/heuristics/H009-x.md")
    errs = check_scope_consistency(ps, "H009", {"kind": "anti_pattern"})
    assert errs and "kind" in errs[0]


# ---- lint end-to-end -------------------------------------------------------

def test_lint_real_hub_is_clean():
    assert lint.main([str(HUB)]) == 0


def test_lint_catches_scope_mismatch_file(tmp_path: Path):
    # memory_item under targets/.../facts but with scope.level=global -> mismatch
    bad = _write(
        tmp_path,
        "knowledge/targets/foo/facts/F099-bad.md",
        "---\nid: F099\ntype: fact\ntitle: bad\n"
        "scope: {level: global}\n"
        "source: [{kind: doc, ref: x}]\n"
        "maturity: L1\nstatus: active\ncreated_at: 2026-01-01T00:00:00Z\n"
        "---\n\nbody\n",
    )
    errs = lint.lint_record_file(bad)
    assert any("scope mismatch" in m for _, m in errs)


def test_lint_catches_schema_violation_file(tmp_path: Path):
    # memory_item missing required `source`
    bad = _write(
        tmp_path,
        "knowledge/targets/foo/facts/F098-bad.md",
        "---\nid: F098\ntype: fact\ntitle: bad\n"
        "scope: {level: function, target_slug: foo}\n"
        "maturity: L1\nstatus: active\ncreated_at: 2026-01-01T00:00:00Z\n"
        "---\n\nbody\n",
    )
    errs = lint.lint_record_file(bad)
    assert any("source" in m for _, m in errs)


def test_lint_unrecognized_path_is_flagged(tmp_path: Path):
    bad = _write(
        tmp_path,
        "knowledge/global/whoops/Z099-bad.md",
        "---\nid: Z099\n---\n\nbody\n",
    )
    errs = lint.lint_record_file(bad)
    assert any("unrecognized" in m or "dispatch" in m for _, m in errs)


# ---- standalone runner -----------------------------------------------------

def _run_standalone() -> int:
    import inspect
    import tempfile

    g = dict(globals())
    tests = [(n, f) for n, f in g.items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in sorted(tests):
        params = inspect.signature(fn).parameters
        try:
            if "tmp_path" in params:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
