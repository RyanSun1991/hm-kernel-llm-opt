"""Read concurrency and bounded scheduling against real SQLite and Git state."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from sqlite3 import OperationalError

import pytest
from test_evolution_mining import GIT
from test_evolution_service import Scenario

from hmopt.evolution.production import ProductionConfig
from hmopt.evolution.runtime import runtime_cycle
from hmopt.evolution.scan import start_scan
from hmopt.evolution.store import EvolutionStore


def test_direct_and_paged_scans_share_retirement_and_successor_rules(tmp_path):
    from hmopt.evolution.mining import Pattern

    scenario = Scenario(tmp_path, GIT)
    service = scenario.service
    original = service.store.read("pattern", scenario.pattern_key)["data"]["pattern"]
    version_two = Pattern.model_validate({**original, "version": 2})
    key = f"{version_two.pattern_id}@2"
    service.import_pattern(version_two)
    service.activate_pattern(key, actor="curator", note="Reviewed the successor version.")
    service.retire_pattern(key, actor="curator", note="Retire this family after adverse evidence.")

    def start(request_id):
        return start_scan(
            service,
            scenario.repo,
            revision="HEAD",
            owners={"**": "owner"},
            request_id=request_id,
            actor="researcher",
        )

    assert service.scan(scenario.repo, owners={"**": "owner"})["candidates"] == []
    assert start("retired")["data"]["status"] == "no_active_patterns"
    successor = Pattern.model_validate({**original, "version": 3})
    service.import_pattern(successor)
    service.activate_pattern(
        f"{successor.pattern_id}@3",
        actor="curator",
        note="Reviewed corrected applicability constraints.",
    )
    direct = service.scan(scenario.repo, owners={"**": "owner"})
    snapshot = service.store.read_evidence(start("successor")["data"]["snapshot_sha256"])
    assert {row["data"]["candidate"]["pattern_version"] for row in direct["candidates"]} == {3}
    assert [row["data"]["pattern"]["version"] for row in snapshot["patterns"]] == [3]


def test_reads_do_not_wait_for_an_uncommitted_writer(tmp_path):
    store = EvolutionStore(tmp_path)
    with store.transaction() as db:
        store.put(db, "fixture", "one", {"value": 1})
        sha = store.evidence(db, {"source": "fixture"})
        store.event(db, "one", "created", "tester", {})

    def read():
        return (
            store.read("fixture", "one"),
            store.list("fixture"),
            store.read_evidence(sha),
            store.audit("one"),
        )

    with ThreadPoolExecutor(max_workers=1) as pool, store.transaction() as db:
        store.put(db, "fixture", "one", {"value": 2}, 1)
        row, page, evidence, audit = pool.submit(read).result(timeout=2)
        assert row["data"] == {"value": 1}
        assert page == [row]
        assert evidence == {"source": "fixture"}
        assert [event["action"] for event in audit] == ["created"]
    assert store.read("fixture", "one")["data"] == {"value": 2}


def test_read_snapshot_is_consistent_and_rejects_writes(tmp_path):
    store = EvolutionStore(tmp_path)
    with store.transaction() as db:
        store.put(db, "fixture", "one", {"value": 1})
    with store.transaction(read_only=True) as snapshot:
        original = store.get(snapshot, "fixture", "one")
        with store.transaction() as writer:
            store.put(writer, "fixture", "one", {"value": 2}, original["version"])
        assert store.get(snapshot, "fixture", "one") == original
        with pytest.raises(OperationalError, match="readonly"):
            snapshot.execute("DELETE FROM records")
    assert store.read("fixture", "one")["version"] == 2


def test_list_page_uses_one_select_instead_of_per_record_fetches(tmp_path, monkeypatch):
    store = EvolutionStore(tmp_path)
    with store.transaction() as db:
        for i in range(120):
            store.put(db, "fixture", str(i), {"value": i})
    statements = []
    transaction = store.transaction

    @contextmanager
    def traced(**kwargs):
        with transaction(**kwargs) as db:
            db.set_trace_callback(statements.append)
            yield db

    monkeypatch.setattr(store, "transaction", traced)
    page = store.list("fixture", limit=100, offset=10)
    assert [row["data"]["value"] for row in page] == list(range(10, 110))
    assert len([sql for sql in statements if sql.startswith("SELECT")]) == 1


@pytest.mark.parametrize("owned_count", [1, 16])
def test_supervisor_advances_owned_and_independent_scans_only_once(
    tmp_path, monkeypatch, owned_count
):
    from hmopt.evolution import runs, scan

    scenario = Scenario(tmp_path, GIT)
    for i in range(3):
        (scenario.repo / f"extra-{i}.c").write_text("int value;\n", encoding="utf-8", newline="\n")
    scenario.commit("ensure scanning requires several pages")
    jobs = [
        start_scan(
            scenario.service,
            scenario.repo,
            revision="HEAD",
            owners={"**": "owner"},
            request_id=name,
            actor="researcher",
        )
        for name in [*(f"owned-{i}" for i in range(owned_count)), "independent"]
    ]
    with scenario.service.store.transaction() as db:
        scenario.service.store.put(db, "workspace_run", "run", {"status": "running"})
    seen = []
    advance = scan.scan_next

    def next_page(service, scan_id, *, expected_version):
        seen.append(scan_id)
        return advance(service, scan_id, expected_version=expected_version, page_size=1)

    def advance_run(service, run_id, **kwargs):
        for job in jobs[:-1]:
            next_page(service, job["id"], expected_version=job["version"])
        return {
            "run": {"data": {"status": "running"}},
            "actions": [],
            "scans": {f"project-{i}": {"scan_id": job["id"]} for i, job in enumerate(jobs[:-1])},
        }

    monkeypatch.setattr(scan, "scan_next", next_page)
    monkeypatch.setattr(runs, "advance_run", advance_run)
    result = runtime_cycle(scenario.service, ProductionConfig(), worker_id="fixture")
    assert not result["errors"]
    assert seen == [job["id"] for job in jobs]
    assert all(scenario.service.store.read("scan", job["id"])["data"]["pages"] == 1 for job in jobs)
