"""Auto-test MCP lmbench runner — exercised against a fake relay (no network)."""
import json
import time

import hmopt.api.auto_test_mcp_service as svc


def _fake_relay_factory(status_sequence):
    """Return a fake _relay_exec: --start -> token; --status -> next status in seq."""
    state = {"i": 0}

    def fake(command, args, *, timeout_s, cwd=None, retries=3):
        if "--start" in args:
            return {"returncode": 0, "duration_s": 0.1, "command": "py --start",
                    "stdout": json.dumps({"success": True, "run_token": "TOK", "pid": 123})}
        if "--status" in args:
            payload = status_sequence[min(state["i"], len(status_sequence) - 1)]
            state["i"] += 1
            return {"returncode": 0, "duration_s": 0.1, "command": "py --status",
                    "stdout": json.dumps(payload)}
        return {"returncode": 1, "stdout": "{}"}

    return fake, state


def test_run_lmbench_polls_until_done(monkeypatch):
    seq = [
        {"success": True, "status": "running", "run_token": "TOK", "elapsed_s": 1.0},
        {"success": True, "status": "done", "run_token": "TOK", "elapsed_s": 2.0,
         "total_xlsx": "D:/r/cur.xlsx", "prev_total_xlsx": "D:/r/prev.xlsx",
         "digest": {"ok": True, "n_metrics": 2,
                    "hm_vs_linux": {"overall_weighted_gap_pct": 5.0},
                    "vs_previous": {"improved": 2, "regressed": 0, "matched": 2}}},
    ]
    fake, state = _fake_relay_factory(seq)
    monkeypatch.setattr(svc, "_relay_exec", fake)
    res = svc.run_lmbench_test(test_dir="D:/LmbenchAutoTest", poll_interval_s=0,
                               overall_timeout_s=30, relay_call_timeout_s=10)
    assert res["status"] == "done" and res["phase"] == "complete"
    assert res["run_token"] == "TOK"
    assert res["digest"]["vs_previous"]["regressed"] == 0
    assert state["i"] == 2  # polled twice: running then done


def test_run_lmbench_start_failure_is_graceful(monkeypatch):
    def fake(command, args, *, timeout_s, cwd=None, retries=3):
        return {"returncode": 1, "duration_s": 0.1, "command": "py --start", "stderr": "boom",
                "stdout": json.dumps({"success": False, "error": "test_dir missing"})}
    monkeypatch.setattr(svc, "_relay_exec", fake)
    res = svc.run_lmbench_test(test_dir="D:/nope", poll_interval_s=0,
                               overall_timeout_s=5, relay_call_timeout_s=5)
    assert res["success"] is False and res["phase"] == "start"
    assert "test_dir missing" in res["error"]


def test_run_lmbench_async_returns_task_id_and_resolves(monkeypatch):
    seq = [{"success": True, "status": "done", "run_token": "TOK", "elapsed_s": 1.0,
            "digest": {"ok": True, "n_metrics": 1}}]
    fake, _ = _fake_relay_factory(seq)
    monkeypatch.setattr(svc, "_relay_exec", fake)
    sub = svc.run_lmbench_test_async(test_dir="D:/x", poll_interval_s=0,
                                     overall_timeout_s=10, relay_call_timeout_s=5)
    assert sub["status"] == "pending" and sub["kind"] == "run_lmbench_test"
    tid = sub["task_id"]
    for _ in range(50):  # let the daemon thread finish
        snap = svc.get_async_task_status(tid)
        if snap["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.02)
    snap = svc.get_async_task_status(tid)
    assert snap["status"] == "succeeded" and snap["kind"] == "run_lmbench_test"
    assert snap["result"]["status"] == "done"


def test_run_lmbench_timeout(monkeypatch):
    seq = [{"success": True, "status": "running", "run_token": "TOK", "elapsed_s": 1.0}]
    fake, _ = _fake_relay_factory(seq)
    monkeypatch.setattr(svc, "_relay_exec", fake)
    res = svc.run_lmbench_test(test_dir="D:/x", poll_interval_s=0,
                               overall_timeout_s=0, relay_call_timeout_s=5)
    assert res["success"] is False and res["phase"] == "timeout"
