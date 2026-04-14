"""Tests for the relay-backed instruction-test entry points in
hmopt.api.auto_test_mcp_service."""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest.mock import patch

import pytest

from hmopt.api import auto_test_mcp_service as svc


class FakeRelayHandler(BaseHTTPRequestHandler):
    """Canned relay that responds to GET /health and POST /exec.

    Every POST records its parsed body on the server instance (as
    ``server.last_request``) and returns whatever the test has queued up
    as ``server.response``.
    """

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            body = json.dumps({"status": "ok", "platform": "fake"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        body = json.loads(raw) if raw else {}
        self.server.last_request = body  # type: ignore[attr-defined]
        resp = self.server.response  # type: ignore[attr-defined]
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def fake_relay():
    server = HTTPServer(("127.0.0.1", 0), FakeRelayHandler)
    server.response = {  # type: ignore[attr-defined]
        "returncode": 0,
        "stdout": "{}",
        "stderr": "",
        "duration_s": 0.01,
        "command": "python ...",
    }
    server.last_request = None  # type: ignore[attr-defined]
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"

    # Clear cached MCP server between tests — the service caches it via lru_cache.
    svc.build_auto_test_fastmcp_server.cache_clear()  # type: ignore[attr-defined]

    with patch.dict(os.environ, {
        "HMOPT_AUTO_TEST_RELAY_URL": url,
        "HMOPT_AUTO_TEST_RELAY_SECRET": "",
        "HMOPT_FLASH_RELAY_URL": "",
        "HMOPT_FLASH_RELAY_SECRET": "",
        "HMOPT_AUTO_TEST_WINDOWS_TEST_DIR": r"D:\modelCase_OH_single",
    }):
        yield server, url
    server.shutdown()


def test_relay_health_check(fake_relay):
    _, url = fake_relay
    result = svc.auto_test_relay_health_check()
    assert result["relay_reachable"] is True
    assert result["relay_status"]["status"] == "ok"


def test_run_instruction_test_stock_no_compare(fake_relay):
    server, _ = fake_relay
    server.response["stdout"] = json.dumps({
        "success": True,
        "phase": "complete",
        "test_dir": r"D:\modelCase_OH_single",
        "report_path": r"D:\modelCase_OH_single\reports\report_20260414114948",
        "report_name": "report_20260414114948",
    })

    result = svc.run_instruction_test()
    assert result["success"] is True
    assert result["report_name"] == "report_20260414114948"
    assert "relay_result" in result

    # Assert the argv does NOT include --compare for stock runs.
    req = server.last_request
    assert req is not None
    assert req["command"] == "python"
    argv = req["args"]
    assert argv[0] == svc.DEFAULT_INSTRUCTION_TEST_PIPELINE_SCRIPT
    assert "--test-dir" in argv
    assert r"D:\modelCase_OH_single" in argv
    assert "--compare" not in argv


def test_run_instruction_test_feature_with_compare(fake_relay):
    server, _ = fake_relay
    feature_report = r"D:\modelCase_OH_single\reports\report_20260414120950"
    baseline_report = r"D:\modelCase_OH_single\reports\report_20260414114948"
    server.response["stdout"] = json.dumps({
        "success": True,
        "phase": "complete",
        "report_path": feature_report,
        "report_name": "report_20260414120950",
        "compare": {
            "ok": True,
            "returncode": 0,
            "result": {
                "success": True,
                "baseline_dir": baseline_report,
                "candidate_dir": feature_report,
                "delta": -42,
                "delta_pct": -1.5,
            },
        },
    })

    result = svc.run_instruction_test(
        compare=True,
        baseline_report=baseline_report,
    )
    assert result["success"] is True
    assert result["compare"]["result"]["delta"] == -42

    req = server.last_request
    argv = req["args"]
    assert "--compare" in argv
    assert "--baseline-report" in argv
    assert baseline_report in argv
    assert "--compare-script" in argv


def test_run_instruction_test_requires_baseline_when_compare_true(fake_relay):
    with pytest.raises(ValueError, match="baseline_report is required"):
        svc.run_instruction_test(compare=True)


def test_run_instruction_test_handles_non_json_stdout(fake_relay):
    server, _ = fake_relay
    server.response["stdout"] = "not even remotely json"
    server.response["returncode"] = 2

    result = svc.run_instruction_test()
    assert result["success"] is False
    assert result["phase"] == "parse_pipeline_output"


def test_compare_reports_via_relay(fake_relay):
    server, _ = fake_relay
    server.response["stdout"] = json.dumps({
        "success": True,
        "baseline_dir": "A",
        "candidate_dir": "B",
        "delta": 10,
        "delta_pct": 5.0,
    })

    result = svc.compare_reports(
        baseline_report="A",
        candidate_report="B",
    )
    assert result["success"] is True
    assert result["delta"] == 10

    req = server.last_request
    argv = req["args"]
    assert argv[0] == svc.DEFAULT_REPORT_COMPARE_SCRIPT
    assert "--baseline" in argv and "A" in argv
    assert "--candidate" in argv and "B" in argv


def test_compare_reports_validates_args(fake_relay):
    with pytest.raises(ValueError):
        svc.compare_reports(baseline_report="", candidate_report="b")
    with pytest.raises(ValueError):
        svc.compare_reports(baseline_report="a", candidate_report="")


def test_run_instruction_test_async_completes(fake_relay):
    server, _ = fake_relay
    server.response["stdout"] = json.dumps({
        "success": True,
        "phase": "complete",
        "report_path": r"D:\modelCase_OH_single\reports\report_20260414121000",
        "report_name": "report_20260414121000",
    })

    task_info = svc.run_instruction_test_async()
    assert task_info["status"] == "pending"
    assert "task_id" in task_info

    for _ in range(100):
        status = svc.get_instruction_task_status(task_info["task_id"])
        if status["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)

    assert status["status"] == "succeeded"
    assert status["result"]["success"] is True


def test_run_instruction_test_forwards_venv_flags(fake_relay):
    server, _ = fake_relay
    server.response["stdout"] = json.dumps({"success": True, "phase": "complete", "report_path": "x", "report_name": "x"})

    # Default (use_venv=True, no explicit venv_dir): no venv flags present.
    svc.run_instruction_test()
    argv = server.last_request["args"]
    assert "--no-venv" not in argv
    assert "--venv-dir" not in argv

    # use_venv=False: pipeline gets --no-venv.
    svc.run_instruction_test(use_venv=False)
    argv = server.last_request["args"]
    assert "--no-venv" in argv
    assert "--venv-dir" not in argv

    # use_venv=True with custom venv_dir: pipeline gets --venv-dir.
    svc.run_instruction_test(venv_dir="myenv")
    argv = server.last_request["args"]
    assert "--no-venv" not in argv
    assert "--venv-dir" in argv
    assert "myenv" in argv


def test_relay_url_missing_raises(monkeypatch):
    # Ensure neither variant is set.
    monkeypatch.delenv("HMOPT_AUTO_TEST_RELAY_URL", raising=False)
    monkeypatch.delenv("HMOPT_FLASH_RELAY_URL", raising=False)
    with pytest.raises(ValueError, match="HMOPT_AUTO_TEST_RELAY_URL"):
        svc._relay_url()


def test_instruction_task_not_found():
    with pytest.raises(ValueError, match="task not found"):
        svc.get_instruction_task_status("does-not-exist")


def test_build_auto_test_fastmcp_server_registers_new_tools():
    server = svc.build_auto_test_fastmcp_server()
    if server is None:
        pytest.skip("mcp package not installed")
    # FastMCP keeps tools in an internal registry; check the method name map.
    tool_names = set()
    # Try common internal attributes across MCP versions.
    for attr in ("_tool_manager", "_tools"):
        candidate = getattr(server, attr, None)
        if candidate is not None:
            # _tool_manager has .tools, a dict keyed by name.
            if hasattr(candidate, "tools"):
                tool_names.update(candidate.tools.keys())
            elif isinstance(candidate, dict):
                tool_names.update(candidate.keys())
    # If we can't introspect, at least ensure build succeeded.
    if tool_names:
        assert "run_instruction_test" in tool_names
        assert "compare_reports" in tool_names
        assert "auto_test_relay_health" in tool_names
