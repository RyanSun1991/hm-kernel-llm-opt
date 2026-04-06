"""Tests for the Windows command relay service."""

from __future__ import annotations

import json
import os
import threading
from http.server import HTTPServer
from typing import Any
from unittest.mock import patch
from urllib.request import Request, urlopen

import pytest

# Import directly from the tools directory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "windows_relay"))
from relay_service import RelayHandler, ALLOWED_COMMANDS


@pytest.fixture()
def relay_server():
    """Start the relay server on a random port."""
    server = HTTPServer(("127.0.0.1", 0), RelayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _get(url: str, path: str) -> dict[str, Any]:
    req = Request(f"{url}{path}", method="GET")
    with urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def _post(url: str, path: str, body: dict[str, Any], headers: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode("utf-8")
    req = Request(f"{url}{path}", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except Exception as exc:
        if hasattr(exc, "code"):
            return exc.code, json.loads(exc.read())
        raise


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health_endpoint(relay_server):
    result = _get(relay_server, "/health")
    assert result["status"] == "ok"
    assert "hostname" in result
    assert "allowed_commands" in result


def test_unknown_path_returns_404(relay_server):
    code, result = _post(relay_server, "/unknown", {})
    assert code == 404


def test_exec_missing_command(relay_server):
    code, result = _post(relay_server, "/exec", {"args": ["--help"]})
    assert code == 400
    assert "command is required" in result.get("error", "")


def test_exec_disallowed_command(relay_server):
    code, result = _post(relay_server, "/exec", {"command": "rm", "args": ["-rf", "/"]})
    assert code == 403
    assert "not allowed" in result.get("error", "")


def test_exec_allowed_command(relay_server):
    # ping should be in the allowlist and available on all platforms
    code, result = _post(relay_server, "/exec", {
        "command": "ping",
        "args": ["-c", "1", "127.0.0.1"] if os.name != "nt" else ["-n", "1", "127.0.0.1"],
        "timeout_s": 10,
    })
    assert code == 200
    assert "returncode" in result
    assert "stdout" in result
    assert "duration_s" in result


def test_exec_command_not_found(relay_server):
    code, result = _post(relay_server, "/exec", {"command": "fastboot", "args": ["devices"]})
    # fastboot likely not installed in CI, so we expect returncode -1
    assert code == 200
    if result["returncode"] == -1:
        assert "not found" in result["stderr"]


def test_secret_auth_rejected(relay_server):
    """When RELAY_SECRET is set, requests without it should be rejected."""
    with patch.dict(os.environ, {"RELAY_SECRET": "test-secret-123"}):
        code, result = _post(relay_server, "/exec", {"command": "ping", "args": ["-c", "1", "127.0.0.1"]})
        assert code == 403
        assert "secret" in result.get("error", "").lower()


def test_secret_auth_accepted(relay_server):
    """When RELAY_SECRET is set, requests with correct header should work."""
    with patch.dict(os.environ, {"RELAY_SECRET": "test-secret-123"}):
        code, result = _post(
            relay_server,
            "/exec",
            {"command": "ping", "args": ["-c", "1", "127.0.0.1"] if os.name != "nt" else ["-n", "1", "127.0.0.1"], "timeout_s": 10},
            headers={"X-Relay-Secret": "test-secret-123"},
        )
        assert code == 200
        assert "returncode" in result


def test_allowed_commands_set():
    assert "fastboot" in ALLOWED_COMMANDS
    assert "hdc" in ALLOWED_COMMANDS
    assert "adb" in ALLOWED_COMMANDS
    assert "ping" in ALLOWED_COMMANDS
    assert "pscp" in ALLOWED_COMMANDS
    assert "scp" in ALLOWED_COMMANDS
    # Dangerous commands should NOT be allowed
    assert "rm" not in ALLOWED_COMMANDS
    assert "bash" not in ALLOWED_COMMANDS
    assert "cmd" not in ALLOWED_COMMANDS
    assert "powershell" not in ALLOWED_COMMANDS


def test_exec_timeout(relay_server):
    """Commands that exceed timeout should return -9."""
    # Use a command that would take a while but is in the allowlist
    code, result = _post(relay_server, "/exec", {
        "command": "ping",
        "args": ["-c", "100", "127.0.0.1"] if os.name != "nt" else ["-n", "100", "127.0.0.1"],
        "timeout_s": 1,
    })
    assert code == 200
    assert result["returncode"] == -9
    assert "timeout" in result["stderr"]
