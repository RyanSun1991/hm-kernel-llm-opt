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


def test_exec_survives_non_cp1252_stdout(relay_server, tmp_path):
    """Relay must not crash the reader thread when a child emits bytes that
    aren't valid cp1252 (e.g. UTF-8 continuation bytes like 0x90 from a
    pipeline whose sys.stdout is UTF-8).  Prior to the fix this showed up
    on Windows hosts as 'UnicodeDecodeError: charmap codec can't decode
    byte 0x90' in the relay's stderr and an empty payload returned."""
    # Add python to the allowlist temporarily — the fixture may have
    # already done so via the pytest process's ALLOWED_COMMANDS import.
    from relay_service import ALLOWED_COMMANDS  # type: ignore

    if "python" not in ALLOWED_COMMANDS and "python3" not in ALLOWED_COMMANDS:
        pytest.skip("python not in relay allowlist for this test host")

    # Child writes UTF-8 bytes containing U+4E2D (中, 0xE4 0xB8 0xAD) and
    # U+521B (创, 0xE5 0x88 0x9B) to stdout.  Both have continuation bytes
    # that cp1252 cannot decode.
    script = tmp_path / "emit_utf8.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.buffer.write('pre 中创建 post'.encode('utf-8'))\n"
        "sys.stdout.buffer.flush()\n",
        encoding="utf-8",
    )

    command = "python3" if "python3" in ALLOWED_COMMANDS else "python"
    code, result = _post(relay_server, "/exec", {
        "command": command,
        "args": [str(script)],
        "timeout_s": 10,
    })
    assert code == 200
    assert result["returncode"] == 0
    # The payload should round-trip the UTF-8 characters intact.
    assert "中创建" in result["stdout"], result


def test_main_installs_openpyxl_when_missing(monkeypatch):
    """relay's main() does `import openpyxl` at startup; on ImportError it
    shells out to `pip install openpyxl`.  Simulate the missing case and
    assert the pip command is issued with the right args."""
    import relay_service  # type: ignore

    # Force the import to fail inside main() by masking openpyxl.
    monkeypatch.setitem(sys.modules, "openpyxl", None)

    pip_calls: list = []

    def fake_run(argv, **kwargs):
        pip_calls.append(argv)
        class _R:
            returncode = 0
        return _R()

    # Patch subprocess.run used by main(); avoid touching real pip.  Also
    # make HTTPServer.serve_forever return immediately so main() exits.
    monkeypatch.setattr(relay_service.subprocess, "run", fake_run)
    monkeypatch.setattr(relay_service.HTTPServer, "serve_forever", lambda self: None)
    monkeypatch.setattr(relay_service.HTTPServer, "server_close", lambda self: None)

    relay_service.main(["--port", "0"])

    assert pip_calls, "pip was not invoked when openpyxl was missing"
    pip_argv = pip_calls[0]
    assert "-m" in pip_argv and "pip" in pip_argv and "install" in pip_argv
    assert "openpyxl" in pip_argv


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
