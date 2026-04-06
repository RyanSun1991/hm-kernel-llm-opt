"""Tests for flash MCP service (relay client and flash orchestration)."""

from __future__ import annotations

import json
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
from unittest.mock import patch

import pytest

from hmopt.api.flash_mcp_service import (
    _translate_image_path,
    build_flash_fastmcp_server,
    flash_and_boot,
    flash_and_boot_async,
    flash_device,
    get_flash_task_status,
    list_fastboot_devices,
    reboot_device,
    relay_health_check,
    wait_for_device_boot,
)


# ---------------------------------------------------------------------------
# Fake relay server for integration-style tests
# ---------------------------------------------------------------------------

class FakeRelayHandler(BaseHTTPRequestHandler):
    """Minimal relay that returns canned responses."""

    responses: dict[str, Any] = {}

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            body = json.dumps({"status": "ok", "platform": "test"}).encode()
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

        command = body.get("command", "")
        args = body.get("args", [])

        # Build a canned response based on command
        if command == "fastboot" and "devices" in args:
            result = {"returncode": 0, "stdout": "ABC123\tfastboot\n", "stderr": "", "duration_s": 0.1, "command": "fastboot devices"}
        elif command == "fastboot" and "flash" in args:
            result = {"returncode": 0, "stdout": "Sending 'boot' ...\nOKAY", "stderr": "", "duration_s": 2.0, "command": " ".join(["fastboot"] + args)}
        elif command == "fastboot" and "reboot" in args:
            result = {"returncode": 0, "stdout": "Rebooting...", "stderr": "", "duration_s": 0.5, "command": "fastboot reboot"}
        elif command == "hdc" and "list" in args:
            result = {"returncode": 0, "stdout": "ABC123\n", "stderr": "", "duration_s": 0.1, "command": "hdc list targets"}
        else:
            result = {"returncode": 1, "stdout": "", "stderr": f"unknown: {command} {args}", "duration_s": 0.01, "command": f"{command} {' '.join(args)}"}

        resp = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


@pytest.fixture()
def fake_relay():
    """Start a fake relay server on a random port and set env vars."""
    server = HTTPServer(("127.0.0.1", 0), FakeRelayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    relay_url = f"http://127.0.0.1:{port}"
    with patch.dict(os.environ, {
        "HMOPT_FLASH_RELAY_URL": relay_url,
        "HMOPT_FLASH_RELAY_SECRET": "",
        "HMOPT_FLASH_SERVER_IMAGE_PREFIX": "",
        "HMOPT_FLASH_WINDOWS_IMAGE_PREFIX": "",
        "HMOPT_FLASH_DEVICE_SERIAL": "",
    }):
        yield relay_url

    server.shutdown()


# ---------------------------------------------------------------------------
# Unit tests for path translation
# ---------------------------------------------------------------------------

def test_translate_no_prefix():
    with patch.dict(os.environ, {"HMOPT_FLASH_SERVER_IMAGE_PREFIX": "", "HMOPT_FLASH_WINDOWS_IMAGE_PREFIX": ""}):
        assert _translate_image_path("/home/user/builds/boot.img") == "/home/user/builds/boot.img"


def test_translate_with_prefix():
    with patch.dict(os.environ, {
        "HMOPT_FLASH_SERVER_IMAGE_PREFIX": "/home/user/builds",
        "HMOPT_FLASH_WINDOWS_IMAGE_PREFIX": r"Z:\builds",
    }):
        result = _translate_image_path("/home/user/builds/charlotte/boot.img")
        assert result == r"Z:\builds\charlotte\boot.img"


def test_translate_no_match():
    with patch.dict(os.environ, {
        "HMOPT_FLASH_SERVER_IMAGE_PREFIX": "/home/user/builds",
        "HMOPT_FLASH_WINDOWS_IMAGE_PREFIX": r"Z:\builds",
    }):
        result = _translate_image_path("/other/path/boot.img")
        assert result == "/other/path/boot.img"


def test_translate_partial_prefix_no_match():
    """Prefix /home/user/builds must not match /home/user/builds-extra."""
    with patch.dict(os.environ, {
        "HMOPT_FLASH_SERVER_IMAGE_PREFIX": "/home/user/builds",
        "HMOPT_FLASH_WINDOWS_IMAGE_PREFIX": r"Z:\builds",
    }):
        result = _translate_image_path("/home/user/builds-extra/boot.img")
        assert result == "/home/user/builds-extra/boot.img"


# ---------------------------------------------------------------------------
# Integration tests with fake relay
# ---------------------------------------------------------------------------

def test_relay_health_check(fake_relay):
    result = relay_health_check()
    assert result["relay_reachable"] is True
    assert result["relay_status"]["status"] == "ok"


def test_list_fastboot_devices(fake_relay):
    result = list_fastboot_devices()
    assert len(result["devices"]) == 1
    assert result["devices"][0]["serial"] == "ABC123"


def test_flash_device(fake_relay):
    result = flash_device(partition="boot", image_path="/tmp/boot.img", device_serial="ABC123")
    assert result["success"] is True
    assert result["partition"] == "boot"


def test_reboot_device(fake_relay):
    result = reboot_device(device_serial="ABC123")
    assert result["success"] is True


def test_wait_for_device_boot(fake_relay):
    result = wait_for_device_boot(device_serial="ABC123", timeout_s=5, poll_interval_s=0.1)
    assert result["success"] is True


def test_flash_and_boot(fake_relay):
    result = flash_and_boot(
        partition="boot",
        image_path="/tmp/boot.img",
        device_serial="ABC123",
        flash_timeout_s=10,
        boot_wait_timeout_s=5,
        poll_interval_s=0.1,
    )
    assert result["success"] is True
    assert result["phase"] == "complete"
    assert "flash" in result["steps"]
    assert "reboot" in result["steps"]
    assert "boot_wait" in result["steps"]


def test_flash_and_boot_async(fake_relay):
    task_info = flash_and_boot_async(
        partition="boot",
        image_path="/tmp/boot.img",
        device_serial="ABC123",
        flash_timeout_s=10,
        boot_wait_timeout_s=5,
        poll_interval_s=0.1,
    )
    assert "task_id" in task_info
    assert task_info["status"] == "pending"

    # Poll until done
    import time
    for _ in range(50):
        status = get_flash_task_status(task_info["task_id"])
        if status["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.1)

    assert status["status"] == "succeeded"
    assert status["result"]["success"] is True


def test_relay_unreachable():
    with patch.dict(os.environ, {"HMOPT_FLASH_RELAY_URL": "http://127.0.0.1:1", "HMOPT_FLASH_RELAY_SECRET": ""}):
        result = relay_health_check()
        assert result["relay_reachable"] is False


def test_build_flash_fastmcp_server():
    # Verify server construction does not raise; may be None if mcp package not installed
    server = build_flash_fastmcp_server()
    if server is not None:
        assert hasattr(server, "settings")
