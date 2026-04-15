"""Minimal command relay HTTP service for Windows.

Receives fastboot/hdc/adb commands from a remote server,
executes them locally, and returns stdout/stderr/returncode.

Zero external dependencies — uses only Python stdlib.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import platform
import socket
import subprocess
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

ALLOWED_COMMANDS: set[str] = {
    "fastboot", "fastboot.exe",
    "hdc", "hdc.exe",
    "adb", "adb.exe",
    "ping", "ping.exe",
    # SCP tools for pulling images from the build server
    "pscp", "pscp.exe",
    "scp", "scp.exe",
    # Python for running integrated pipeline scripts
    "python", "python.exe",
    "python3", "python3.exe",
}

DEFAULT_PORT = 9100
DEFAULT_TIMEOUT_S = 120
MAX_TIMEOUT_S = 3600


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    length = int(handler.headers.get("Content-Length", 0))
    return handler.rfile.read(length) if length > 0 else b""


def _json_response(handler: BaseHTTPRequestHandler, code: int, data: Any) -> None:
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _check_secret(handler: BaseHTTPRequestHandler) -> bool:
    expected = os.environ.get("RELAY_SECRET", "").strip()
    if not expected:
        return True
    provided = (handler.headers.get("X-Relay-Secret") or "").strip()
    return hmac.compare_digest(provided, expected)


def _exec_command(command: str, args: list[str], timeout_s: int, cwd: str | None) -> dict[str, Any]:
    argv = [command] + args
    started = time.time()
    try:
        # NOTE: encoding="utf-8", errors="replace" is load-bearing.  Without
        # it Python uses locale.getpreferredencoding() to decode the child's
        # stdout/stderr pipes — on a typical Windows host that's cp1252, and
        # any non-ASCII byte (UTF-8 continuation bytes like 0x90 from our
        # pipeline, device output from fastboot/hdc, etc.) kills the reader
        # thread with UnicodeDecodeError and silently drops the payload.
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=min(timeout_s, MAX_TIMEOUT_S),
            cwd=cwd or None,
            check=False,
        )
        duration = round(time.time() - started, 3)
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_s": duration,
            "command": " ".join(argv),
        }
    except FileNotFoundError:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"command not found: {command}",
            "duration_s": round(time.time() - started, 3),
            "command": " ".join(argv),
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -9,
            "stdout": "",
            "stderr": f"timeout after {timeout_s}s",
            "duration_s": round(time.time() - started, 3),
            "command": " ".join(argv),
        }


class RelayHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the command relay."""

    server_version = "HMOpt-DeviceRelay/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {self.address_string()} - {format % args}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            _json_response(self, 200, {
                "status": "ok",
                "hostname": socket.gethostname(),
                "platform": platform.system(),
                "allowed_commands": sorted(ALLOWED_COMMANDS),
            })
        elif self.path == "/devices":
            if not _check_secret(self):
                _json_response(self, 403, {"error": "invalid or missing X-Relay-Secret"})
                return
            result = _exec_command("fastboot", ["devices", "-l"], timeout_s=10, cwd=None)
            devices = []
            for line in result.get("stdout", "").strip().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    devices.append({"serial": parts[0], "state": parts[1]})
            _json_response(self, 200, {"devices": devices, "raw": result})
        else:
            _json_response(self, 404, {"error": f"unknown path: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/exec":
            _json_response(self, 404, {"error": f"unknown path: {self.path}"})
            return

        if not _check_secret(self):
            _json_response(self, 403, {"error": "invalid or missing X-Relay-Secret"})
            return

        try:
            body = json.loads(_read_body(self))
        except (json.JSONDecodeError, ValueError) as exc:
            _json_response(self, 400, {"error": f"invalid JSON: {exc}"})
            return

        command = str(body.get("command", "")).strip()
        if not command:
            _json_response(self, 400, {"error": "command is required"})
            return

        cmd_basename = os.path.basename(command).lower()
        if cmd_basename not in ALLOWED_COMMANDS:
            _json_response(self, 403, {
                "error": f"command not allowed: {command}",
                "allowed": sorted(ALLOWED_COMMANDS),
            })
            return

        args = [str(a) for a in body.get("args", [])]
        timeout_s = int(body.get("timeout_s", DEFAULT_TIMEOUT_S))
        cwd = body.get("cwd")

        result = _exec_command(command, args, timeout_s, cwd)
        _json_response(self, 200, result)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="HMOpt Device Command Relay Service")
    parser.add_argument("--port", type=int, default=int(os.environ.get("RELAY_PORT", DEFAULT_PORT)))
    parser.add_argument("--bind", default=os.environ.get("RELAY_BIND", "0.0.0.0"))
    args = parser.parse_args(argv)

    server = HTTPServer((args.bind, args.port), RelayHandler)
    secret_status = "enabled" if os.environ.get("RELAY_SECRET", "").strip() else "disabled"
    print(f"HMOpt Device Relay listening on {args.bind}:{args.port}")
    print(f"Secret auth: {secret_status}")
    print(f"Allowed commands: {sorted(ALLOWED_COMMANDS)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down relay service.")
        server.server_close()


if __name__ == "__main__":
    main()
