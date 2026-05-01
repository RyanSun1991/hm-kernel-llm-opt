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
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
    # Stamp request_id into the response body so callers can correlate with
    # server-side logs, and into a response header so curl --include surfaces
    # it without reading the body.  Falls back gracefully if the handler did
    # not assign one (e.g. legacy callers / tests).
    rid = getattr(handler, "_rid", None)
    if isinstance(data, dict) and rid and "request_id" not in data:
        data = {**data, "request_id": rid}
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    # Force per-request connection close (no keep-alive).  Eliminates the
    # half-open keep-alive failure mode (client vanishes mid-idle, server's
    # readline blocks on a dead socket until the OS-level TCP keepalive
    # eventually reaps it — 2 hours by default on Windows).  For an admin
    # relay the extra TCP setup per request is irrelevant; the simplicity
    # of "every request is a fresh connection" is worth more.
    handler.send_header("Connection", "close")
    if rid:
        handler.send_header("X-Request-ID", rid)
    handler.end_headers()
    handler.wfile.write(body)
    # Tell the keep-alive loop in BaseHTTPRequestHandler.handle() to terminate
    # after this response.  send_header alone is advisory to the client; this
    # flag is what actually breaks the server-side `while not close_connection`
    # loop and frees the worker thread.
    handler.close_connection = True


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
        #
        # CREATE_NO_WINDOW (Windows-only flag, 0x08000000) tells CreateProcess
        # not to allocate a new console for the child.  Without it, every
        # subprocess.run for fastboot / hdc / scp briefly flashes a console
        # window if the relay is run in any context other than a normal
        # foreground cmd.exe (services, scheduled tasks, pythonw, IDE) — and
        # those flashes can cause focus thrash that makes operators
        # accidentally click the relay's own console (re-triggering the
        # QuickEdit Mode freeze).  getattr() makes this portable to POSIX,
        # where the flag does not exist and creationflags=0 is the default.
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=min(timeout_s, MAX_TIMEOUT_S),
            cwd=cwd or None,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
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

    # Per-connection idle timeout.  StreamRequestHandler.setup() reads this
    # attribute and calls socket.settimeout(), so any read (including the
    # keep-alive loop's readline that waits for the next pipelined request)
    # raises socket.timeout / TimeoutError after this many seconds of
    # inactivity.  BaseHTTPRequestHandler.handle_one_request already catches
    # TimeoutError and closes the connection cleanly, so this is a safe
    # belt-and-suspenders against half-open keep-alive sockets that the
    # OS-level TCP keepalive (default 2 hours on Windows) is too slow to
    # reap.  Symptom this prevents: after 1-2 hours of operation the relay
    # accumulates idle keep-alive threads stuck in readline() forever, and
    # under the legacy single-threaded HTTPServer the entire serve_forever
    # loop wedges with no log of new requests reaching the handler.
    #
    # Pick a value comfortably larger than any realistic client think-time
    # between requests on a single keep-alive connection (urllib3 / requests
    # default pool idle timeout is ~5s; httpx is ~5s).  60s leaves slack.
    timeout = 60

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # Include the per-request id so a single grep against the log
        # reconstructs everything that happened to one request — entry,
        # subprocess outcome, response, any disconnect.  '--------' (8 dashes)
        # for connection-level events that fire before do_GET / do_POST
        # assigns an id (e.g. keep-alive RST notification from handle()).
        rid = getattr(self, "_rid", "--------")
        print(f"[{timestamp}] [{rid}] {self.address_string()} - {format % args}")

    def _new_request_id(self) -> str:
        rid = uuid.uuid4().hex[:8]
        self._rid = rid
        return rid

    def handle(self) -> None:
        # BaseHTTPRequestHandler runs a keep-alive loop here.  When a remote
        # client (urllib3 connection pool, requests Session, etc.) closes a
        # keep-alive socket between requests — common after pool eviction or
        # process exit — Windows reports it as a TCP RST, which surfaces as
        # ConnectionResetError (WinError 10054) on the next readline().  The
        # stdlib only swallows TimeoutError; everything else escapes and the
        # socketserver framework prints a full stack trace as if it were a
        # fatal handler error.  These disconnects are benign — every request
        # that already completed is unaffected — so we catch the family and
        # close the connection cleanly with a single-line log entry.
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
            self.log_message("client disconnected (%s); closing connection",
                             exc.__class__.__name__)
            self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        self._new_request_id()
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
        self._new_request_id()
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

    # report_compare.py needs openpyxl.  Install it on first startup so
    # operators don't have to remember.
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("[deps] openpyxl not found, installing ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "openpyxl"], check=False)

    # ThreadingHTTPServer (stdlib, daemon_threads = True by default) gives us
    # one thread per request.  The default HTTPServer is single-threaded and
    # blocks every other request — including /health — behind whatever long-
    # running fastboot/hdc command happens to be in flight.  The handler is
    # thread-safe: ALLOWED_COMMANDS is read-only, environment reads are
    # atomic, and each subprocess.run spawns its own child process with no
    # shared state.  Note: device-level mutual exclusion (e.g. don't flash
    # while a test is running on the same serial) belongs to the upstream
    # scheduler — the relay must not pretend to enforce it via single-
    # threaded transport, since that also blocks unrelated commands.
    server = ThreadingHTTPServer((args.bind, args.port), RelayHandler)
    secret_status = "enabled" if os.environ.get("RELAY_SECRET", "").strip() else "disabled"
    print(f"HMOpt Device Relay listening on {args.bind}:{args.port}")
    print(f"Secret auth: {secret_status}")
    print("Concurrency: ThreadingHTTPServer (one thread per request)")
    print(f"Per-connection idle timeout: {RelayHandler.timeout}s")
    print(f"Allowed commands: {sorted(ALLOWED_COMMANDS)}")
    if platform.system() == "Windows":
        # Operator footgun warning: clicking the console window puts cmd.exe
        # into Mark / Selection mode, which freezes this process's stdout —
        # subsequent print()s block indefinitely and the relay appears
        # unresponsive even though the listening socket is fine.  Recommend
        # running through relay_service.bat (which redirects stdout to a log
        # file) or disabling QuickEdit Mode in the console properties.
        print(
            "[warn] Windows console QuickEdit Mode can freeze this process if "
            "you click in the console window.  If the relay stops responding, "
            "press Esc or Enter in the console — buffered output will flush.  "
            "For unattended operation, prefer relay_service.bat (logs to file)."
        )
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down relay service.")
        server.server_close()


if __name__ == "__main__":
    main()
