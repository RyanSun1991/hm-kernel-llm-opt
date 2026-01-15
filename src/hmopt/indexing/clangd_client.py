"""Minimal clangd LSP client for symbol extraction."""

from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClangdConfig:
    binary: str
    compile_commands_dir: Optional[Path]
    extra_args: list[str]
    timeout_sec: int
    symbol_kinds: list[str] = field(default_factory=list)
    call_hierarchy_enabled: bool = True
    call_hierarchy_max_functions: int = 2000
    call_hierarchy_max_calls: int = 100
    call_hierarchy_max_depth: int = 1
    usage_scan_enabled: bool = True
    usage_scan_max_names: int = 2000
    relation_max_per_symbol: int = 200
    file_summary_enabled: bool = True
    relation_summary_enabled: bool = True
    relation_summary_max_items: int = 50


class ClangdClient:
    def __init__(self, config: ClangdConfig, root: Path) -> None:
        self.config = config
        self.root = root
        self._opened: set[str] = set()
        cmd = [config.binary]
        if config.compile_commands_dir:
            cmd.append(f"--compile-commands-dir={config.compile_commands_dir}")
        cmd.extend(config.extra_args or [])
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._id = 0
        self._initialize()

    def _initialize(self) -> None:
        params = {
            "processId": os.getpid(),
            "rootUri": self.root.as_uri(),
            "capabilities": {},
        }
        _ = self.request("initialize", params)
        self.notify("initialized", {})

    def close(self) -> None:
        try:
            self.request("shutdown", {})
            self.notify("exit", {})
        except Exception:
            pass
        if self.proc:
            self.proc.terminate()

    def _ensure_open(self, path: Path) -> str:
        uri = path.as_uri()
        if uri in self._opened:
            return uri
        text = path.read_text(encoding="utf-8", errors="ignore")
        self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "cpp",
                    "version": 1,
                    "text": text,
                }
            },
        )
        self._opened.add(uri)
        return uri

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._id += 1
        req_id = self._id
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        return self._wait_for_response(req_id)

    def _send(self, payload: dict[str, Any]) -> None:
        if not self.proc.stdin:
            raise RuntimeError("clangd stdin not available")
        data = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
        self.proc.stdin.write(header + data)
        self.proc.stdin.flush()

    def _wait_for_response(self, req_id: int) -> dict[str, Any]:
        end = time.monotonic() + self.config.timeout_sec
        while time.monotonic() < end:
            msg = self._read_message(timeout=0.5)
            if not msg:
                continue
            if msg.get("id") == req_id:
                return msg
        raise TimeoutError(f"clangd response timeout for id={req_id}")

    def _read_message(self, timeout: float) -> Optional[dict[str, Any]]:
        if not self.proc.stdout:
            return None
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            return None
        headers = {}
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                break
            parts = line.decode("ascii").split(":", 1)
            if len(parts) == 2:
                headers[parts[0].strip().lower()] = parts[1].strip()
        length = int(headers.get("content-length", "0"))
        if length <= 0:
            return None
        body = self.proc.stdout.read(length)
        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def document_symbols(self, path: Path) -> list[dict[str, Any]]:
        uri = self._ensure_open(path)
        resp = self.request("textDocument/documentSymbol", {"textDocument": {"uri": uri}})
        return resp.get("result", []) or []

    def prepare_call_hierarchy(self, path: Path, line: int, character: int) -> list[dict[str, Any]]:
        uri = self._ensure_open(path)
        resp = self.request(
            "textDocument/prepareCallHierarchy",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
            },
        )
        return resp.get("result", []) or []

    def outgoing_calls(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        resp = self.request("callHierarchy/outgoingCalls", {"item": item})
        return resp.get("result", []) or []

    def incoming_calls(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        resp = self.request("callHierarchy/incomingCalls", {"item": item})
        return resp.get("result", []) or []
