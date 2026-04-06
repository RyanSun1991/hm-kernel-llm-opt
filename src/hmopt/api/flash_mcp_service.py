"""FastMCP service wrapper for flashing device images via a Windows relay."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from copy import deepcopy
from functools import lru_cache
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)
DEFAULT_SERVER_NAME = "hmopt-flash-mcp"

_FLASH_TASKS: dict[str, dict[str, Any]] = {}
_FLASH_TASKS_LOCK = threading.Lock()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _relay_url() -> str:
    url = _env("HMOPT_FLASH_RELAY_URL")
    if not url:
        raise ValueError("HMOPT_FLASH_RELAY_URL is required (e.g. http://10.x.x.x:9100)")
    return url.rstrip("/")


def _relay_secret() -> str:
    return _env("HMOPT_FLASH_RELAY_SECRET")


def _relay_exec(
    command: str,
    args: list[str],
    *,
    timeout_s: int = 120,
    cwd: str | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    """Send a command to the Windows relay service via HTTP POST /exec."""
    url = f"{_relay_url()}/exec"
    payload = json.dumps({
        "command": command,
        "args": args,
        "timeout_s": timeout_s,
        "cwd": cwd,
    }).encode("utf-8")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    secret = _relay_secret()
    if secret:
        headers["X-Relay-Secret"] = secret

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, data=payload, headers=headers, method="POST")
            # Allow extra time for the HTTP round-trip on top of command timeout
            http_timeout = timeout_s + 30
            with urlopen(req, timeout=http_timeout) as resp:
                body = resp.read()
                return json.loads(body)
        except (URLError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning("relay request failed (attempt %d/%d), retrying in %ds: %s", attempt + 1, retries, wait, exc)
                time.sleep(wait)

    raise ConnectionError(f"relay unreachable after {retries} attempts: {last_error}") from last_error


def _translate_image_path(server_path: str) -> str:
    """Convert a server-side image path to the Windows-side equivalent."""
    server_prefix = _env("HMOPT_FLASH_SERVER_IMAGE_PREFIX")
    windows_prefix = _env("HMOPT_FLASH_WINDOWS_IMAGE_PREFIX")
    if not server_prefix or not windows_prefix:
        return server_path
    # Ensure prefix ends with / to avoid partial directory matches
    normalized_prefix = server_prefix.rstrip("/") + "/"
    if server_path.startswith(normalized_prefix):
        relative = server_path[len(normalized_prefix):]
        return windows_prefix.rstrip("\\") + "\\" + relative.replace("/", "\\")
    # Exact match (path == prefix without trailing slash)
    if server_path == server_prefix.rstrip("/"):
        return windows_prefix.rstrip("\\")
    return server_path


def _fastboot_serial_args(device_serial: str | None) -> list[str]:
    serial = (device_serial or _env("HMOPT_FLASH_DEVICE_SERIAL")).strip()
    if serial:
        return ["-s", serial]
    return []


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def relay_health_check() -> dict[str, Any]:
    """Check connectivity to the Windows relay service."""
    url = f"{_relay_url()}/health"
    headers: dict[str, str] = {}
    secret = _relay_secret()
    if secret:
        headers["X-Relay-Secret"] = secret
    try:
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=10) as resp:
            return {"relay_reachable": True, "relay_status": json.loads(resp.read())}
    except (URLError, OSError, json.JSONDecodeError) as exc:
        return {"relay_reachable": False, "error": str(exc)}


def list_fastboot_devices() -> dict[str, Any]:
    """List devices visible to fastboot on the Windows relay host."""
    result = _relay_exec("fastboot", ["devices", "-l"], timeout_s=10)
    devices = []
    for line in result.get("stdout", "").strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            devices.append({"serial": parts[0], "state": parts[1]})
    return {"devices": devices, "raw": result}


def flash_device(
    *,
    partition: str = "boot",
    image_path: str,
    device_serial: str | None = None,
    timeout_s: int = 600,
) -> dict[str, Any]:
    """Flash a single partition on the target device via fastboot."""
    win_path = _translate_image_path(image_path)
    serial_args = _fastboot_serial_args(device_serial)
    args = serial_args + ["flash", partition, win_path]
    result = _relay_exec("fastboot", args, timeout_s=timeout_s)
    return {
        "success": result.get("returncode") == 0,
        "partition": partition,
        "image_path": image_path,
        "windows_path": win_path,
        "device_serial": device_serial,
        "relay_result": result,
    }


def reboot_device(
    *,
    device_serial: str | None = None,
    timeout_s: int = 60,
) -> dict[str, Any]:
    """Reboot the device via fastboot."""
    serial_args = _fastboot_serial_args(device_serial)
    args = serial_args + ["reboot"]
    result = _relay_exec("fastboot", args, timeout_s=timeout_s)
    return {
        "success": result.get("returncode") == 0,
        "device_serial": device_serial,
        "relay_result": result,
    }


def wait_for_device_boot(
    *,
    device_serial: str | None = None,
    timeout_s: int = 300,
    poll_interval_s: int = 5,
) -> dict[str, Any]:
    """Poll hdc list targets until the device appears or timeout."""
    started = time.time()
    serial = (device_serial or _env("HMOPT_FLASH_DEVICE_SERIAL")).strip()

    while True:
        elapsed = time.time() - started
        if elapsed >= timeout_s:
            return {
                "success": False,
                "error": f"device did not appear within {timeout_s}s",
                "elapsed_s": round(elapsed, 1),
                "device_serial": serial,
            }

        try:
            result = _relay_exec("hdc", ["list", "targets"], timeout_s=10, retries=1)
            stdout = result.get("stdout", "")
            if serial:
                if serial in stdout:
                    return {
                        "success": True,
                        "elapsed_s": round(time.time() - started, 1),
                        "device_serial": serial,
                        "hdc_output": stdout.strip(),
                    }
            else:
                # Accept any device
                lines = [l.strip() for l in stdout.strip().splitlines() if l.strip() and l.strip() != "[Empty]"]
                if lines:
                    return {
                        "success": True,
                        "elapsed_s": round(time.time() - started, 1),
                        "device_serial": lines[0],
                        "hdc_output": stdout.strip(),
                    }
        except ConnectionError:
            pass  # relay temporarily unreachable, keep polling

        time.sleep(poll_interval_s)


def flash_and_boot(
    *,
    partition: str = "boot",
    image_path: str,
    device_serial: str | None = None,
    flash_timeout_s: int = 600,
    boot_wait_timeout_s: int = 300,
    poll_interval_s: int = 5,
) -> dict[str, Any]:
    """Flash a partition, reboot the device, and wait for it to come online."""
    steps: dict[str, Any] = {}

    # Step 1: Flash
    flash_result = flash_device(
        partition=partition,
        image_path=image_path,
        device_serial=device_serial,
        timeout_s=flash_timeout_s,
    )
    steps["flash"] = flash_result
    if not flash_result.get("success"):
        return {
            "success": False,
            "phase": "flash",
            "partition": partition,
            "image_path": image_path,
            "steps": steps,
        }

    # Step 2: Reboot
    reboot_result = reboot_device(device_serial=device_serial)
    steps["reboot"] = reboot_result
    if not reboot_result.get("success"):
        return {
            "success": False,
            "phase": "reboot",
            "partition": partition,
            "image_path": image_path,
            "steps": steps,
        }

    # Step 3: Wait for boot
    boot_result = wait_for_device_boot(
        device_serial=device_serial,
        timeout_s=boot_wait_timeout_s,
        poll_interval_s=poll_interval_s,
    )
    steps["boot_wait"] = boot_result

    return {
        "success": boot_result.get("success", False),
        "phase": "complete" if boot_result.get("success") else "boot_wait",
        "partition": partition,
        "image_path": image_path,
        "device_serial": boot_result.get("device_serial", device_serial),
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Async task management (same pattern as build_mcp_service.py)
# ---------------------------------------------------------------------------

def _register_async_task(kind: str, payload: dict[str, Any]) -> str:
    task_id = str(uuid.uuid4())
    now = time.time()
    with _FLASH_TASKS_LOCK:
        _FLASH_TASKS[task_id] = {
            "task_id": task_id,
            "kind": kind,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "payload": payload,
            "result": None,
            "error": None,
        }
    return task_id


def _set_task_running(task_id: str) -> None:
    with _FLASH_TASKS_LOCK:
        task = _FLASH_TASKS.get(task_id)
        if task is None:
            return
        task["status"] = "running"
        task["updated_at"] = time.time()


def _set_task_result(task_id: str, *, result: dict[str, Any] | None = None, error: str | None = None) -> None:
    with _FLASH_TASKS_LOCK:
        task = _FLASH_TASKS.get(task_id)
        if task is None:
            return
        task["status"] = "failed" if error else "succeeded"
        task["result"] = result
        task["error"] = error
        task["updated_at"] = time.time()


def _run_async_task(task_id: str, runner: Callable[[], dict[str, Any]]) -> None:
    _set_task_running(task_id)
    try:
        result = runner()
        _set_task_result(task_id, result=result)
    except Exception as exc:
        logger.exception("flash mcp async task failed: task_id=%s", task_id)
        _set_task_result(task_id, error=str(exc))


def _submit_async_task(kind: str, payload: dict[str, Any], runner: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    task_id = _register_async_task(kind, payload)
    thread = threading.Thread(target=_run_async_task, args=(task_id, runner), daemon=True)
    thread.start()
    return {"task_id": task_id, "status": "pending", "kind": kind}


def flash_and_boot_async(
    *,
    partition: str = "boot",
    image_path: str,
    device_serial: str | None = None,
    flash_timeout_s: int = 600,
    boot_wait_timeout_s: int = 300,
    poll_interval_s: int = 5,
) -> dict[str, Any]:
    """Submit a flash-and-boot operation as an async task."""
    payload = {
        "partition": partition,
        "image_path": image_path,
        "device_serial": device_serial,
        "flash_timeout_s": flash_timeout_s,
        "boot_wait_timeout_s": boot_wait_timeout_s,
        "poll_interval_s": poll_interval_s,
    }
    return _submit_async_task(
        "flash_and_boot",
        payload,
        lambda: flash_and_boot(
            partition=partition,
            image_path=image_path,
            device_serial=device_serial,
            flash_timeout_s=flash_timeout_s,
            boot_wait_timeout_s=boot_wait_timeout_s,
            poll_interval_s=poll_interval_s,
        ),
    )


def get_flash_task_status(task_id: str) -> dict[str, Any]:
    """Query the status of an async flash task."""
    with _FLASH_TASKS_LOCK:
        task = _FLASH_TASKS.get(task_id)
        if task is None:
            raise ValueError(f"task not found: {task_id}")
        snapshot = deepcopy(task)
    snapshot["duration_s"] = round(snapshot["updated_at"] - snapshot["created_at"], 3)
    return snapshot


# ---------------------------------------------------------------------------
# FastMCP server builder
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def build_flash_fastmcp_server() -> Any | None:
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError:
        logger.warning(
            "mcp package is not installed; flash MCP protocol endpoint is unavailable. "
            "Install with: pip install 'mcp[cli]'"
        )
        return None

    server_name = _env("HMOPT_FLASH_MCP_SERVER_NAME", DEFAULT_SERVER_NAME) or DEFAULT_SERVER_NAME
    mcp = FastMCP(server_name, stateless_http=True, json_response=True)

    @mcp.tool(
        name="flash_device",
        description="Flash a single partition on the target device via fastboot through the Windows relay.",
    )
    def mcp_flash_device(
        partition: str = "boot",
        image_path: str = "",
        device_serial: str | None = None,
        timeout_s: int = 600,
    ) -> dict[str, Any]:
        return flash_device(
            partition=partition,
            image_path=image_path,
            device_serial=device_serial,
            timeout_s=timeout_s,
        )

    @mcp.tool(
        name="flash_and_boot",
        description="Flash partition, reboot device, and wait for boot completion. Returns combined result.",
    )
    def mcp_flash_and_boot(
        partition: str = "boot",
        image_path: str = "",
        device_serial: str | None = None,
        flash_timeout_s: int = 600,
        boot_wait_timeout_s: int = 300,
        poll_interval_s: int = 5,
    ) -> dict[str, Any]:
        return flash_and_boot(
            partition=partition,
            image_path=image_path,
            device_serial=device_serial,
            flash_timeout_s=flash_timeout_s,
            boot_wait_timeout_s=boot_wait_timeout_s,
            poll_interval_s=poll_interval_s,
        )

    @mcp.tool(
        name="flash_and_boot_async",
        description="Submit flash-and-boot as async task, returns task_id immediately.",
    )
    def mcp_flash_and_boot_async(
        partition: str = "boot",
        image_path: str = "",
        device_serial: str | None = None,
        flash_timeout_s: int = 600,
        boot_wait_timeout_s: int = 300,
        poll_interval_s: int = 5,
    ) -> dict[str, Any]:
        return flash_and_boot_async(
            partition=partition,
            image_path=image_path,
            device_serial=device_serial,
            flash_timeout_s=flash_timeout_s,
            boot_wait_timeout_s=boot_wait_timeout_s,
            poll_interval_s=poll_interval_s,
        )

    @mcp.tool(
        name="flash_status",
        description="Query async flash task status by task_id.",
    )
    def mcp_flash_status(task_id: str) -> dict[str, Any]:
        return get_flash_task_status(task_id=task_id)

    @mcp.tool(
        name="device_reboot",
        description="Reboot device via fastboot through the Windows relay.",
    )
    def mcp_device_reboot(device_serial: str | None = None, timeout_s: int = 60) -> dict[str, Any]:
        return reboot_device(device_serial=device_serial, timeout_s=timeout_s)

    @mcp.tool(
        name="device_wait_boot",
        description="Poll until device appears in hdc list targets after reboot.",
    )
    def mcp_device_wait_boot(
        device_serial: str | None = None,
        timeout_s: int = 300,
        poll_interval_s: int = 5,
    ) -> dict[str, Any]:
        return wait_for_device_boot(
            device_serial=device_serial,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )

    @mcp.tool(
        name="relay_health",
        description="Check Windows relay service connectivity and status.",
    )
    def mcp_relay_health() -> dict[str, Any]:
        return relay_health_check()

    @mcp.tool(
        name="list_devices",
        description="List devices visible to fastboot on the Windows relay host.",
    )
    def mcp_list_devices() -> dict[str, Any]:
        return list_fastboot_devices()

    return mcp
