"""FastMCP service wrapper for flashing device images via a Windows relay.

Real flash workflow (matching production scripts):
  1. pscp images from build server → Windows PC
  2. hdc shell reboot bootloader  (enter fastboot mode)
  3. wait for device in fastboot devices
  4. fastboot flash boot / modem_driver / …  (multi-partition)
  5. fastboot reboot
  6. wait for device in hdc list targets  (boot complete)
"""

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
    normalized_prefix = server_prefix.rstrip("/") + "/"
    if server_path.startswith(normalized_prefix):
        relative = server_path[len(normalized_prefix):]
        return windows_prefix.rstrip("\\") + "\\" + relative.replace("/", "\\")
    if server_path == server_prefix.rstrip("/"):
        return windows_prefix.rstrip("\\")
    return server_path


def _hdc_target_args(device_serial: str | None) -> list[str]:
    serial = (device_serial or _env("HMOPT_FLASH_DEVICE_SERIAL")).strip()
    if serial:
        return ["-t", serial]
    return []


def _fastboot_serial_args(device_serial: str | None) -> list[str]:
    serial = (device_serial or _env("HMOPT_FLASH_DEVICE_SERIAL")).strip()
    if serial:
        return ["-s", serial]
    return []


# ---------------------------------------------------------------------------
# SCP config helpers
# ---------------------------------------------------------------------------

def _scp_tool() -> str:
    return _env("HMOPT_FLASH_SCP_TOOL", "pscp")


def _scp_host() -> str:
    return _env("HMOPT_FLASH_SCP_HOST")


def _scp_user() -> str:
    return _env("HMOPT_FLASH_SCP_USER")


def _scp_password() -> str:
    return _env("HMOPT_FLASH_SCP_PASSWORD")


def _windows_image_dir() -> str:
    return _env("HMOPT_FLASH_WINDOWS_IMAGE_DIR", ".")


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


def list_hdc_targets() -> dict[str, Any]:
    """List devices visible to hdc on the Windows relay host."""
    result = _relay_exec("hdc", ["list", "targets"], timeout_s=10)
    stdout = result.get("stdout", "").strip()
    targets = [
        line.strip()
        for line in stdout.splitlines()
        if line.strip() and line.strip() != "[Empty]"
    ]
    return {"targets": targets, "raw": result}


def transfer_images(
    *,
    images: list[dict[str, str]],
    scp_host: str | None = None,
    scp_user: str | None = None,
    scp_password: str | None = None,
    windows_image_dir: str | None = None,
    timeout_s: int = 300,
) -> dict[str, Any]:
    """Pull image files from the build server to the Windows PC via pscp/scp.

    Each entry in *images* is {"server_path": "...", "local_filename": "..."}.
    Example:
        images=[
            {"server_path": "/home/damon/ws/hm/signing/nsv_user/boot.img", "local_filename": "boot_plr.img"},
            {"server_path": "/home/damon/ws/hm/signing/nsv_user/modem_driver.img", "local_filename": "modem_driver_plr.img"},
        ]
    """
    tool = _scp_tool()
    host = scp_host or _scp_host()
    user = scp_user or _scp_user()
    password = scp_password or _scp_password()
    dest_dir = windows_image_dir or _windows_image_dir()

    if not host:
        raise ValueError("SCP host is required (set scp_host argument or HMOPT_FLASH_SCP_HOST)")
    if not user:
        raise ValueError("SCP user is required (set scp_user argument or HMOPT_FLASH_SCP_USER)")

    results: list[dict[str, Any]] = []
    all_ok = True

    for entry in images:
        server_path = entry["server_path"]
        local_filename = entry["local_filename"]
        remote_spec = f"{user}@{host}:{server_path}"
        local_path = f"{dest_dir}\\{local_filename}" if dest_dir != "." else local_filename

        # Build pscp command: pscp -pw <password> user@host:/path local_file
        args: list[str] = []
        if password and tool in ("pscp", "pscp.exe"):
            args.extend(["-pw", password])
        args.extend([remote_spec, local_path])

        result = _relay_exec(tool, args, timeout_s=timeout_s, cwd=dest_dir)
        success = result.get("returncode") == 0
        if not success:
            all_ok = False
        results.append({
            "server_path": server_path,
            "local_filename": local_filename,
            "local_path": local_path,
            "success": success,
            "relay_result": result,
        })

    return {"success": all_ok, "transfers": results}


def enter_bootloader(
    *,
    device_serial: str | None = None,
    timeout_s: int = 30,
) -> dict[str, Any]:
    """Reboot the device into fastboot/bootloader mode via hdc."""
    target_args = _hdc_target_args(device_serial)
    args = target_args + ["shell", "reboot", "bootloader"]
    result = _relay_exec("hdc", args, timeout_s=timeout_s)
    return {
        "success": result.get("returncode") == 0,
        "device_serial": device_serial,
        "relay_result": result,
    }


def wait_for_fastboot_device(
    *,
    device_serial: str | None = None,
    timeout_s: int = 30,
    poll_interval_s: int = 2,
) -> dict[str, Any]:
    """Poll fastboot devices until the target device appears in bootloader."""
    started = time.time()
    serial = (device_serial or _env("HMOPT_FLASH_DEVICE_SERIAL")).strip()

    while True:
        elapsed = time.time() - started
        if elapsed >= timeout_s:
            return {
                "success": False,
                "error": f"device did not appear in fastboot within {timeout_s}s",
                "elapsed_s": round(elapsed, 1),
                "device_serial": serial,
            }

        try:
            result = _relay_exec("fastboot", ["devices"], timeout_s=10, retries=1)
            stdout = result.get("stdout", "")
            if serial:
                if serial in stdout:
                    return {
                        "success": True,
                        "elapsed_s": round(time.time() - started, 1),
                        "device_serial": serial,
                    }
            else:
                lines = [l.strip() for l in stdout.strip().splitlines() if l.strip()]
                if lines:
                    found_serial = lines[0].split()[0] if lines[0].split() else ""
                    return {
                        "success": True,
                        "elapsed_s": round(time.time() - started, 1),
                        "device_serial": found_serial,
                    }
        except ConnectionError:
            pass

        time.sleep(poll_interval_s)


def flash_device(
    *,
    partition: str = "boot",
    image_path: str,
    device_serial: str | None = None,
    timeout_s: int = 600,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Flash a single partition on the target device via fastboot."""
    serial_args = _fastboot_serial_args(device_serial)
    args = serial_args + ["flash", partition, image_path]
    win_cwd = cwd or _windows_image_dir() or None
    result = _relay_exec("fastboot", args, timeout_s=timeout_s, cwd=win_cwd)
    return {
        "success": result.get("returncode") == 0,
        "partition": partition,
        "image_path": image_path,
        "device_serial": device_serial,
        "relay_result": result,
    }


def flash_partitions(
    *,
    partitions: list[dict[str, str]],
    device_serial: str | None = None,
    timeout_s: int = 600,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Flash multiple partitions sequentially.

    Each entry: {"partition": "boot", "image_path": "boot_plr.img"}
    """
    results: list[dict[str, Any]] = []
    all_ok = True

    for entry in partitions:
        r = flash_device(
            partition=entry["partition"],
            image_path=entry["image_path"],
            device_serial=device_serial,
            timeout_s=timeout_s,
            cwd=cwd,
        )
        results.append(r)
        if not r.get("success"):
            all_ok = False
            break  # stop on first failure

    return {"success": all_ok, "partitions_flashed": results}


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
                lines = [l.strip() for l in stdout.strip().splitlines() if l.strip() and l.strip() != "[Empty]"]
                if lines:
                    return {
                        "success": True,
                        "elapsed_s": round(time.time() - started, 1),
                        "device_serial": lines[0],
                        "hdc_output": stdout.strip(),
                    }
        except ConnectionError:
            pass

        time.sleep(poll_interval_s)


def flash_and_boot(
    *,
    partitions: list[dict[str, str]],
    server_images: list[dict[str, str]] | None = None,
    device_serial: str | None = None,
    scp_host: str | None = None,
    scp_user: str | None = None,
    scp_password: str | None = None,
    windows_image_dir: str | None = None,
    pipeline_script: str | None = None,
    bootloader_wait_s: int = 10,
    fastboot_wait_s: int = 30,
    flash_timeout_s: int = 600,
    boot_wait_timeout_s: int = 300,
    poll_interval_s: int = 5,
) -> dict[str, Any]:
    """Full flash pipeline via a single integrated script call on the Windows relay.

    Instead of making multiple relay HTTP calls (one per step), this sends
    a SINGLE command to the relay that runs flash_pipeline.py on the Windows
    PC.  The script handles the entire sequence locally with maximum
    determinism:
      pscp → hdc reboot bootloader → wait fastboot → flash all → reboot → wait hdc

    Args:
        partitions: [{"partition": "boot", "image_path": "boot_plr.img"}, ...]
        server_images: [{"server_path": "...", "local_filename": "..."}, ...] (optional)
        device_serial: Target device serial.
        scp_host/scp_user/scp_password: SCP credentials (falls back to env vars).
        windows_image_dir: Local dir on Windows for images.
        pipeline_script: Path to flash_pipeline.py on Windows (falls back to env var).
        bootloader_wait_s: Seconds to wait after entering bootloader.
        fastboot_wait_s: Max seconds to wait for fastboot device.
        flash_timeout_s: Timeout per partition flash.
        boot_wait_timeout_s: Max seconds to wait for hdc after reboot.
        poll_interval_s: Poll interval for boot wait.
    """
    script = pipeline_script or _env("HMOPT_FLASH_PIPELINE_SCRIPT", "flash_pipeline.py")
    host = scp_host or _scp_host()
    user = scp_user or _scp_user()
    password = scp_password or _scp_password()
    img_dir = windows_image_dir or _windows_image_dir()
    serial = (device_serial or _env("HMOPT_FLASH_DEVICE_SERIAL")).strip()

    # Build --image specs: server_path:partition:local_filename
    image_args: list[str] = []
    for p in partitions:
        # Find matching server_image for this partition's local file
        server_path = ""
        if server_images:
            for si in server_images:
                if si.get("local_filename") == p.get("image_path"):
                    server_path = si.get("server_path", "")
                    break
        spec = f"{server_path}:{p['partition']}:{p['image_path']}"
        image_args.extend(["--image", spec])

    args: list[str] = [script] + image_args + [
        "--image-dir", img_dir,
        "--bootloader-wait", str(bootloader_wait_s),
        "--fastboot-wait", str(fastboot_wait_s),
        "--flash-timeout", str(flash_timeout_s),
        "--boot-wait-timeout", str(boot_wait_timeout_s),
        "--boot-poll-interval", str(poll_interval_s),
    ]

    if host:
        args.extend(["--scp-host", host])
    if user:
        args.extend(["--scp-user", user])
    if password:
        args.extend(["--scp-pw", password])
    scp_tool = _env("HMOPT_FLASH_SCP_TOOL", "pscp")
    if scp_tool:
        args.extend(["--scp-tool", scp_tool])
    if serial:
        args.extend(["--serial", serial])

    # Estimate total timeout: transfer + bootloader_wait + fastboot_wait + flash + boot_wait + margin
    total_timeout = 300 + bootloader_wait_s + fastboot_wait_s + (flash_timeout_s * len(partitions)) + boot_wait_timeout_s + 60

    result = _relay_exec("python", args, timeout_s=total_timeout, cwd=img_dir)

    # Parse the JSON output from the pipeline script
    pipeline_result: dict[str, Any] = {}
    stdout = result.get("stdout", "").strip()
    if stdout:
        try:
            pipeline_result = json.loads(stdout)
        except json.JSONDecodeError:
            pipeline_result = {"success": False, "error": "failed to parse pipeline output", "raw_stdout": stdout}

    # If the relay call itself failed (e.g., python not found)
    if result.get("returncode") not in (0, 2):
        return {
            "success": False,
            "phase": "relay_exec",
            "error": result.get("stderr", ""),
            "relay_result": result,
            "pipeline_result": pipeline_result,
        }

    # Return the pipeline result, augmented with relay info
    pipeline_result["relay_result"] = {
        "returncode": result.get("returncode"),
        "duration_s": result.get("duration_s"),
        "stderr": result.get("stderr", ""),
    }
    return pipeline_result


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
    partitions: list[dict[str, str]],
    server_images: list[dict[str, str]] | None = None,
    device_serial: str | None = None,
    scp_host: str | None = None,
    scp_user: str | None = None,
    scp_password: str | None = None,
    windows_image_dir: str | None = None,
    pipeline_script: str | None = None,
    bootloader_wait_s: int = 10,
    fastboot_wait_s: int = 30,
    flash_timeout_s: int = 600,
    boot_wait_timeout_s: int = 300,
    poll_interval_s: int = 5,
) -> dict[str, Any]:
    """Submit a full flash-and-boot pipeline as an async task."""
    payload = {
        "partitions": partitions,
        "server_images": server_images,
        "device_serial": device_serial,
        "bootloader_wait_s": bootloader_wait_s,
        "fastboot_wait_s": fastboot_wait_s,
        "flash_timeout_s": flash_timeout_s,
        "boot_wait_timeout_s": boot_wait_timeout_s,
    }
    return _submit_async_task(
        "flash_and_boot",
        payload,
        lambda: flash_and_boot(
            partitions=partitions,
            server_images=server_images,
            device_serial=device_serial,
            scp_host=scp_host,
            scp_user=scp_user,
            scp_password=scp_password,
            windows_image_dir=windows_image_dir,
            pipeline_script=pipeline_script,
            bootloader_wait_s=bootloader_wait_s,
            fastboot_wait_s=fastboot_wait_s,
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
        name="transfer_images",
        description=(
            "Pull image files from the build server to the Windows PC via pscp/scp. "
            "Each image entry: {server_path, local_filename}."
        ),
    )
    def mcp_transfer_images(
        images: list[dict[str, str]],
        scp_host: str | None = None,
        scp_user: str | None = None,
        scp_password: str | None = None,
        windows_image_dir: str | None = None,
        timeout_s: int = 300,
    ) -> dict[str, Any]:
        return transfer_images(
            images=images,
            scp_host=scp_host,
            scp_user=scp_user,
            scp_password=scp_password,
            windows_image_dir=windows_image_dir,
            timeout_s=timeout_s,
        )

    @mcp.tool(
        name="enter_bootloader",
        description="Reboot device into fastboot/bootloader mode via hdc shell reboot bootloader.",
    )
    def mcp_enter_bootloader(device_serial: str | None = None, timeout_s: int = 30) -> dict[str, Any]:
        return enter_bootloader(device_serial=device_serial, timeout_s=timeout_s)

    @mcp.tool(
        name="wait_for_fastboot",
        description="Poll fastboot devices until target device appears in bootloader mode.",
    )
    def mcp_wait_for_fastboot(
        device_serial: str | None = None,
        timeout_s: int = 30,
        poll_interval_s: int = 2,
    ) -> dict[str, Any]:
        return wait_for_fastboot_device(
            device_serial=device_serial,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )

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
        name="flash_partitions",
        description="Flash multiple partitions sequentially. Each entry: {partition, image_path}.",
    )
    def mcp_flash_partitions(
        partitions: list[dict[str, str]],
        device_serial: str | None = None,
        timeout_s: int = 600,
    ) -> dict[str, Any]:
        return flash_partitions(
            partitions=partitions,
            device_serial=device_serial,
            timeout_s=timeout_s,
        )

    @mcp.tool(
        name="flash_and_boot",
        description=(
            "Full flash pipeline: transfer images via pscp, enter bootloader, "
            "flash partitions, reboot, wait for hdc. Matches production flash workflow."
        ),
    )
    def mcp_flash_and_boot(
        partitions: list[dict[str, str]],
        server_images: list[dict[str, str]] | None = None,
        device_serial: str | None = None,
        scp_host: str | None = None,
        scp_user: str | None = None,
        scp_password: str | None = None,
        windows_image_dir: str | None = None,
        bootloader_wait_s: int = 15,
        fastboot_wait_s: int = 30,
        flash_timeout_s: int = 600,
        boot_wait_timeout_s: int = 300,
        poll_interval_s: int = 5,
    ) -> dict[str, Any]:
        return flash_and_boot(
            partitions=partitions,
            server_images=server_images,
            device_serial=device_serial,
            scp_host=scp_host,
            scp_user=scp_user,
            scp_password=scp_password,
            windows_image_dir=windows_image_dir,
            bootloader_wait_s=bootloader_wait_s,
            fastboot_wait_s=fastboot_wait_s,
            flash_timeout_s=flash_timeout_s,
            boot_wait_timeout_s=boot_wait_timeout_s,
            poll_interval_s=poll_interval_s,
        )

    @mcp.tool(
        name="flash_and_boot_async",
        description="Submit full flash pipeline as async task, returns task_id immediately.",
    )
    def mcp_flash_and_boot_async(
        partitions: list[dict[str, str]],
        server_images: list[dict[str, str]] | None = None,
        device_serial: str | None = None,
        scp_host: str | None = None,
        scp_user: str | None = None,
        scp_password: str | None = None,
        windows_image_dir: str | None = None,
        bootloader_wait_s: int = 15,
        fastboot_wait_s: int = 30,
        flash_timeout_s: int = 600,
        boot_wait_timeout_s: int = 300,
        poll_interval_s: int = 5,
    ) -> dict[str, Any]:
        return flash_and_boot_async(
            partitions=partitions,
            server_images=server_images,
            device_serial=device_serial,
            scp_host=scp_host,
            scp_user=scp_user,
            scp_password=scp_password,
            windows_image_dir=windows_image_dir,
            bootloader_wait_s=bootloader_wait_s,
            fastboot_wait_s=fastboot_wait_s,
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

    @mcp.tool(
        name="list_hdc_targets",
        description="List devices visible to hdc on the Windows relay host.",
    )
    def mcp_list_hdc_targets() -> dict[str, Any]:
        return list_hdc_targets()

    return mcp
