"""FastMCP service wrapper for running phone test scripts through HDC.

Also exposes the modelCase-style instruction-test workflow that runs on a
Windows host via the shared command relay service
(``tools/windows_relay/relay_service.bat``).  The relay client talks to
the same ``/exec`` endpoint used by the flash MCP, but dispatches the
two pipeline helpers bundled next to ``relay_service.bat``:

  - ``instruction_test_pipeline.py`` — runs ``python main.py`` under the
    configured test workspace and returns the newly-created
    ``reports\\report_YYYYMMDDHHMMSS`` directory.
  - ``report_compare.py`` — compares two report directories on
    instruction count.

Linux-side orchestration (stock → flash feature → feature → compare) is
what drives this MCP.  The relay layer is kept thin so the Windows host
only needs Python and the two bundled scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import threading
import time
import uuid
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)
DEFAULT_SERVER_NAME = "hmopt-auto-test-mcp"
DEFAULT_TOOL_NAME = "phone_test_run"
BUILTIN_SWIPE_CASE = "basic_swipe"
BUILTIN_SWIPE_REMOTE_SCRIPT = "/data/local/tmp/hmopt_basic_swipe.sh"

# modelCase-style defaults for the Windows relay workflow.
DEFAULT_WINDOWS_TEST_DIR = r"D:\modelCase_OH_single"
DEFAULT_INSTRUCTION_TEST_PIPELINE_SCRIPT = "instruction_test_pipeline.py"
DEFAULT_REPORT_COMPARE_SCRIPT = "report_compare.py"
DEFAULT_INSTRUCTION_MAIN_SCRIPT = "main.py"
DEFAULT_INSTRUCTION_RUN_TIMEOUT_S = 3600 * 2  # 2h, matches pipeline default
DEFAULT_COMPARE_TIMEOUT_S = 300

_INSTRUCTION_TASKS: dict[str, dict[str, Any]] = {}
_INSTRUCTION_TASKS_LOCK = threading.Lock()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _resolve_hdc_bin() -> str:
    return _env("HMOPT_AUTO_TEST_HDC_BIN", "hdc")


def _resolve_default_target() -> str | None:
    value = _env("HMOPT_AUTO_TEST_TARGET", "")
    return value or None


def _run_cmd(argv: list[str], timeout_s: int) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    duration = round(time.time() - started, 3)
    return {
        "command": " ".join(shlex.quote(item) for item in argv),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "duration_s": duration,
    }


def _resolve_local_result_dir(local_result_dir: str | None = None) -> Path:
    raw_path = (local_result_dir or _env("HMOPT_AUTO_TEST_RESULT_DIR", "outputs/phone_test_results")).strip()
    path = Path(raw_path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_test_case(test_case: str) -> str:
    value = test_case.strip()
    if not value:
        raise ValueError("test_case is required")
    return value


def _build_shell_command(remote_script: str, test_case: str, extra_args: list[str]) -> str:
    script = remote_script.strip()
    if not script:
        raise ValueError("remote_script is required")

    argv = [script, test_case]
    argv.extend(extra_args)
    return " ".join(shlex.quote(arg) for arg in argv)


def _needs_legacy_tconn(result: dict[str, Any]) -> bool:
    if result["returncode"] == 0:
        return False
    merged = (result.get("stdout", "") + "\n" + result.get("stderr", "")).lower()
    return "unknown operation command" in merged or "tconn" in merged


def _normalize_target(target: str | None) -> str | None:
    value = str(target or "").strip()
    return value or None


def _resolve_connect_mode() -> str:
    mode = _env("HMOPT_AUTO_TEST_HDC_CONNECT_MODE", "none").lower()
    if mode not in {"auto", "connect", "tconn", "none"}:
        raise ValueError("HMOPT_AUTO_TEST_HDC_CONNECT_MODE must be one of auto/connect/tconn/none")
    return mode


def _hdc_target_prefix(target: str | None, use_target_flag: bool) -> list[str]:
    if target and use_target_flag:
        return ["-t", target]
    return []


def _connect_target(hdc: str, target: str, connect_mode: str, connect_timeout_s: int) -> dict[str, Any] | None:
    if connect_mode == "none":
        return None

    if connect_mode == "connect":
        result = _run_cmd([hdc, "connect", target], timeout_s=connect_timeout_s)
        if result["returncode"] != 0:
            raise ValueError(f"hdc connect failed: {result['stderr'].strip() or result['stdout'].strip()}")
        return result

    if connect_mode == "tconn":
        result = _run_cmd([hdc, "tconn", target], timeout_s=connect_timeout_s)
        if result["returncode"] != 0:
            raise ValueError(f"hdc tconn failed: {result['stderr'].strip() or result['stdout'].strip()}")
        return result

    connect_result = _run_cmd([hdc, "connect", target], timeout_s=connect_timeout_s)
    if connect_result["returncode"] == 0:
        return connect_result

    if not _needs_legacy_tconn(connect_result):
        raise ValueError(f"hdc connect failed: {connect_result['stderr'].strip() or connect_result['stdout'].strip()}")

    tconn_result = _run_cmd([hdc, "tconn", target], timeout_s=connect_timeout_s)
    if tconn_result["returncode"] != 0:
        raise ValueError(
            "hdc connect/tconn failed: "
            f"connect=({connect_result['stderr'].strip() or connect_result['stdout'].strip()}); "
            f"tconn=({tconn_result['stderr'].strip() or tconn_result['stdout'].strip()})"
        )
    return tconn_result


def _builtin_swipe_local_script() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "phone_tests" / "basic_swipe.sh"


def _prepare_builtin_swipe_script(
    *,
    hdc: str,
    target_value: str | None,
    use_target_flag: bool,
    timeout_s: int,
) -> tuple[str, dict[str, Any]]:
    local_script = _builtin_swipe_local_script()
    if not local_script.exists():
        raise ValueError(f"builtin swipe script not found: {local_script}")

    remote_script = _env("HMOPT_AUTO_TEST_BUILTIN_SWIPE_REMOTE_SCRIPT", BUILTIN_SWIPE_REMOTE_SCRIPT)
    send_argv = [
        hdc,
        *_hdc_target_prefix(target_value, use_target_flag),
        "file",
        "send",
        str(local_script),
        remote_script,
    ]
    send_result = _run_cmd(send_argv, timeout_s=timeout_s)
    if send_result["returncode"] != 0:
        raise ValueError(
            "hdc file send builtin swipe script failed: "
            f"{send_result['stderr'].strip() or send_result['stdout'].strip()}"
        )

    chmod_argv = [
        hdc,
        *_hdc_target_prefix(target_value, use_target_flag),
        "shell",
        f"chmod +x {shlex.quote(remote_script)}",
    ]
    chmod_result = _run_cmd(chmod_argv, timeout_s=timeout_s)
    if chmod_result["returncode"] != 0:
        raise ValueError(
            "hdc shell chmod builtin swipe script failed: "
            f"{chmod_result['stderr'].strip() or chmod_result['stdout'].strip()}"
        )

    return remote_script, {"send": send_result, "chmod": chmod_result}


def _resolve_builtin_swipe_args(remote_result_path: str, extra_args: list[str]) -> list[str]:
    duration_s = extra_args[0] if extra_args else "60"
    swipe_count = extra_args[1] if len(extra_args) > 1 else "1050"
    return [remote_result_path, duration_s, swipe_count]


def run_phone_test(
    *,
    target: str | None = None,
    test_case: str,
    remote_script: str,
    remote_result_path: str,
    local_result_dir: str | None = None,
    extra_args: list[str] | None = None,
    connect_before_shell: bool = False,
    use_target_flag: bool = True,
    connect_timeout_s: int = 20,
    shell_timeout_s: int = 1800,
    recv_timeout_s: int = 120,
) -> dict[str, Any]:
    """Run phone test script through HDC and retrieve result artifact."""
    target_value = _normalize_target(target) or _resolve_default_target()
    if not target_value and use_target_flag:
        raise ValueError("target is required (set tool argument `target` or env `HMOPT_AUTO_TEST_TARGET`)")

    case_name = _validate_test_case(test_case)
    result_path = remote_result_path.strip() or f"/data/local/tmp/{case_name}.result"

    safe_extra_args = [str(item) for item in (extra_args or [])]
    local_dir = _resolve_local_result_dir(local_result_dir)

    hdc = _resolve_hdc_bin()
    connect_mode = _resolve_connect_mode()
    connect_result = None
    if connect_before_shell:
        if not target_value:
            raise ValueError("target is required when connect_before_shell=true")
        connect_result = _connect_target(hdc, str(target_value), connect_mode, connect_timeout_s)

    resolved_remote_script = remote_script
    prepare_result = None
    if case_name == BUILTIN_SWIPE_CASE and not remote_script.strip():
        resolved_remote_script, prepare_result = _prepare_builtin_swipe_script(
            hdc=hdc,
            target_value=target_value,
            use_target_flag=use_target_flag,
            timeout_s=recv_timeout_s,
        )
        safe_extra_args = _resolve_builtin_swipe_args(result_path, safe_extra_args)

    shell_cmd = _build_shell_command(resolved_remote_script, case_name, safe_extra_args)
    run_argv = [hdc, *_hdc_target_prefix(target_value, use_target_flag), "shell", shell_cmd]
    run_result = _run_cmd(run_argv, timeout_s=shell_timeout_s)

    artifact_local_path = local_dir / f"{case_name}.result"
    recv_argv = [
        hdc,
        *_hdc_target_prefix(target_value, use_target_flag),
        "file",
        "recv",
        result_path,
        str(artifact_local_path),
    ]
    recv_result = _run_cmd(recv_argv, timeout_s=recv_timeout_s)

    success = run_result["returncode"] == 0 and recv_result["returncode"] == 0
    return {
        "success": success,
        "target": target_value,
        "test_case": case_name,
        "remote_script": resolved_remote_script,
        "remote_result_path": result_path,
        "local_result_path": str(artifact_local_path),
        "connect_before_shell": connect_before_shell,
        "use_target_flag": use_target_flag,
        "steps": {
            "connect": connect_result,
            "prepare": prepare_result,
            "run": run_result,
            "recv": recv_result,
        },
    }


# ---------------------------------------------------------------------------
# Windows relay helpers (modelCase instruction-test workflow)
# ---------------------------------------------------------------------------

def _relay_url() -> str:
    """Resolve the Windows relay URL.

    Falls back to the flash MCP relay URL so deployments that already
    configured flashing don't need a second env var.
    """
    url = (os.getenv("HMOPT_AUTO_TEST_RELAY_URL") or os.getenv("HMOPT_FLASH_RELAY_URL") or "").strip()
    if not url:
        raise ValueError(
            "HMOPT_AUTO_TEST_RELAY_URL (or HMOPT_FLASH_RELAY_URL) is required, e.g. http://10.x.x.x:9100"
        )
    return url.rstrip("/")


def _relay_secret() -> str:
    return (
        os.getenv("HMOPT_AUTO_TEST_RELAY_SECRET") or os.getenv("HMOPT_FLASH_RELAY_SECRET") or ""
    ).strip()


def _relay_exec(
    command: str,
    args: list[str],
    *,
    timeout_s: int,
    cwd: str | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    """POST /exec on the Windows relay and return the parsed response."""
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
                return json.loads(resp.read())
        except (URLError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning(
                    "auto-test relay request failed (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1,
                    retries,
                    wait,
                    exc,
                )
                time.sleep(wait)

    raise ConnectionError(f"auto-test relay unreachable after {retries} attempts: {last_error}") from last_error


def auto_test_relay_health_check() -> dict[str, Any]:
    """Check connectivity to the Windows relay host used for instruction tests."""
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


def _default_windows_test_dir() -> str:
    return os.getenv("HMOPT_AUTO_TEST_WINDOWS_TEST_DIR", DEFAULT_WINDOWS_TEST_DIR).strip() or DEFAULT_WINDOWS_TEST_DIR


def _default_pipeline_script() -> str:
    return (
        os.getenv("HMOPT_AUTO_TEST_PIPELINE_SCRIPT", DEFAULT_INSTRUCTION_TEST_PIPELINE_SCRIPT).strip()
        or DEFAULT_INSTRUCTION_TEST_PIPELINE_SCRIPT
    )


def _default_compare_script() -> str:
    return (
        os.getenv("HMOPT_AUTO_TEST_COMPARE_SCRIPT", DEFAULT_REPORT_COMPARE_SCRIPT).strip()
        or DEFAULT_REPORT_COMPARE_SCRIPT
    )


def _default_main_script() -> str:
    return (
        os.getenv("HMOPT_AUTO_TEST_MAIN_SCRIPT", DEFAULT_INSTRUCTION_MAIN_SCRIPT).strip()
        or DEFAULT_INSTRUCTION_MAIN_SCRIPT
    )


def _parse_pipeline_stdout(stdout: str) -> tuple[dict[str, Any], str | None]:
    """Parse the JSON document the pipeline prints to stdout.

    Returns ``(result_dict, parse_error_or_none)``.  If the script
    printed extra logging before the JSON, only the final JSON object is
    considered.
    """
    text = (stdout or "").strip()
    if not text:
        return {}, "empty stdout"
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        # Fallback: find the last "{" that can be balanced to the end.
        last_brace = text.rfind("{")
        if last_brace > 0:
            snippet = text[last_brace:]
            try:
                return json.loads(snippet), None
            except json.JSONDecodeError as exc:
                return {}, f"failed to parse trailing JSON: {exc}"
        return {}, "no JSON object in stdout"


def run_instruction_test(
    *,
    compare: bool = False,
    baseline_report: str | None = None,
    test_dir: str | None = None,
    main_script: str | None = None,
    pipeline_script: str | None = None,
    compare_script: str | None = None,
    extra_args: list[str] | None = None,
    run_timeout_s: int = DEFAULT_INSTRUCTION_RUN_TIMEOUT_S,
    compare_level: str = "total",
    compare_process: str | None = None,
    compare_thread: str | None = None,
    compare_lib: str | None = None,
    compare_function: str | None = None,
    python_exe: str | None = None,
    use_venv: bool = True,
    venv_dir: str | None = None,
    force_utf8: bool = True,
) -> dict[str, Any]:
    """Trigger modelCase's ``main.py`` on the Windows host via the relay.

    The Windows-side ``instruction_test_pipeline.py`` is responsible for
    snapshotting the ``reports/`` directory, running ``main.py``, then
    locating the freshly-written ``report_YYYYMMDDHHMMSS`` directory and
    optionally calling ``report_compare.py`` against a previous baseline.

    Args:
        compare: When True, ask the pipeline to compare the new report
            against *baseline_report*.  For stock runs keep this False;
            for feature runs pass the path returned by the prior stock
            call here.
        baseline_report: Absolute path (on the Windows host) of the
            baseline report directory to compare against.
        test_dir: Override the workspace (default
            ``D:\\modelCase_OH_single`` or
            ``HMOPT_AUTO_TEST_WINDOWS_TEST_DIR``).
        main_script: Override the script name (default ``main.py``).
        pipeline_script: Filename of the pipeline script next to
            ``relay_service.bat`` (default
            ``instruction_test_pipeline.py``).
        compare_script: Filename of the compare script (default
            ``report_compare.py``).
        extra_args: Extra CLI args forwarded to ``main.py``.
        run_timeout_s: Ceiling on the ``main.py`` execution time.
        compare_level: Granularity when *compare* is True. One of
            ``total`` (default) / ``process`` / ``thread`` / ``lib`` /
            ``function``; forwarded to ``report_compare.py --level``.
        compare_process / compare_thread / compare_lib / compare_function:
            Target names needed at and above the chosen level (e.g.
            ``compare_level="thread"`` requires both ``compare_process``
            and ``compare_thread``).
        python_exe: Override the python interpreter on Windows. Takes
            precedence over venv auto-detection.
        use_venv: When True (default) the pipeline probes
            ``<test_dir>\\.venv\\Scripts\\python.exe`` (and ``venv/``
            plus POSIX equivalents) and uses it if present, which is
            equivalent to running ``activate.bat`` first.  Set False to
            force the system interpreter.
        venv_dir: Override the venv directory name (absolute or
            relative to ``test_dir``).
        force_utf8: When True (default) the pipeline launches main.py
            with ``-X utf8`` and ``PYTHONUTF8=1`` /
            ``PYTHONIOENCODING=utf-8`` so multiprocessing workers
            spawned by main.py don't hit ``UnicodeEncodeError`` when
            their stdout is piped on a cp1252 Windows host.  Set
            False only if a workload actually depends on a legacy
            codepage.

    The pipeline always wakes the device to the home screen before
    launching main.py — this is baked into ``instruction_test_pipeline.py``
    and not a tunable here.
    """
    if compare and not baseline_report:
        raise ValueError("baseline_report is required when compare=True")

    ws_dir = (test_dir or _default_windows_test_dir()).strip()
    pipeline = (pipeline_script or _default_pipeline_script()).strip()
    compare_name = (compare_script or _default_compare_script()).strip()
    main_name = (main_script or _default_main_script()).strip()

    argv: list[str] = [
        pipeline,
        "--test-dir", ws_dir,
        "--main-script", main_name,
        "--run-timeout", str(int(run_timeout_s)),
    ]
    if python_exe:
        argv.extend(["--python-exe", python_exe])
    if not use_venv:
        argv.append("--no-venv")
    elif venv_dir:
        argv.extend(["--venv-dir", venv_dir])
    if not force_utf8:
        argv.append("--no-utf8")
    if compare:
        argv.append("--compare")
        argv.extend(["--baseline-report", baseline_report or ""])
        argv.extend(["--compare-script", compare_name])
        argv.extend(["--compare-level", compare_level])
        if compare_process:
            argv.extend(["--compare-process", compare_process])
        if compare_thread:
            argv.extend(["--compare-thread", compare_thread])
        if compare_lib:
            argv.extend(["--compare-lib", compare_lib])
        if compare_function:
            argv.extend(["--compare-function", compare_function])
    for extra in extra_args or []:
        argv.extend(["--extra-arg", str(extra)])

    # Allow the relay a little more headroom than the pipeline itself so it can
    # return the partial structured output on timeout.
    relay_timeout = int(run_timeout_s) + DEFAULT_COMPARE_TIMEOUT_S + 120

    relay_result = _relay_exec("python", argv, timeout_s=relay_timeout, cwd=None)

    pipeline_result, parse_error = _parse_pipeline_stdout(relay_result.get("stdout", ""))
    if parse_error and not pipeline_result:
        return {
            "success": False,
            "phase": "parse_pipeline_output",
            "error": parse_error,
            "relay_result": {
                "returncode": relay_result.get("returncode"),
                "duration_s": relay_result.get("duration_s"),
                "stderr": relay_result.get("stderr", ""),
                "stdout": relay_result.get("stdout", ""),
                "command": relay_result.get("command", ""),
            },
        }

    # Propagate relay-level diagnostics alongside the pipeline result.
    pipeline_result["relay_result"] = {
        "returncode": relay_result.get("returncode"),
        "duration_s": relay_result.get("duration_s"),
        "stderr": relay_result.get("stderr", ""),
        "command": relay_result.get("command", ""),
    }

    # Surface a relay-level failure if the pipeline did not self-report.
    if relay_result.get("returncode") not in (0, 2) and pipeline_result.get("success") is None:
        pipeline_result.setdefault("success", False)
        pipeline_result.setdefault("phase", "relay_exec")
        pipeline_result.setdefault("error", relay_result.get("stderr", ""))

    return pipeline_result


def compare_reports(
    *,
    baseline_report: str,
    candidate_report: str,
    compare_script: str | None = None,
    level: str = "total",
    process: str | None = None,
    thread: str | None = None,
    lib: str | None = None,
    function: str | None = None,
    timeout_s: int = DEFAULT_COMPARE_TIMEOUT_S,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Run ``report_compare.py`` on the Windows host via the relay.

    For every hiperf xlsx pair found under *baseline_report* and
    *candidate_report* (keyed by ``(case, round, step)``) this extracts
    the count at the requested granularity and returns a single
    baseline/candidate/delta per pair plus an aggregate.

    Levels map to names that must be supplied alongside:
      - ``total``:    no names
      - ``process``:  *process*
      - ``thread``:   *process*, *thread*
      - ``lib``:      *process*, *thread*, *lib*
      - ``function``: *process*, *thread*, *lib*, *function*
    """
    if not baseline_report or not candidate_report:
        raise ValueError("baseline_report and candidate_report are both required")

    script = (compare_script or _default_compare_script()).strip()
    argv: list[str] = [
        script,
        "--baseline", baseline_report,
        "--candidate", candidate_report,
        "--level", level,
    ]
    if process:
        argv.extend(["--process", process])
    if thread:
        argv.extend(["--thread", thread])
    if lib:
        argv.extend(["--lib", lib])
    if function:
        argv.extend(["--function", function])

    relay_result = _relay_exec("python", argv, timeout_s=int(timeout_s), cwd=cwd)
    compare_result, parse_error = _parse_pipeline_stdout(relay_result.get("stdout", ""))
    if parse_error and not compare_result:
        return {
            "success": False,
            "phase": "parse_compare_output",
            "error": parse_error,
            "relay_result": {
                "returncode": relay_result.get("returncode"),
                "duration_s": relay_result.get("duration_s"),
                "stderr": relay_result.get("stderr", ""),
                "stdout": relay_result.get("stdout", ""),
                "command": relay_result.get("command", ""),
            },
        }

    compare_result["relay_result"] = {
        "returncode": relay_result.get("returncode"),
        "duration_s": relay_result.get("duration_s"),
        "stderr": relay_result.get("stderr", ""),
        "command": relay_result.get("command", ""),
    }
    return compare_result


# ---------------------------------------------------------------------------
# Async task management for the long-running instruction test
# ---------------------------------------------------------------------------

def _register_instruction_task(payload: dict[str, Any]) -> str:
    task_id = str(uuid.uuid4())
    now = time.time()
    with _INSTRUCTION_TASKS_LOCK:
        _INSTRUCTION_TASKS[task_id] = {
            "task_id": task_id,
            "kind": "run_instruction_test",
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "payload": payload,
            "result": None,
            "error": None,
        }
    return task_id


def _set_instruction_task_running(task_id: str) -> None:
    with _INSTRUCTION_TASKS_LOCK:
        task = _INSTRUCTION_TASKS.get(task_id)
        if task is None:
            return
        task["status"] = "running"
        task["updated_at"] = time.time()


def _set_instruction_task_result(
    task_id: str,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    with _INSTRUCTION_TASKS_LOCK:
        task = _INSTRUCTION_TASKS.get(task_id)
        if task is None:
            return
        task["status"] = "failed" if error else "succeeded"
        task["result"] = result
        task["error"] = error
        task["updated_at"] = time.time()


def _run_instruction_task(task_id: str, runner: Callable[[], dict[str, Any]]) -> None:
    _set_instruction_task_running(task_id)
    try:
        result = runner()
        _set_instruction_task_result(task_id, result=result)
    except Exception as exc:  # noqa: BLE001 — propagate to task status
        logger.exception("instruction test async task failed: task_id=%s", task_id)
        _set_instruction_task_result(task_id, error=str(exc))


def run_instruction_test_async(**kwargs: Any) -> dict[str, Any]:
    """Submit ``run_instruction_test`` on a background thread and return immediately."""
    payload = {k: v for k, v in kwargs.items() if k != "python_exe"}
    task_id = _register_instruction_task(payload)
    thread = threading.Thread(
        target=_run_instruction_task,
        args=(task_id, lambda: run_instruction_test(**kwargs)),
        daemon=True,
    )
    thread.start()
    return {"task_id": task_id, "status": "pending", "kind": "run_instruction_test"}


def get_instruction_task_status(task_id: str) -> dict[str, Any]:
    """Query an async instruction-test task by id."""
    with _INSTRUCTION_TASKS_LOCK:
        task = _INSTRUCTION_TASKS.get(task_id)
        if task is None:
            raise ValueError(f"task not found: {task_id}")
        snapshot = deepcopy(task)
    snapshot["duration_s"] = round(snapshot["updated_at"] - snapshot["created_at"], 3)
    return snapshot


@lru_cache(maxsize=1)
def build_auto_test_fastmcp_server() -> Any | None:
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError:
        logger.warning(
            "mcp package is not installed; auto-test MCP endpoint is unavailable. "
            "Install with: pip install 'mcp[cli]'"
        )
        return None

    server_name = (
        os.getenv("HMOPT_AUTO_TEST_MCP_SERVER_NAME", DEFAULT_SERVER_NAME).strip() or DEFAULT_SERVER_NAME
    )
    tool_name = os.getenv("HMOPT_AUTO_TEST_TOOL_NAME", DEFAULT_TOOL_NAME).strip() or DEFAULT_TOOL_NAME

    mcp = FastMCP(server_name, stateless_http=True, json_response=True)

    @mcp.tool(
        name=tool_name,
        description=(
            "Run one phone test case via hdc shell and receive remote result artifact with hdc file recv. "
            "Supports legacy hdc tconn and optional connect-skip mode."
        ),
    )
    def mcp_run_phone_test(
        target: str | None = None,
        test_case: str = "",
        remote_script: str = "",
        remote_result_path: str = "",
        local_result_dir: str | None = None,
        extra_args: list[str] | None = None,
        connect_before_shell: bool = False,
        use_target_flag: bool = True,
        connect_timeout_s: int = 20,
        shell_timeout_s: int = 1800,
        recv_timeout_s: int = 120,
    ) -> dict[str, Any]:
        return run_phone_test(
            target=target,
            test_case=test_case,
            remote_script=remote_script,
            remote_result_path=remote_result_path,
            local_result_dir=local_result_dir,
            extra_args=extra_args,
            connect_before_shell=connect_before_shell,
            use_target_flag=use_target_flag,
            connect_timeout_s=connect_timeout_s,
            shell_timeout_s=shell_timeout_s,
            recv_timeout_s=recv_timeout_s,
        )

    # ── Windows-relay-based modelCase instruction-test workflow ──────────

    @mcp.tool(
        name="run_instruction_test",
        description=(
            "Trigger modelCase main.py on the Windows host via the relay. "
            "Runs instruction_test_pipeline.py which executes python main.py under "
            "the test workspace (default D:\\modelCase_OH_single) and returns the "
            "newly-created reports\\report_YYYYMMDDHHMMSS directory. "
            "For the first (stock) call leave compare=False. On the feature run set "
            "compare=True and pass baseline_report to the stock report_path returned earlier."
        ),
    )
    def mcp_run_instruction_test(
        compare: bool = False,
        baseline_report: str | None = None,
        test_dir: str | None = None,
        main_script: str | None = None,
        pipeline_script: str | None = None,
        compare_script: str | None = None,
        extra_args: list[str] | None = None,
        run_timeout_s: int = DEFAULT_INSTRUCTION_RUN_TIMEOUT_S,
        compare_level: str = "total",
        compare_process: str | None = None,
        compare_thread: str | None = None,
        compare_lib: str | None = None,
        compare_function: str | None = None,
        python_exe: str | None = None,
        use_venv: bool = True,
        venv_dir: str | None = None,
        force_utf8: bool = True,
    ) -> dict[str, Any]:
        return run_instruction_test(
            compare=compare,
            baseline_report=baseline_report,
            test_dir=test_dir,
            main_script=main_script,
            pipeline_script=pipeline_script,
            compare_script=compare_script,
            extra_args=extra_args,
            run_timeout_s=run_timeout_s,
            compare_level=compare_level,
            compare_process=compare_process,
            compare_thread=compare_thread,
            compare_lib=compare_lib,
            compare_function=compare_function,
            python_exe=python_exe,
            use_venv=use_venv,
            venv_dir=venv_dir,
            force_utf8=force_utf8,
        )

    @mcp.tool(
        name="run_instruction_test_async",
        description=(
            "Submit the modelCase instruction-test pipeline as an async task and "
            "return a task_id immediately. Use instruction_test_status to poll. "
            "Recommended for typical 15–60 minute runs that would time out a sync call."
        ),
    )
    def mcp_run_instruction_test_async(
        compare: bool = False,
        baseline_report: str | None = None,
        test_dir: str | None = None,
        main_script: str | None = None,
        pipeline_script: str | None = None,
        compare_script: str | None = None,
        extra_args: list[str] | None = None,
        run_timeout_s: int = DEFAULT_INSTRUCTION_RUN_TIMEOUT_S,
        compare_level: str = "total",
        compare_process: str | None = None,
        compare_thread: str | None = None,
        compare_lib: str | None = None,
        compare_function: str | None = None,
        python_exe: str | None = None,
        use_venv: bool = True,
        venv_dir: str | None = None,
        force_utf8: bool = True,
    ) -> dict[str, Any]:
        return run_instruction_test_async(
            compare=compare,
            baseline_report=baseline_report,
            test_dir=test_dir,
            main_script=main_script,
            pipeline_script=pipeline_script,
            compare_script=compare_script,
            extra_args=extra_args,
            run_timeout_s=run_timeout_s,
            compare_level=compare_level,
            compare_process=compare_process,
            compare_thread=compare_thread,
            compare_lib=compare_lib,
            compare_function=compare_function,
            python_exe=python_exe,
            use_venv=use_venv,
            venv_dir=venv_dir,
            force_utf8=force_utf8,
        )

    @mcp.tool(
        name="instruction_test_status",
        description="Query an async instruction-test task by task_id.",
    )
    def mcp_instruction_test_status(task_id: str) -> dict[str, Any]:
        return get_instruction_task_status(task_id=task_id)

    @mcp.tool(
        name="compare_reports",
        description=(
            "Invoke report_compare.py on the Windows host against two existing "
            "report_YYYYMMDDHHMMSS directories and return the instruction-count diff. "
            "Useful to re-run comparison with a different baseline without re-running the test."
        ),
    )
    def mcp_compare_reports(
        baseline_report: str,
        candidate_report: str,
        compare_script: str | None = None,
        level: str = "total",
        process: str | None = None,
        thread: str | None = None,
        lib: str | None = None,
        function: str | None = None,
        timeout_s: int = DEFAULT_COMPARE_TIMEOUT_S,
    ) -> dict[str, Any]:
        return compare_reports(
            baseline_report=baseline_report,
            candidate_report=candidate_report,
            compare_script=compare_script,
            level=level,
            process=process,
            thread=thread,
            lib=lib,
            function=function,
            timeout_s=timeout_s,
        )

    @mcp.tool(
        name="auto_test_relay_health",
        description="Check connectivity to the Windows relay service used for modelCase tests.",
    )
    def mcp_auto_test_relay_health() -> dict[str, Any]:
        return auto_test_relay_health_check()

    return mcp


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run phone test once without starting MCP server")
    parser.add_argument("--target", default="", help="hdc target; falls back to HMOPT_AUTO_TEST_TARGET")
    parser.add_argument("--test-case", required=True, help="test case name, e.g. basic_swipe")
    parser.add_argument("--remote-script", default="", help="script path on phone (empty for builtin basic_swipe)")
    parser.add_argument("--remote-result-path", default="", help="result path on phone")
    parser.add_argument("--local-result-dir", default=None, help="local dir to save pulled artifact")
    parser.add_argument("--extra-arg", action="append", default=[], help="append one extra arg; can repeat")
    parser.add_argument("--connect-before-shell", action="store_true", help="run connect/tconn before shell")
    parser.add_argument("--no-target-flag", action="store_true", help="disable hdc -t target")
    parser.add_argument("--connect-timeout-s", type=int, default=20)
    parser.add_argument("--shell-timeout-s", type=int, default=1800)
    parser.add_argument("--recv-timeout-s", type=int, default=120)
    return parser


def _main(argv: list[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    try:
        result = run_phone_test(
            target=args.target,
            test_case=args.test_case,
            remote_script=args.remote_script,
            remote_result_path=args.remote_result_path,
            local_result_dir=args.local_result_dir,
            extra_args=args.extra_arg,
            connect_before_shell=args.connect_before_shell,
            use_target_flag=not args.no_target_flag,
            connect_timeout_s=args.connect_timeout_s,
            shell_timeout_s=args.shell_timeout_s,
            recv_timeout_s=args.recv_timeout_s,
        )
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(_main())
