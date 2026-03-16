"""FastMCP service wrapper for running phone test scripts through HDC."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
DEFAULT_SERVER_NAME = "hmopt-auto-test-mcp"
DEFAULT_TOOL_NAME = "phone_test_run"
BUILTIN_SWIPE_CASE = "basic_swipe"
BUILTIN_SWIPE_REMOTE_SCRIPT = "/data/local/tmp/hmopt_basic_swipe.sh"


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
