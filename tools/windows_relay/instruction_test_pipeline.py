"""All-in-one instruction-test pipeline script for Windows.

This runs the modelCase-style test harness on Windows and locates the
freshly-written report directory so the Linux-side auto-test MCP
service can consume the result deterministically.

Flow:

  1. Snapshot existing ``reports\\report_YYYYMMDDHHMMSS`` directories in
     the test workspace (e.g. ``D:\\modelCase_OH_single\\reports``).
  2. Run ``python main.py`` in the workspace (15 min – 1 hr typical).
  3. Re-scan the reports dir, pick the new ``report_YYYYMMDDHHMMSS``
     directory that was created during/after the run.
  4. Optionally invoke ``report_compare.py`` against a prior baseline
     report directory to produce an instruction-count diff.
  5. Emit a single JSON document to stdout for the caller to parse.

Zero external dependencies — uses only Python stdlib.  Kept independent
from the rest of the codebase so it can live alongside
``relay_service.bat`` on any Windows host with only Python installed.

Usage:
    python instruction_test_pipeline.py --test-dir D:\\modelCase_OH_single
    python instruction_test_pipeline.py --test-dir D:\\modelCase_OH_single \\
        --compare --baseline-report D:\\modelCase_OH_single\\reports\\report_20260414114948
    python instruction_test_pipeline.py --test-dir D:\\modelCase_OH_single \\
        --main-script main.py --run-timeout 3600
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

REPORT_DIR_NAME = "reports"
REPORT_NAME_PATTERN = re.compile(r"^report_(\d{14})$")
DEFAULT_MAIN_SCRIPT = "main.py"
DEFAULT_RUN_TIMEOUT_S = 3600 * 2  # 2 hours to allow long instruction runs
DEFAULT_TEST_DIR = r"D:\modelCase_OH_single"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Log to stderr so stdout stays pure JSON."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[itest][{ts}] {msg}", file=sys.stderr, flush=True)


def _snapshot_reports(reports_dir: str) -> set[str]:
    """Return the set of existing report_* directory names."""
    if not os.path.isdir(reports_dir):
        return set()
    return {
        name
        for name in os.listdir(reports_dir)
        if REPORT_NAME_PATTERN.match(name)
        and os.path.isdir(os.path.join(reports_dir, name))
    }


def _parse_report_timestamp(name: str) -> datetime | None:
    match = REPORT_NAME_PATTERN.match(name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _pick_new_report(
    reports_dir: str,
    *,
    baseline_names: set[str],
    started_at: datetime,
    grace_seconds: int = 60,
) -> str | None:
    """Pick the ``report_YYYYMMDDHHMMSS`` directory associated with the run we
    just started.

    Strategy:
      1. Of the directories present now but not in *baseline_names*, pick
         the one with the most recent timestamp parsed from its name.
         (This is the normal, unambiguous case.)
      2. If no new names appear (e.g. the tool reuses a name), fall back
         to scanning *all* report dirs and returning the one whose
         embedded timestamp is the latest AND is within *grace_seconds*
         seconds of *started_at* or later.
      3. If neither works, return None.
    """
    if not os.path.isdir(reports_dir):
        return None

    current = _snapshot_reports(reports_dir)
    new_dirs = sorted(current - baseline_names, key=lambda n: _parse_report_timestamp(n) or datetime.min)
    if new_dirs:
        return os.path.join(reports_dir, new_dirs[-1])

    # Fallback: newest existing dir that could plausibly have been written by this run.
    cutoff = started_at.timestamp() - grace_seconds
    candidates: list[tuple[datetime, str]] = []
    for name in current:
        ts = _parse_report_timestamp(name)
        if ts is None:
            continue
        if ts.timestamp() >= cutoff:
            candidates.append((ts, name))
    if candidates:
        candidates.sort()
        return os.path.join(reports_dir, candidates[-1][1])
    return None


def _run(
    argv: list[str],
    *,
    cwd: str | None,
    timeout_s: int,
    capture_output: bool = True,
) -> dict[str, Any]:
    """Run a subprocess and return a structured result dict."""
    cmd_str = subprocess.list2cmdline(argv)
    started = time.time()
    stdout = ""
    stderr = ""
    returncode: int
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
    except FileNotFoundError:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"not found: {argv[0]}",
            "duration_s": round(time.time() - started, 3),
            "command": cmd_str,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        return {
            "ok": False,
            "returncode": -9,
            "stdout": stdout,
            "stderr": (stderr + f"\ntimeout after {timeout_s}s").strip(),
            "duration_s": round(time.time() - started, 3),
            "command": cmd_str,
        }

    return {
        "ok": returncode == 0,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "duration_s": round(time.time() - started, 3),
        "command": cmd_str,
    }


def _truncate(text: str, max_chars: int = 4000) -> str:
    """Trim long stdout/stderr blobs to stay under the relay payload cap."""
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return text[:keep] + f"\n... [truncated {len(text) - max_chars} chars] ...\n" + text[-keep:]


def _resolve_python_exe(override: str | None = None) -> str:
    """Return a python executable suitable for launching main.py."""
    if override:
        return override
    # Prefer the exact interpreter currently running this pipeline.
    if sys.executable:
        return sys.executable
    return "python"


def _resolve_compare_script(override: str | None = None) -> str:
    """Locate the bundled report_compare.py script."""
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "report_compare.py")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_instruction_test(
    *,
    test_dir: str,
    main_script: str = DEFAULT_MAIN_SCRIPT,
    python_exe: str | None = None,
    extra_args: list[str] | None = None,
    run_timeout_s: int = DEFAULT_RUN_TIMEOUT_S,
    compare: bool = False,
    baseline_report: str | None = None,
    compare_script: str | None = None,
    metric_file: str | None = None,
    metric_key: str | None = None,
    metric_pattern: str | None = None,
) -> dict[str, Any]:
    """Run ``main.py`` under *test_dir* and locate the fresh report directory.

    If *compare* is True and *baseline_report* is supplied, invoke
    ``report_compare.py`` to produce an instruction-count diff.

    Always returns a JSON-serializable dict.
    """
    test_dir_abs = os.path.abspath(test_dir)
    if not os.path.isdir(test_dir_abs):
        return {
            "success": False,
            "phase": "validate",
            "error": f"test_dir does not exist: {test_dir_abs}",
            "test_dir": test_dir_abs,
        }

    script_path = main_script if os.path.isabs(main_script) else os.path.join(test_dir_abs, main_script)
    if not os.path.isfile(script_path):
        return {
            "success": False,
            "phase": "validate",
            "error": f"main script not found: {script_path}",
            "test_dir": test_dir_abs,
        }

    reports_dir = os.path.join(test_dir_abs, REPORT_DIR_NAME)
    os.makedirs(reports_dir, exist_ok=True)

    baseline_names = _snapshot_reports(reports_dir)
    started_at = datetime.now()
    _log(f"snapshot: {len(baseline_names)} existing report(s) in {reports_dir}")

    python = _resolve_python_exe(python_exe)
    argv = [python, os.path.basename(main_script) if not os.path.isabs(main_script) else main_script]
    if extra_args:
        argv.extend(str(a) for a in extra_args)

    _log(f"launching: {subprocess.list2cmdline(argv)} (cwd={test_dir_abs}, timeout={run_timeout_s}s)")
    main_result = _run(argv, cwd=test_dir_abs, timeout_s=run_timeout_s)
    main_result_summary = {
        "ok": main_result["ok"],
        "returncode": main_result["returncode"],
        "duration_s": main_result["duration_s"],
        "command": main_result["command"],
        "stdout_tail": _truncate(main_result["stdout"]),
        "stderr_tail": _truncate(main_result["stderr"]),
    }

    if not main_result["ok"]:
        return {
            "success": False,
            "phase": "run_main",
            "error": f"main.py exited with code {main_result['returncode']}",
            "test_dir": test_dir_abs,
            "reports_dir": reports_dir,
            "started_at": started_at.isoformat(timespec="seconds"),
            "run_result": main_result_summary,
        }

    # Give the filesystem a moment to finalize the report directory.
    time.sleep(1)

    report_path = _pick_new_report(
        reports_dir,
        baseline_names=baseline_names,
        started_at=started_at,
    )
    if report_path is None:
        return {
            "success": False,
            "phase": "locate_report",
            "error": "no new report directory was detected under reports/",
            "test_dir": test_dir_abs,
            "reports_dir": reports_dir,
            "started_at": started_at.isoformat(timespec="seconds"),
            "run_result": main_result_summary,
        }

    report_name = os.path.basename(report_path)
    report_ts = _parse_report_timestamp(report_name)
    _log(f"new report located: {report_path}")

    output: dict[str, Any] = {
        "success": True,
        "phase": "complete",
        "test_dir": test_dir_abs,
        "reports_dir": reports_dir,
        "report_path": report_path,
        "report_name": report_name,
        "report_timestamp": report_ts.isoformat(timespec="seconds") if report_ts else None,
        "started_at": started_at.isoformat(timespec="seconds"),
        "run_result": main_result_summary,
    }

    # ── Optional comparison against a baseline report directory ──────────
    if compare:
        if not baseline_report:
            output["compare"] = {
                "success": False,
                "error": "compare=True but --baseline-report was not supplied",
            }
            return output

        compare_path = _resolve_compare_script(compare_script)
        if not os.path.isfile(compare_path):
            output["compare"] = {
                "success": False,
                "error": f"compare script not found: {compare_path}",
            }
            return output

        cmp_argv = [
            python,
            compare_path,
            "--baseline", baseline_report,
            "--candidate", report_path,
        ]
        if metric_file:
            cmp_argv.extend(["--metric-file", metric_file])
        if metric_key:
            cmp_argv.extend(["--metric-key", metric_key])
        if metric_pattern:
            cmp_argv.extend(["--metric-pattern", metric_pattern])

        _log(f"comparing: {subprocess.list2cmdline(cmp_argv)}")
        cmp_result = _run(cmp_argv, cwd=None, timeout_s=300)

        compare_payload: dict[str, Any] = {
            "ok": cmp_result["ok"],
            "returncode": cmp_result["returncode"],
            "duration_s": cmp_result["duration_s"],
            "command": cmp_result["command"],
            "baseline_report": os.path.abspath(baseline_report),
            "candidate_report": report_path,
        }
        stdout = cmp_result["stdout"].strip()
        if stdout:
            try:
                compare_payload["result"] = json.loads(stdout)
            except json.JSONDecodeError:
                compare_payload["stdout_raw"] = _truncate(cmp_result["stdout"])
        if cmp_result["stderr"]:
            compare_payload["stderr_tail"] = _truncate(cmp_result["stderr"])
        output["compare"] = compare_payload

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run modelCase main.py and locate the fresh report directory.",
    )
    parser.add_argument(
        "--test-dir",
        default=os.environ.get("HMOPT_AUTO_TEST_WINDOWS_TEST_DIR", DEFAULT_TEST_DIR),
        help=f"Test workspace (default: {DEFAULT_TEST_DIR}).",
    )
    parser.add_argument("--main-script", default=DEFAULT_MAIN_SCRIPT, help="Main script to run (default: main.py).")
    parser.add_argument("--python-exe", default=None, help="Override python executable.")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append one extra arg to main.py (repeatable).",
    )
    parser.add_argument(
        "--run-timeout",
        type=int,
        default=DEFAULT_RUN_TIMEOUT_S,
        help="Timeout in seconds for the main.py invocation.",
    )
    parser.add_argument("--compare", action="store_true", help="Compare the new report against --baseline-report.")
    parser.add_argument("--baseline-report", default=None, help="Baseline report directory for comparison.")
    parser.add_argument("--compare-script", default=None, help="Override path to report_compare.py.")
    parser.add_argument("--metric-file", default=None, help="Relative metric filename inside each report dir.")
    parser.add_argument("--metric-key", default=None, help="Dot-path key inside the metric JSON file.")
    parser.add_argument("--metric-pattern", default=None, help="Regex with one capture group for text extraction.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    result = run_instruction_test(
        test_dir=args.test_dir,
        main_script=args.main_script,
        python_exe=args.python_exe,
        extra_args=list(args.extra_arg or []),
        run_timeout_s=args.run_timeout,
        compare=args.compare,
        baseline_report=args.baseline_report,
        compare_script=args.compare_script,
        metric_file=args.metric_file,
        metric_key=args.metric_key,
        metric_pattern=args.metric_pattern,
    )

    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
