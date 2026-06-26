# -*- coding: utf-8 -*-
"""Windows-side lmbench full-suite runner for the auto-test relay.

lmbench runs 2-5 hours, but the relay's per-request ceiling is 1 hour. So this
script does NOT block: it launches the lmbench autotest (D:\\LmbenchAutoTest\\
main.py) as a DETACHED background process and returns immediately; the Linux MCP
side then polls --status (each call is sub-second) until the run finishes.

Sub-commands (each emits one JSON document to stdout):

  --start              snapshot existing results, spawn main.py detached, write a
                       run-state file, return {started, run_token, pid, ...}
  --status RUN_TOKEN   is the process alive? did a new total_result xlsx appear?
                       when done, also computes the compact digest (HM-vs-Linux +
                       this-run-vs-previous) via lmbench_digest.py
  --digest ...         standalone: parse given xlsx into the digest

Result layout produced by the framework:
  <test_dir>\\result\\hongmeng\\total_result_Hongmeng_<ts>.xlsx        (per-run)
  <test_dir>\\result\\hongmeng_linux_result\\HM_Linux_lmbench_result_<ts>.xlsx

Zero deps beyond stdlib + (for --digest) the co-located lmbench_digest.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import lmbench_digest  # type: ignore
except Exception:  # pragma: no cover
    lmbench_digest = None

DEFAULT_TEST_DIR = r"D:\LmbenchAutoTest"
TOTAL_SUBDIR = os.path.join("result", "hongmeng")
HMLINUX_SUBDIR = os.path.join("result", "hongmeng_linux_result")
TOTAL_TAG = "total_result"          # total_result_Hongmeng_<ts>.xlsx
HMLINUX_TAG = "HM_Linux"            # HM_Linux_lmbench_result_<ts>.xlsx
STATE_DIRNAME = ".lmbench_runs"
DEFAULT_RUN_CMD = ["main.py"]       # framework entry (RunTest.bat -> main.py)


def _reconfigure_stdio_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if s is not None and hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass


_reconfigure_stdio_utf8()


def _log(msg: str) -> None:
    print(f"[lmbench][{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _utf8_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _truncate(text: str, max_chars: int = 4000) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    keep = max_chars // 2
    return text[:keep] + f"\n... [truncated {len(text) - max_chars} chars] ...\n" + text[-keep:]


def _python_exe(override: str | None, test_dir: str, use_venv: bool) -> str:
    if override:
        return override
    if use_venv:
        for d in (".venv", "venv"):
            for rel in (os.path.join("Scripts", "python.exe"), os.path.join("bin", "python")):
                p = os.path.join(test_dir, d, rel)
                if os.path.isfile(p):
                    return p
    return sys.executable or "python"


def _list_xlsx(dir_path: str, tag: str) -> list[str]:
    if not os.path.isdir(dir_path):
        return []
    out = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
           if f.lower().endswith(".xlsx") and tag.lower() in f.lower()]
    return sorted(out, key=lambda p: os.path.getmtime(p))


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        try:
            r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                               capture_output=True, text=True, timeout=15)
            return str(pid) in (r.stdout or "")
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except (OSError, ProcessLookupError):
        return False


def _spawn_detached(argv: list[str], cwd: str, log_path: str, env: dict[str, str]) -> int:
    log = open(log_path, "ab")
    kw: dict[str, Any] = dict(cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, env=env)
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | NEW_PROCESS_GROUP
        kw["close_fds"] = True
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(argv, **kw).pid


def _state_path(test_dir: str, token: str) -> str:
    d = os.path.join(test_dir, STATE_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{token}.json")


def cmd_start(test_dir: str, run_cmd: list[str], python_exe: str | None,
              use_venv: bool, force_utf8: bool) -> dict[str, Any]:
    test_dir = os.path.abspath(test_dir)
    if not os.path.isdir(test_dir):
        return {"success": False, "phase": "validate", "error": f"test_dir missing: {test_dir}"}
    total_dir = os.path.join(test_dir, TOTAL_SUBDIR)
    baseline = [os.path.basename(p) for p in _list_xlsx(total_dir, TOTAL_TAG)]

    py = _python_exe(python_exe, test_dir, use_venv)
    argv = [py]
    if force_utf8:
        argv += ["-X", "utf8"]
    argv += [str(a) for a in run_cmd]

    token = datetime.now().strftime("%Y%m%d%H%M%S") + f"_{os.getpid()}"
    log_path = os.path.join(test_dir, STATE_DIRNAME, f"{token}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        pid = _spawn_detached(argv, test_dir, log_path, _utf8_env() if force_utf8 else os.environ.copy())
    except Exception as exc:
        return {"success": False, "phase": "spawn", "error": f"{type(exc).__name__}: {exc}"}

    state = {
        "run_token": token, "pid": pid, "started_at": datetime.now().isoformat(timespec="seconds"),
        "test_dir": test_dir, "total_dir": total_dir,
        "hmlinux_dir": os.path.join(test_dir, HMLINUX_SUBDIR),
        "baseline": baseline, "log_path": log_path,
        "command": subprocess.list2cmdline(argv),
    }
    with open(_state_path(test_dir, token), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)
    _log(f"started pid={pid} token={token}: {state['command']}")
    return {"success": True, "phase": "started", "run_token": token, "pid": pid,
            "started_at": state["started_at"], "baseline_count": len(baseline),
            "command": state["command"], "note": "poll --status until status=done/failed"}


def _locate_results(state: dict, *, require_new: bool):
    totals = _list_xlsx(state["total_dir"], TOTAL_TAG)
    baseline = set(state.get("baseline") or [])
    new_totals = [p for p in totals if os.path.basename(p) not in baseline]
    current = new_totals[-1] if new_totals else (totals[-1] if (totals and not require_new) else None)
    prev = None
    if current is not None:
        rest = [p for p in totals if p != current]
        prev = rest[-1] if rest else None
    hmlinux = _list_xlsx(state["hmlinux_dir"], HMLINUX_TAG)
    hm = hmlinux[-1] if hmlinux else None
    return current, prev, hm


def cmd_status(test_dir: str, token: str, top_n: int) -> dict[str, Any]:
    sp = _state_path(os.path.abspath(test_dir), token)
    if not os.path.isfile(sp):
        return {"success": False, "error": f"unknown run_token: {token}"}
    with open(sp, encoding="utf-8") as f:
        state = json.load(f)
    alive = _pid_alive(state.get("pid"))
    current, prev, hm = _locate_results(state, require_new=True)
    started = datetime.fromisoformat(state["started_at"])
    elapsed = round((datetime.now() - started).total_seconds(), 1)

    if current is not None and not alive:
        status = "done"
    elif alive:
        status = "running"
    else:
        status = "failed"

    log_tail = ""
    try:
        if os.path.isfile(state.get("log_path", "")):
            with open(state["log_path"], "rb") as f:
                log_tail = _truncate(f.read()[-6000:].decode("utf-8", "replace"))
    except Exception:
        pass

    out = {
        "success": status != "failed", "status": status, "run_token": token,
        "pid": state.get("pid"), "pid_alive": alive, "elapsed_s": elapsed,
        "total_xlsx": current, "prev_total_xlsx": prev, "hm_linux_xlsx": hm,
        "log_tail": log_tail,
    }
    if status == "done":
        if lmbench_digest is None:
            out["digest"] = {"ok": False, "error": "lmbench_digest unavailable on host"}
        else:
            out["digest"] = lmbench_digest.build_digest(current, hm, prev, top_n=top_n)
    elif status == "failed":
        out["error"] = "process exited before a new total_result xlsx appeared; see log_tail"
    return out


def cmd_digest(total: str, hm_linux: str | None, prev: str | None, top_n: int) -> dict[str, Any]:
    if lmbench_digest is None:
        return {"ok": False, "error": "lmbench_digest unavailable on host"}
    return lmbench_digest.build_digest(total, hm_linux, prev, top_n=top_n)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Windows-side lmbench full-suite runner (detached).")
    p.add_argument("--test-dir", default=os.environ.get("HMOPT_LMBENCH_TEST_DIR", DEFAULT_TEST_DIR))
    sub = p.add_mutually_exclusive_group(required=True)
    sub.add_argument("--start", action="store_true", help="launch the suite detached")
    sub.add_argument("--status", metavar="RUN_TOKEN", help="poll a started run")
    sub.add_argument("--digest", action="store_true", help="parse given xlsx into a digest")
    p.add_argument("--run-cmd", nargs=argparse.REMAINDER, default=None,
                   help="override the launch command (default: main.py)")
    p.add_argument("--python-exe", default=None)
    p.add_argument("--no-venv", action="store_true")
    p.add_argument("--no-utf8", action="store_true")
    p.add_argument("--top", type=int, default=8)
    # --digest inputs
    p.add_argument("--total")
    p.add_argument("--hm-linux")
    p.add_argument("--prev")
    return p


def main(argv: list[str] | None = None) -> int:
    a = _build_parser().parse_args(argv)
    if a.start:
        res = cmd_start(a.test_dir, a.run_cmd or DEFAULT_RUN_CMD, a.python_exe,
                        use_venv=not a.no_venv, force_utf8=not a.no_utf8)
        ok = res.get("success")
    elif a.status:
        res = cmd_status(a.test_dir, a.status, a.top)
        ok = res.get("success")
    else:
        if not a.total:
            res, ok = {"ok": False, "error": "--digest requires --total"}, False
        else:
            res = cmd_digest(a.total, a.hm_linux, a.prev, a.top)
            ok = res.get("ok")
    json.dump(res, sys.stdout, ensure_ascii=True, indent=2)
    sys.stdout.write("\n")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
