"""Compare two modelCase test report directories on instruction count.

Usage:
  python report_compare.py --baseline <dir> --candidate <dir>
  python report_compare.py --baseline <dir> --candidate <dir> \\
      --depth function --top-n 20

Report layout produced by the modelCase test pipeline looks like::

  <report_dir>/
    PerfLoad_<case>_round<N>/
      hiperf/
        step<M>/
          PerfLoad_<case>_round<N>_step<M>_nosym_hiperfReport.xlsx

Each workbook has four worksheets:

  1. ``process_instructions_info``
       pid | processName | count
       (first data row has pid="TOTAL" — that row's count is the overall
       instruction total for the run.)
  2. ``thread_instructions_info``
       pid | processName | processCount | tid | threadName | threadCount
  3. ``lib_instructions_info``
       pid | processName | processCount | tid | threadName | threadCount |
       fileId | libName | libCount
  4. ``functions_instructions_info``
       pid | processName | processCount | tid | threadName | threadCount |
       fileId | libName | libCount | functionId | functionName |
       functionCount_self | functionCount_total

The script walks both report dirs, pairs up workbooks with the same
``(case, round, step)`` triple, and for each pair reports:

  - baseline_total / candidate_total / delta / delta_pct (from the TOTAL
    row on page 1)
  - top_processes: per-process diff (aggregated by processName)
  - optionally per-thread / per-lib / per-function diffs, depending on
    the ``--depth`` argument

An aggregate across all pairs is also reported so a single number can be
surfaced upstream.

Requires ``openpyxl`` on the executing host (``pip install openpyxl``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Iterable

try:  # pragma: no cover — import is trivial, availability is what we test
    from openpyxl import load_workbook  # type: ignore
except ImportError:  # pragma: no cover
    load_workbook = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Report-file discovery
# ---------------------------------------------------------------------------

REPORT_FILE_PATTERN = re.compile(
    r"^PerfLoad_(?P<case>.+?)_round(?P<round>\d+)_step(?P<step>\d+)"
    r"_nosym_hiperfReport\.xlsx$",
    re.IGNORECASE,
)


def find_reports(report_dir: str) -> list[dict[str, Any]]:
    """Walk *report_dir* and return every matching hiperf xlsx."""
    if not os.path.isdir(report_dir):
        return []
    found: list[dict[str, Any]] = []
    for root, _dirs, files in os.walk(report_dir):
        for name in files:
            match = REPORT_FILE_PATTERN.match(name)
            if not match:
                continue
            found.append({
                "case": match.group("case"),
                "round": int(match.group("round")),
                "step": int(match.group("step")),
                "path": os.path.join(root, name),
            })
    found.sort(key=lambda r: (r["case"], r["round"], r["step"]))
    return found


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------

def _to_int(value: Any) -> int | None:
    """Best-effort coerce a cell value to int.

    Handles plain ints, floats (Excel often stores integers as floats
    and may display them in scientific notation, e.g. 1.01886e+11), and
    numeric strings with commas or underscores.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        return int(round(value))
    if isinstance(value, str):
        text = value.replace(",", "").replace("_", "").strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            try:
                return int(round(float(text)))
            except ValueError:
                return None
    return None


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


# ---------------------------------------------------------------------------
# Worksheet parsing
# ---------------------------------------------------------------------------

PROCESS_SHEET = "process_instructions_info"
THREAD_SHEET = "thread_instructions_info"
LIB_SHEET = "lib_instructions_info"
FUNCTION_SHEET = "functions_instructions_info"


def _iter_data_rows(ws: Any, min_cols: int) -> Iterable[tuple]:
    """Yield data rows (header skipped) padded to at least *min_cols*."""
    for idx, row in enumerate(ws.iter_rows(values_only=True)):
        if idx == 0:  # header
            continue
        if not row or all(v is None or _clean_str(v) == "" for v in row):
            continue
        padded = tuple(row) + (None,) * max(0, min_cols - len(row))
        yield padded


def _parse_process_sheet(ws: Any) -> tuple[int | None, dict[str, int]]:
    """Return (total, {processName: sum_count})."""
    total: int | None = None
    processes: dict[str, int] = {}
    for row in _iter_data_rows(ws, min_cols=3):
        pid_raw, process_name, count = row[0], row[1], row[2]
        pid = _clean_str(pid_raw)
        if pid.upper() == "TOTAL":
            total = _to_int(count)
            continue
        value = _to_int(count)
        if value is None:
            continue
        key = _clean_str(process_name) or pid or "<unknown>"
        processes[key] = processes.get(key, 0) + value
    return total, processes


def _parse_thread_sheet(ws: Any) -> dict[str, int]:
    """Return {'<processName>|<threadName>': sum_threadCount}."""
    threads: dict[str, int] = {}
    for row in _iter_data_rows(ws, min_cols=6):
        _pid, process_name, _pcnt, _tid, thread_name, thread_count = row[:6]
        value = _to_int(thread_count)
        if value is None:
            continue
        key = f"{_clean_str(process_name)}|{_clean_str(thread_name)}"
        threads[key] = threads.get(key, 0) + value
    return threads


def _parse_lib_sheet(ws: Any) -> dict[str, int]:
    """Return {'<processName>|<threadName>|<libName>': sum_libCount}."""
    libs: dict[str, int] = {}
    for row in _iter_data_rows(ws, min_cols=9):
        _pid, process_name, _pcnt, _tid, thread_name, _tcnt, _fid, lib_name, lib_count = row[:9]
        value = _to_int(lib_count)
        if value is None:
            continue
        key = "|".join((
            _clean_str(process_name),
            _clean_str(thread_name),
            _clean_str(lib_name),
        ))
        libs[key] = libs.get(key, 0) + value
    return libs


def _parse_function_sheet(ws: Any) -> dict[str, int]:
    """Return {'<processName>|<threadName>|<libName>|<functionName>': functionCount_total}."""
    functions: dict[str, int] = {}
    for row in _iter_data_rows(ws, min_cols=13):
        (
            _pid, process_name, _pcnt, _tid, thread_name, _tcnt,
            _fid, lib_name, _lcnt, _funcid, function_name,
            _fc_self, fc_total,
        ) = row[:13]
        value = _to_int(fc_total)
        if value is None:
            continue
        key = "|".join((
            _clean_str(process_name),
            _clean_str(thread_name),
            _clean_str(lib_name),
            _clean_str(function_name),
        ))
        functions[key] = functions.get(key, 0) + value
    return functions


# ---------------------------------------------------------------------------
# Workbook loading
# ---------------------------------------------------------------------------

DEPTH_CHOICES = ("total", "process", "thread", "lib", "function")


def _ensure_openpyxl() -> None:
    if load_workbook is None:
        raise RuntimeError(
            "openpyxl is required for report_compare.py. "
            "Install it on the Windows host with: pip install openpyxl"
        )


def parse_hiperf_workbook(path: str, depth: str = "process") -> dict[str, Any]:
    """Load a single hiperf xlsx and return total + breakdown maps.

    *depth* controls how many sheets are read; the TOTAL from page 1 is
    always returned regardless of depth.
    """
    _ensure_openpyxl()
    if depth not in DEPTH_CHOICES:
        raise ValueError(f"invalid depth {depth!r}; choose from {DEPTH_CHOICES}")

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        out: dict[str, Any] = {
            "path": path,
            "total": None,
            "processes": {},
            "threads": {},
            "libs": {},
            "functions": {},
        }
        if PROCESS_SHEET in wb.sheetnames:
            total, processes = _parse_process_sheet(wb[PROCESS_SHEET])
            out["total"] = total
            out["processes"] = processes

        if depth in ("thread", "lib", "function") and THREAD_SHEET in wb.sheetnames:
            out["threads"] = _parse_thread_sheet(wb[THREAD_SHEET])
        if depth in ("lib", "function") and LIB_SHEET in wb.sheetnames:
            out["libs"] = _parse_lib_sheet(wb[LIB_SHEET])
        if depth == "function" and FUNCTION_SHEET in wb.sheetnames:
            out["functions"] = _parse_function_sheet(wb[FUNCTION_SHEET])
    finally:
        wb.close()

    return out


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------

def _pct(delta: int, baseline: int | None) -> float | None:
    if not baseline:
        return None
    return round(delta / baseline * 100, 4)


def diff_maps(
    baseline: dict[str, int],
    candidate: dict[str, int],
    *,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Return per-key diffs sorted by ``abs(delta)`` descending, capped at *top_n*."""
    rows: list[dict[str, Any]] = []
    for key in sorted(set(baseline) | set(candidate)):
        b = baseline.get(key, 0)
        c = candidate.get(key, 0)
        delta = c - b
        if delta == 0 and b == 0 and c == 0:
            continue
        rows.append({
            "key": key,
            "baseline": b,
            "candidate": c,
            "delta": delta,
            "delta_pct": _pct(delta, b),
        })
    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    if top_n is not None and top_n > 0:
        rows = rows[:top_n]
    return rows


# ---------------------------------------------------------------------------
# Top-level compare
# ---------------------------------------------------------------------------

def _pair_key(report: dict[str, Any]) -> tuple[str, int, int]:
    return (report["case"], report["round"], report["step"])


def compare_reports(
    *,
    baseline_dir: str,
    candidate_dir: str,
    depth: str = "process",
    top_n: int = 20,
) -> dict[str, Any]:
    """Compare two report directories and summarize the per-pair delta.

    Pairs workbooks by ``(case, round, step)``.  Missing members are
    reported with ``missing: baseline|candidate``.
    """
    if depth not in DEPTH_CHOICES:
        raise ValueError(f"invalid depth {depth!r}; choose from {DEPTH_CHOICES}")

    baseline_reports = find_reports(baseline_dir)
    candidate_reports = find_reports(candidate_dir)
    b_map = {_pair_key(r): r for r in baseline_reports}
    c_map = {_pair_key(r): r for r in candidate_reports}
    all_keys = sorted(set(b_map) | set(c_map))

    pair_results: list[dict[str, Any]] = []
    agg_baseline = 0
    agg_candidate = 0
    totals_complete = True
    pair_errors = False

    for key in all_keys:
        entry: dict[str, Any] = {
            "case": key[0],
            "round": key[1],
            "step": key[2],
        }
        b = b_map.get(key)
        c = c_map.get(key)

        if b is None or c is None:
            entry["missing"] = "baseline" if b is None else "candidate"
            entry["baseline_path"] = b["path"] if b else None
            entry["candidate_path"] = c["path"] if c else None
            pair_errors = True
            pair_results.append(entry)
            continue

        entry["baseline_path"] = b["path"]
        entry["candidate_path"] = c["path"]

        try:
            b_data = parse_hiperf_workbook(b["path"], depth=depth)
            c_data = parse_hiperf_workbook(c["path"], depth=depth)
        except Exception as exc:  # noqa: BLE001 — surface parse errors per-pair
            entry["error"] = f"failed to parse xlsx: {exc}"
            pair_errors = True
            pair_results.append(entry)
            continue

        b_total = b_data.get("total")
        c_total = c_data.get("total")
        entry["baseline_total"] = b_total
        entry["candidate_total"] = c_total
        if b_total is not None and c_total is not None:
            delta = c_total - b_total
            entry["delta"] = delta
            entry["delta_pct"] = _pct(delta, b_total)
            agg_baseline += b_total
            agg_candidate += c_total
        else:
            totals_complete = False

        if depth in ("process", "thread", "lib", "function"):
            entry["top_processes"] = diff_maps(
                b_data.get("processes", {}),
                c_data.get("processes", {}),
                top_n=top_n,
            )
        if depth in ("thread", "lib", "function"):
            entry["top_threads"] = diff_maps(
                b_data.get("threads", {}),
                c_data.get("threads", {}),
                top_n=top_n,
            )
        if depth in ("lib", "function"):
            entry["top_libs"] = diff_maps(
                b_data.get("libs", {}),
                c_data.get("libs", {}),
                top_n=top_n,
            )
        if depth == "function":
            entry["top_functions"] = diff_maps(
                b_data.get("functions", {}),
                c_data.get("functions", {}),
                top_n=top_n,
            )

        pair_results.append(entry)

    agg_delta = agg_candidate - agg_baseline
    aggregate = {
        "baseline_total": agg_baseline,
        "candidate_total": agg_candidate,
        "delta": agg_delta,
        "delta_pct": _pct(agg_delta, agg_baseline),
        "pairs_compared": sum(1 for r in pair_results if "missing" not in r and "error" not in r),
        "pairs_missing_baseline": sum(1 for r in pair_results if r.get("missing") == "baseline"),
        "pairs_missing_candidate": sum(1 for r in pair_results if r.get("missing") == "candidate"),
    }

    return {
        "success": totals_complete and not pair_errors and bool(pair_results),
        "baseline_dir": os.path.abspath(baseline_dir),
        "candidate_dir": os.path.abspath(candidate_dir),
        "depth": depth,
        "top_n": top_n,
        "aggregate": aggregate,
        "reports": pair_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two modelCase test report directories on instruction count.",
    )
    parser.add_argument("--baseline", required=True, help="Baseline (stock) report directory.")
    parser.add_argument("--candidate", required=True, help="Candidate (feature) report directory.")
    parser.add_argument(
        "--depth",
        choices=DEPTH_CHOICES,
        default="process",
        help=(
            "How deep to diff.  'total' = only TOTAL row; 'process' (default) "
            "adds per-process; 'thread' adds per-thread; 'lib' adds per-library; "
            "'function' adds per-function (functionCount_total)."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Keep top N entries by abs(delta) in each breakdown (default 20).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = compare_reports(
            baseline_dir=args.baseline,
            candidate_dir=args.candidate,
            depth=args.depth,
            top_n=args.top_n,
        )
    except RuntimeError as exc:  # openpyxl missing
        json.dump({"success": False, "error": str(exc)}, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 1
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
