"""Compare two modelCase test report directories on instruction count.

Usage:
  python report_compare.py --baseline <dir> --candidate <dir>
  python report_compare.py --baseline <dir> --candidate <dir> \
      --metric-file summary.json --metric-key total_instructions

The script tries several extraction strategies in order:

  1. Explicit ``--metric-file`` + ``--metric-key``: read the named JSON
     file from each report directory and extract the given key
     (supports dot-notation paths, e.g. ``summary.total_instructions``).
  2. Well-known filenames scanned for well-known keys
     (``summary.json``, ``metrics.json``, ``report.json``,
     ``instruction_count.json``, ``result.json``).
  3. Regex scan of ``.txt`` / ``.log`` / ``.csv`` / ``.out`` / ``.md``
     files for patterns like ``Total Instructions: 12345``.

If the JSON contains a list/dict of per-test-case entries each with an
instruction count, a per-case breakdown is included in the output.

Zero external dependencies — only Python stdlib.  Always emits a single
JSON document to stdout so the caller (Linux MCP) can parse it
deterministically.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Known conventions
# ---------------------------------------------------------------------------

DEFAULT_SUMMARY_FILES: tuple[str, ...] = (
    "summary.json",
    "metrics.json",
    "report.json",
    "instruction_count.json",
    "result.json",
    "results.json",
)

# Top-level keys that may hold an overall instruction count.
DEFAULT_TOTAL_KEYS: tuple[str, ...] = (
    "total_instructions",
    "total_instruction_count",
    "instruction_count",
    "instructions",
    "inst_count",
    "total_inst",
    "total_inst_count",
)

# Regex patterns used when scanning plain-text files.
TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)total\s+instructions?\s*[:=]\s*([0-9][0-9_,]*)"),
    re.compile(r"(?i)instruction\s*count\s*[:=]\s*([0-9][0-9_,]*)"),
    re.compile(r"(?i)\binstructions?\b\s*[:=]\s*([0-9][0-9_,]*)"),
    re.compile(r"(?i)inst[_\s-]*count\s*[:=]\s*([0-9][0-9_,]*)"),
)

TEXT_EXTENSIONS: tuple[str, ...] = (".txt", ".log", ".csv", ".out", ".md", ".tsv")


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _parse_int(value: Any) -> int | None:
    """Best-effort coerce to int (handle strings with commas/underscores)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == int(value) else None
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("_", "").strip()
        if cleaned.isdigit():
            return int(cleaned)
        try:
            return int(float(cleaned))
        except ValueError:
            return None
    return None


def _walk_json(obj: Any, path: str = "") -> list[tuple[str, Any]]:
    """Flatten a JSON object into (dot-path, value) pairs."""
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else str(key)
            out.append((child_path, value))
            out.extend(_walk_json(value, child_path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            child_path = f"{path}[{idx}]"
            out.append((child_path, value))
            out.extend(_walk_json(value, child_path))
    return out


def _get_by_dot_path(obj: Any, dot_path: str) -> Any:
    """Look up a dot-notation path in a nested dict. Returns None if missing."""
    cursor: Any = obj
    for segment in dot_path.split("."):
        if isinstance(cursor, dict) and segment in cursor:
            cursor = cursor[segment]
        else:
            return None
    return cursor


def _find_total_in_json(data: Any, keys: tuple[str, ...]) -> tuple[int | None, str | None]:
    """Search a loaded JSON object for a known total-instruction key."""
    # Try top-level keys first.
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                parsed = _parse_int(data[key])
                if parsed is not None:
                    return parsed, key

    # Then any nested key whose leaf name matches.
    for path, value in _walk_json(data):
        leaf = path.rsplit(".", 1)[-1].split("[")[0]
        if leaf in keys:
            parsed = _parse_int(value)
            if parsed is not None:
                return parsed, path
    return None, None


def _extract_per_case(data: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Best-effort extraction of per-test-case instruction counts from JSON.

    Looks for lists or dicts where each entry has both a name-like field
    and one of the known instruction-count keys.
    """
    cases: list[dict[str, Any]] = []

    def add_case(name: Any, value: Any) -> None:
        parsed = _parse_int(value)
        if parsed is None:
            return
        case_name = str(name) if name is not None else f"case_{len(cases)}"
        cases.append({"case": case_name, "instructions": parsed})

    # Shape 1: top-level list of case dicts.
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            name = entry.get("case") or entry.get("name") or entry.get("test") or entry.get("test_case")
            for key in keys:
                if key in entry:
                    add_case(name, entry[key])
                    break

    # Shape 2: dict of {case_name: {instruction_count: N}} or {case_name: N}.
    if isinstance(data, dict):
        for name, value in data.items():
            if isinstance(value, dict):
                for key in keys:
                    if key in value:
                        add_case(name, value[key])
                        break
            elif any(marker in str(name).lower() for marker in ("instructions", "inst_count", "instruction_count")):
                continue  # skip obvious non-case keys
            elif isinstance(value, (int, float, str)):
                # Only treat as a case if it looks like a plain number.
                # Avoid swallowing unrelated top-level metadata.
                parsed = _parse_int(value)
                if parsed is not None and "case" in str(name).lower():
                    add_case(name, value)

        # Shape 3: dict with a "cases"/"tests" sub-list.
        for container_key in ("cases", "tests", "test_cases", "results"):
            container = data.get(container_key)
            if isinstance(container, list):
                for entry in container:
                    if not isinstance(entry, dict):
                        continue
                    name = entry.get("case") or entry.get("name") or entry.get("test") or entry.get("test_case")
                    for key in keys:
                        if key in entry:
                            add_case(name, entry[key])
                            break

    return cases


def _extract_from_text(path: str) -> tuple[int | None, str | None]:
    """Scan a plain text file for an instruction-count line and return the
    first match as an int."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
    except OSError:
        return None, None

    for pattern in TEXT_PATTERNS:
        match = pattern.search(content)
        if match:
            parsed = _parse_int(match.group(1))
            if parsed is not None:
                return parsed, pattern.pattern
    return None, None


def _iter_candidate_files(report_dir: str) -> list[str]:
    """Enumerate files in a report dir, depth-first, returning absolute paths."""
    results: list[str] = []
    for root, _dirs, files in os.walk(report_dir):
        for name in files:
            results.append(os.path.join(root, name))
    return results


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_instruction_count(
    report_dir: str,
    *,
    metric_file: str | None = None,
    metric_key: str | None = None,
    metric_pattern: str | None = None,
    total_keys: tuple[str, ...] = DEFAULT_TOTAL_KEYS,
    summary_files: tuple[str, ...] = DEFAULT_SUMMARY_FILES,
) -> dict[str, Any]:
    """Extract a total instruction count and per-case breakdown from *report_dir*.

    Returns a dict with fields:
      total: int | None
      source: relative path of file used for extraction (or None)
      strategy: one of {"explicit", "summary_key", "json_scan", "text_scan"}
      per_case: list of {"case": str, "instructions": int}
      notes: list of human-readable messages
    """
    report_dir = os.path.abspath(report_dir)
    notes: list[str] = []
    per_case: list[dict[str, Any]] = []

    if not os.path.isdir(report_dir):
        return {
            "total": None,
            "source": None,
            "strategy": None,
            "per_case": [],
            "notes": [f"report directory does not exist: {report_dir}"],
        }

    # ── Strategy 1: explicit metric file + key (or pattern) ──────────────
    if metric_file:
        abs_file = os.path.join(report_dir, metric_file)
        if not os.path.isfile(abs_file):
            notes.append(f"explicit metric file not found: {abs_file}")
        else:
            ext = os.path.splitext(abs_file)[1].lower()
            if ext == ".json":
                try:
                    with open(abs_file, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except (OSError, json.JSONDecodeError) as exc:
                    notes.append(f"failed to parse {metric_file}: {exc}")
                else:
                    if metric_key:
                        value = _get_by_dot_path(data, metric_key)
                        parsed = _parse_int(value)
                        if parsed is not None:
                            per_case = _extract_per_case(data, total_keys)
                            return {
                                "total": parsed,
                                "source": os.path.relpath(abs_file, report_dir),
                                "strategy": "explicit",
                                "key": metric_key,
                                "per_case": per_case,
                                "notes": notes,
                            }
                        notes.append(f"explicit metric_key {metric_key!r} not found in {metric_file}")
                    else:
                        total, key = _find_total_in_json(data, total_keys)
                        if total is not None:
                            per_case = _extract_per_case(data, total_keys)
                            return {
                                "total": total,
                                "source": os.path.relpath(abs_file, report_dir),
                                "strategy": "summary_key",
                                "key": key,
                                "per_case": per_case,
                                "notes": notes,
                            }
                        notes.append(f"no known total key found in {metric_file}")
            else:
                # Text file: use pattern or defaults.
                if metric_pattern:
                    try:
                        pat = re.compile(metric_pattern)
                    except re.error as exc:
                        notes.append(f"invalid --metric-pattern: {exc}")
                        pat = None
                    if pat is not None:
                        try:
                            with open(abs_file, "r", encoding="utf-8", errors="replace") as fh:
                                content = fh.read()
                        except OSError as exc:
                            notes.append(f"failed to read {metric_file}: {exc}")
                        else:
                            match = pat.search(content)
                            if match and match.groups():
                                parsed = _parse_int(match.group(1))
                                if parsed is not None:
                                    return {
                                        "total": parsed,
                                        "source": os.path.relpath(abs_file, report_dir),
                                        "strategy": "explicit",
                                        "pattern": metric_pattern,
                                        "per_case": per_case,
                                        "notes": notes,
                                    }
                            notes.append(f"pattern did not match in {metric_file}")
                total, pat_name = _extract_from_text(abs_file)
                if total is not None:
                    return {
                        "total": total,
                        "source": os.path.relpath(abs_file, report_dir),
                        "strategy": "text_scan",
                        "pattern": pat_name,
                        "per_case": per_case,
                        "notes": notes,
                    }
                notes.append(f"no known instruction-count pattern found in {metric_file}")

    # ── Strategy 2: well-known summary filenames at top of report dir ────
    for name in summary_files:
        abs_file = os.path.join(report_dir, name)
        if not os.path.isfile(abs_file):
            continue
        try:
            with open(abs_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            notes.append(f"skipping {name}: {exc}")
            continue
        total, key = _find_total_in_json(data, total_keys)
        if total is not None:
            return {
                "total": total,
                "source": os.path.relpath(abs_file, report_dir),
                "strategy": "summary_key",
                "key": key,
                "per_case": _extract_per_case(data, total_keys),
                "notes": notes,
            }

    # ── Strategy 3: scan every JSON file anywhere under the dir ──────────
    for path in _iter_candidate_files(report_dir):
        if not path.lower().endswith(".json"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        total, key = _find_total_in_json(data, total_keys)
        if total is not None:
            return {
                "total": total,
                "source": os.path.relpath(path, report_dir),
                "strategy": "json_scan",
                "key": key,
                "per_case": _extract_per_case(data, total_keys),
                "notes": notes,
            }

    # ── Strategy 4: regex scan of plain-text files ───────────────────────
    for path in _iter_candidate_files(report_dir):
        if not path.lower().endswith(TEXT_EXTENSIONS):
            continue
        total, pat_name = _extract_from_text(path)
        if total is not None:
            return {
                "total": total,
                "source": os.path.relpath(path, report_dir),
                "strategy": "text_scan",
                "pattern": pat_name,
                "per_case": per_case,
                "notes": notes,
            }

    notes.append("no instruction-count metric could be extracted")
    return {
        "total": None,
        "source": None,
        "strategy": None,
        "per_case": [],
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _diff_per_case(
    baseline_cases: list[dict[str, Any]],
    candidate_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return per-case diffs when the two sets of cases share names."""
    base_map = {str(c["case"]): int(c["instructions"]) for c in baseline_cases}
    cand_map = {str(c["case"]): int(c["instructions"]) for c in candidate_cases}
    names = sorted(set(base_map) | set(cand_map))
    diffs: list[dict[str, Any]] = []
    for name in names:
        base = base_map.get(name)
        cand = cand_map.get(name)
        entry: dict[str, Any] = {"case": name, "baseline": base, "candidate": cand}
        if base is not None and cand is not None:
            entry["delta"] = cand - base
            entry["delta_pct"] = round((cand - base) / base * 100, 4) if base else None
        diffs.append(entry)
    return diffs


def compare_reports(
    *,
    baseline_dir: str,
    candidate_dir: str,
    metric_file: str | None = None,
    metric_key: str | None = None,
    metric_pattern: str | None = None,
) -> dict[str, Any]:
    """Compare baseline vs candidate report directories and summarize delta."""
    baseline = extract_instruction_count(
        baseline_dir,
        metric_file=metric_file,
        metric_key=metric_key,
        metric_pattern=metric_pattern,
    )
    candidate = extract_instruction_count(
        candidate_dir,
        metric_file=metric_file,
        metric_key=metric_key,
        metric_pattern=metric_pattern,
    )

    b_total = baseline.get("total")
    c_total = candidate.get("total")

    delta: int | None = None
    delta_pct: float | None = None
    if b_total is not None and c_total is not None:
        delta = c_total - b_total
        delta_pct = round((c_total - b_total) / b_total * 100, 4) if b_total else None

    success = b_total is not None and c_total is not None
    return {
        "success": success,
        "baseline_dir": os.path.abspath(baseline_dir),
        "candidate_dir": os.path.abspath(candidate_dir),
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "delta_pct": delta_pct,
        "per_case_diff": _diff_per_case(
            baseline.get("per_case", []),
            candidate.get("per_case", []),
        ),
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
        "--metric-file",
        default=None,
        help="Optional relative filename inside each report dir to read first (e.g. summary.json).",
    )
    parser.add_argument(
        "--metric-key",
        default=None,
        help="Optional dot-path key inside the metric JSON (e.g. total.instructions).",
    )
    parser.add_argument(
        "--metric-pattern",
        default=None,
        help="Optional regex with one capture group for text-file extraction.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = compare_reports(
        baseline_dir=args.baseline,
        candidate_dir=args.candidate,
        metric_file=args.metric_file,
        metric_key=args.metric_key,
        metric_pattern=args.metric_pattern,
    )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
