"""Tests for tools/windows_relay/report_compare.py (hiperf xlsx version)."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "windows_relay"))

import report_compare  # type: ignore  # noqa: E402

try:
    from openpyxl import Workbook  # type: ignore
except ImportError:  # pragma: no cover — skip the whole module
    Workbook = None  # type: ignore


pytestmark = pytest.mark.skipif(Workbook is None, reason="openpyxl not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_hiperf_xlsx(
    path: Path,
    *,
    total: int,
    processes: list[tuple[str | int, str, int]],
    threads: list[tuple] | None = None,
    libs: list[tuple] | None = None,
    functions: list[tuple] | None = None,
) -> None:
    """Create a hiperf-style workbook with four sheets.

    *processes* entries are (pid, processName, count).  Other lists may be
    None — the sheet is still created (headers only) so the parser sees
    it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()

    # page 1
    ws1 = wb.active
    ws1.title = "process_instructions_info"
    ws1.append(["pid", "processName", "count"])
    ws1.append(["TOTAL", "", total])
    for pid, name, count in processes:
        ws1.append([pid, name, count])

    # page 2
    ws2 = wb.create_sheet("thread_instructions_info")
    ws2.append(["pid", "processName", "processCount", "tid", "threadName", "threadCount"])
    for row in threads or []:
        ws2.append(list(row))

    # page 3
    ws3 = wb.create_sheet("lib_instructions_info")
    ws3.append([
        "pid", "processName", "processCount", "tid", "threadName", "threadCount",
        "fileId", "libName", "libCount",
    ])
    for row in libs or []:
        ws3.append(list(row))

    # page 4
    ws4 = wb.create_sheet("functions_instructions_info")
    ws4.append([
        "pid", "processName", "processCount", "tid", "threadName", "threadCount",
        "fileId", "libName", "libCount",
        "functionId", "functionName", "functionCount_self", "functionCount_total",
    ])
    for row in functions or []:
        ws4.append(list(row))

    wb.save(path)


def _make_report_tree(
    root: Path,
    *,
    case: str,
    round_n: int,
    step: int,
    total: int,
    processes: list[tuple[str | int, str, int]],
    **extra,
) -> Path:
    """Mirror the production layout:
    <root>/PerfLoad_<case>_round<N>/hiperf/step<M>/<xlsx>
    """
    case_dir = root / f"PerfLoad_{case}_round{round_n}" / "hiperf" / f"step{step}"
    xlsx = case_dir / f"PerfLoad_{case}_round{round_n}_step{step}_nosym_hiperfReport.xlsx"
    _write_hiperf_xlsx(xlsx, total=total, processes=processes, **extra)
    return xlsx


class _TmpBase(unittest.TestCase):
    def setUp(self) -> None:
        self._root = Path(tempfile.mkdtemp(prefix="hmopt_hiperf_cmp_"))
        self.addCleanup(shutil.rmtree, self._root, ignore_errors=True)

    def tmpdir(self, name: str) -> Path:
        d = self._root / name
        d.mkdir(parents=True, exist_ok=True)
        return d


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------

class ToIntTests(unittest.TestCase):
    def test_int(self) -> None:
        self.assertEqual(report_compare._to_int(1000), 1000)

    def test_float_scientific(self) -> None:
        # 1.01886e+11 should round-trip to 101_886_000_000
        self.assertEqual(report_compare._to_int(1.01886e11), 101_886_000_000)

    def test_comma_string(self) -> None:
        self.assertEqual(report_compare._to_int("1,234,567"), 1_234_567)

    def test_nan_rejected(self) -> None:
        self.assertIsNone(report_compare._to_int(float("nan")))

    def test_empty_rejected(self) -> None:
        self.assertIsNone(report_compare._to_int(""))
        self.assertIsNone(report_compare._to_int(None))


# ---------------------------------------------------------------------------
# find_reports
# ---------------------------------------------------------------------------

class FindReportsTests(_TmpBase):
    def test_walks_report_dir(self) -> None:
        d = self.tmpdir("reports")
        _make_report_tree(d, case="bilibili_0020", round_n=0, step=1, total=10, processes=[(1, "init", 10)])
        _make_report_tree(d, case="bilibili_0020", round_n=0, step=2, total=20, processes=[(1, "init", 20)])
        _make_report_tree(d, case="bilibili_0020", round_n=1, step=1, total=30, processes=[(1, "init", 30)])

        found = report_compare.find_reports(str(d))
        triples = [(r["case"], r["round"], r["step"]) for r in found]
        self.assertEqual(
            triples,
            [("bilibili_0020", 0, 1), ("bilibili_0020", 0, 2), ("bilibili_0020", 1, 1)],
        )

    def test_missing_directory_returns_empty(self) -> None:
        self.assertEqual(report_compare.find_reports(str(self._root / "nope")), [])

    def test_case_name_with_underscores_parsed_correctly(self) -> None:
        d = self.tmpdir("reports2")
        _make_report_tree(d, case="my_complex_case_0042", round_n=3, step=5, total=1, processes=[(1, "x", 1)])
        [entry] = report_compare.find_reports(str(d))
        self.assertEqual(entry["case"], "my_complex_case_0042")
        self.assertEqual(entry["round"], 3)
        self.assertEqual(entry["step"], 5)


# ---------------------------------------------------------------------------
# parse_hiperf_workbook
# ---------------------------------------------------------------------------

class ParseWorkbookTests(_TmpBase):
    def test_process_sheet_and_total(self) -> None:
        xlsx = self._root / "w.xlsx"
        _write_hiperf_xlsx(
            xlsx,
            total=101_886_000_000,
            processes=[
                (0, "swapper", 4_069_975_650),
                (1, "init", 3_417_038),
                (2, "[sysmgr-main]", 2_926_131_503),
            ],
        )
        out = report_compare.parse_hiperf_workbook(str(xlsx), depth="process")
        self.assertEqual(out["total"], 101_886_000_000)
        self.assertEqual(out["processes"]["swapper"], 4_069_975_650)
        self.assertEqual(out["processes"]["[sysmgr-main]"], 2_926_131_503)
        # threads/libs/functions empty at depth=process.
        self.assertEqual(out["threads"], {})

    def test_total_depth_skips_deeper_sheets(self) -> None:
        xlsx = self._root / "t.xlsx"
        _write_hiperf_xlsx(
            xlsx,
            total=100,
            processes=[(1, "init", 100)],
            threads=[(1, "init", 100, 1, "/bin/init", 100)],
        )
        out = report_compare.parse_hiperf_workbook(str(xlsx), depth="total")
        self.assertEqual(out["total"], 100)
        self.assertEqual(out["processes"], {"init": 100})
        self.assertEqual(out["threads"], {})

    def test_thread_lib_function_depth(self) -> None:
        xlsx = self._root / "deep.xlsx"
        _write_hiperf_xlsx(
            xlsx,
            total=10,
            processes=[(1, "init", 10)],
            threads=[(1, "init", 10, 1, "/bin/init", 10)],
            libs=[(1, "init", 10, 1, "/bin/init", 10, 2, "ld-musl.so", 5)],
            functions=[
                (1, "init", 10, 1, "/bin/init", 10, 2, "ld-musl.so", 5,
                 59, "libc_start_main_stage2", 0, 10),
                (1, "init", 10, 1, "/bin/init", 10, 2, "ld-musl.so", 5,
                 182, "strlen", 3, 3),
            ],
        )
        out = report_compare.parse_hiperf_workbook(str(xlsx), depth="function")
        self.assertIn("init|/bin/init", out["threads"])
        self.assertIn("init|/bin/init|ld-musl.so", out["libs"])
        # functionCount_total is what we key on.
        self.assertEqual(out["functions"]["init|/bin/init|ld-musl.so|libc_start_main_stage2"], 10)
        self.assertEqual(out["functions"]["init|/bin/init|ld-musl.so|strlen"], 3)

    def test_process_names_aggregate_when_pid_differs(self) -> None:
        xlsx = self._root / "agg.xlsx"
        _write_hiperf_xlsx(
            xlsx,
            total=30,
            processes=[
                (10, "samgr", 12),
                (11, "samgr", 18),  # same processName, different pid
            ],
        )
        out = report_compare.parse_hiperf_workbook(str(xlsx), depth="process")
        self.assertEqual(out["processes"]["samgr"], 30)


# ---------------------------------------------------------------------------
# diff_maps
# ---------------------------------------------------------------------------

class DiffMapsTests(unittest.TestCase):
    def test_sorted_by_abs_delta_descending(self) -> None:
        diffs = report_compare.diff_maps(
            {"a": 100, "b": 50, "c": 10},
            {"a": 90, "b": 80, "c": 11},
            top_n=20,
        )
        keys = [d["key"] for d in diffs]
        self.assertEqual(keys, ["b", "a", "c"])

    def test_top_n_caps_output(self) -> None:
        b = {f"k{i}": 100 for i in range(10)}
        c = {f"k{i}": 100 + (10 - i) for i in range(10)}  # descending deltas
        diffs = report_compare.diff_maps(b, c, top_n=3)
        self.assertEqual(len(diffs), 3)
        # Largest deltas first: k0 (+10), k1 (+9), k2 (+8).
        self.assertEqual([d["key"] for d in diffs], ["k0", "k1", "k2"])

    def test_new_and_disappeared_keys(self) -> None:
        diffs = report_compare.diff_maps({"x": 50}, {"y": 30}, top_n=20)
        by_key = {d["key"]: d for d in diffs}
        self.assertEqual(by_key["x"]["baseline"], 50)
        self.assertEqual(by_key["x"]["candidate"], 0)
        self.assertEqual(by_key["x"]["delta"], -50)
        self.assertEqual(by_key["y"]["delta"], 30)


# ---------------------------------------------------------------------------
# compare_reports
# ---------------------------------------------------------------------------

class CompareReportsTests(_TmpBase):
    def test_single_pair_happy_path(self) -> None:
        base = self.tmpdir("base")
        cand = self.tmpdir("cand")
        _make_report_tree(
            base, case="bilibili_0020", round_n=0, step=1,
            total=1000, processes=[(1, "init", 400), (2, "samgr", 600)],
        )
        _make_report_tree(
            cand, case="bilibili_0020", round_n=0, step=1,
            total=900, processes=[(1, "init", 350), (2, "samgr", 550)],
        )

        result = report_compare.compare_reports(
            baseline_dir=str(base), candidate_dir=str(cand),
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["aggregate"]["delta"], -100)
        self.assertEqual(result["aggregate"]["delta_pct"], -10.0)
        self.assertEqual(result["aggregate"]["pairs_compared"], 1)
        [pair] = result["reports"]
        self.assertEqual(pair["case"], "bilibili_0020")
        self.assertEqual(pair["baseline_total"], 1000)
        self.assertEqual(pair["candidate_total"], 900)
        top = {p["key"]: p for p in pair["top_processes"]}
        self.assertEqual(top["init"]["delta"], -50)
        self.assertEqual(top["samgr"]["delta"], -50)

    def test_multiple_pairs_sum_to_aggregate(self) -> None:
        base = self.tmpdir("base2")
        cand = self.tmpdir("cand2")
        _make_report_tree(base, case="a", round_n=0, step=1, total=10, processes=[(1, "p", 10)])
        _make_report_tree(base, case="b", round_n=0, step=1, total=20, processes=[(1, "p", 20)])
        _make_report_tree(cand, case="a", round_n=0, step=1, total=12, processes=[(1, "p", 12)])
        _make_report_tree(cand, case="b", round_n=0, step=1, total=15, processes=[(1, "p", 15)])

        result = report_compare.compare_reports(
            baseline_dir=str(base), candidate_dir=str(cand),
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["aggregate"]["baseline_total"], 30)
        self.assertEqual(result["aggregate"]["candidate_total"], 27)
        self.assertEqual(result["aggregate"]["delta"], -3)
        self.assertEqual(result["aggregate"]["pairs_compared"], 2)

    def test_missing_on_one_side_flagged(self) -> None:
        base = self.tmpdir("base3")
        cand = self.tmpdir("cand3")
        _make_report_tree(base, case="a", round_n=0, step=1, total=10, processes=[(1, "p", 10)])
        _make_report_tree(base, case="b", round_n=0, step=1, total=10, processes=[(1, "p", 10)])
        _make_report_tree(cand, case="a", round_n=0, step=1, total=11, processes=[(1, "p", 11)])
        # case b is missing in candidate

        result = report_compare.compare_reports(
            baseline_dir=str(base), candidate_dir=str(cand),
        )
        self.assertFalse(result["success"])  # unpaired → not successful overall
        self.assertEqual(result["aggregate"]["pairs_missing_candidate"], 1)
        by_case = {r["case"]: r for r in result["reports"]}
        self.assertEqual(by_case["b"]["missing"], "candidate")

    def test_depth_function_includes_function_breakdown(self) -> None:
        base = self.tmpdir("baseF")
        cand = self.tmpdir("candF")
        common_kwargs = dict(
            processes=[(1, "init", 100)],
            threads=[(1, "init", 100, 1, "/bin/init", 100)],
            libs=[(1, "init", 100, 1, "/bin/init", 100, 2, "ld-musl.so", 100)],
        )
        _make_report_tree(
            base, case="a", round_n=0, step=1, total=100,
            functions=[
                (1, "init", 100, 1, "/bin/init", 100, 2, "ld-musl.so", 100,
                 1, "foo", 0, 60),
                (1, "init", 100, 1, "/bin/init", 100, 2, "ld-musl.so", 100,
                 2, "bar", 0, 40),
            ],
            **common_kwargs,
        )
        _make_report_tree(
            cand, case="a", round_n=0, step=1, total=80,
            functions=[
                (1, "init", 100, 1, "/bin/init", 100, 2, "ld-musl.so", 100,
                 1, "foo", 0, 30),
                (1, "init", 100, 1, "/bin/init", 100, 2, "ld-musl.so", 100,
                 2, "bar", 0, 50),
            ],
            **common_kwargs,
        )

        result = report_compare.compare_reports(
            baseline_dir=str(base), candidate_dir=str(cand), depth="function",
        )
        [pair] = result["reports"]
        self.assertIn("top_threads", pair)
        self.assertIn("top_libs", pair)
        self.assertIn("top_functions", pair)
        func_by_key = {f["key"]: f for f in pair["top_functions"]}
        self.assertEqual(func_by_key["init|/bin/init|ld-musl.so|foo"]["delta"], -30)
        self.assertEqual(func_by_key["init|/bin/init|ld-musl.so|bar"]["delta"], 10)

    def test_cli_main_emits_json(self) -> None:
        base = self.tmpdir("cli_base")
        cand = self.tmpdir("cli_cand")
        _make_report_tree(base, case="a", round_n=0, step=1, total=100, processes=[(1, "p", 100)])
        _make_report_tree(cand, case="a", round_n=0, step=1, total=120, processes=[(1, "p", 120)])

        buf = StringIO()
        original = sys.stdout
        sys.stdout = buf
        try:
            rc = report_compare.main(
                ["--baseline", str(base), "--candidate", str(cand)],
            )
        finally:
            sys.stdout = original
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue().strip())
        self.assertTrue(payload["success"])
        self.assertEqual(payload["aggregate"]["delta"], 20)


if __name__ == "__main__":
    unittest.main()
