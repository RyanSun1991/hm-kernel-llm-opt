"""Tests for tools/windows_relay/instruction_test_pipeline.py.

Focus on the pure helpers that can be exercised in-process on any OS
(the subprocess-invoking parts are covered by the MCP service tests
using a fake relay).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "windows_relay"))

import instruction_test_pipeline as itp  # type: ignore  # noqa: E402


class ParseReportTimestampTests(unittest.TestCase):
    def test_valid_name(self) -> None:
        ts = itp._parse_report_timestamp("report_20260414114948")
        self.assertIsNotNone(ts)
        assert ts is not None
        self.assertEqual(ts, datetime(2026, 4, 14, 11, 49, 48))

    def test_invalid_name_returns_none(self) -> None:
        self.assertIsNone(itp._parse_report_timestamp("notareport"))
        self.assertIsNone(itp._parse_report_timestamp("report_2026"))

    def test_non_digit_suffix_rejected(self) -> None:
        self.assertIsNone(itp._parse_report_timestamp("report_abcdefghijklmn"))


class PickNewReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._root = Path(tempfile.mkdtemp(prefix="hmopt_itp_pick_"))
        self.addCleanup(shutil.rmtree, self._root, ignore_errors=True)
        self.reports = self._root / "reports"
        self.reports.mkdir()

    def _make_report(self, name: str) -> Path:
        d = self.reports / name
        d.mkdir()
        return d

    def test_returns_none_when_dir_missing(self) -> None:
        ghost = self._root / "nope"
        picked = itp._pick_new_report(
            str(ghost),
            baseline_names=set(),
            started_at=datetime.now(),
        )
        self.assertIsNone(picked)

    def test_picks_newest_new_directory(self) -> None:
        self._make_report("report_20260414114948")
        baseline = itp._snapshot_reports(str(self.reports))
        self._make_report("report_20260414120950")
        self._make_report("report_20260414120000")

        picked = itp._pick_new_report(
            str(self.reports),
            baseline_names=baseline,
            started_at=datetime(2026, 4, 14, 12, 9, 49),
        )
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertTrue(picked.endswith("report_20260414120950"))

    def test_fallback_to_recent_when_no_new_names(self) -> None:
        # Pretend both directories existed before the run started, but one of
        # them has a timestamp newer than the started_at grace cutoff.
        self._make_report("report_20260414114948")
        self._make_report("report_20260414120950")
        baseline = itp._snapshot_reports(str(self.reports))

        picked = itp._pick_new_report(
            str(self.reports),
            baseline_names=baseline,
            started_at=datetime(2026, 4, 14, 12, 9, 0),
            grace_seconds=60,
        )
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertTrue(picked.endswith("report_20260414120950"))

    def test_fallback_returns_none_when_nothing_recent(self) -> None:
        self._make_report("report_20200101010101")
        baseline = itp._snapshot_reports(str(self.reports))
        picked = itp._pick_new_report(
            str(self.reports),
            baseline_names=baseline,
            started_at=datetime(2026, 4, 14, 12, 0, 0),
            grace_seconds=60,
        )
        self.assertIsNone(picked)


class RunInstructionTestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._root = Path(tempfile.mkdtemp(prefix="hmopt_itp_run_"))
        self.addCleanup(shutil.rmtree, self._root, ignore_errors=True)

    def test_rejects_missing_test_dir(self) -> None:
        result = itp.run_instruction_test(
            test_dir=str(self._root / "nope"),
            main_script="main.py",
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["phase"], "validate")

    def test_rejects_missing_main_script(self) -> None:
        test_dir = self._root / "ws"
        test_dir.mkdir()
        result = itp.run_instruction_test(
            test_dir=str(test_dir),
            main_script="main.py",
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["phase"], "validate")

    def test_main_failure_surfaces_run_result(self) -> None:
        test_dir = self._root / "ws2"
        test_dir.mkdir()
        (test_dir / "main.py").write_text("# placeholder", encoding="utf-8")

        def fake_run(argv: list, *, cwd: str, timeout_s: int, capture_output: bool = True) -> dict:
            return {
                "ok": False,
                "returncode": 7,
                "stdout": "",
                "stderr": "boom",
                "duration_s": 0.1,
                "command": "python main.py",
            }

        with mock.patch.object(itp, "_run", side_effect=fake_run):
            result = itp.run_instruction_test(
                test_dir=str(test_dir),
                main_script="main.py",
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["phase"], "run_main")
        self.assertEqual(result["run_result"]["returncode"], 7)

    def test_happy_path_locates_new_report(self) -> None:
        test_dir = self._root / "ws3"
        (test_dir / "reports").mkdir(parents=True)
        (test_dir / "main.py").write_text("# placeholder", encoding="utf-8")
        # Pre-existing report should not be picked.
        (test_dir / "reports" / "report_20260101010101").mkdir()

        def fake_run(argv: list, *, cwd: str, timeout_s: int, capture_output: bool = True) -> dict:
            # Simulate the script creating a new report dir during its run.
            stamp = (datetime.now() + timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")
            (Path(cwd) / "reports" / f"report_{stamp}").mkdir()
            return {
                "ok": True,
                "returncode": 0,
                "stdout": "done",
                "stderr": "",
                "duration_s": 0.05,
                "command": "python main.py",
            }

        with mock.patch.object(itp, "_run", side_effect=fake_run):
            result = itp.run_instruction_test(
                test_dir=str(test_dir),
                main_script="main.py",
            )

        self.assertTrue(result["success"], msg=json.dumps(result, indent=2))
        self.assertIsNotNone(result["report_path"])
        self.assertTrue(result["report_name"].startswith("report_"))
        self.assertNotEqual(result["report_name"], "report_20260101010101")


if __name__ == "__main__":
    unittest.main()
