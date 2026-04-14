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


class VenvDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._root = Path(tempfile.mkdtemp(prefix="hmopt_itp_venv_"))
        self.addCleanup(shutil.rmtree, self._root, ignore_errors=True)

    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def test_no_venv_returns_none(self) -> None:
        self.assertIsNone(itp._detect_venv_python(str(self._root)))
        self.assertIsNone(itp._detect_activate_bat(str(self._root)))

    def test_dotvenv_scripts_python_exe_detected(self) -> None:
        # Simulate a Windows-style venv layout next to main.py.
        py = self._root / ".venv" / "Scripts" / "python.exe"
        self._touch(py)
        self._touch(self._root / ".venv" / "Scripts" / "activate.bat")

        self.assertEqual(itp._detect_venv_python(str(self._root)), str(py))
        self.assertEqual(
            itp._detect_activate_bat(str(self._root)),
            str(self._root / ".venv" / "Scripts" / "activate.bat"),
        )

    def test_plain_venv_dir_fallback(self) -> None:
        py = self._root / "venv" / "Scripts" / "python.exe"
        self._touch(py)
        self.assertEqual(itp._detect_venv_python(str(self._root)), str(py))

    def test_posix_venv_layout(self) -> None:
        py = self._root / ".venv" / "bin" / "python"
        self._touch(py)
        self.assertEqual(itp._detect_venv_python(str(self._root)), str(py))

    def test_explicit_venv_dir_override(self) -> None:
        py = self._root / "custom-env" / "bin" / "python"
        self._touch(py)
        # Relative override.
        self.assertEqual(
            itp._detect_venv_python(str(self._root), venv_dir="custom-env"),
            str(py),
        )
        # Absolute override also works.
        self.assertEqual(
            itp._detect_venv_python(str(self._root), venv_dir=str(self._root / "custom-env")),
            str(py),
        )

    def test_dotvenv_preferred_over_venv(self) -> None:
        self._touch(self._root / ".venv" / "Scripts" / "python.exe")
        self._touch(self._root / "venv" / "Scripts" / "python.exe")
        self.assertIn(".venv", itp._detect_venv_python(str(self._root)) or "")

    def test_resolve_python_exe_honors_override(self) -> None:
        self._touch(self._root / ".venv" / "Scripts" / "python.exe")
        exe, venv_python = itp._resolve_python_exe(
            override="C:\\python.exe",
            test_dir=str(self._root),
        )
        self.assertEqual(exe, "C:\\python.exe")
        self.assertIsNone(venv_python)

    def test_resolve_python_exe_uses_venv_when_present(self) -> None:
        target = self._root / ".venv" / "Scripts" / "python.exe"
        self._touch(target)
        exe, venv_python = itp._resolve_python_exe(
            override=None,
            test_dir=str(self._root),
        )
        self.assertEqual(exe, str(target))
        self.assertEqual(venv_python, str(target))

    def test_resolve_python_exe_use_venv_false_skips_detection(self) -> None:
        target = self._root / ".venv" / "Scripts" / "python.exe"
        self._touch(target)
        exe, venv_python = itp._resolve_python_exe(
            override=None,
            test_dir=str(self._root),
            use_venv=False,
        )
        self.assertNotEqual(exe, str(target))
        self.assertIsNone(venv_python)


class RunInstructionTestVenvIntegrationTests(unittest.TestCase):
    """End-to-end: run_instruction_test should pick up the venv python
    and record it in the result."""

    def setUp(self) -> None:
        self._root = Path(tempfile.mkdtemp(prefix="hmopt_itp_venv_e2e_"))
        self.addCleanup(shutil.rmtree, self._root, ignore_errors=True)

    def test_venv_python_is_used_and_reported(self) -> None:
        test_dir = self._root
        (test_dir / "main.py").write_text("# placeholder", encoding="utf-8")
        (test_dir / "reports").mkdir()
        venv_python = test_dir / ".venv" / "Scripts" / "python.exe"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("", encoding="utf-8")
        (test_dir / ".venv" / "Scripts" / "activate.bat").write_text("", encoding="utf-8")

        captured: dict = {}

        def fake_run(argv: list, *, cwd: str, timeout_s: int, capture_output: bool = True) -> dict:
            captured["argv"] = argv
            # Create a fresh report dir so the pipeline reports success.
            stamp = (datetime.now() + timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")
            (Path(cwd) / "reports" / f"report_{stamp}").mkdir()
            return {
                "ok": True,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "duration_s": 0.01,
                "command": "python main.py",
            }

        with mock.patch.object(itp, "_run", side_effect=fake_run):
            result = itp.run_instruction_test(test_dir=str(test_dir), main_script="main.py")

        self.assertTrue(result["success"])
        self.assertEqual(result["python_exe"], str(venv_python))
        self.assertEqual(result["venv_python"], str(venv_python))
        self.assertEqual(result["activate_bat"], str(test_dir / ".venv" / "Scripts" / "activate.bat"))
        # argv[0] must be the venv python, not the system one.
        self.assertEqual(captured["argv"][0], str(venv_python))

    def test_no_venv_falls_back_to_system_python(self) -> None:
        test_dir = self._root / "ws_novenv"
        test_dir.mkdir()
        (test_dir / "main.py").write_text("# placeholder", encoding="utf-8")
        (test_dir / "reports").mkdir()

        def fake_run(argv: list, *, cwd: str, timeout_s: int, capture_output: bool = True) -> dict:
            stamp = (datetime.now() + timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")
            (Path(cwd) / "reports" / f"report_{stamp}").mkdir()
            return {
                "ok": True,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "duration_s": 0.01,
                "command": "python main.py",
            }

        with mock.patch.object(itp, "_run", side_effect=fake_run):
            result = itp.run_instruction_test(test_dir=str(test_dir), main_script="main.py")

        self.assertTrue(result["success"])
        self.assertIsNone(result["venv_python"])
        self.assertIsNone(result["activate_bat"])
        # python_exe should be set to a real system interpreter.
        self.assertTrue(result["python_exe"])

    def test_use_venv_false_skips_even_when_present(self) -> None:
        test_dir = self._root / "ws_disable"
        test_dir.mkdir()
        (test_dir / "main.py").write_text("# placeholder", encoding="utf-8")
        (test_dir / "reports").mkdir()
        venv_python = test_dir / ".venv" / "Scripts" / "python.exe"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("", encoding="utf-8")

        def fake_run(argv: list, *, cwd: str, timeout_s: int, capture_output: bool = True) -> dict:
            stamp = (datetime.now() + timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")
            (Path(cwd) / "reports" / f"report_{stamp}").mkdir()
            return {
                "ok": True,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "duration_s": 0.01,
                "command": "python main.py",
            }

        with mock.patch.object(itp, "_run", side_effect=fake_run):
            result = itp.run_instruction_test(
                test_dir=str(test_dir),
                main_script="main.py",
                use_venv=False,
            )
        self.assertTrue(result["success"])
        self.assertIsNone(result["venv_python"])
        self.assertIsNone(result["activate_bat"])


if __name__ == "__main__":
    unittest.main()
