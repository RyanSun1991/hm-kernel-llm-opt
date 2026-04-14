"""Tests for tools/windows_relay/report_compare.py."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "windows_relay"))

import report_compare  # type: ignore  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class _TmpBase(unittest.TestCase):
    def setUp(self) -> None:
        self._root = Path(tempfile.mkdtemp(prefix="hmopt_report_compare_"))
        self.addCleanup(shutil.rmtree, self._root, ignore_errors=True)

    def tmpdir(self, name: str) -> Path:
        d = self._root / name
        d.mkdir(parents=True, exist_ok=True)
        return d


class ParseIntTests(unittest.TestCase):
    def test_plain_int(self) -> None:
        self.assertEqual(report_compare._parse_int(12345), 12345)

    def test_comma_string(self) -> None:
        self.assertEqual(report_compare._parse_int("12,345,678"), 12_345_678)

    def test_underscore_string(self) -> None:
        self.assertEqual(report_compare._parse_int("12_345"), 12_345)

    def test_float_like_string(self) -> None:
        self.assertEqual(report_compare._parse_int("1.0"), 1)

    def test_non_numeric(self) -> None:
        self.assertIsNone(report_compare._parse_int("abc"))

    def test_bool_rejected(self) -> None:
        self.assertIsNone(report_compare._parse_int(True))


class ExtractInstructionCountTests(_TmpBase):
    def test_missing_directory(self) -> None:
        result = report_compare.extract_instruction_count("/no/such/path/abcxyz123")
        self.assertIsNone(result["total"])
        self.assertTrue(any("does not exist" in note for note in result["notes"]))

    def test_summary_json_top_level_key(self) -> None:
        d = self.tmpdir("top")
        _write_json(d / "summary.json", {"total_instructions": "1,000,000"})
        result = report_compare.extract_instruction_count(str(d))
        self.assertEqual(result["total"], 1_000_000)
        self.assertEqual(result["strategy"], "summary_key")
        self.assertEqual(result["source"], "summary.json")

    def test_nested_json_fallback_scan(self) -> None:
        d = self.tmpdir("nested")
        _write_json(d / "weird" / "metrics.json", {
            "meta": "x",
            "stats": {"total_instructions": 250},
        })
        result = report_compare.extract_instruction_count(str(d))
        self.assertEqual(result["total"], 250)
        self.assertEqual(result["strategy"], "json_scan")

    def test_per_case_list_in_summary(self) -> None:
        d = self.tmpdir("percase")
        _write_json(d / "summary.json", {
            "total_instructions": 300,
            "cases": [
                {"case": "foo", "instruction_count": 100},
                {"case": "bar", "instruction_count": 200},
            ],
        })
        result = report_compare.extract_instruction_count(str(d))
        self.assertEqual(result["total"], 300)
        cases = {c["case"]: c["instructions"] for c in result["per_case"]}
        self.assertEqual(cases, {"foo": 100, "bar": 200})

    def test_text_file_regex_scan(self) -> None:
        d = self.tmpdir("textscan")
        _write_text(d / "run.log", "some header\nTotal Instructions: 98,765\nbye\n")
        result = report_compare.extract_instruction_count(str(d))
        self.assertEqual(result["total"], 98_765)
        self.assertEqual(result["strategy"], "text_scan")
        self.assertEqual(result["source"], "run.log")

    def test_explicit_metric_file_with_dot_key(self) -> None:
        d = self.tmpdir("dotkey")
        _write_json(d / "report.json", {"summary": {"instructions": 42}})
        result = report_compare.extract_instruction_count(
            str(d),
            metric_file="report.json",
            metric_key="summary.instructions",
        )
        self.assertEqual(result["total"], 42)
        self.assertEqual(result["strategy"], "explicit")
        self.assertEqual(result["key"], "summary.instructions")

    def test_explicit_missing_key_falls_through_to_notes(self) -> None:
        d = self.tmpdir("missing")
        _write_json(d / "report.json", {"something_else": 1})
        result = report_compare.extract_instruction_count(
            str(d),
            metric_file="report.json",
            metric_key="summary.instructions",
        )
        self.assertIsNone(result["total"])
        self.assertTrue(any("not found" in n for n in result["notes"]))


class CompareReportsTests(_TmpBase):
    def test_happy_path_delta(self) -> None:
        base_dir = self.tmpdir("base")
        cand_dir = self.tmpdir("cand")
        _write_json(base_dir / "summary.json", {
            "total_instructions": 1000,
            "cases": [{"case": "a", "instruction_count": 400}, {"case": "b", "instruction_count": 600}],
        })
        _write_json(cand_dir / "summary.json", {
            "total_instructions": 900,
            "cases": [{"case": "a", "instruction_count": 350}, {"case": "b", "instruction_count": 550}],
        })
        result = report_compare.compare_reports(
            baseline_dir=str(base_dir),
            candidate_dir=str(cand_dir),
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["delta"], -100)
        self.assertEqual(result["delta_pct"], -10.0)
        per_case = {c["case"]: c for c in result["per_case_diff"]}
        self.assertEqual(per_case["a"]["delta"], -50)
        self.assertEqual(per_case["b"]["delta"], -50)

    def test_missing_candidate_total_reports_failure(self) -> None:
        base_dir = self.tmpdir("base2")
        cand_dir = self.tmpdir("cand2")
        _write_json(base_dir / "summary.json", {"total_instructions": 10})
        _write_text(cand_dir / "README.txt", "no useful data here")
        result = report_compare.compare_reports(
            baseline_dir=str(base_dir),
            candidate_dir=str(cand_dir),
        )
        self.assertFalse(result["success"])
        self.assertIsNone(result["delta"])

    def test_cli_main_emits_json(self) -> None:
        base_dir = self.tmpdir("cli_base")
        cand_dir = self.tmpdir("cli_cand")
        _write_json(base_dir / "summary.json", {"total_instructions": 100})
        _write_json(cand_dir / "summary.json", {"total_instructions": 120})

        buf = StringIO()
        original = sys.stdout
        sys.stdout = buf
        try:
            rc = report_compare.main(["--baseline", str(base_dir), "--candidate", str(cand_dir)])
        finally:
            sys.stdout = original

        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue().strip())
        self.assertTrue(payload["success"])
        self.assertEqual(payload["delta"], 20)


if __name__ == "__main__":
    unittest.main()
