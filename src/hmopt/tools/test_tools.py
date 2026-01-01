"""Test runner adapters."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TestResult:
    success: bool
    log: str


class TestAdapter:
    def run_tests(self, repo_path: Path, test_plan: Optional[dict] = None) -> TestResult:
        raise NotImplementedError


class ShellTestAdapter(TestAdapter):
    def __init__(self, command: str):
        self.command = command

    def run_tests(self, repo_path: Path, test_plan: Optional[dict] = None) -> TestResult:
        try:
            proc = subprocess.run(
                self.command,
                cwd=repo_path,
                shell=True,
                text=True,
                capture_output=True,
                check=False,
            )
            return TestResult(success=proc.returncode == 0, log=proc.stdout + "\n" + proc.stderr)
        except Exception as exc:
            return TestResult(success=False, log=f"test run failed: {exc}")


class DummyTestAdapter(TestAdapter):
    def run_tests(self, repo_path: Path, test_plan: Optional[dict] = None) -> TestResult:
        return TestResult(success=True, log="dummy tests passed")
