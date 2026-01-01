"""Verifier agent (non-LLM)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hmopt.tools.build_tools import BuildAdapter, BuildResult
from hmopt.tools.test_tools import TestAdapter, TestResult


@dataclass
class VerificationResult:
    build: BuildResult
    tests: TestResult

    @property
    def success(self) -> bool:
        return self.build.success and self.tests.success


class VerifierAgent:
    def __init__(self, build_adapter: BuildAdapter, test_adapter: TestAdapter):
        self.build_adapter = build_adapter
        self.test_adapter = test_adapter

    def verify(self, repo_path: Path, build_config: Optional[dict] = None, test_plan: Optional[dict] = None) -> VerificationResult:
        build = self.build_adapter.build(repo_path, build_config)
        tests = self.test_adapter.run_tests(repo_path, test_plan)
        return VerificationResult(build=build, tests=tests)
