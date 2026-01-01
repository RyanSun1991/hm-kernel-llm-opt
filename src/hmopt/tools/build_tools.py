"""Build integration helpers and adapters."""

from __future__ import annotations

import subprocess
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class BuildResult:
    success: bool
    log: str
    artifact_paths: list[Path]


class BuildAdapter:
    def build(self, repo_path: Path, build_config: Optional[dict] = None) -> BuildResult:
        raise NotImplementedError


class ShellBuildAdapter(BuildAdapter):
    """Run a shell command for builds."""

    def __init__(self, command: str):
        self.command = command

    def build(self, repo_path: Path, build_config: Optional[dict] = None) -> BuildResult:
        try:
            logger.info("Shell build start: cmd=%s", self.command)
            proc = subprocess.run(
                self.command,
                cwd=repo_path,
                shell=True,
                text=True,
                capture_output=True,
                check=False,
            )
            success = proc.returncode == 0
            log = proc.stdout + "\n" + proc.stderr
        except Exception as exc:
            success = False
            log = f"build failed: {exc}"
            logger.exception("Shell build error")
        return BuildResult(success=success, log=log, artifact_paths=[])


class DummyBuildAdapter(BuildAdapter):
    """Fast, side-effect free adapter for CI/dry runs."""

    def build(self, repo_path: Path, build_config: Optional[dict] = None) -> BuildResult:
        logger.info("Dummy build invoked")
        return BuildResult(success=True, log="dummy build succeeded", artifact_paths=[])
