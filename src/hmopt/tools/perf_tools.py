"""Performance tool adapters."""

from __future__ import annotations

import json
import subprocess
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

@dataclass
class ProfileResult:
    success: bool
    artifacts: Dict[str, Path]
    log: str


class ProfilerAdapter:
    def profile(self, workload_id: str, output_dir: Path, options: Optional[dict] = None) -> ProfileResult:
        raise NotImplementedError


# Filename globs -> artifact kind, in priority order. The shell profiler writes
# its outputs into output_dir; we discover them here so the analysis stage
# actually receives them (previously profile() always returned artifacts={}, so
# the real profiling path produced no metrics/hotspots at all).
_ARTIFACT_GLOBS: list[tuple[str, str]] = [
    ("flamegraph", "*__sysmgr_hiperfReport.html"),
    ("flamegraph", "*flamegraph*.json"),
    ("flamegraph", "*flamegraph*.html"),
    ("hitrace", "*hitrace*.json"),
    ("hitrace", "*hitrace*.txt"),
    ("hitrace", "*.ftrace"),
    ("hiperf", "*hiperf*.json"),
    ("perf", "perf.data"),
    ("perf", "*.perf"),
]


def _discover_artifacts(output_dir: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for kind, pattern in _ARTIFACT_GLOBS:
        if kind in found:
            continue
        matches = sorted(output_dir.rglob(pattern))
        if matches:
            found[kind] = matches[0]
    return found


class ShellProfilerAdapter(ProfilerAdapter):
    def __init__(self, command: str):
        self.command = command

    def profile(self, workload_id: str, output_dir: Path, options: Optional[dict] = None) -> ProfileResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("Shell profiler start: cmd=%s", self.command)
            proc = subprocess.run(
                self.command,
                cwd=output_dir,
                shell=True,
                text=True,
                capture_output=True,
                check=False,
            )
            success = proc.returncode == 0
            # Allow an explicit override (options['artifacts'] = {kind: relpath});
            # otherwise auto-discover the files the command wrote into output_dir.
            artifacts: Dict[str, Path] = {}
            override = (options or {}).get("artifacts")
            if isinstance(override, dict):
                for kind, rel in override.items():
                    p = output_dir / str(rel)
                    if p.exists():
                        artifacts[str(kind)] = p
            if not artifacts:
                artifacts = _discover_artifacts(output_dir)
            if success and not artifacts:
                logger.warning(
                    "Shell profiler succeeded but produced no recognized artifacts in %s",
                    output_dir,
                )
            logger.info("Shell profiler discovered artifacts=%s", list(artifacts))
            return ProfileResult(
                success=success, artifacts=artifacts, log=proc.stdout + "\n" + proc.stderr
            )
        except Exception as exc:
            logger.exception("Shell profiler error")
            return ProfileResult(success=False, artifacts={}, log=f"profile failed: {exc}")


class DummyProfilerAdapter(ProfilerAdapter):
    """Generate small synthetic traces for local testing."""

    def profile(self, workload_id: str, output_dir: Path, options: Optional[dict] = None) -> ProfileResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Path] = {}

        # flamegraph
        frames = [{"ts": i * 16.6, "dur": 16.6 + (i % 10) * 0.5} for i in range(120)]
        frame_path = output_dir / "flamegraph.json"
        frame_path.write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")
        artifacts["flamegraph"] = frame_path

        # hitrace (scheduler latency samples)
        hitrace = [{"ts": i * 1000, "dur": 80 + (i % 5) * 20, "name": "sched"} for i in range(200)]
        hitrace_path = output_dir / "hitrace.json"
        hitrace_path.write_text(json.dumps({"events": hitrace}, indent=2), encoding="utf-8")
        artifacts["hitrace"] = hitrace_path

        # hiperf (sample stacks)
        hiperf_samples = [
            {"stack": ["main", "render", "driver"], "weight": 10},
            {"stack": ["main", "update", "memcpy"], "weight": 5},
            {"stack": ["main", "render", "blend"], "weight": 8},
        ]
        hiperf_path = output_dir / "hiperf.json"
        hiperf_path.write_text(json.dumps({"samples": hiperf_samples}, indent=2), encoding="utf-8")
        artifacts["hiperf"] = hiperf_path

        logger.info("Dummy profiler emitted artifacts=%d", len(artifacts))
        return ProfileResult(success=True, artifacts=artifacts, log="dummy profiler emitted synthetic traces")
