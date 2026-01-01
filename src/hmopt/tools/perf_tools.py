"""Performance tool adapters."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ProfileResult:
    success: bool
    artifacts: Dict[str, Path]
    log: str


class ProfilerAdapter:
    def profile(self, workload_id: str, output_dir: Path, options: Optional[dict] = None) -> ProfileResult:
        raise NotImplementedError


class ShellProfilerAdapter(ProfilerAdapter):
    def __init__(self, command: str):
        self.command = command

    def profile(self, workload_id: str, output_dir: Path, options: Optional[dict] = None) -> ProfileResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(
                self.command,
                cwd=output_dir,
                shell=True,
                text=True,
                capture_output=True,
                check=False,
            )
            success = proc.returncode == 0
            return ProfileResult(success=success, artifacts={}, log=proc.stdout + "\n" + proc.stderr)
        except Exception as exc:
            return ProfileResult(success=False, artifacts={}, log=f"profile failed: {exc}")


class DummyProfilerAdapter(ProfilerAdapter):
    """Generate small synthetic traces for local testing."""

    def profile(self, workload_id: str, output_dir: Path, options: Optional[dict] = None) -> ProfileResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Path] = {}

        # framegraph
        frames = [{"ts": i * 16.6, "dur": 16.6 + (i % 10) * 0.5} for i in range(120)]
        frame_path = output_dir / "framegraph.json"
        frame_path.write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")
        artifacts["framegraph"] = frame_path

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

        return ProfileResult(success=True, artifacts=artifacts, log="dummy profiler emitted synthetic traces")
