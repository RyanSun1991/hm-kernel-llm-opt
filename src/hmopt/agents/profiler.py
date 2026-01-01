"""Profiler agent (non-LLM)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from hmopt.tools.perf_tools import ProfilerAdapter, ProfileResult


class ProfilerAgent:
    def __init__(self, adapter: ProfilerAdapter):
        self.adapter = adapter

    def profile(self, workload_id: str, output_dir: Path, options: Optional[dict] = None) -> ProfileResult:
        return self.adapter.profile(workload_id, output_dir, options)
