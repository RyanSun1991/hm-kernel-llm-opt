"""Profiler agent (non-LLM)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import logging

from hmopt.tools.perf_tools import ProfilerAdapter, ProfileResult

logger = logging.getLogger(__name__)


class ProfilerAgent:
    def __init__(self, adapter: ProfilerAdapter):
        self.adapter = adapter

    def profile(self, workload_id: str, output_dir: Path, options: Optional[dict] = None) -> ProfileResult:
        logger.info("Profiler collecting workload=%s", workload_id)
        result = self.adapter.profile(workload_id, output_dir, options)
        logger.info("Profiler done: success=%s artifacts=%d", result.success, len(result.artifacts))
        return result
