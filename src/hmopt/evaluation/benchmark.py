"""Benchmark runner."""

from __future__ import annotations

from hmopt.core.config import AppConfig
from hmopt.orchestration import run_pipeline


def run_dummy_benchmark(config_path: str = "configs/app.yaml") -> str:
    """Execute the pipeline with dummy adapters to validate plumbing."""
    cfg = AppConfig.from_yaml(config_path)
    cfg.adapters.dummy = True
    cfg.iterations = 1
    return run_pipeline(cfg)
