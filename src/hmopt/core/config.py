"""Configuration loading and normalization.

The repository ships sample configs under ``configs/``.  This module
turns those YAML files into typed objects that other layers can rely on.
Environment variables in YAML values are expanded to keep secrets (LLM
keys) out of the repo.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


def _expand_env(text: str) -> str:
    """Expand ${VARS} inside YAML text."""
    return os.path.expandvars(text)


class ProjectConfig(BaseModel):
    name: str
    repo_path: str
    build_system: str = "auto"
    workload: Optional[str] = None


class ModelConfig(BaseModel):
    provider: str = "openai_compatible"
    base_url: str = "http://10.90.56.33:20010/v1"
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("HMOPT_LLM_API_KEY")
    )
    model: str = "qwen3-coder-30b"
    embedding_model: str = "qwen3-embedding-8b"
    allow_external_proxy_models: bool = False


class StorageConfig(BaseModel):
    db_url: str = "sqlite:///data/hmopt.db"
    artifacts_root: Path = Path("data/artifacts")
    vector_store_path: Path = Path("data/vector_store.db")
    run_root: Path = Path("data/runs")


class AdapterConfig(BaseModel):
    build_command: Optional[str] = None
    test_command: Optional[str] = None
    workload_command: Optional[str] = None
    profile_command: Optional[str] = None
    dummy: bool = True


class AppConfig(BaseModel):
    project: ProjectConfig
    llm: ModelConfig
    storage: StorageConfig = StorageConfig()
    adapters: AdapterConfig = AdapterConfig()
    iterations: int = 2
    profiling_enabled: bool = True
    pipeline: str = "optimize_kernel"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        raw = load_yaml(path)
        normalized = normalize_raw_config(raw)
        return cls(**normalized)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML and expand environment variables."""
    p = Path(path)
    text = _expand_env(p.read_text(encoding="utf-8"))
    return yaml.safe_load(text)


def normalize_raw_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept legacy/sample config shapes and map them to AppConfig fields."""
    project_cfg = raw.get("project", {})
    llm_cfg = raw.get("llm", {})
    storage_cfg = raw.get("storage", {})

    # Resolve API key from env if provided via env name
    api_key_env = llm_cfg.get("api_key_env")
    api_key = llm_cfg.get("api_key") or (os.getenv(api_key_env) if api_key_env else None)
    llm_norm = {
        "provider": llm_cfg.get("provider", "openai_compatible"),
        "base_url": llm_cfg.get("base_url")
        or llm_cfg.get("api_base")
        or "http://10.90.56.33:20010/v1",
        "api_key": api_key,
        "model": llm_cfg.get("model", "qwen3-coder-30b"),
        "embedding_model": llm_cfg.get("embedding_model", "qwen3-embedding-8b"),
        "allow_external_proxy_models": bool(
            llm_cfg.get("allow_external_proxy_models")
            or llm_cfg.get("security", {}).get("allow_external_proxy_models", False)
        ),
    }

    db_url = storage_cfg.get("db_url")
    if not db_url:
        db_url = storage_cfg.get("db", {}).get("url", "sqlite:///data/hmopt.db")
    artifacts_root = storage_cfg.get("artifacts_root") or storage_cfg.get("artifacts", {}).get(
        "root_dir", "data/artifacts"
    )
    vector_store_path = storage_cfg.get("vector_store_path") or storage_cfg.get(
        "vector_store", {}
    ).get("path", "data/vector_store.db")
    run_root = storage_cfg.get("run_root", "data/runs")

    storage_norm = {
        "db_url": db_url,
        "artifacts_root": Path(artifacts_root),
        "vector_store_path": Path(vector_store_path),
        "run_root": Path(run_root),
    }

    adapters_cfg = raw.get("adapters", {})
    adapters_norm = {
        "build_command": adapters_cfg.get("build_command"),
        "test_command": adapters_cfg.get("test_command"),
        "workload_command": adapters_cfg.get("workload_command"),
        "profile_command": adapters_cfg.get("profile_command"),
        "dummy": adapters_cfg.get("dummy", True),
    }

    return {
        "project": project_cfg,
        "llm": llm_norm,
        "storage": storage_norm,
        "adapters": adapters_norm,
        "iterations": raw.get("iterations", raw.get("max_iterations", 2)),
        "profiling_enabled": raw.get("profiling", {}).get("enabled", True),
        "pipeline": raw.get("pipelines", {}).get("default", raw.get("pipeline", "optimize_kernel")),
    }
