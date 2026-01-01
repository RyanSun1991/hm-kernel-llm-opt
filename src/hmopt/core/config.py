"""Configuration loading and normalization.

The repository ships sample configs under ``configs/``.  This module
turns those YAML files into typed objects that other layers can rely on.
Environment variables in YAML values are expanded to keep secrets (LLM
keys) out of the repo.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


class WorkloadConfig(BaseModel):
    name: str
    kind: Optional[str] = None
    command: Optional[str] = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    objectives: list[dict[str, Any]] = Field(default_factory=list)


class AppConfig(BaseModel):
    project: ProjectConfig
    llm: ModelConfig
    storage: StorageConfig = StorageConfig()
    adapters: AdapterConfig = AdapterConfig()
    workloads: list[WorkloadConfig] = Field(default_factory=list)
    iterations: int = 2
    profiling_enabled: bool = True
    pipeline: str = "optimize_kernel"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        raw = load_yaml(path)
        merged = merge_includes(raw, Path(path).resolve().parent)
        normalized = normalize_raw_config(merged)
        return cls(**normalized)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML and expand environment variables."""
    p = Path(path)
    text = _expand_env(p.read_text(encoding="utf-8"))
    data = yaml.safe_load(text) or {}
    logger.debug("Loaded YAML: path=%s keys=%s", p, list(data.keys()))
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries (override wins)."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def merge_includes(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Merge optional include files declared under ``includes``."""
    merged = dict(raw)
    includes = merged.get("includes", {}) or {}

    def _resolve(path_like: str) -> Path:
        return (base_dir / path_like).resolve()

    # Model/LLM settings
    if "model_server" in includes:
        model_path = _resolve(includes["model_server"])
        model_cfg = load_yaml(model_path)
        llm_cfg = model_cfg.get("llm", model_cfg)
        merged["llm"] = deep_merge(llm_cfg, merged.get("llm", {}))
        logger.info("Merged model_server config from %s", model_path)

    # Workloads
    if "workloads" in includes:
        workloads_path = _resolve(includes["workloads"])
        workloads_cfg = load_yaml(workloads_path)
        merged["workloads"] = workloads_cfg.get("workloads", merged.get("workloads", []))
        logger.info("Merged workloads config from %s", workloads_path)

    return merged


def normalize_raw_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept legacy/sample config shapes and map them to AppConfig fields."""
    project_cfg = raw.get("project", {})
    llm_cfg = raw.get("llm", {})
    storage_cfg = raw.get("storage", {})
    logger.debug("Normalizing config: project_keys=%s llm_keys=%s storage_keys=%s", list(project_cfg.keys()), list(llm_cfg.keys()), list(storage_cfg.keys()))

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

    workloads_cfg = raw.get("workloads", [])

    return {
        "project": project_cfg,
        "llm": llm_norm,
        "storage": storage_norm,
        "adapters": adapters_norm,
        "workloads": workloads_cfg,
        "iterations": raw.get("iterations", raw.get("max_iterations", 2)),
        "profiling_enabled": raw.get("profiling", {}).get("enabled", True),
        "pipeline": raw.get("pipelines", {}).get("default", raw.get("pipeline", "optimize_kernel")),
    }
