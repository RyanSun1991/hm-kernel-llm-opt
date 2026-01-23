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


class RepoConfig(BaseModel):
    name: str
    repo_path: str
    compile_commands_dir: Optional[Path] = None


class ProjectConfig(BaseModel):
    name: str
    repo_path: str
    build_system: str = "auto"
    workload: Optional[str] = None
    repos: list[RepoConfig] = Field(default_factory=list)


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


class Neo4jConfig(BaseModel):
    enabled: bool = False
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: Optional[str] = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD"))
    database: str = "neo4j"


class ClangdConfig(BaseModel):
    enabled: bool = True
    binary: str = "clangd"
    compile_commands_dir: Optional[Path] = None
    extra_args: list[str] = Field(default_factory=list)
    timeout_sec: int = 10
    max_files: int = 5000
    symbol_kinds: list[str] = Field(
        default_factory=lambda: [
            "function",
            "method",
            "constructor",
            "class",
            "struct",
            "enum",
            "enum_member",
            "interface",
            "type_parameter",
            "variable",
            "constant",
            "field",
            "property",
        ]
    )
    call_hierarchy_enabled: bool = True
    call_hierarchy_max_functions: int = 2000
    call_hierarchy_max_calls: int = 100
    call_hierarchy_max_depth: int = 1
    usage_scan_enabled: bool = True
    usage_scan_max_names: int = 2000
    relation_max_per_symbol: int = 200
    file_summary_enabled: bool = True
    relation_summary_enabled: bool = True
    relation_summary_max_items: int = 50


class IndexingConfig(BaseModel):
    enabled: bool = True
    persist_dir: Path = Path("data/llamaindex")
    code_index_root: Optional[Path] = None
    runtime_index_root: Optional[Path] = None
    code_index_version: Optional[str] = None
    runtime_index_version: Optional[str] = None
    versioning_scheme: str = "git_commit"
    include_dirty_suffix: bool = True
    allow_legacy_paths: bool = True
    index_registry_path: Path = Path("data/llamaindex/index_registry.json")
    incremental_base_ref: str = "HEAD~1"
    incremental_mode: str = "rebuild"  # rebuild|merge
    incremental_max_changed_files: int = 5000
    llm_enrich: bool = False
    llm_enrich_limit: int = 50
    runtime_evidence_max_chars: int = 20000
    hotspot_top_k: int = 20
    hotspot_min_ratio: float = 0.001
    hotspot_min_abs: float = 10.0
    query_code_top_k: int = 10
    query_runtime_top_k: int = 10
    query_graph_top_k: int = 3
    query_code_filter_top_k: int = 30
    query_runtime_symbol_top_k: int = 20
    query_runtime_path_top_k: int = 20
    neo4j: Neo4jConfig = Neo4jConfig()
    clangd: ClangdConfig = ClangdConfig()


class PromptConfig(BaseModel):
    dir: Path = Path("configs/prompts")
    overrides: dict[str, dict[str, str]] = Field(default_factory=dict)
    profile: str = "analysis"
    profiles: dict[str, dict[str, str]] = Field(default_factory=dict)
    stage_profiles: dict[str, str] = Field(default_factory=dict)
    review_enabled: bool = False
    review_block_on_reject: bool = False


class AppConfig(BaseModel):
    project: ProjectConfig
    llm: ModelConfig
    storage: StorageConfig = StorageConfig()
    adapters: AdapterConfig = AdapterConfig()
    workloads: list[WorkloadConfig] = Field(default_factory=list)
    indexing: IndexingConfig = IndexingConfig()
    prompts: PromptConfig = PromptConfig()
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
    prompts_cfg = raw.get("prompts", {}) or {}
    prompts_norm = {
        "dir": Path(prompts_cfg.get("dir", "configs/prompts")),
        "overrides": prompts_cfg.get("overrides", {}) or {},
        "profile": prompts_cfg.get("profile", "analysis"),
        "profiles": prompts_cfg.get("profiles", {}) or {},
        "stage_profiles": prompts_cfg.get("stage_profiles", {}) or {},
        "review_enabled": bool(prompts_cfg.get("review_enabled", False)),
        "review_block_on_reject": bool(prompts_cfg.get("review_block_on_reject", False)),
    }
    indexing_cfg = raw.get("indexing", {})

    neo4j_cfg = indexing_cfg.get("neo4j", {})
    neo4j_norm = {
        "enabled": bool(neo4j_cfg.get("enabled", False)),
        "uri": neo4j_cfg.get("uri", "bolt://localhost:7687"),
        "user": neo4j_cfg.get("user", "neo4j"),
        "password": neo4j_cfg.get("password")
        or (os.getenv(neo4j_cfg.get("password_env")) if neo4j_cfg.get("password_env") else None),
        "database": neo4j_cfg.get("database", "neo4j"),
    }

    clangd_cfg = indexing_cfg.get("clangd", {})
    clangd_norm = {
        "enabled": clangd_cfg.get("enabled", True),
        "binary": clangd_cfg.get("binary", "clangd"),
        "compile_commands_dir": Path(clangd_cfg["compile_commands_dir"])
        if clangd_cfg.get("compile_commands_dir")
        else None,
        "extra_args": clangd_cfg.get("extra_args", []),
        "timeout_sec": int(clangd_cfg.get("timeout_sec", 10)),
        "max_files": int(clangd_cfg.get("max_files", 5000)),
        "symbol_kinds": clangd_cfg.get(
            "symbol_kinds",
            [
                "function",
                "method",
                "constructor",
                "class",
                "struct",
                "enum",
                "enum_member",
                "interface",
                "type_parameter",
                "variable",
                "constant",
                "field",
                "property",
            ],
        ),
        "call_hierarchy_enabled": bool(clangd_cfg.get("call_hierarchy_enabled", True)),
        "call_hierarchy_max_functions": int(clangd_cfg.get("call_hierarchy_max_functions", 2000)),
        "call_hierarchy_max_calls": int(clangd_cfg.get("call_hierarchy_max_calls", 100)),
        "call_hierarchy_max_depth": int(clangd_cfg.get("call_hierarchy_max_depth", 1)),
        "usage_scan_enabled": bool(clangd_cfg.get("usage_scan_enabled", True)),
        "usage_scan_max_names": int(clangd_cfg.get("usage_scan_max_names", 2000)),
        "relation_max_per_symbol": int(clangd_cfg.get("relation_max_per_symbol", 200)),
        "file_summary_enabled": bool(clangd_cfg.get("file_summary_enabled", True)),
        "relation_summary_enabled": bool(clangd_cfg.get("relation_summary_enabled", True)),
        "relation_summary_max_items": int(clangd_cfg.get("relation_summary_max_items", 50)),
    }

    indexing_norm = {
        "enabled": indexing_cfg.get("enabled", True),
        "persist_dir": Path(indexing_cfg.get("persist_dir", "data/llamaindex")),
        "code_index_root": Path(indexing_cfg["code_index_root"])
        if indexing_cfg.get("code_index_root")
        else None,
        "runtime_index_root": Path(indexing_cfg["runtime_index_root"])
        if indexing_cfg.get("runtime_index_root")
        else None,
        "code_index_version": indexing_cfg.get("code_index_version"),
        "runtime_index_version": indexing_cfg.get("runtime_index_version"),
        "versioning_scheme": indexing_cfg.get("versioning_scheme", "git_commit"),
        "include_dirty_suffix": bool(indexing_cfg.get("include_dirty_suffix", True)),
        "allow_legacy_paths": bool(indexing_cfg.get("allow_legacy_paths", True)),
        "index_registry_path": Path(
            indexing_cfg.get("index_registry_path", "data/llamaindex/index_registry.json")
        ),
        "incremental_base_ref": indexing_cfg.get("incremental_base_ref", "HEAD~1"),
        "incremental_mode": indexing_cfg.get("incremental_mode", "rebuild"),
        "incremental_max_changed_files": int(
            indexing_cfg.get("incremental_max_changed_files", 5000)
        ),
        "llm_enrich": indexing_cfg.get("llm_enrich", False),
        "llm_enrich_limit": int(indexing_cfg.get("llm_enrich_limit", 50)),
        "runtime_evidence_max_chars": int(indexing_cfg.get("runtime_evidence_max_chars", 20000)),
        "hotspot_top_k": int(indexing_cfg.get("hotspot_top_k", 20)),
        "hotspot_min_ratio": float(indexing_cfg.get("hotspot_min_ratio", 0.001)),
        "hotspot_min_abs": float(indexing_cfg.get("hotspot_min_abs", 10.0)),
        "query_code_top_k": int(indexing_cfg.get("query_code_top_k", 10)),
        "query_runtime_top_k": int(indexing_cfg.get("query_runtime_top_k", 10)),
        "query_graph_top_k": int(indexing_cfg.get("query_graph_top_k", 3)),
        "query_code_filter_top_k": int(indexing_cfg.get("query_code_filter_top_k", 30)),
        "query_runtime_symbol_top_k": int(indexing_cfg.get("query_runtime_symbol_top_k", 20)),
        "query_runtime_path_top_k": int(indexing_cfg.get("query_runtime_path_top_k", 20)),
        "neo4j": neo4j_norm,
        "clangd": clangd_norm,
    }

    # Normalize multi-repo project config.
    repos_cfg = project_cfg.get("repos") or []
    normalized_repos: list[dict[str, Any]] = []
    if isinstance(repos_cfg, list):
        for entry in repos_cfg:
            if isinstance(entry, str):
                repo_path = entry
                name = Path(repo_path).name
                normalized_repos.append(
                    {
                        "name": name,
                        "repo_path": repo_path,
                        "compile_commands_dir": None,
                    }
                )
            elif isinstance(entry, dict):
                repo_path = entry.get("repo_path") or entry.get("path") or ""
                name = entry.get("name") or Path(repo_path).name or "repo"
                normalized_repos.append(
                    {
                        "name": name,
                        "repo_path": repo_path,
                        "compile_commands_dir": entry.get("compile_commands_dir"),
                    }
                )
    if normalized_repos:
        project_cfg = dict(project_cfg)
        project_cfg["repos"] = normalized_repos
        if not project_cfg.get("repo_path") and normalized_repos:
            project_cfg["repo_path"] = normalized_repos[0]["repo_path"]

    return {
        "project": project_cfg,
        "llm": llm_norm,
        "storage": storage_norm,
        "adapters": adapters_norm,
        "workloads": workloads_cfg,
        "prompts": prompts_norm,
        "indexing": indexing_norm,
        "iterations": raw.get("iterations", raw.get("max_iterations", 2)),
        "profiling_enabled": raw.get("profiling", {}).get("enabled", True),
        "pipeline": raw.get("pipelines", {}).get("default", raw.get("pipeline", "optimize_kernel")),
    }
