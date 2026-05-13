"""Tests for the Phase 4 IndexingConfig schema additions.

Verifies that `normalize_raw_config` correctly extracts the new
`indexing.backend` and `indexing.scip_clang` sections, that defaults
preserve existing behavior when those fields are missing, and that
explicit values override them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "hmopt" / "core" / "config.py"
)


def _load_config_module():
    """Load `hmopt.core.config` without importing the rest of hmopt."""
    if "config_under_test" in sys.modules:
        return sys.modules["config_under_test"]
    spec = importlib.util.spec_from_file_location("config_under_test", _CONFIG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["config_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def config_mod():
    try:
        import pydantic  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        pytest.skip("pydantic / yaml not installed")
    return _load_config_module()


def _minimal_raw(extra_indexing: dict | None = None) -> dict:
    """Build a minimal raw config dict accepted by normalize_raw_config."""
    base = {
        "project": {"name": "test", "repo_path": "/tmp/repo"},
        "llm": {},
        "storage": {},
        "indexing": {},
    }
    if extra_indexing:
        base["indexing"].update(extra_indexing)
    return base


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_backend_is_clangd(config_mod):
    """When no backend is specified, default to 'clangd'."""
    normalized = config_mod.normalize_raw_config(_minimal_raw())
    assert normalized["indexing"]["backend"] == "clangd"


def test_default_scip_clang_block_present(config_mod):
    """The scip_clang section always exists in normalized output, even when
    omitted from YAML, so downstream code never has to handle missing keys."""
    normalized = config_mod.normalize_raw_config(_minimal_raw())
    sc = normalized["indexing"]["scip_clang"]
    assert sc["enabled"] is True
    assert sc["binary"] == "scip-clang"
    assert sc["timeout_sec"] == 7200
    assert sc["extra_args"] == []
    assert sc["keep_scip_file"] is False


# ---------------------------------------------------------------------------
# Explicit override
# ---------------------------------------------------------------------------


def test_explicit_backend_scip_clang(config_mod):
    raw = _minimal_raw({"backend": "scip-clang"})
    normalized = config_mod.normalize_raw_config(raw)
    assert normalized["indexing"]["backend"] == "scip-clang"


def test_explicit_scip_clang_block_overrides_defaults(config_mod):
    raw = _minimal_raw(
        {
            "backend": "scip-clang",
            "scip_clang": {
                "enabled": True,
                "binary": "/opt/scip-clang/bin/scip-clang",
                "timeout_sec": 14400,
                "extra_args": ["-j", "8"],
                "keep_scip_file": True,
            },
        }
    )
    normalized = config_mod.normalize_raw_config(raw)
    sc = normalized["indexing"]["scip_clang"]
    assert sc["binary"] == "/opt/scip-clang/bin/scip-clang"
    assert sc["timeout_sec"] == 14400
    assert sc["extra_args"] == ["-j", "8"]
    assert sc["keep_scip_file"] is True


# ---------------------------------------------------------------------------
# Pydantic model construction
# ---------------------------------------------------------------------------


def test_appconfig_accepts_scip_clang_backend(config_mod):
    """Round-trip through AppConfig: normalize_raw_config + Pydantic
    validation should yield a usable AppConfig with the new fields."""
    raw = _minimal_raw(
        {
            "backend": "scip-clang",
            "scip_clang": {"timeout_sec": 600, "keep_scip_file": True},
        }
    )
    normalized = config_mod.normalize_raw_config(raw)
    app_cfg = config_mod.AppConfig(**normalized)
    assert app_cfg.indexing.backend == "scip-clang"
    assert app_cfg.indexing.scip_clang.timeout_sec == 600
    assert app_cfg.indexing.scip_clang.keep_scip_file is True
    # Clangd config still has its defaults — both backend configs coexist.
    assert app_cfg.indexing.clangd.enabled is True


def test_appconfig_default_backend_clangd(config_mod):
    """Backward-compat: a YAML with no `backend` field still produces
    a working AppConfig pointing at clangd."""
    raw = _minimal_raw()
    normalized = config_mod.normalize_raw_config(raw)
    app_cfg = config_mod.AppConfig(**normalized)
    assert app_cfg.indexing.backend == "clangd"
    # ScipClangConfig defaults are populated even when not used.
    assert app_cfg.indexing.scip_clang.binary == "scip-clang"
