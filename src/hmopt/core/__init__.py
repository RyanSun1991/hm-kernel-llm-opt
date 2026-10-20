"""Core exports, loaded on demand so configuration reads do not start the runtime."""

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "AppConfig": "config",
    "HMOptError": "errors",
    "ConfigError": "errors",
    "PipelineError": "errors",
    "LLMClient": "llm",
    "ChatMessage": "llm",
    "RunContext": "run_context",
    "build_context": "run_context",
    "register_run": "run_context",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
