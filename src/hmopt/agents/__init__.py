"""Agent exports, loaded on demand so offline operator commands stay lightweight."""

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "CoderAgent": "coder",
    "ConductorAgent": "conductor",
    "ProfilerAgent": "profiler",
    "ReviewerAgent": "reviewer",
    "SafetyGuard": "safety",
    "TraceAnalystAgent": "trace_analyst",
    "VerifierAgent": "verifier",
}

__all__ = [
    "CoderAgent",
    "ConductorAgent",
    "ProfilerAgent",
    "ReviewerAgent",
    "SafetyGuard",
    "TraceAnalystAgent",
    "VerifierAgent",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
