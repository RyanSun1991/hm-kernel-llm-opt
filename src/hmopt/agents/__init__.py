"""Agent implementations.

Imports are resolved lazily so lightweight helpers such as
``hmopt.agents.prompting`` can be imported without loading the full runtime
analysis stack.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CoderAgent",
    "ConductorAgent",
    "ProfilerAgent",
    "ReviewerAgent",
    "SafetyGuard",
    "TraceAnalystAgent",
    "VerifierAgent",
]

_MODULES = {
    "CoderAgent": "coder",
    "ConductorAgent": "conductor",
    "ProfilerAgent": "profiler",
    "ReviewerAgent": "reviewer",
    "SafetyGuard": "safety",
    "TraceAnalystAgent": "trace_analyst",
    "VerifierAgent": "verifier",
}


def __getattr__(name: str) -> Any:
    if name not in _MODULES:
        raise AttributeError(name)
    module_name = _MODULES[name]
    module = __import__(f"{__name__}.{module_name}", fromlist=[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
