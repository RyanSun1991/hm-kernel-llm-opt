"""Indexing and query routing with LlamaIndex.

LlamaIndex is a heavyweight optional dependency. Importing it eagerly here made
the whole ``hmopt`` package (and CLI) fail to import on a clean checkout that
hadn't installed the full indexing stack. We now import the real implementations
lazily and, if the stack is absent, expose stub callables that fail loudly with
an actionable message *when invoked* — so ``hmopt --help`` and the non-indexing
commands keep working, while an actual index command tells the user what to
install instead of crashing at import time.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_EXPORTS = [
    "build_kernel_index",
    "build_runtime_index",
    "fetch_code_snippets",
    "lookup_code_symbols",
    "retrieve_call_chain",
    "route_query",
]

__all__ = list(_EXPORTS)

try:
    from .llamaindex_pipeline import (  # noqa: F401
        build_kernel_index,
        build_runtime_index,
        fetch_code_snippets,
        lookup_code_symbols,
        retrieve_call_chain,
        route_query,
    )

    _INDEXING_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - optional heavy stack (llama_index etc.)
    _INDEXING_IMPORT_ERROR = exc
    logger.warning(
        "indexing stack unavailable (%s); index commands will raise if invoked", exc
    )

    def _missing(name: str):
        def _stub(*_args, **_kwargs):
            raise RuntimeError(
                f"hmopt.indexing.{name} requires the indexing extra. "
                f"Install it (e.g. `pip install -e '.[indexing]'` or the llama-index "
                f"packages) to use code/runtime indexing. Original import error: "
                f"{_INDEXING_IMPORT_ERROR}"
            )

        _stub.__name__ = name
        return _stub

    build_kernel_index = _missing("build_kernel_index")
    build_runtime_index = _missing("build_runtime_index")
    fetch_code_snippets = _missing("fetch_code_snippets")
    lookup_code_symbols = _missing("lookup_code_symbols")
    retrieve_call_chain = _missing("retrieve_call_chain")
    route_query = _missing("route_query")
