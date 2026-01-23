"""Compatibility wrapper for indexing pipeline exports."""

from __future__ import annotations

from hmopt.indexing.kernel_index import build_kernel_index, build_kernel_indexes
from hmopt.indexing.query import route_query
from hmopt.indexing.runtime_index import build_runtime_aggregate_index, build_runtime_index

__all__ = [
    "build_kernel_index",
    "build_kernel_indexes",
    "build_runtime_index",
    "build_runtime_aggregate_index",
    "route_query",
]
