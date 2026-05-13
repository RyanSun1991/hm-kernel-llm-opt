"""Code-indexing backends.

Each backend implements the `CodeIndexBackend` Protocol defined in
`hmopt.indexing.models` and produces a unified `CodeIndex` containing
`CodeChunk` / `CodeRelation` records. Both backends feed the same Neo4j +
LlamaIndex vector store via `llamaindex_pipeline.build_kernel_index`.

Available backends:

- `ClangdBackend` — wraps the existing clangd LSP pipeline (Phase 2).
- `ScipClangBackend` — wraps the scip-clang batch indexer (Phase 3, planned).

See `docs/scip_clang_integration_plan.md` for the full rollout plan.
"""

from .clangd import ClangdBackend

__all__ = ["ClangdBackend"]
