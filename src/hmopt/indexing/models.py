"""Unified data model for code-indexing backends.

Both the clangd-based backend (existing) and the scip-clang backend (planned)
emit instances of these dataclasses. Backend-specific signal lives in Optional
fields that one backend leaves None — the downstream Neo4j upsert path only
writes a property when the field is non-None, so the graph stays clean.

The `backend_origin` field on every record tags the producing backend; this
makes A/B comparison and mixed-backend graphs auditable.

See docs/scip_clang_integration_plan.md for the full schema rationale and
phase-by-phase rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Node-bearing dataclass — becomes a `:symbol` (or `:external`) node in Neo4j
# and a TextNode in the LlamaIndex vector store.
# ---------------------------------------------------------------------------


@dataclass
class CodeChunk:
    """A code symbol together with its source body.

    Core fields are filled by every backend. Optional fields after `container`
    are scip-clang-exclusive at the moment; the clangd backend leaves them
    None. Adding a new Optional field here is a backward-compatible schema
    extension — existing consumers ignore unknown attrs and Neo4j upsert
    skips None values.
    """

    # === Core (every backend fills these) ===
    path: Path
    symbol_id: str
    symbol_name: str
    symbol_qualname: str
    kind: str
    start_line: int
    end_line: int
    text: str
    parser: str
    detail: Optional[str] = None
    container: Optional[str] = None

    # === Backend provenance ===
    backend_origin: str = "clangd"

    # === scip-clang-exclusive (Optional; None when not applicable) ===
    scip_symbol: Optional[str] = None
    """Canonical SCIP symbol descriptor, e.g. ``cxx . . . path/foo.c/bar().``."""

    documentation: Optional[list[str]] = None
    """SCIP SymbolInformation.documentation strings."""

    signature: Optional[str] = None
    """Full signature with parameters from SCIP; richer than `detail`."""

    is_forward_decl: Optional[bool] = None
    """True if only forward declarations were observed for this symbol."""

    is_generated: Optional[bool] = None
    """From SymbolRole.Generated bit."""

    header_origin_tu: Optional[str] = None
    """Which translation unit first indexed this header symbol (cross-TU dedup)."""


# ---------------------------------------------------------------------------
# Edge dataclass — becomes a relationship in Neo4j.
# ---------------------------------------------------------------------------


@dataclass
class CodeRelation:
    """A directed relationship between two symbols.

    Same Optional-extension contract as CodeChunk: scip-only fields default
    to None and are skipped at upsert time when None.
    """

    # === Core (every backend fills these) ===
    src_id: str
    dst_id: str
    kind: str  # calls | uses_type | uses_macro | contains | uses_field | implements
    src_name: str
    dst_name: str
    src_kind: str
    dst_kind: str
    src_path: Optional[str] = None
    dst_path: Optional[str] = None

    # === Backend provenance ===
    backend_origin: str = "clangd"

    # === scip-clang-exclusive: per-occurrence call-site precision ===
    call_site_path: Optional[str] = None
    """File where this reference occurs (usually equal to src_path, but may
    differ for references inside an inlined header)."""

    call_site_line: Optional[int] = None
    """1-based line number of the occurrence."""

    call_site_col: Optional[int] = None
    """1-based column number of the occurrence."""

    # === scip-clang-exclusive: occurrence semantics ===
    syntax_kind: Optional[int] = None
    """SCIP SyntaxKind enum value; distinguishes function call from function
    pointer take, identifier from punctuation, etc."""

    role_bits: Optional[int] = None
    """SCIP SymbolRole bit field: Definition / Reference / WriteAccess /
    ReadAccess / Import / ForwardDefinition / Generated / Test."""

    is_write: Optional[bool] = None
    """True if the WriteAccess role bit is set."""

    occurrence_count: Optional[int] = None
    """How many occurrences of (src, dst, kind) live in the same src body.
    Useful for ranking repeat-callers."""


# ---------------------------------------------------------------------------
# Aggregate types — kept identical to the existing CodeIndex shape so the
# llamaindex_pipeline.py consumer is byte-compatible. A new `diagnostics`
# field is the only schema addition.
# ---------------------------------------------------------------------------


@dataclass
class FileSummary:
    path: Path
    text: str


@dataclass
class RelationSummary:
    symbol_id: str
    symbol_name: str
    symbol_kind: str
    path: str
    text: str


@dataclass
class CodeIndex:
    """The unified output of a code-indexing backend.

    This is what `CodeIndexBackend.build()` returns and what
    `llamaindex_pipeline.build_kernel_index()` feeds to Neo4j + the
    LlamaIndex vector store.
    """

    chunks: list[CodeChunk]
    relations: list[CodeRelation]
    file_summaries: list[FileSummary]
    relation_summaries: list[RelationSummary]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    """Free-form backend telemetry: backend_name, files_indexed, duration_s,
    fallback_invoked, failed_tus, etc. Logged but not persisted to Neo4j."""


# ---------------------------------------------------------------------------
# Backend contract — both ClangdBackend and ScipClangBackend implement this.
# ---------------------------------------------------------------------------


@runtime_checkable
class CodeIndexBackend(Protocol):
    """Protocol every code-index backend must satisfy.

    Implementations:
      - hmopt.indexing.backends.clangd.ClangdBackend
      - hmopt.indexing.backends.scip_clang.ScipClangBackend  (Phase 3)
    """

    name: str
    """Stable backend identifier, e.g. "clangd" or "scip-clang"."""

    def build(
        self,
        *,
        repo_path: Path,
        compile_commands_dir: Path,
        config: Any,
    ) -> CodeIndex:
        """Run indexing and return a populated CodeIndex.

        Parameters
        ----------
        repo_path:
            Root of the source tree being indexed.
        compile_commands_dir:
            Directory containing `compile_commands.json`.
        config:
            Backend-specific config object (e.g. ClangdConfig, ScipClangConfig).

        Returns
        -------
        A CodeIndex whose chunks / relations carry the appropriate
        `backend_origin` tag. Optional scip-only fields are populated when
        the backend has the relevant data; otherwise they remain None.
        """
        ...


__all__ = [
    "CodeChunk",
    "CodeIndex",
    "CodeIndexBackend",
    "CodeRelation",
    "FileSummary",
    "RelationSummary",
]
