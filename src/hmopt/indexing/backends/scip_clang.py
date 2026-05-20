"""ScipClangBackend — invokes scip-clang and parses its .scip output into the
unified `CodeIndex` model.

Runtime requirements:
- The `scip-clang` binary on PATH (or supplied as `binary=...`)
- The `protobuf` Python package (declared in pyproject.toml)

The pure SCIP↔project translation lives in `_scip_translation.py` and is
unit-tested in isolation. This module glues those helpers to the protobuf-
decoded objects, runs the subprocess, and assembles a `CodeIndex`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Optional

from ..models import CodeChunk, CodeIndex, CodeRelation, FileSummary

# Absolute submodule import (NOT `from . import _scip_translation`):
# when `backends/__init__.py` re-exports ScipClangBackend, this module is
# loaded while the `backends` package is still partially initialized.
# Under Python 3.11+ the relative-attribute form raises
#   ImportError: cannot import name '_scip_translation' from partially
#   initialized module 'hmopt.indexing.backends'
# because `getattr` on the in-flight package can't yet see the sibling.
# Importing the submodule by its fully qualified name goes through
# sys.modules directly and is safe even mid-init.
import hmopt.indexing.backends._scip_translation as tx

logger = logging.getLogger(__name__)


@dataclass
class ScipClangConfig:
    """Configuration for ScipClangBackend.

    Wired into the application config in Phase 4; for now constructed
    directly by callers / tests.
    """

    binary: str = "scip-clang"
    timeout_s: int = 7200
    extra_args: list[str] = field(default_factory=list)
    keep_scip_file: bool = False
    """If True, the intermediate `index.scip` is preserved at
    `<repo_path>/.hmopt/scip/index.scip` for debugging. Default False —
    use a tempdir and clean up."""


class ScipClangError(RuntimeError):
    """Raised when scip-clang fails to produce an index."""


class ScipClangBackend:
    """Wraps the `scip-clang` batch indexer behind the `CodeIndexBackend`
    Protocol so `llamaindex_pipeline.build_kernel_index` can dispatch
    by `config.indexing.backend`.
    """

    name: str = "scip-clang"

    def __init__(self, config: Optional[ScipClangConfig] = None) -> None:
        self.config = config or ScipClangConfig()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        repo_path: Path,
        compile_commands_dir: Optional[Path],
        config: Any = None,  # ignored; we use self.config
    ) -> CodeIndex:
        if compile_commands_dir is None:
            raise ScipClangError("scip-clang backend requires compile_commands_dir")
        compdb = compile_commands_dir / "compile_commands.json"
        if not compdb.exists():
            raise ScipClangError(f"compile_commands.json not found: {compdb}")
        if shutil.which(self.config.binary) is None:
            raise ScipClangError(
                f"scip-clang binary not on PATH (looked for {self.config.binary!r}). "
                "Install from https://github.com/sourcegraph/scip-clang/releases "
                "or switch to backend=clangd."
            )

        diagnostics: dict[str, Any] = {
            "backend_name": self.name,
            "binary": self.config.binary,
        }

        t_start = monotonic()

        # Run scip-clang into either a tempdir or a persistent .hmopt dir.
        if self.config.keep_scip_file:
            out_dir = repo_path / ".hmopt" / "scip"
            out_dir.mkdir(parents=True, exist_ok=True)
            scip_path = out_dir / "index.scip"
            self._invoke_indexer(repo_path, compdb, scip_path, diagnostics)
            index = self._parse_scip(scip_path, repo_path, diagnostics)
        else:
            with tempfile.TemporaryDirectory(prefix="hmopt-scip-") as tmp:
                scip_path = Path(tmp) / "index.scip"
                self._invoke_indexer(repo_path, compdb, scip_path, diagnostics)
                index = self._parse_scip(scip_path, repo_path, diagnostics)

        diagnostics["duration_s"] = round(monotonic() - t_start, 3)
        diagnostics["chunks_emitted"] = len(index.chunks)
        diagnostics["relations_emitted"] = len(index.relations)
        index.diagnostics.update(diagnostics)

        logger.info(
            "ScipClangBackend.build complete: chunks=%d relations=%d duration_s=%.2f "
            "failed_tus=%d",
            len(index.chunks),
            len(index.relations),
            diagnostics["duration_s"],
            diagnostics.get("failed_tu_count", 0),
        )
        return index

    # ------------------------------------------------------------------
    # Internal — subprocess invocation
    # ------------------------------------------------------------------

    def _invoke_indexer(
        self,
        repo_path: Path,
        compdb: Path,
        output_path: Path,
        diagnostics: dict[str, Any],
    ) -> None:
        cmd = [
            self.config.binary,
            f"--compdb-path={compdb}",
            # scip-clang's output flag is `--index-output-path`, NOT `-o`.
            # See https://github.com/sourcegraph/scip-clang `--help` —
            # passing `-o ...` (or `-o=...`) is rejected with
            # `error: unknown argument(s) ["-o", ...]` and scip-clang
            # exits non-zero without producing any output.
            f"--index-output-path={output_path}",
            *self.config.extra_args,
        ]
        logger.info("Running scip-clang: cwd=%s cmd=%s", repo_path, " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                timeout=self.config.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ScipClangError(
                f"scip-clang exceeded timeout of {self.config.timeout_s}s"
            ) from exc

        stderr_text = proc.stderr.decode("utf-8", errors="replace")
        # scip-clang skips TUs that fail to parse; capture those lines for
        # diagnostics but do not treat non-zero exit as fatal unless the
        # output file is missing.
        failed_tus: list[str] = []
        for line in stderr_text.splitlines():
            lower = line.lower()
            if "failed" in lower or "error:" in lower:
                failed_tus.append(line.strip())
        diagnostics["failed_tu_count"] = len(failed_tus)
        diagnostics["failed_tus"] = failed_tus[:50]
        diagnostics["scip_clang_exit_code"] = proc.returncode

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise ScipClangError(
                f"scip-clang produced no output (exit={proc.returncode}). "
                f"stderr tail: {stderr_text[-2000:]}"
            )

    # ------------------------------------------------------------------
    # Internal — SCIP parsing into CodeIndex
    # ------------------------------------------------------------------

    def _parse_scip(
        self,
        scip_path: Path,
        repo_path: Path,
        diagnostics: dict[str, Any],
    ) -> CodeIndex:
        # Lazy import so the project loads without protobuf at runtime when
        # this backend isn't being used.
        from .._generated import scip_pb2

        index_pb = scip_pb2.Index()
        with open(scip_path, "rb") as fh:
            index_pb.ParseFromString(fh.read())

        diagnostics["scip_file_bytes"] = scip_path.stat().st_size
        diagnostics["documents_total"] = len(index_pb.documents)

        # ----- Pass 1: build a global symbol → (path, line, qualname, kind)
        # table by walking all definition occurrences across all Documents.
        # This lets pass 2 produce correct dst_id values for cross-TU refs.
        global_defs = _build_global_definition_index(index_pb, scip_pb2)
        diagnostics["definitions_total"] = len(global_defs)

        all_chunks: list[CodeChunk] = []
        all_relations: list[CodeRelation] = []
        file_summaries: list[FileSummary] = []

        # ----- Pass 2: walk each Document, emitting chunks for definitions
        # and relations for references, with cross-TU lookups against
        # global_defs.
        for doc in index_pb.documents:
            doc_chunks, doc_relations = _process_document(
                doc=doc,
                repo_path=repo_path,
                scip_pb2=scip_pb2,
                global_defs=global_defs,
            )
            all_chunks.extend(doc_chunks)
            all_relations.extend(doc_relations)

            # File summary mirrors the clangd backend's output for downstream
            # vector-store compatibility.
            file_path = repo_path / doc.relative_path
            if file_path.exists():
                try:
                    text = (
                        doc.text
                        if doc.text
                        else file_path.read_text(encoding="utf-8", errors="ignore")
                    )
                except OSError:
                    text = doc.text or ""
                file_summaries.append(FileSummary(path=file_path, text=text))

        return CodeIndex(
            chunks=all_chunks,
            relations=all_relations,
            file_summaries=file_summaries,
            relation_summaries=[],
            diagnostics={},
        )


# ---------------------------------------------------------------------------
# Helpers (module-level so they're cheap to test and reuse)
# ---------------------------------------------------------------------------


@dataclass
class _GlobalDefRecord:
    """Cross-TU info about where a SCIP symbol is defined."""

    relative_path: str
    start_line: int  # 0-based
    qualname: str
    kind: str


def _build_global_definition_index(index_pb, scip_pb2) -> dict[str, _GlobalDefRecord]:
    """Walk all Documents and collect a global map from SCIP symbol string to
    its canonical definition site (path + line + qualname + kind).

    When a symbol is defined in multiple Documents (well-behaved headers), the
    first one wins — scip-clang already deduplicates these, so we just take
    whatever comes first in the index.
    """
    global_defs: dict[str, _GlobalDefRecord] = {}
    for doc in index_pb.documents:
        info_by_symbol = {info.symbol: info for info in doc.symbols}
        for occ in doc.occurrences:
            if not (occ.symbol_roles & scip_pb2.Definition):
                continue
            if occ.symbol in global_defs:
                continue
            rng = tx.normalize_range(occ.range)
            if rng is None:
                continue
            parsed = tx.parse_scip_symbol(occ.symbol)
            if parsed is None or parsed.is_local:
                continue
            info = info_by_symbol.get(occ.symbol)
            kind_id = int(info.kind) if info else 0
            kind = (
                tx.scip_symbol_kind_to_node_kind(kind_id) if kind_id else parsed.kind
            )
            global_defs[occ.symbol] = _GlobalDefRecord(
                relative_path=doc.relative_path,
                start_line=rng[0],
                qualname=parsed.qualname,
                kind=kind,
            )
    return global_defs


def _process_document(
    *,
    doc,
    repo_path: Path,
    scip_pb2,
    global_defs: dict[str, _GlobalDefRecord],
) -> tuple[list[CodeChunk], list[CodeRelation]]:
    info_by_symbol = {info.symbol: info for info in doc.symbols}

    # Read source text once for chunk body extraction.
    source_lines: list[str] = []
    if doc.text:
        source_lines = doc.text.splitlines()
    else:
        file_path = repo_path / doc.relative_path
        if file_path.exists():
            try:
                source_lines = file_path.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
            except OSError:
                source_lines = []

    # ----- Pass 1 (within this doc): definition occurrences → chunks +
    # def_extents (for containment scan in pass 2).
    chunks: list[CodeChunk] = []
    def_extents: list[tx.DefinitionExtent] = []
    for occ in doc.occurrences:
        if not (occ.symbol_roles & scip_pb2.Definition):
            continue
        chunk_and_extent = _build_chunk(
            occ=occ,
            doc=doc,
            scip_pb2=scip_pb2,
            info_by_symbol=info_by_symbol,
            source_lines=source_lines,
        )
        if chunk_and_extent is None:
            continue
        chunk, extent = chunk_and_extent
        chunks.append(chunk)
        if extent is not None:
            def_extents.append(extent)
    def_extents.sort(key=lambda d: d.extent)

    # ----- Pass 2 (within this doc): reference occurrences → relations.
    relations: list[CodeRelation] = []
    for occ in doc.occurrences:
        if occ.symbol_roles & scip_pb2.Definition:
            continue
        relation = _build_relation(
            occ=occ,
            doc=doc,
            scip_pb2=scip_pb2,
            def_extents=def_extents,
            info_by_symbol=info_by_symbol,
            global_defs=global_defs,
        )
        if relation is not None:
            relations.append(relation)

    return chunks, relations


def _build_chunk(
    *,
    occ,
    doc,
    scip_pb2,
    info_by_symbol: dict,
    source_lines: list[str],
) -> Optional[tuple[CodeChunk, Optional[tx.DefinitionExtent]]]:
    parsed = tx.parse_scip_symbol(occ.symbol)
    if parsed is None or parsed.is_local:
        return None
    if not parsed.descriptors:
        return None

    info = info_by_symbol.get(occ.symbol)
    kind_id = int(info.kind) if info else 0
    kind = tx.scip_symbol_kind_to_node_kind(kind_id) if kind_id else parsed.kind

    rng = tx.normalize_range(occ.range)
    if rng is None:
        return None
    enc = tx.normalize_range(occ.enclosing_range) if occ.enclosing_range else rng

    start_line_0, start_col_0, _, _ = rng
    if enc is not None:
        _, _, end_line_0, _ = enc
    else:
        end_line_0 = start_line_0

    body_lines = source_lines[start_line_0 : end_line_0 + 1] if source_lines else []
    text = "\n".join(body_lines)

    qualname = parsed.qualname
    name = parsed.descriptors[-1].name if parsed.descriptors else qualname
    relative_path = doc.relative_path
    symbol_id = f"{relative_path}:{qualname}:{start_line_0 + 1}:{kind}"

    documentation = list(info.documentation) if info else None
    signature = None
    if info and info.HasField("signature_documentation"):
        signature = info.signature_documentation.text or None

    chunk = CodeChunk(
        path=Path(relative_path),
        symbol_id=symbol_id,
        symbol_name=name,
        symbol_qualname=qualname,
        kind=kind,
        start_line=start_line_0 + 1,
        end_line=end_line_0 + 1,
        text=text,
        parser="scip-clang",
        detail=None,
        container=None,
        backend_origin="scip-clang",
        scip_symbol=occ.symbol,
        documentation=documentation if documentation else None,
        signature=signature,
        is_forward_decl=bool(occ.symbol_roles & scip_pb2.ForwardDefinition),
        is_generated=bool(occ.symbol_roles & scip_pb2.Generated),
    )
    extent = (
        tx.DefinitionExtent(symbol=occ.symbol, extent=enc) if enc is not None else None
    )
    return chunk, extent


def _build_relation(
    *,
    occ,
    doc,
    scip_pb2,
    def_extents: list[tx.DefinitionExtent],
    info_by_symbol: dict,
    global_defs: dict[str, _GlobalDefRecord],
) -> Optional[CodeRelation]:
    if not occ.symbol:
        return None

    # Parse the destination symbol up front so we can also use it as the
    # source of truth for relation-kind inference when syntax_kind is
    # unspecified.
    dst_parsed = tx.parse_scip_symbol(occ.symbol)
    if dst_parsed is None or dst_parsed.is_local:
        return None

    rel_kind = tx.scip_syntax_kind_to_relation_kind(occ.syntax_kind)
    if rel_kind is None:
        # Only fall back when scip-clang left syntax_kind unspecified (=0).
        # Other table misses (Comment=1, PunctuationDelimiter=2, ...) are
        # explicit "this is not a code reference" signals from scip-clang
        # and must be respected — otherwise we'd manufacture spurious calls
        # edges from incidental occurrences of identifier-shaped tokens.
        #
        # scip-clang 0.3.1 + BiSheng emits most genuine reference
        # occurrences with syntax_kind=UnspecifiedSyntaxKind(0); without
        # this fallback ~80% of refs silently fail to become graph edges.
        if occ.syntax_kind != 0:
            return None
        rel_kind = tx.infer_relation_kind_from_descriptor(dst_parsed)
        if rel_kind is None:
            return None

    occ_range = tx.normalize_range(occ.range)
    if occ_range is None:
        return None

    src_def = tx.find_enclosing_definition(occ_range, def_extents)
    if src_def is None:
        # File-scope reference (e.g. at top of file before any function body).
        # We don't have a src symbol to attach the edge to; skip.
        return None

    src_record = global_defs.get(src_def.symbol)
    dst_record = global_defs.get(occ.symbol)

    src_parsed = tx.parse_scip_symbol(src_def.symbol)
    if src_parsed is None or src_parsed.is_local:
        return None

    # src is always defined in this doc (def_extents only holds local defs).
    src_path = src_record.relative_path if src_record else doc.relative_path
    src_kind = src_record.kind if src_record else src_parsed.kind
    src_line = (src_record.start_line + 1) if src_record else (src_def.extent[0] + 1)
    src_id = f"{src_path}:{src_parsed.qualname}:{src_line}:{src_kind}"
    src_name = src_parsed.descriptors[-1].name if src_parsed.descriptors else ""

    # dst: prefer global lookup; fall back to local info; mark external when
    # no definition is known anywhere in the index.
    if dst_record is not None:
        dst_id = (
            f"{dst_record.relative_path}:{dst_parsed.qualname}:"
            f"{dst_record.start_line + 1}:{dst_record.kind}"
        )
        dst_path = dst_record.relative_path
        dst_kind = dst_record.kind
    else:
        dst_info = info_by_symbol.get(occ.symbol)
        if dst_info is not None and dst_info.kind:
            dst_kind = tx.scip_symbol_kind_to_node_kind(int(dst_info.kind))
        else:
            dst_kind = dst_parsed.kind
        dst_id = f"external::{occ.symbol}"
        dst_path = None
    dst_name = dst_parsed.descriptors[-1].name if dst_parsed.descriptors else ""

    return CodeRelation(
        src_id=src_id,
        dst_id=dst_id,
        kind=rel_kind,
        src_name=src_name,
        dst_name=dst_name,
        src_kind=src_kind,
        dst_kind=dst_kind,
        src_path=src_path,
        dst_path=dst_path,
        backend_origin="scip-clang",
        call_site_path=doc.relative_path,
        call_site_line=occ_range[0] + 1,
        call_site_col=occ_range[1] + 1,
        syntax_kind=occ.syntax_kind,
        role_bits=occ.symbol_roles,
        is_write=bool(occ.symbol_roles & scip_pb2.WriteAccess),
    )
