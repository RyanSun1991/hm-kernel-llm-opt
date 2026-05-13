"""ClangdBackend — adapts the existing clangd LSP indexer to the unified
`CodeIndexBackend` Protocol.

This is a thin shim: all real indexing work still happens in
`hmopt.indexing.clangd_indexer.index_kernel_code`. The backend wrapper exists
so `llamaindex_pipeline.build_kernel_index` can dispatch by backend name
without caring which implementation actually ran.

`backend_origin` tagging is automatic — `CodeChunk` and `CodeRelation` in
`models.py` default the field to `"clangd"`, so every record emitted by the
clangd path is correctly tagged without changes to `clangd_indexer.py`.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import Optional

from ..clangd_client import ClangdConfig
from ..clangd_indexer import index_kernel_code
from ..models import CodeIndex

logger = logging.getLogger(__name__)


class ClangdBackend:
    """Wraps `clangd_indexer.index_kernel_code` as a `CodeIndexBackend`.

    Parameters
    ----------
    max_files:
        Cap on number of compile-commands entries to index per run.
        Matches the existing default in `index_kernel_code`.
    """

    name: str = "clangd"

    def __init__(self, *, max_files: int = 5000) -> None:
        self.max_files = max_files

    def build(
        self,
        *,
        repo_path: Path,
        compile_commands_dir: Optional[Path],
        config: Optional[ClangdConfig],
    ) -> CodeIndex:
        """Run the clangd LSP indexer and return a populated `CodeIndex`.

        `compile_commands_dir` is allowed to be None when `config` already
        carries a value (legacy callers). When both are provided and differ,
        the explicit argument wins — this matches the dispatch pattern that
        the future scip-clang backend will use.
        """

        effective_config: Optional[ClangdConfig]
        if config is not None and compile_commands_dir is not None:
            effective_config = replace(
                config, compile_commands_dir=str(compile_commands_dir)
            )
        else:
            effective_config = config

        t_start = monotonic()
        index = index_kernel_code(
            repo_path,
            clangd_config=effective_config,
            max_files=self.max_files,
        )
        duration_s = monotonic() - t_start

        # Populate diagnostics. `backend_origin="clangd"` is already on every
        # CodeChunk / CodeRelation via dataclass default; no per-record fixup
        # needed. If a future change ever moves backend_origin out of the
        # default, re-add a tagging loop here.
        fallback_invoked = (
            effective_config is None
            or not effective_config.compile_commands_dir
            or len(index.chunks) == 0
        )
        index.diagnostics.update(
            {
                "backend_name": self.name,
                "duration_s": round(duration_s, 3),
                "max_files": self.max_files,
                "chunks_emitted": len(index.chunks),
                "relations_emitted": len(index.relations),
                "fallback_invoked": fallback_invoked,
            }
        )

        logger.info(
            "ClangdBackend.build complete: chunks=%d relations=%d duration_s=%.2f "
            "fallback=%s",
            len(index.chunks),
            len(index.relations),
            duration_s,
            fallback_invoked,
        )
        return index
