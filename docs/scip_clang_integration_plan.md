# scip-clang Integration Plan

## Goal

Add scip-clang as an alternative code-index backend alongside the existing clangd-based pipeline. Both backends feed the same unified data model (`CodeNode` / `CodeRelation`) consumed by Neo4j + LlamaIndex Vector Store. The clangd pipeline stays intact; scip-clang is selectable at index time via config or CLI flag.

## Design Principles

1. **Zero break** — every existing CLI / MCP / pipeline run keeps working with `backend = clangd` (the default). No call-site outside `src/hmopt/indexing/` needs to change.
2. **Unified data model** — both backends emit the same `CodeNode` / `CodeRelation` dataclasses. scip-clang's richer signal (per-occurrence call-site, role bits, syntax kind, documentation) lives in **Optional fields** that clangd backend leaves `None`. Neo4j only writes the property when the dataclass field is non-None — no null bloat.
3. **Symbol ID compatibility** — scip-clang backend translates SCIP symbol descriptors into the existing canonical `path:qualname:line:kind` form. Downstream queries are unaware which backend produced the index.
4. **Forward compatibility** — every record gets a `backend_origin` tag so A/B comparison, audit, and selective re-index by backend are trivial. Cross-backend MERGE in Neo4j is `COALESCE`-friendly so a scip-clang re-index over a clangd graph upgrades records without losing data.

## Target Data Flow

```
                        ┌──────────────────┐
compile_commands.json ──┤ ClangdBackend    │──┐
(+ regex fallback)      └──────────────────┘  │
                                              ├──► build_kernel_index() in
                                              │    llamaindex_pipeline.py
                        ┌──────────────────┐  │      ├─ pick backend by cfg/flag
compile_commands.json ──┤ ScipClangBackend │──┘      ├─ collect CodeIndexBuildResult
                        │  + scip_pb2 parse│         ├─ existing Neo4j upsert path
                        └──────────────────┘         └─ existing Vector upsert path
```

`build_kernel_index()` becomes the unified provider: it picks a backend, calls `backend.build()`, gets back `(CodeNode[], CodeRelation[], diagnostics)`, and feeds the existing upsert routines. No new orchestrator class is needed — the existing function-level dispatch IS the unified provider.

---

## Phase 0 — Scaffolding & Plan (this phase)

### Deliverables

- `docs/scip_clang_integration_plan.md` — this document
- `third_party/scip/scip.proto` — vendored from `sourcegraph/scip@main`
- `scripts/gen_scip_pb2.sh` — protoc invocation for generating Python bindings
- `pyproject.toml` — add `protobuf>=4.25` as a runtime dep (cheap; even clangd-only installs pull it)
- Documentation note on installing `protoc` and `scip-clang` binary (Phase 3 prerequisite)

### Non-goals for Phase 0

- Do NOT generate `_generated/scip_pb2.py` yet. Generation requires `protoc`; this is a build-time concern handled in Phase 3 when the user is also installing `scip-clang`. The script makes the generation reproducible.

### Exit criteria

- `cat third_party/scip/scip.proto | wc -l` ≥ 800
- `bash scripts/gen_scip_pb2.sh --check` reports protoc missing gracefully (does not crash)
- existing test suite passes (no behavior change yet)

---

## Phase 1 — Unified Data Model (`models.py`)

### Deliverables

- New file `src/hmopt/indexing/models.py` containing the dataclasses listed below.
- `clangd_indexer.py` rewired to import its dataclasses from `models.py` and re-export them for backward compat.
- `llamaindex_pipeline.py` import path (`from hmopt.indexing.clangd_indexer import CodeIndex, CodeChunk, index_kernel_code`) continues to work unchanged.

### Unified schema (the key Phase 1 contract)

#### `CodeNode` — Neo4j `:symbol` / `:external` node

| Field | Type | clangd | scip-clang | Notes |
|---|---|---|---|---|
| `symbol_id` | str | ✓ | ✓ | canonical `path:qualname:line:kind` |
| `name` | str | ✓ | ✓ | |
| `qualname` | str | ✓ | ✓ | |
| `kind_id` | int | ✓ LSP SymbolKind | ✓ mapped from SCIP `SymbolKind` | |
| `kind` | str | ✓ | ✓ | function / type / macro / field / ... |
| `path` | Path | ✓ | ✓ | |
| `start_line` | int | ✓ | ✓ | |
| `end_line` | int | ✓ | ✓ | |
| `start_char` | int | ✓ | ✓ | |
| `selection_line` | int | ✓ | ✓ | |
| `selection_char` | int | ✓ | ✓ | |
| `detail` | Optional[str] | ✓ | ✓ | |
| `container` | Optional[str] | ✓ | ✓ | |
| `backend_origin` | str | "clangd" | "scip-clang" | **NEW**, default `"clangd"` |
| `scip_symbol` | Optional[str] | None | ✓ | scip-only, e.g. `cxx . . . path/foo.c/bar().` |
| `documentation` | Optional[list[str]] | None | ✓ | scip-only, `SymbolInformation.documentation` |
| `signature` | Optional[str] | None | ✓ | scip-only |
| `is_forward_decl` | Optional[bool] | None | ✓ | scip-only |
| `is_generated` | Optional[bool] | None | ✓ | scip-only, from `SymbolRole.Generated` |
| `header_origin_tu` | Optional[str] | None | ✓ | scip-only, cross-TU dedup metadata |

#### `CodeRelation` — Neo4j edge

| Field | Type | clangd | scip-clang | Notes |
|---|---|---|---|---|
| `src_id` | str | ✓ | ✓ | |
| `dst_id` | str | ✓ | ✓ | |
| `kind` | str | ✓ | ✓ | calls / uses_type / uses_macro / contains / uses_field / implements |
| `src_name` | str | ✓ | ✓ | |
| `dst_name` | str | ✓ | ✓ | |
| `src_kind` | str | ✓ | ✓ | |
| `dst_kind` | str | ✓ | ✓ | |
| `src_path` | Optional[str] | ✓ | ✓ | |
| `dst_path` | Optional[str] | ✓ | ✓ | |
| `backend_origin` | str | "clangd" | "scip-clang" | **NEW**, default `"clangd"` |
| `call_site_path` | Optional[str] | None | ✓ | **scip-only**, most important new field |
| `call_site_line` | Optional[int] | None | ✓ | **scip-only** |
| `call_site_col` | Optional[int] | None | ✓ | **scip-only** |
| `syntax_kind` | Optional[int] | None | ✓ | scip-only, distinguishes call vs function-pointer take |
| `role_bits` | Optional[int] | None | ✓ | scip-only, `SymbolRole` bitfield |
| `is_write` | Optional[bool] | None | ✓ | scip-only, from `WriteAccess` role |
| `occurrence_count` | Optional[int] | None | ✓ | scip-only, repeat-call count in same body |

#### `CodeIndexBackend` Protocol

```python
class CodeIndexBackend(Protocol):
    name: str  # "clangd" | "scip-clang"

    def build(
        self,
        repo_path: Path,
        compile_commands_dir: Path,
        config,  # backend-specific config object
    ) -> CodeIndexBuildResult: ...
```

#### `CodeIndexBuildResult` dataclass

```python
@dataclass
class CodeIndexBuildResult:
    nodes: list[CodeNode]
    relations: list[CodeRelation]
    chunks: list[CodeChunk]                # text bodies for vector embed
    file_summaries: list[FileSummary]
    relation_summaries: list[RelationSummary]
    diagnostics: dict[str, Any]            # backend_name, files_indexed, duration_s,
                                           # fallback_invoked, failed_tus, ...
```

The first three fields (`nodes`, `relations`, plus the existing companions `chunks` / `file_summaries` / `relation_summaries`) align with what `clangd_indexer.CodeIndex` already produces today — Phase 1 simply renames/repurposes that dataclass and surfaces it through the Backend protocol.

### Exit criteria

- `python -c "from hmopt.indexing.models import CodeNode, CodeRelation, CodeIndexBackend, CodeIndexBuildResult"` works
- `python -c "from hmopt.indexing.clangd_indexer import CodeIndex, CodeChunk, index_kernel_code"` still works (re-export shim)
- existing pytest suite passes
- no callsite in `src/` or `tests/` was modified except `clangd_indexer.py` itself

---

## Phase 2 — `ClangdBackend` adapter

### Deliverables

- `src/hmopt/indexing/backends/__init__.py`
- `src/hmopt/indexing/backends/clangd.py`:
  - `class ClangdBackend` implements `CodeIndexBackend`
  - `.build(repo_path, compile_commands_dir, clangd_config)` delegates to existing `index_kernel_code(...)`
  - Wraps the returned `CodeIndex` dataclass into a `CodeIndexBuildResult` with:
    - `backend_origin="clangd"` on every node and relation
    - all scip-only fields left as `None`
    - `diagnostics={"backend_name": "clangd", "files_indexed": N, "fallback_invoked": bool, ...}`

### Exit criteria

- `from hmopt.indexing.backends.clangd import ClangdBackend` works
- `ClangdBackend(...).build(...)` on a small fixture returns a `CodeIndexBuildResult` whose `nodes` / `relations` content matches direct `index_kernel_code(...)` output (byte-for-byte on the core fields)
- existing pytest suite passes

---

## Phase 3 — `ScipClangBackend` (next milestone, not in this commit)

### Deliverables (preview)

- `src/hmopt/indexing/_generated/scip_pb2.py` — generated from `third_party/scip/scip.proto`
- `src/hmopt/indexing/backends/scip_clang.py`:
  - `ScipClangBackend` runs `scip-clang --compdb-path=... -o index.scip` as a subprocess
  - parses `index.scip` via `scip_pb2`
  - for each `Document`:
    - definition occurrences → `CodeNode` (with scip-only fields filled)
    - reference occurrences → `CodeRelation` (with `call_site_*` filled)
    - containment derived via interval-overlap of definition ranges
  - SCIP symbol descriptor → canonical `path:qualname:line:kind` (translation function isolated and unit-tested)
- backend selection in `llamaindex_pipeline.build_kernel_index()` via `config.indexing.backend`
- CLI flag `hmopt index-kernel --backend scip-clang`
- Neo4j upsert path updated to conditionally write the new Optional properties

### Exit criteria (preview)

- On a small C fixture, scip-clang backend produces ≥80% the same `(src_id, dst_id, kind)` tuples as the clangd backend
- `call_site_line` non-None coverage ≥ 95% on the scip-clang output
- A/B verification doc at `docs/scip_clang_eval.md`

---

## Phase 4 — Query-side enrichment (`kernel_call_chain`)

- `retrieve_call_chain` in `llamaindex_pipeline.py:2192-2502` returns `call_site_path` / `call_site_line` per edge when available
- `kernel_call_chain` MCP tool response schema extended
- `kernel-source-research` agent spec's promised `call_site_path:call_site_line` becomes truthful

## Phase 5 — Production rollout

- Run scip-clang on `memmgr_reclaim` and `hyperhold_io` targets
- Compare Neo4j edge counts, call-site coverage, index duration
- If favorable: switch default `backend` in `configs/app.yaml` to `scip-clang`
- Keep clangd as the documented fallback for kernel < v5.10 or environments without `scip-clang` binary

---

## File-by-file changeset summary

| Path | Phase | Operation | Description |
|---|---|---|---|
| `third_party/scip/scip.proto` | 0 | new | vendored from sourcegraph/scip |
| `scripts/gen_scip_pb2.sh` | 0 | new | protoc → `_generated/scip_pb2.py` |
| `docs/scip_clang_integration_plan.md` | 0 | new | this plan |
| `pyproject.toml` | 0 | edit | add `protobuf>=4.25` |
| `src/hmopt/indexing/models.py` | 1 | new | unified dataclasses + Backend Protocol |
| `src/hmopt/indexing/clangd_indexer.py` | 1 | edit | move dataclasses to models, re-export |
| `src/hmopt/indexing/backends/__init__.py` | 2 | new | empty package marker |
| `src/hmopt/indexing/backends/clangd.py` | 2 | new | `ClangdBackend` adapter |
| `src/hmopt/indexing/_generated/scip_pb2.py` | 3 | new (generated) | protobuf bindings |
| `src/hmopt/indexing/backends/scip_clang.py` | 3 | new | `ScipClangBackend` |
| `src/hmopt/indexing/llamaindex_pipeline.py` | 3 | edit | backend dispatch in `build_kernel_index` + Neo4j upsert extensions |
| `src/hmopt/core/config.py` | 3 | edit | `IndexingConfig.backend` + `ScipClangConfig` |
| `src/hmopt/cli.py` | 3 | edit | `index-kernel --backend` flag |
| `configs/app.yaml` | 3 | edit | `indexing.backend` + `indexing.scip_clang` block |
| `src/hmopt/api/mcp_service.py` | 4 | edit | `kernel_call_chain` response carries call-site |

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| scip-clang Beta, some TUs may crash | scip-clang skips crashed TUs and continues; record failed TUs in diagnostics |
| Kernel < v5.10 not supported by scip-clang | `ScipClangBackend.build()` pre-flight checks kernel version; falls back error message points user back to clangd |
| `.scip` file at kernel scale (~375 MB) — Python protobuf slow | Phase 3 uses length-delimited streaming read of the `Index` message |
| SCIP symbol descriptor → canonical ID translation incomplete | isolate translation in a pure function, unit-test on edge cases (anonymous namespaces, template instances, static-with-same-name-across-TUs) |
| Mixing backends in one Neo4j graph causes duplicate edges | upsert uses `MERGE ... ON MATCH SET COALESCE(...)`; `backend_origin` tags every record |
| protoc not installed in user env | `gen_scip_pb2.sh` checks for protoc, prints install instructions; Phase 3 will document `apt install protobuf-compiler` / `brew install protobuf` |
